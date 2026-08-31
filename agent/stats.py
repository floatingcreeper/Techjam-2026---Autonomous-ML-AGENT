"""Statistical inference over saved node predictions -- docs/SYSTEM.md §13 (statistical decision machinery).

Why this module exists
----------------------
There are THREE distinct variances in this benchmark and the old code conflated them
(docs/RESEARCH.md §4 (noise and uncertainty model)):

  1. training stochasticity     -- same cfg, same seed. EXACTLY 0.00000 for FM and LightGBM
                                   (measured over 24 and 14 nodes); sigma ~ 0.00025 for DIN (torch).
                                   Reduced by re-training under more seeds.
  2. validation-sample noise    -- the finite valid set. Paired user-bootstrap SE ~ 0.0009.
                                   NOT reduced by re-training. This is the binding constraint.
  3. cross-seed generalisation  -- the published FM test std, 0.0008.

`agent/reeval.py` attacks (1) at the cost of a full training pass per seed. This module attacks (2)
for free, from the `val_scores.npy` every node already writes. The two are complementary, never
interchangeable, and are reported separately everywhere.

Everything here is a CHEAP ANALYSIS OPERATION: it reads saved predictions and never trains.

Exactness
---------
`Evaluator` reproduces the frozen `evaluate.evaluate` semantics: Mann-Whitney U with average-rank tie
correction, positive-weighted GAUC over discriminative users only, nDCG@5 over ALL users with a stable
descending sort. It agrees with the frozen evaluator to <1e-5 (the residual is float32 accumulation
inside the frozen implementation); see tests/test_stats.py. 1e-5 is two orders of magnitude below the
smallest effect this project can resolve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UserStats:
    """Per-user decomposition of the primary metric for ONE score vector."""
    auc: np.ndarray        # (G,) per-user AUC; 0.0 where not discriminative
    ndcg: np.ndarray       # (G,) per-user nDCG@k
    npos: np.ndarray       # (G,) positives per user
    disc: np.ndarray       # (G,) bool: 0 < npos < n  (GAUC counts these only)


class Evaluator:
    """Owns the per-user grouping for one (users, labels) pair so repeated scoring is cheap.

    Two models evaluated through the SAME Evaluator are guaranteed to be compared on identical user
    groups in identical order -- which is what makes the bootstrap in `paired_bootstrap` genuinely
    paired.
    """

    def __init__(self, users, labels, k: int = 5):
        u = np.asarray(users)
        y = np.asarray(labels, np.float64)
        self.N = len(u)
        self.k = k
        # group rows by user, preserving original row order inside a user (stable) -- this matches
        # evaluate.evaluate, which builds per-user lists in row order.
        self._order = np.argsort(u, kind="stable")
        us = u[self._order]
        new = np.r_[True, us[1:] != us[:-1]]
        self._gid = np.cumsum(new) - 1                     # group id per position in sorted space
        self._gstart = np.flatnonzero(new)
        self.G = int(self._gid[-1]) + 1 if self.N else 0
        self._y = y[self._order]
        self.n = np.bincount(self._gid, minlength=self.G).astype(np.float64)
        self.npos = np.bincount(self._gid, weights=self._y, minlength=self.G)
        self.disc = (self.npos > 0) & (self.npos < self.n)
        self._dw = 1.0 / np.log2(np.arange(k) + 2.0)       # nDCG discounts
        cw = np.r_[0.0, np.cumsum(self._dw)]
        self._idcg = cw[np.minimum(self.npos, k).astype(int)]
        self._arange = np.arange(self.N)

    # ---------------------------------------------------------------- per-user decomposition
    def user_stats(self, scores) -> UserStats:
        s = np.asarray(scores, np.float64)[self._order]

        # ---- GAUC: ascending within group, average ranks over score ties (Mann-Whitney U) ----
        asc = np.lexsort((s, self._gid))
        ga, sa = self._gid[asc], s[asc]
        run = np.r_[True, (ga[1:] != ga[:-1]) | (sa[1:] != sa[:-1])]
        r1 = (self._arange - self._gstart[ga] + 1).astype(np.float64)   # 1-based rank in group
        rs = np.flatnonzero(run)
        rlen = np.diff(np.r_[rs, self.N]).astype(np.float64)
        ranks = np.repeat(np.add.reduceat(r1, rs) / rlen, rlen.astype(int))
        rpos = np.bincount(ga, weights=ranks * self._y[asc], minlength=self.G)
        nneg = self.n - self.npos
        with np.errstate(invalid="ignore", divide="ignore"):
            auc = (rpos - self.npos * (self.npos + 1.0) / 2.0) / (self.npos * nneg)
        auc = np.where(self.disc, auc, 0.0)

        # ---- nDCG@k: descending within group, ties broken by row order (stable) ----
        desc = np.lexsort((self._arange, -s, self._gid))
        gd = self._gid[desc]
        pos = self._arange - self._gstart[gd]
        top = pos < self.k
        dcg = np.bincount(gd[top], weights=self._y[desc][top] * self._dw[pos[top]], minlength=self.G)
        ndcg = np.where(self._idcg > 0, dcg / np.where(self._idcg > 0, self._idcg, 1.0), 0.0)

        return UserStats(auc=auc, ndcg=ndcg, npos=self.npos, disc=self.disc)

    # ---------------------------------------------------------------- aggregation
    @staticmethod
    def aggregate(st: UserStats, idx=None) -> tuple[float, float, float]:
        """(primary, GAUC, nDCG) over the users in `idx` (all users when None).

        `idx` may contain repeats -- that is exactly how the bootstrap resamples users.
        """
        if idx is None:
            auc, ndcg, npos, disc = st.auc, st.ndcg, st.npos, st.disc
        else:
            auc, ndcg, npos, disc = st.auc[idx], st.ndcg[idx], st.npos[idx], st.disc[idx]
        w = npos * disc
        tot = w.sum()
        gauc = float((auc * w).sum() / tot) if tot > 0 else 0.5
        nd = float(ndcg.mean()) if len(ndcg) else 0.0
        return 0.5 * (gauc + nd), gauc, nd

    def primary(self, scores) -> tuple[float, float, float]:
        return self.aggregate(self.user_stats(scores))


# ---------------------------------------------------------------------- paired bootstrap
def paired_bootstrap(ev: Evaluator, st_treat: UserStats, st_ctrl: UserStats,
                     B: int = 1000, seed: int = 0) -> dict:
    """Paired user-level bootstrap of the primary DELTA (treatment - control).

    Resamples USERS with replacement and rescores BOTH models on the same resample, so the estimate
    is of the paired difference and the strong per-user correlation between two models cancels.

    Returns delta / se / ci / p_gt0 for primary and the per-metric deltas for GAUC and nDCG (point
    estimates only -- the CI is on primary, which is what decisions are made on).

    This quantifies validation-sample uncertainty ONLY. It says nothing about training stochasticity
    (see agent/reeval.py) and nothing about cross-seed generalisation.
    """
    d0, g0, n0 = Evaluator.aggregate(st_treat)
    dc, gc, nc = Evaluator.aggregate(st_ctrl)
    G = ev.G
    if G == 0 or B <= 0:
        return {"delta_primary": d0 - dc, "boot_se": float("nan"), "p_gt0": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "B": 0,
                "delta_GAUC": g0 - gc, "delta_nDCG": n0 - nc,
                "treat_primary": d0, "ctrl_primary": dc}
    rng = np.random.default_rng(seed)
    deltas = np.empty(B, np.float64)
    for b in range(B):
        idx = rng.integers(0, G, size=G)
        deltas[b] = (Evaluator.aggregate(st_treat, idx)[0] - Evaluator.aggregate(st_ctrl, idx)[0])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_primary": float(d0 - dc),
        "delta_GAUC": float(g0 - gc),
        "delta_nDCG": float(n0 - nc),
        "boot_se": float(deltas.std(ddof=1)),
        "p_gt0": float((deltas > 0).mean()),
        "ci_lo": float(lo), "ci_hi": float(hi), "B": int(B),
        "treat_primary": float(d0), "ctrl_primary": float(dc),
    }


# ---------------------------------------------------------------------- evidence classification
CONFIRMED, PROMISING, INCONCLUSIVE, REJECTED = "confirmed", "promising", "inconclusive", "rejected"


def classify_evidence(p_gt0: float, hi: float = 0.90, lo_band: float = 0.60) -> str:
    """Map P(delta>0) onto an evidence class (docs/SYSTEM.md §13-14).

    This is EVIDENCE about an effect. It is deliberately NOT the tree's adoption status: a node can be
    `inconclusive` and still be the current champion, and a node can be `rejected` standalone and still
    earn a place in the portfolio on ensemble contribution.
    """
    eps = 1e-9                               # 1.0 - 0.90 == 0.09999999999999998 in binary float
    if p_gt0 != p_gt0:                       # NaN -- no comparison was possible
        return INCONCLUSIVE
    if p_gt0 >= hi - eps:
        return CONFIRMED
    if p_gt0 <= (1.0 - hi) + eps:
        return REJECTED
    if p_gt0 >= lo_band - eps:
        return PROMISING
    return INCONCLUSIVE


# ---------------------------------------------------------------------- rank-space helpers
def per_user_rank(scores, users) -> np.ndarray:
    """Within-user percentile rank in [0,1] -- monotone and scale-free, so models on incomparable
    score scales can be blended. Mirrors orchestrator._per_user_rank."""
    s = np.asarray(scores)
    u = np.asarray(users)
    out = np.empty(len(s), np.float32)
    order = np.argsort(u, kind="stable")
    us = u[order]
    for grp in np.split(order, np.flatnonzero(np.diff(us)) + 1):
        x = s[grp]
        out[grp] = np.argsort(np.argsort(x)) / max(1, len(x) - 1)
    return out


def rank_corr(ra, rb) -> float:
    """Correlation of two within-user percentile-rank vectors: the diversity statistic (docs/RESEARCH.md §12)."""
    a, b = np.asarray(ra, np.float64), np.asarray(rb, np.float64)
    if a.std() == 0 or b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])
