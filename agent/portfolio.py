"""Ensemble-aware node valuation and statistically honest portfolio assembly.

docs/RESEARCH.md §12 (portfolio findings) measured the thing this module exists for: in the reference run the **weakest
standalone member** (LightGBM, 0.60205, 3rd of 3) had the **largest leave-one-out contribution**
(+0.00121). Ranking experiments by standalone primary therefore discards the models that make the
portfolio work, and calling such a node "rejected" is a scientific error.

Two jobs:

1. `valuation()` -- cheap per-node portfolio statistics computed from saved `val_scores.npy`:
   `rank_corr_to_best`, `pair_blend_gain`, `emc`. No training, ever.

2. `assemble_cv()` -- honest assembly. docs/RESEARCH.md §12 measured that grid-searching blend weights on all of valid
   and then reporting that same maximised number carries **+0.00072** of optimism, roughly half the
   apparent ensemble gain. An earlier draft of docs/SYSTEM.md §15 proposed greedy selection scored on a "holdout"
   half -- but a subset consulted once per greedy step to choose members is a SELECTION set, and
   reporting on it is biased by exactly that mechanism. What is implemented here instead is user-level
   K-fold cross-validation of the whole procedure:

       for each fold k:  members_k, weights_k = A(valid \\ fold_k)   # fold_k never consulted
                         score_k              = primary on fold_k
       honest estimate = mean(score_k) +/- sd/sqrt(K)
       final artefact  = A(all of valid)                            # for test prediction only

   Four data roles, never conflated: choose members / tune weights / report honestly / refit for the
   final submission.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agent.stats import Evaluator, per_user_rank, rank_corr


@dataclass
class Member:
    node_id: str
    model_type: str
    lever: str
    primary: float
    ranks: np.ndarray = field(repr=False)          # within-user percentile ranks on valid


def _weight_grid(m: int, step: float = 0.25):
    """Compositions of 1.0 into m non-negative parts on a `step` lattice.

    Coarse on purpose: docs/RESEARCH.md §12 measured that a 66-point simplex on 22k users overfits the weights. A
    coarser lattice loses ~nothing and generalises better.
    """
    n = int(round(1.0 / step))

    def rec(k, left):
        if k == 1:
            yield (left,)
            return
        for i in range(left + 1):
            for rest in rec(k - 1, left - i):
                yield (i,) + rest

    for comb in rec(m, n):
        yield tuple(c / n for c in comb)


def _blend(members, weights):
    out = np.zeros_like(members[0].ranks, dtype=np.float32)
    for w, m in zip(weights, members):
        if w:
            out += np.float32(w) * m.ranks
    return out


def _score(ev: Evaluator, members, weights, idx) -> float:
    return Evaluator.aggregate(ev.user_stats(_blend(members, weights)), idx)[0]


def best_weights(ev: Evaluator, members, idx, step: float = 0.25):
    """Grid-search blend weights for a fixed member set on the users in `idx`."""
    bw, bp = None, -1.0
    for w in _weight_grid(len(members), step):
        if sum(w) <= 0:
            continue
        p = _score(ev, members, w, idx)
        if p > bp:
            bp, bw = p, w
    return bw, bp


def greedy_select(ev: Evaluator, pool, idx, max_members: int = 4, step: float = 0.25):
    """The assembly procedure `A(S)`: greedy forward selection + weight tuning, using ONLY `idx`.

    Returns (members, weights, score_on_idx). Deterministic given the pool order.
    """
    if not pool:
        return [], (), -1.0
    singles = [(Evaluator.aggregate(ev.user_stats(m.ranks), idx)[0], i) for i, m in enumerate(pool)]
    singles.sort(key=lambda t: (-t[0], t[1]))
    chosen = [pool[singles[0][1]]]
    weights, score = (1.0,), singles[0][0]

    remaining = [m for m in pool if m is not chosen[0]]
    while remaining and len(chosen) < max_members:
        best = None
        for cand in remaining:
            trial = chosen + [cand]
            w, p = best_weights(ev, trial, idx, step)
            if best is None or p > best[2]:
                best = (cand, w, p)
        if best is None or best[2] <= score + 1e-9:
            break                                   # no candidate improves the fold -> stop
        cand, weights, score = best
        chosen.append(cand)
        remaining = [m for m in remaining if m is not cand]
    return chosen, tuple(weights), float(score)


def assemble_cv(ev: Evaluator, pool, K: int = 5, seed: int = 0,
                max_members: int = 4, step: float = 0.25) -> dict:
    """K-fold CV of the assembly procedure + a full-valid refit for the test submission."""
    G = ev.G
    if len(pool) < 2 or G < 2 * K:
        return {"ok": False, "reason": f"need >=2 candidates and >={2 * K} users "
                                       f"(have {len(pool)}, {G})"}
    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(G) % K
    all_idx = np.arange(G)

    fold_scores, fold_single, fold_members = [], [], []
    for k in range(K):
        tr = all_idx[fold_of != k]
        te = all_idx[fold_of == k]
        members, weights, _ = greedy_select(ev, pool, tr, max_members, step)
        if not members:
            continue
        fold_scores.append(_score(ev, members, weights, te))
        # like-for-like control: the best SINGLE member chosen on the same training users
        best_single_tr = max(pool, key=lambda m: Evaluator.aggregate(ev.user_stats(m.ranks), tr)[0])
        fold_single.append(Evaluator.aggregate(ev.user_stats(best_single_tr.ranks), te)[0])
        fold_members.append([m.node_id for m in members])

    if not fold_scores:
        return {"ok": False, "reason": "no fold produced a portfolio"}

    fs, fsing = np.array(fold_scores), np.array(fold_single)
    # Final artefact: refit on ALL users. This is what produces the test blend. It is NOT an estimate
    # and is never reported as one.
    members, weights, tuned = greedy_select(ev, pool, all_idx, max_members, step)

    return {
        "ok": True,
        "members": [m.node_id for m in members],
        "member_families": [m.model_type for m in members],
        "weights": [float(w) for w in weights],
        "valid_primary_tuned": float(tuned),                 # in-sample, optimistic
        "cv_mean": float(fs.mean()),                         # honest
        "cv_se": float(fs.std(ddof=1) / np.sqrt(len(fs))) if len(fs) > 1 else float("nan"),
        "cv_folds": [float(x) for x in fs],
        "cv_best_single_mean": float(fsing.mean()),
        "cv_gain_over_best_single": float((fs - fsing).mean()),
        "cv_gain_se": (float((fs - fsing).std(ddof=1) / np.sqrt(len(fs))) if len(fs) > 1
                       else float("nan")),
        "K": int(K),
        "fold_members": fold_members,
    }


def valuation(ev: Evaluator, members, champion_id: str | None = None) -> dict:
    """Per-node portfolio statistics. Cheap analysis on saved predictions -- never trains.

    `emc` is the leave-one-out contribution to the full-pool blend: how much the portfolio loses if
    this member is removed. This is the statistic that made LightGBM's value visible.
    """
    if len(members) < 2:
        return {m.node_id: {"rank_corr_to_best": 1.0, "pair_blend_gain": 0.0, "emc": 0.0}
                for m in members}
    idx = np.arange(ev.G)
    champ = next((m for m in members if m.node_id == champion_id), None)
    if champ is None:
        champ = max(members, key=lambda m: m.primary)

    _, full = best_weights(ev, members, idx)
    out = {}
    for m in members:
        rc = 1.0 if m is champ else rank_corr(m.ranks, champ.ranks)
        if m is champ:
            pair_gain = 0.0
        else:
            _, p2 = best_weights(ev, [champ, m], idx, step=0.1)
            pair_gain = p2 - max(champ.primary, m.primary)
        rest = [x for x in members if x is not m]
        _, without = best_weights(ev, rest, idx) if len(rest) >= 2 else (None, rest[0].primary)
        out[m.node_id] = {
            "rank_corr_to_best": round(float(rc), 4),
            "pair_blend_gain": round(float(pair_gain), 6),
            "emc": round(float(full - without), 6),
            "standalone_primary": round(float(m.primary), 6),
        }
    return out


def build_members(nodes, users, load_scores) -> list[Member]:
    """Turn viable nodes into portfolio members, de-duplicated by near-identical ranking.

    Two nodes whose within-user rankings correlate above 0.999 contribute nothing to each other in a
    blend, so keeping both only adds selection noise.
    """
    out: list[Member] = []
    for n in sorted(nodes, key=lambda n: -n.score()):
        s = load_scores(n)
        if s is None or len(s) != len(users):
            continue
        r = per_user_rank(s, users)
        if any(rank_corr(r, m.ranks) > 0.999 for m in out):
            continue
        out.append(Member(node_id=n.id, model_type=n.cfg.model_type, lever=n.lever,
                          primary=float(n.score()), ranks=r))
    return out
