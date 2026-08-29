"""KuaiRand-Pure baselines.
  --model pop   : item popularity (official baseline, no training)
  --model fm    : Factorization Machine trained with a HYBRID loss: a
                  pointwise-logloss gradient computed over ALL sampled
                  users' training rows (so every user, including ones that
                  are all-positive or all-negative in train, still
                  contributes to learning the shared video_id/author_id
                  embeddings), blended with a PAIRWISE BPR gradient
                  computed ONLY from users in that batch who have both a
                  positive and a negative training row (the users
                  GAUC/nDCG@5 actually care about).

                  THIS ITERATION'S CHANGE: the previous best-known hybrid
                  used a ListNet-style listwise softmax cross-entropy term
                  for its ranking-aware component. This iteration REPLACES
                  that listwise term with a PAIRWISE BPR term instead,
                  while keeping the exact same curriculum alpha schedule
                  (alpha_start=0.0 -> alpha_end=0.7 over ramp_epochs=12)
                  that produced the current best score. For every mixed
                  user in a batch, each of that user's positive training
                  rows is paired with one randomly resampled negative row
                  from the same user, and the BPR loss -log(sigmoid(score(
                  pos) - score(neg))) is used as the ranking-aware term.

                  WHY: GAUC is, by construction, an aggregation of
                  per-user pairwise concordance (the probability a random
                  positive scores above a random negative for that user).
                  BPR's loss and gradient are a direct, exact surrogate for
                  that exact pairwise quantity, whereas the previous
                  ListNet-style softmax cross-entropy term optimizes a
                  listwise probability distribution over an entire user's
                  candidate set -- a more nDCG-flavored objective that only
                  indirectly pushes on pairwise concordance. Swapping in
                  BPR for the ranking-aware half of the hybrid loss (still
                  combined with the same all-users pointwise term, under
                  the same alpha curriculum) should more directly target
                  the GAUC component of the primary metric without
                  changing anything else about the pipeline.
  --model random: random scores (sanity floor)
Only depends on numpy. See README.md for usage.
"""

# ============================================================================
# WHAT THIS FILE DOES (plain English)
# ----------------------------------------------------------------------------
# This file defines the three reference models used as scoring anchors:
#   random -> the floor (sanity check that evaluate.py itself isn't broken)
#   pop    -> a trivial "always recommend popular videos" baseline
#   fm     -> the baseline model you need to beat. It is a Factorization
#             Machine trained with a HYBRID pointwise + PAIRWISE-BPR loss
#             whose blend weight follows a CURRICULUM SCHEDULE across
#             epochs (see module docstring above). Per batch, we sample a
#             set of users (ALL training users, not filtered), pull every
#             one of their training rows for the pointwise term, and
#             separately sample positive/negative row pairs from just the
#             mixed-label users in that same batch for the BPR term:
#               (1) pointwise logloss gradient, over every row of every
#                   sampled user (all-positive/all-negative users included)
#               (2) pairwise BPR gradient, over (pos,neg) row pairs drawn
#                   only from users in the batch who have both labels
#             These are combined as (1-alpha_ep)*grad1 + alpha_ep*grad2
#             before a single Adam update, where alpha_ep depends on the
#             current epoch per the curriculum schedule.
#
# HOW IT CONNECTS TO THE OTHER FILES:
#   - data.py       supplies load() (raw rows) and encode() (turns those
#                   rows into the integer feature matrix the FM model reads)
#   - evaluate.py   supplies evaluate(), called after every training epoch
#                   to check validation score, and again at the end for the
#                   final valid/test report
#   - submit.py     imports FM and run_fm-style training logic from here
#                   when generating an example submission with --make
#   - ablation_features.py imports the FM class from this file directly
#                   (`import baseline as B`) and reuses it, but builds its
#                   own encoded features instead of using data.py's encode()
#
# Running this file directly (`python3 baseline.py --model fm`) is how you
# reproduce the FM baseline number described in the problem statement.
# ============================================================================

import argparse, collections, time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
# clip avoids overflow warnings/inf for very large |x| -- result is the same
# sigmoid function, just numerically safe.


# ---------------- item popularity (official baseline) ----------------
def run_pop(splits, prior=20.0):
    """Trivial baseline: score each video by how often it was long_view'd in
    training, smoothed toward the global average so rarely-seen videos don't
    get an extreme score just from a handful of observations (Bayesian /
    "add prior pseudo-counts" smoothing)."""
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[2]] += 1; pos[x[2]] += x[6]                 # x[2]=video_id, x[6]=label
    gmean = sum(pos.values()) / sum(imp.values())           # overall long_view rate across all videos
    # smoothed rate: blends this video's own rate with the global rate,
    # weighted by `prior` -- a video with 0 impressions just gets gmean.
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             [score(x[2]) for x in rws])
    return out


def run_random(splits, seed=0):
    """The floor baseline -- pure random scores. Used only as a sanity check:
    if this doesn't land near primary ~= 0.475 on test, something is wrong
    with the evaluation harness itself (see README self-check note)."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             rng.random(len(rws)))
    return out


# ---------------- Factorization Machine ----------------
class FM:
    """A from-scratch Factorization Machine, trained with Adam. This is the
    baseline model -- the number your agent's iterations need to beat on the
    hidden test set. Supports the original pointwise logloss step() (kept
    for backward compatibility / other scripts that may call it), the
    legacy pure-pairwise bpr_step and pure listwise step (kept for
    reference/comparison from earlier iterations), and the CURRENT
    hybrid_step_bpr (pointwise-on-all-users + pairwise-BPR-on-mixed-users)
    used by run_fm / train_and_predict below, driven by a curriculum alpha
    schedule.
    """

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        # dim = total size of the shared embedding table (sum of field_dims
        # from data.encode()). k = embedding dimension per feature id.
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)  # per-id embedding vectors (for interactions)
        self.W = np.zeros(dim, dtype=np.float32)                    # per-id linear (first-order) weight
        self.b = np.float32(0.0)                                    # global bias term
        self.lr, self.l2 = lr, l2
        # Adam optimizer's moving-average state (first and second moments),
        # one pair per parameter tensor (V and W).
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0                                                   # Adam time step counter

    def logits(self, X):
        # X is (batch, num_fields) of integer feature ids.
        E = self.V[X]                                   # (B,F,k) -- embedding of each field's id in this batch
        S = E.sum(1)                                    # (B,k)  -- sum of embeddings across fields
        # Standard FM pairwise-interaction trick: sum over all field pairs of
        # <v_i, v_j> equals 0.5 * ((sum of v)^2 - sum of v^2), computed in
        # O(F*k) instead of O(F^2*k).
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        # final score = bias + linear term (sum of per-id weights) + pairwise
        # interaction term. Also returns E, S since the gradient step reuses them.
        return self.b + self.W[X].sum(1) + inter, E, S

    def _adam_update(self, gV, gW):
        gV = gV + self.l2 * self.V
        gW = gW + self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8                     # standard Adam hyperparameters
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G                      # update biased first moment estimate
            Vv *= b2; Vv += (1 - b2) * (G * G)               # update biased second moment estimate
            # bias-corrected Adam update, applied in place to the parameter P
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

    def step(self, X, y):
        """[Legacy] one mini-batch pointwise-logloss gradient step. Kept for
        backward compatibility with other scripts (e.g. ablation_features.py)
        that may call it directly; NOT used by run_fm/train_and_predict."""
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)    # (B,) -- gradient of logloss w.r.t. the logit, per example
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        self._adam_update(gV, gW)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def bpr_step(self, Xp, Xn):
        """[Legacy, standalone] one mini-batch pairwise (BPR-style) gradient
        step trained on ONLY the pairwise term (no pointwise blend). Kept
        for reference/comparison; NOT used by run_fm/train_and_predict
        (hybrid_step_bpr below folds this same math into the hybrid loss)."""
        B = len(Xp)
        zp, Ep, Sp = self.logits(Xp)
        zn, En, Sn = self.logits(Xn)
        diff = zp - zn
        g = ((sigmoid(diff) - 1.0) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, Xp, g[:, None])
        np.add.at(gW, Xn, -g[:, None])
        np.add.at(gV, Xp, g[:, None, None] * (Sp[:, None, :] - Ep))
        np.add.at(gV, Xn, -g[:, None, None] * (Sn[:, None, :] - En))
        self._adam_update(gV, gW)
        return float(-np.mean(np.log(sigmoid(diff) + 1e-9)))

    def listwise_step(self, X, y, groups, G):
        """[Legacy] one mini-batch PURE LISTWISE (ListNet-style) gradient
        step, trained only on mixed-label user groups. Kept for
        reference/comparison; NOT used by run_fm/train_and_predict (this
        iteration replaces the listwise ranking term with pairwise BPR,
        see hybrid_step_bpr below)."""
        z, E, S = self.logits(X)
        groupmax = np.full(G, -np.inf, dtype=np.float32)
        np.maximum.at(groupmax, groups, z)
        shifted = z - groupmax[groups]
        exps = np.exp(shifted)
        groupsum = np.zeros(G, dtype=np.float32)
        np.add.at(groupsum, groups, exps)
        softmax = exps / groupsum[groups]
        grouppos = np.zeros(G, dtype=np.float32)
        np.add.at(grouppos, groups, y)
        target = y / grouppos[groups]
        g = ((softmax - target) / G).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        self._adam_update(gV, gW)
        return float(np.mean(-target * np.log(softmax + 1e-9)) * (len(y) / max(G, 1)))

    def hybrid_step(self, X, y, groups, G, alpha=0.5):
        """[Legacy] the PREVIOUS best-known hybrid step: pointwise-on-all +
        ListNet-style listwise softmax cross-entropy on mixed-label groups.
        Kept for reference/comparison; NOT used by run_fm/train_and_predict
        any more -- see hybrid_step_bpr below, which is this iteration's
        replacement for the ranking-aware half of the loss."""
        B = len(y)
        z, E, S = self.logits(X)
        p = sigmoid(z)

        g_point = (p - y) / B

        groupmax = np.full(G, -np.inf, dtype=np.float32)
        np.maximum.at(groupmax, groups, z)
        shifted = z - groupmax[groups]
        exps = np.exp(shifted)
        groupsum = np.zeros(G, dtype=np.float32)
        np.add.at(groupsum, groups, exps)
        softmax = exps / groupsum[groups]

        grouppos = np.zeros(G, dtype=np.float32)
        np.add.at(grouppos, groups, y)
        grouplen = np.zeros(G, dtype=np.float32)
        np.add.at(grouplen, groups, np.ones_like(y))
        is_mixed = (grouppos > 0) & (grouppos < grouplen)
        mixed_row_mask = is_mixed[groups].astype(np.float32)
        G_mixed = max(int(is_mixed.sum()), 1)

        safe_grouppos = np.maximum(grouppos[groups], 1e-9)
        target = mixed_row_mask * (y / safe_grouppos)
        g_list = mixed_row_mask * (softmax - target) / G_mixed

        g = ((1.0 - alpha) * g_point + alpha * g_list).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        self._adam_update(gV, gW)
        self.b -= self.lr * ((1.0 - alpha) * g_point.sum())

        point_loss = float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
        if G_mixed > 0:
            list_loss = float(np.sum(mixed_row_mask * -target * np.log(softmax + 1e-9)) / G_mixed)
        else:
            list_loss = 0.0
        return float((1.0 - alpha) * point_loss + alpha * list_loss)

    def hybrid_step_bpr(self, Xpt, ypt, Xbp, Xbn, alpha=0.5):
        """THIS ITERATION: one mini-batch gradient step combining
        (1) a pointwise logloss gradient over EVERY row of every sampled
            user in this batch (rows from ALL sampled users, including ones
            whose sampled rows happen to be all-positive or all-negative --
            keeps the shared video_id/author_id embeddings trained on the
            full population), and
        (2) a PAIRWISE BPR gradient computed from (positive, negative) row
            pairs drawn only from the mixed-label users present in this
            batch: every positive row of such a user is paired with one
            randomly resampled negative row from that SAME user, and the
            loss is -log(sigmoid(score(pos) - score(neg))).

        This directly targets pairwise concordance -- exactly what GAUC
        measures -- rather than the previous ListNet-style listwise softmax
        cross-entropy, which optimized a whole-group probability
        distribution (a more nDCG-flavored objective).

        Xpt, ypt : pointwise batch (rows for ALL sampled users, all labels)
        Xbp, Xbn : parallel arrays of positive-row / negative-row feature
                   ids for the sampled BPR pairs (may be length 0 if this
                   batch happened to contain no mixed users)
        alpha    : blend weight in [0,1] on the BPR term for THIS CALL;
                   (1-alpha) on the pointwise term. Callers pass a
                   per-epoch value from the curriculum schedule (see
                   _alpha_schedule / _train_fm_hybrid below).

        Returns the alpha-blended training loss value for logging.
        """
        Bpt = len(ypt)
        z_pt, E_pt, S_pt = self.logits(Xpt)
        p = sigmoid(z_pt)
        g_point = ((p - ypt) / Bpt).astype(np.float32)

        Bbpr = len(Xbp)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)

        # ---- (1) pointwise term, over every row ----
        np.add.at(gW, Xpt, ((1.0 - alpha) * g_point)[:, None])
        np.add.at(gV, Xpt, ((1.0 - alpha) * g_point)[:, None, None] * (S_pt[:, None, :] - E_pt))

        # ---- (2) pairwise BPR term, over sampled mixed-user pairs ----
        if Bbpr > 0:
            zp, Ep, Sp = self.logits(Xbp)
            zn, En, Sn = self.logits(Xbn)
            diff = zp - zn
            gbpr = ((sigmoid(diff) - 1.0) / Bbpr).astype(np.float32)
            np.add.at(gW, Xbp, (alpha * gbpr)[:, None])
            np.add.at(gW, Xbn, (-alpha * gbpr)[:, None])
            np.add.at(gV, Xbp, (alpha * gbpr)[:, None, None] * (Sp[:, None, :] - Ep))
            np.add.at(gV, Xbn, (-alpha * gbpr)[:, None, None] * (Sn[:, None, :] - En))
            bpr_loss = float(-np.mean(np.log(sigmoid(diff) + 1e-9)))
        else:
            bpr_loss = 0.0

        self._adam_update(gV, gW)
        # bias term only gets a gradient from the pointwise part: BPR's
        # pairwise diff is invariant to a shared additive bias shift across
        # a single user's pos/neg pair, so its contribution to the shared
        # global bias gradient is ~0 (same reasoning as the listwise/BPR
        # legacy steps above).
        self.b -= self.lr * ((1.0 - alpha) * g_point.sum())

        point_loss = float(-np.mean(ypt * np.log(p + 1e-9) + (1 - ypt) * np.log(1 - p + 1e-9)))
        return float((1.0 - alpha) * point_loss + alpha * bpr_loss)

    def predict(self, X, bs=200_000):
        # Predicts in chunks (bs = batch size) so large splits (like test,
        # ~170k rows) don't need one giant array all at once.
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def _build_user_pos_neg(users, y):
    """Groups training row-indices by user_id into positive-label and
    negative-label index lists (global indices into the train arrays).
    Only users with at least one of each label are kept -- these are the
    'mixed' users usable for the pairwise BPR term."""
    pos = collections.defaultdict(list)
    neg = collections.defaultdict(list)
    for i, (u, yy) in enumerate(zip(users, y)):
        (pos if yy > 0.5 else neg)[u].append(i)
    usable = [u for u in pos if u in neg]
    pos = {u: np.asarray(pos[u], dtype=np.int64) for u in usable}
    neg = {u: np.asarray(neg[u], dtype=np.int64) for u in usable}
    return pos, neg


def _sample_pairs(pos, neg, rng):
    """For every user in `pos` (assumed also present in `neg`), pairs EVERY
    positive row with a randomly resampled negative row from the same
    user. Returns (pos_idx, neg_idx), two parallel arrays of GLOBAL row
    indices suitable for indexing directly into the train feature matrix."""
    pos_idx_chunks, neg_idx_chunks = [], []
    for u, plist in pos.items():
        nlist = neg[u]
        chosen = nlist[rng.integers(0, len(nlist), size=len(plist))]
        pos_idx_chunks.append(plist)
        neg_idx_chunks.append(chosen)
    if not pos_idx_chunks:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(pos_idx_chunks), np.concatenate(neg_idx_chunks)


def _build_user_groups(users, y):
    """[Legacy, used by the pure listwise path] Groups training row-indices
    by user_id, keeping only users that have AT LEAST ONE positive and AT
    LEAST ONE negative row. Returns dict {user_id: np.array of row indices}."""
    d = collections.defaultdict(list)
    for i, u in enumerate(users):
        d[u].append(i)
    keep = {}
    for u, idxs in d.items():
        idxs = np.asarray(idxs, dtype=np.int64)
        s = y[idxs].sum()
        if 0 < s < len(idxs):
            keep[u] = idxs
    return keep


def _build_all_user_groups(users):
    """Groups training row-indices by user_id, keeping EVERY user (no
    filtering by label mix) -- this is what lets the pointwise term see the
    full training population, including users who are all-positive or
    all-negative among their own training rows. Returns dict
    {user_id: np.array of row indices}."""
    d = collections.defaultdict(list)
    for i, u in enumerate(users):
        d[u].append(i)
    return {u: np.asarray(idxs, dtype=np.int64) for u, idxs in d.items()}


def _alpha_schedule(ep, alpha_start, alpha_end, ramp_epochs):
    """Computes the ranking-term blend weight for a given 1-indexed epoch
    `ep`, following a linear curriculum from alpha_start (epoch 1) to
    alpha_end (epoch >= ramp_epochs), instead of one fixed alpha used for
    the whole run. If ramp_epochs <= 1, jumps straight to alpha_end from
    epoch 1 (degenerates to fixed-alpha behavior when alpha_start ==
    alpha_end). Unchanged from the previous best-known iteration -- only
    the ranking loss term itself (BPR instead of listwise softmax) changes
    this iteration."""
    if ramp_epochs <= 1:
        return alpha_end
    frac = min(1.0, ep / float(ramp_epochs))
    return alpha_start + frac * (alpha_end - alpha_start)


def _train_fm_hybrid(splits, k=16, lr=0.0015, alpha_start=0.0, alpha_end=0.7, ramp_epochs=12,
                      epochs=40, batch_users=256, patience=5, seed=0, verbose=True):
    """Shared training loop used by both run_fm() and train_and_predict():
    trains an FM with the HYBRID pointwise(all-users) + PAIRWISE-BPR
    (mixed-users) loss (see FM.hybrid_step_bpr / module docstring), where
    the BPR blend weight alpha follows a linear CURRICULUM SCHEDULE from
    alpha_start (at epoch 1) to alpha_end (reached by epoch ramp_epochs and
    held thereafter). Early-stops on validation primary score, and returns
    (model_with_best_weights, enc).

    batch_users: number of distinct users grouped together per gradient
    step (ALL of these users' training rows feed the pointwise term; the
    subset of them that are mixed-label additionally feed sampled BPR
    pairs).
    alpha_start / alpha_end / ramp_epochs: curriculum schedule parameters,
    see _alpha_schedule above. Setting alpha_start == alpha_end reproduces
    a fixed-alpha hybrid-BPR run.
    """
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    keep = _build_all_user_groups(utr)                 # user -> ALL of their train row indices
    pos_by_user, neg_by_user = _build_user_pos_neg(utr, ytr)   # user -> pos/neg indices (mixed users only)
    user_ids = list(keep.keys())
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        alpha_ep = _alpha_schedule(ep, alpha_start, alpha_end, ramp_epochs)
        order = rng.permutation(len(user_ids))
        shuffled_users = [user_ids[i] for i in order]
        losses = []
        for i in range(0, len(shuffled_users), batch_users):
            batch_user_ids = shuffled_users[i:i + batch_users]
            # pointwise batch: every training row of every sampled user
            idx_pt = np.concatenate([keep[u] for u in batch_user_ids])
            # pairwise BPR batch: sampled (pos,neg) pairs, mixed users only
            mixed_users = [u for u in batch_user_ids if u in pos_by_user]
            if mixed_users:
                sub_pos = {u: pos_by_user[u] for u in mixed_users}
                sub_neg = {u: neg_by_user[u] for u in mixed_users}
                idx_p, idx_n = _sample_pairs(sub_pos, sub_neg, rng)
            else:
                idx_p = np.zeros(0, dtype=np.int64); idx_n = np.zeros(0, dtype=np.int64)
            losses.append(m.hybrid_step_bpr(Xtr[idx_pt], ytr[idx_pt], Xtr[idx_p], Xtr[idx_n], alpha=alpha_ep))
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            # epoch    -> which pass over the shuffled user batches this is
            # alpha    -> this epoch's curriculum-scheduled BPR weight
            # loss     -> training hybrid loss (health signal, not a scored metric)
            # GAUC / nDCG@5 / primary -> official metrics on VALIDATION, used
            #             to decide early stopping
            print(f"  epoch {ep:2d} | alpha {alpha_ep:.3f} | users {len(user_ids)} | "
                  f"loss {np.mean(losses) if losses else float('nan'):.4f} "
                  f"| valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} "
                  f"| {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return m, enc


def run_fm(splits, k=16, lr=0.0015, alpha_start=0.0, alpha_end=0.7, ramp_epochs=12,
           epochs=40, batch_users=256, patience=5, seed=0, verbose=True):
    """Trains the FM model with the hybrid pointwise(all) + pairwise-BPR(mixed)
    loss under a curriculum alpha schedule and early stopping on validation
    `primary` score, then reports both valid and test metrics using the
    best-validation checkpoint (not necessarily the last epoch)."""
    m, enc = _train_fm_hybrid(splits, k=k, lr=lr, alpha_start=alpha_start, alpha_end=alpha_end,
                               ramp_epochs=ramp_epochs, epochs=epochs, batch_users=batch_users,
                               patience=patience, seed=seed, verbose=verbose)
    Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


def train_and_predict(splits, predict_split='test', k=16, lr=0.0015, alpha_start=0.0, alpha_end=0.7,
                       ramp_epochs=12, epochs=40, batch_users=256, patience=5, seed=0, bs=None,
                       alpha=None):
    """AGENT-AUTOMATION CONTRACT: keep this function's signature and return
    shape stable even if you replace the model/training approach entirely.
    agent/finalize.py calls this ONCE at the very end (not every iteration)
    to get real prediction scores for the submission file, without having
    to know anything about your model's internal class/weights. It must
    return (metrics, scores) where metrics is the same {'valid':...,
    'test':...} shape as run_fm() above, and scores is a 1-D array of
    per-row prediction scores for `predict_split`, using the SAME
    best-validation checkpoint reflected in metrics (not some other
    epoch/model). If you swap FM for a different model class, reimplement
    the body of this function against your new model/training loop, but
    keep the signature and return shape identical.

    `bs` is accepted (and ignored) purely for backward compatibility with
    any older caller that still passes the previous per-row batch-size
    keyword; batch size is now expressed as `batch_users` (users per batch).
    `alpha` is accepted (and, if given, used as BOTH alpha_start and
    alpha_end, i.e. disables the curriculum and reproduces a fixed-alpha
    hybrid-BPR run) purely for backward compatibility with any older caller
    that still passes a single flat alpha value.
    """
    if alpha is not None:
        alpha_start = alpha_end = alpha
    m, enc = _train_fm_hybrid(splits, k=k, lr=lr, alpha_start=alpha_start, alpha_end=alpha_end,
                               ramp_epochs=ramp_epochs, epochs=epochs, batch_users=batch_users,
                               patience=patience, seed=seed, verbose=False)
    Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    metrics = {'valid': evaluate(uva, yva, m.predict(Xva)),
               'test':  evaluate(ute, yte, m.predict(Xte))}
    Xp, _, _ = enc[predict_split]
    scores = m.predict(Xp)
    return metrics, scores


if __name__ == '__main__':
    # Command-line entry point: `python3 baseline.py --model fm` etc.
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='fm', choices=['pop', 'fm', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.0015)
    ap.add_argument('--alpha_start', type=float, default=0.0,
                     help='BPR blend weight at epoch 1 (curriculum start)')
    ap.add_argument('--alpha_end', type=float, default=0.7,
                     help='BPR blend weight once ramp_epochs is reached (curriculum end)')
    ap.add_argument('--ramp_epochs', type=int, default=12,
                     help='number of epochs over which alpha linearly ramps from alpha_start to alpha_end')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_users', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)                                # data.py: read + date-split the raw CSVs
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, alpha_start=a.alpha_start, alpha_end=a.alpha_end,
                                   ramp_epochs=a.ramp_epochs, epochs=a.epochs,
                                   batch_users=a.batch_users, seed=a.seed)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
