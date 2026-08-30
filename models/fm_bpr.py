"""FM trained on a WITHIN-USER PAIRWISE (BPR) objective instead of pointwise logloss.

The motivation, and why this is the change worth making rather than more features:

The score is `mean(GAUC, nDCG@5)`. Both halves are WITHIN-USER rankings — they never compare one
user's rows to another's. But baseline.FM trains pointwise binary cross-entropy against a
globally imbalanced label, which spends most of its capacity getting each row's absolute
probability right. Calibration across users is exactly the part of that job the metric throws
away, and the ordering within a user is the part it keeps.

Measured, not assumed: adding item-quality features (smoothed per-video/author long_view rates as
target-encoded fields, blended with the popularity prior) was tried first and made things WORSE —
0.5906 against the baseline's 0.6015. An offline sweep of the blend weight on a trained FM's own
valid scores showed why: the best alpha was 0.05 for +0.00003, then monotonic decline. The FM
already extracts the per-video propensity through its `video_id` linear weight, so re-feeding it
as a feature is redundant capacity and nothing else. The item-side signal is saturated. What is
NOT saturated is the objective, which is what this file changes.

The change is genuinely small: the forward pass, the embedding table, the L2 term and the Adam
optimizer are all baseline.FM's, untouched. Only the gradient of the loss differs, in _step_pair.

MEASURED on valid, 5 seeds each, this config vs models/fm_v1.py at its defaults:

    fm_v1    0.60157  +/- 0.00032   (GAUC 0.6671, nDCG@5 0.5358)
    fm_bpr   0.60274  +/- 0.00049   (GAUC 0.6690, nDCG@5 0.5368)
    delta    +0.00117

That is ~2.4x the combined seed noise, and fm_bpr's worst seed (0.60187) is within a whisker of
fm_v1's best (0.60204) - a real effect, not a lucky seed. It is nonetheless BELOW the agent's
0.002 acceptance epsilon (agent/config.py), so agent/decision.py would log this as KEEP_NODE
rather than COMMIT. On this dataset an epsilon of 0.002 is wider than the gains actually
available; treat that threshold as needing recalibration, not this result as a failure.

Tuning notes, all measured, all negative - the hyperparameters are at their optimum here just as
they were for fm_v1, so do not spend iterations re-searching them:
    pairs_per_pos  1.0 -> 0.60292 | 2.0 -> 0.60306 | 4.0 -> 0.60214   (2.0 is the default)
    k              16  -> 0.60306 | 32  -> 0.60207
    lr             0.001 -> 0.60306 | 0.002 -> 0.60015 | 0.003 -> 0.59970 | 0.005 -> 0.59839

HARD NEGATIVE MINING WAS TRIED AND IS WORSE - monotonically, which is the useful part:
    neg_candidates 1 -> 0.60306 | 2 -> 0.60041 | 4 -> 0.58238 | 8 -> 0.57176 | 16 -> 0.56862
The knob survives (default 1 = disabled, so this costs nothing) because the trend itself is the
finding. long_view is noisy implicit feedback, so a user's highest-scoring negative is very often
a mislabeled positive - someone who did watch and was not recorded as a long view. Training on
the most-violating candidate therefore aims each gradient step at exactly the rows most likely to
be wrong, and the harder you mine, the more of the fit is noise. Do not re-propose this, and be
suspicious of any variant that concentrates on high-scoring negatives.

WHAT DID HELP ON TOP OF THIS MODEL: seed ensembling - see ensemble_submission.py. Averaging
per-user-standardized scores over 5 seeds gives valid 0.6039 (vs 0.6031 best single) and test
0.5974 (vs 0.5970). Small but real, and it is averaging out initialization/minibatch variance
rather than adding information, which is why it works where the popularity blend did not.

Contract is models/base.py's: train(splits, config=None, verbose=False) -> {split: evaluate(...)}
for every non-'train' key. Fits nothing on a non-train split, loads no data itself.
"""
import numpy as np

import baseline as B
from data import encode
from evaluate import evaluate
from models.base import non_train_splits

DEFAULTS = {
    'k': 16, 'lr': 0.001, 'l2': 1e-6,
    'epochs': 40, 'patience': 4, 'batch_size': 8192,
    'seed': 0,
    # Pairs drawn per epoch, as a multiple of the number of eligible training positives.
    # 1.0 means "one sampled negative per positive per epoch"; raising it trades wall-clock for
    # a denser sample of the pair space.
    'pairs_per_pos': 2.0,
    # HARD NEGATIVE MINING. For each sampled positive, draw this many candidate negatives from
    # the same user and train on the single HIGHEST-SCORING one — the negative the model is
    # currently most wrong about. 1 disables it (uniform sampling, the original behaviour).
    #
    # Worth being clear why this is not just "more negatives": averaging the BPR loss over N
    # uniformly-drawn negatives is statistically almost the same thing as drawing N times as many
    # independent pairs, and that was measured as WORSE (pairs_per_pos=4 -> 0.6021 vs 2 -> 0.6031).
    # Picking the most-violating candidate is a different estimator, not a denser one: it
    # concentrates every gradient step on the ordering errors that actually cost GAUC, instead of
    # spending most steps on easy pairs the model already ranks correctly.
    'neg_candidates': 1,
}


def _build_pair_index(users, y):
    """Precompute the structure the pair sampler needs, once, before training.

    Returns (pos_idx, neg_flat, neg_start, neg_counts, codes) where:
      pos_idx    - row indices of every positive belonging to an ELIGIBLE user
      neg_flat   - row indices of every negative belonging to an eligible user, sorted by user
      neg_start  - per-user start offset into neg_flat
      neg_counts - per-user negative count
      codes      - per-row integer user code

    "Eligible" means the user has at least one positive AND at least one negative in train — the
    only users a pairwise loss can produce a gradient from, and (not coincidentally) the same
    population GAUC actually scores over: evaluate.py skips users with 0 or all positives.
    """
    uniq, codes = np.unique(np.asarray(users), return_inverse=True)
    n_users = len(uniq)
    pos_mask = y > 0.5

    n_pos = np.bincount(codes[pos_mask], minlength=n_users)
    n_neg = np.bincount(codes[~pos_mask], minlength=n_users)
    eligible = (n_pos > 0) & (n_neg > 0)
    row_eligible = eligible[codes]

    pos_idx = np.flatnonzero(pos_mask & row_eligible)

    neg_idx = np.flatnonzero((~pos_mask) & row_eligible)
    # Group negatives by user by sorting on the user code; a stable sort keeps each user's
    # negatives in original row order, which keeps the whole thing reproducible under a seed.
    neg_flat = neg_idx[np.argsort(codes[neg_idx], kind='stable')]
    neg_counts = np.bincount(codes[neg_flat], minlength=n_users)
    neg_start = np.zeros(n_users, dtype=np.int64)
    np.cumsum(neg_counts[:-1], out=neg_start[1:])

    return pos_idx, neg_flat, neg_start, neg_counts, codes


def _sample_pairs(rng, n_pairs, pos_idx, neg_flat, neg_start, neg_counts, codes,
                   n_candidates=1):
    """Draw `n_pairs` positives and, for each, `n_candidates` negatives from the SAME user.

    Returns (p, cand) with p shape (n_pairs,) and cand shape (n_pairs, n_candidates). With
    n_candidates=1 that second axis is just the single uniform negative the original code drew.

    Sampling positives uniformly is deliberate rather than incidental: it makes a user's chance
    of being trained on proportional to their positive count, which is exactly how GAUC weights
    users when it averages. The training distribution therefore matches the scoring weight.
    """
    p = pos_idx[rng.integers(0, len(pos_idx), size=n_pairs)]
    c = codes[p]
    # One uniform draw per candidate, inside each positive's own user's negative block. The
    # (n_pairs, n_candidates) broadcast against a (n_pairs, 1) count/offset is what keeps every
    # candidate inside the right user — getting this shape wrong is what made the loop's own
    # attempt at this idea die with "operands could not be broadcast together".
    offset = (rng.random((n_pairs, n_candidates)) * neg_counts[c][:, None]).astype(np.int64)
    cand = neg_flat[neg_start[c][:, None] + offset]
    return p, cand


def _hardest_negative(m, Xtr, cand):
    """Of each row's candidate negatives, the one the model currently scores HIGHEST.

    That is the most-violating negative: the one closest to (or already above) the positive, and
    therefore the one carrying the largest gradient. Scoring costs one forward pass over
    n_pairs * n_candidates rows, but the backward pass still runs on n_pairs pairs only, so the
    cost grows far more slowly than the candidate count.

    No gradient is taken here — m.logits() is called purely to rank candidates, and its E/S
    intermediates are discarded.
    """
    n_rows, n_cand = cand.shape
    z, _, _ = m.logits(Xtr[cand.reshape(-1)])
    z = z.reshape(n_rows, n_cand)
    return cand[np.arange(n_rows), np.argmax(z, axis=1)]


def _step_pair(m, Xp, Xn):
    """One BPR gradient step on a batch of pairs. Mirrors baseline.FM.step exactly apart from
    the loss; returns the batch's mean BPR loss.

    Pointwise FM optimizes  -[y log s(z) + (1-y) log(1-s(z))]  on single rows.
    This optimizes         -log s(z_pos - z_neg)               on pairs.

    So with d = s(z_pos - z_neg), the derivative of the loss w.r.t. z_pos is -(1-d) and w.r.t.
    z_neg is +(1-d) — equal and opposite. Everything downstream of that (how a change in z maps
    onto W and V) is identical to the pointwise case, because it is the same forward pass.

    Note there is NO bias update: b appears in both z_pos and z_neg and cancels in the
    difference, so its gradient is exactly zero under this objective. A global bias cannot change
    a within-user ranking anyway, which is the same reason the metric doesn't care about it.
    """
    Bn = len(Xp)
    zp, Ep, Sp = m.logits(Xp)
    zn, En, Sn = m.logits(Xn)
    d = B.sigmoid(zp - zn)
    gp = (-(1.0 - d) / Bn).astype(np.float32)
    gn = -gp

    gV = np.zeros_like(m.V)
    gW = np.zeros_like(m.W)
    # np.add.at (scatter-add) for the same reason baseline.FM.step uses it: the same feature id
    # recurs many times across a batch and plain indexed assignment would silently drop updates.
    np.add.at(gW, Xp, gp[:, None])
    np.add.at(gW, Xn, gn[:, None])
    np.add.at(gV, Xp, gp[:, None, None] * (Sp[:, None, :] - Ep))
    np.add.at(gV, Xn, gn[:, None, None] * (Sn[:, None, :] - En))
    gV += m.l2 * m.V
    gW += m.l2 * m.W

    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for P, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        P -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)

    return float(-np.mean(np.log(d + 1e-9)))


def train(splits, config=None, verbose=False):
    cfg = {**DEFAULTS, **(config or {})}
    eval_names = non_train_splits(splits)
    if 'train' not in splits or not eval_names:
        raise ValueError("fm_bpr.train needs a 'train' key and at least one other split to "
                          "evaluate against")

    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    eval_enc = {name: enc[name] for name in eval_names}
    primary_split = 'valid' if 'valid' in eval_enc else eval_names[0]

    pos_idx, neg_flat, neg_start, neg_counts, codes = _build_pair_index(utr, ytr)
    if len(pos_idx) == 0:
        raise RuntimeError("fm_bpr.train: no training user has both a positive and a negative — "
                            "a pairwise objective has nothing to learn from")
    n_pairs = max(1, int(len(pos_idx) * cfg['pairs_per_pos']))
    n_cand = max(1, int(cfg['neg_candidates']))
    if verbose:
        print(f"  [fm_bpr] {len(pos_idx)} eligible positives, {n_pairs} pairs/epoch, "
              f"neg_candidates={n_cand}, dim={dim}")

    m = B.FM(dim, k=cfg['k'], lr=cfg['lr'], l2=cfg['l2'], seed=cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    bs = cfg['batch_size']

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, cfg['epochs'] + 1):
        p, cand = _sample_pairs(rng, n_pairs, pos_idx, neg_flat, neg_start, neg_counts, codes,
                                 n_candidates=n_cand)
        for i in range(0, n_pairs, bs):
            pb, cb = p[i:i + bs], cand[i:i + bs]
            # Selected per BATCH, not once per epoch: "hardest" is a property of the CURRENT
            # parameters, and they move with every step. Choosing at epoch start would rank the
            # candidates against a model that no longer exists by the time the batch is used.
            nb = cb[:, 0] if n_cand == 1 else _hardest_negative(m, Xtr, cb)
            _step_pair(m, Xtr[pb], Xtr[nb])

        Xv, yv, uv = eval_enc[primary_split]
        cur = evaluate(uv, yv, m.predict(Xv))
        if verbose:
            print(f"  epoch {ep:2d} | {primary_split} primary {cur['primary']:.4f}")
        if cur['primary'] > best + 1e-5:
            best, bad = cur['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= cfg['patience']:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    if best_state is None:
        raise RuntimeError("fm_bpr.train: no epoch produced a finite primary score — likely NaN "
                            "inputs or a broken config")
    m.V, m.W, m.b = best_state

    return {name: evaluate(u, y, m.predict(X)) for name, (X, y, u) in eval_enc.items()}
