"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine（起步模型，学生从这里往上改）
  --model random: 随机打分（下界，用来自检评测代码没坏）
只依赖 numpy。用法见 README.md
"""

# ============================================================================
# WHAT THIS FILE DOES (plain English)
# ----------------------------------------------------------------------------
# This file defines the three reference models used as scoring anchors:
#   random -> the floor (sanity check that evaluate.py itself isn't broken)
#   pop    -> a trivial "always recommend popular videos" baseline
#   fm     -> the OFFICIAL baseline you actually need to beat (Factorization
#             Machine, trained with Adam + early stopping)
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
# reproduce the official baseline number described in the problem statement.
# ============================================================================

import argparse, collections, time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
# clip avoids overflow warnings/inf for very large |x| — result is the same
# sigmoid function, just numerically safe.


# ---------------- item popularity（官方 baseline） ----------------
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
    # weighted by `prior` — a video with 0 impressions just gets gmean.
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             [score(x[2]) for x in rws])
    return out


def run_random(splits, seed=0):
    """The floor baseline — pure random scores. Used only as a sanity check:
    if this doesn't land near primary ≈ 0.475 on test, something is wrong
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
    OFFICIAL baseline model — the number your agent's iterations need to
    beat on the hidden test set."""

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
        E = self.V[X]                                   # (B,F,k) — embedding of each field's id in this batch
        S = E.sum(1)                                    # (B,k)  — sum of embeddings across fields
        # Standard FM pairwise-interaction trick: sum over all field pairs of
        # <v_i, v_j> equals 0.5 * ((sum of v)^2 - sum of v^2), computed in
        # O(F*k) instead of O(F^2*k).
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        # final score = bias + linear term (sum of per-id weights) + pairwise
        # interaction term. Also returns E, S since the gradient step reuses them.
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        """One mini-batch gradient step (forward + backward + Adam update),
        returns this batch's logloss for monitoring."""
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)    # (B,) — gradient of logloss w.r.t. the logit, per example
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        # Scatter-add each example's gradient into the rows of W/V that were
        # actually used (np.add.at handles the case where the same id
        # appears more than once in a batch, unlike plain indexed assignment).
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W    # L2 weight decay
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8                     # standard Adam hyperparameters
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G                      # update biased first moment estimate
            Vv *= b2; Vv += (1 - b2) * (G * G)               # update biased second moment estimate
            # bias-corrected Adam update, applied in place to the parameter P
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        # return this batch's binary cross-entropy loss (for the printed log)
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def predict(self, X, bs=200_000):
        # Predicts in chunks (bs = batch size) so large splits (like test,
        # ~170k rows) don't need one giant array all at once.
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
    """Trains the FM model with early stopping on validation `primary`
    score, then reports both valid and test metrics using the
    best-validation checkpoint (not necessarily the last epoch)."""

    enc, dim = encode(splits)                              # data.py does all feature encoding here
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()   # shuffle training order each epoch
        # one gradient step per mini-batch of size bs, covering the full
        # training set once (this is what "one epoch" means here)
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))              # evaluate.py scores this epoch's validation predictions
        if verbose:
            # What each printed column means, in simple words:
            #   epoch    -> which pass over the full training set this is
            #   loss     -> training error (binary cross-entropy) on this
            #               epoch's training batches — how wrong the model's
            #               predictions were, NOT a competition metric, just
            #               a training-health signal. Lower is better, and
            #               it should mostly go down every epoch since the
            #               model is directly fitting to minimize it.
            #   GAUC / nDCG@5 / primary -> the official metrics (see
            #               evaluate.py), computed here on the VALIDATION
            #               split only — this is the number that decides
            #               whether training keeps going or stops.
            #   the trailing "Xs"  -> wall-clock seconds this epoch took
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            # new best validation score — save a copy of the model weights.
            # This snapshot (not necessarily the final epoch's weights) is
            # what gets used at the very end — see best_state below.
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            # validation primary did NOT improve enough this epoch
            bad += 1
            if bad >= patience:
                # stop early once validation primary has failed to improve
                # for `patience` (default 4) epochs in a row. This is
                # "early stopping": it protects against overfitting, where
                # training loss keeps dropping but validation score starts
                # to flatten or worsen because the model is memorizing
                # training data rather than learning something general.
                if verbose: print(f"  early stop at epoch {ep}")
                break
    # Roll back to the BEST validation checkpoint seen (best_state), which
    # may be several epochs earlier than where training actually stopped —
    # e.g. if epoch 7 was best and epochs 8-11 failed to beat it, the
    # weights used below are epoch 7's, not epoch 11's.
    m.V, m.W, m.b = best_state
    # Final report: score that same best checkpoint on BOTH validation
    # (what training already saw) and test (the truly held-out split) so
    # you can compare "how good it looked" vs "how good it actually is".
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


if __name__ == '__main__':
    # Command-line entry point: `python3 baseline.py --model fm` etc.
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='fm', choices=['pop', 'fm', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)                                # data.py: read + date-split the raw CSVs
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
