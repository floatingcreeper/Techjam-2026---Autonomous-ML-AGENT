"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine（起步模型，学生从这里往上改）
  --model random: 随机打分（下界，用来自检评测代码没坏）
只依赖 numpy。用法见 README.md
"""
# NOTE: three "difficulty tiers" live in this one file: random (no intelligence at
# all, just checks the scoring code isn't broken), pop (a lookup table, no
# training/learning involved), and fm (an actual trained model). Run them in that
# order once (--model random, then --model pop, then --model fm) to build
# intuition for what each resulting score number means.
import argparse, collections, time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate

# NOTE: sigmoid squashes any real number into (0, 1) — i.e. turns a raw model
# output ("logit") into something readable as a probability. `np.clip(x, -30, 30)`
# exists purely so `np.exp()` doesn't overflow on a huge input; it doesn't change
# the math for any x you'd realistically see (exp(30) is already astronomical).
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- item popularity（官方 baseline） ----------------
# NOTE: "popularity" baseline = no ML at all. For every video it just computes
# "what fraction of the time people who saw this video watched it long?" and uses
# that fraction as the score for every row with that video. It's a lookup table,
# not a trained model — there's nothing here that generalizes beyond videos it
# has already seen (a video with 0 impressions falls back to the global average).
def run_pop(splits, prior=20.0):
    # pos[v] = how many times video v got long_view=1; imp[v] = how many times
    # video v was shown at all (impressions). x[2]=video_id, x[6]=label (see the
    # NOTE on row tuples in data.py's load()).
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[2]] += 1; pos[x[2]] += x[6]
    gmean = sum(pos.values()) / sum(imp.values())
    # NOTE: `prior` is Bayesian/Laplace smoothing. A video seen only twice with
    # 2/2 long-views shouldn't score as "100% guaranteed long view" — that's just
    # noise from a tiny sample size. Blending in `prior` fake observations' worth
    # of the global average (gmean) pulls low-sample videos back toward a sane
    # average instead of letting a couple of lucky/unlucky impressions dominate.
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             [score(x[2]) for x in rws])
    return out

# NOTE: pure noise as scores — literally random numbers, no learning. If this
# doesn't come out to ~0.475 primary (per README), the bug is in evaluate.py or
# data.py, not in a model. Always the first thing to rerun after touching either.
def run_random(splits, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             rng.random(len(rws)))
    return out

# ---------------- Factorization Machine ----------------
# NOTE: FM is the first *actual* trainable model here. Background: every
# categorical value (a specific user_id, a specific video_id, ...) gets turned
# into a row-id by data.encode(), and the model learns a small vector of numbers
# ("embedding" / "latent vector", length k) for every single one of those ids.
# Two ids that behave similarly in the data end up with similar vectors — that's
# the whole point, it's how the model represents "similarity" numerically.
# The model's prediction = a bias term + a per-feature linear weight (like plain
# logistic regression) + a sum of "interaction" terms between every PAIR of
# active features (e.g. this-user × this-video), each computed as a dot product
# of their embedding vectors. That pairwise term is what lets it learn things
# like "this user tends to like videos from this author" without anyone
# hand-writing that as an explicit feature.
class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        # NOTE: V = the embedding table. Shape (dim, k): one length-k vector per
        # possible feature value across ALL fields combined (dim = sum of all
        # field vocab sizes, computed in data.encode() — see the offsets NOTE
        # there). Tiny random init (std=0.01) so training starts from "almost
        # nothing" rather than some arbitrary bias.
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        # NOTE: W = one plain scalar weight per feature value — the "linear"
        # part, like a coefficient in linear/logistic regression, no interactions.
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)   # NOTE: b = single global bias/intercept term.
        self.lr, self.l2 = lr, l2
        # NOTE: l2 = "L2 regularization" strength — a penalty added to the loss
        # that discourages weights from growing huge, which helps avoid
        # overfitting (memorizing training data instead of learning patterns
        # that generalize to valid/test).
        # mV/vV/mW/vW/t below are internal bookkeeping for the Adam optimizer
        # (explained in step()) — NOT part of the model's predictions, just
        # state the optimizer needs to carry between updates.
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        # X: (B, F) batch of B rows, each row listing F feature-ids (F=5, see
        # FIELDS in data.py: user_id, video_id, author_id, tab, dur_bucket).
        E = self.V[X]                                   # (B,F,k) NOTE: fancy-indexing —
        # for every row, look up the embedding vector of each of its F features.
        S = E.sum(1)                                    # (B,k)  NOTE: sum those F vectors.
        # NOTE: this line is the "weird looking" part of FM — it's a standard
        # algebra identity, not magic. Naively, "sum over every PAIR of features
        # of (their embeddings dotted together)" costs O(F^2) work. It's
        # mathematically identical to 0.5 * [ (sum of all vectors)^2 - sum of
        # (each vector)^2 ], which only costs O(F) work — same answer, cheaper
        # to compute. This is *the* trick that makes FM practical at scale.
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S   # bias + linear part + interaction part

    def step(self, X, y):
        # NOTE: one gradient-descent update on one batch (X, y). There's no
        # autograd/PyTorch here — the gradient formulas below were derived by
        # hand with calculus ahead of time; this is literally "the math for how
        # much to nudge each number to reduce the error," precomputed and
        # hardcoded (this is what "backpropagation" would do automatically in a
        # deep learning framework — here it's written out manually instead).
        B = len(y)
        z, E, S = self.logits(X)
        # NOTE: g = derivative of the loss w.r.t. the logit z. For sigmoid +
        # binary cross-entropy loss this simplifies neatly to
        # (predicted_probability - true_label) / batch_size — a well-known
        # shortcut, not something you'd need to re-derive by hand.
        g = ((sigmoid(z) - y) / B).astype(np.float32)    # (B,)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        # NOTE: np.add.at is a "scatter-add": normal indexed assignment
        # (arr[idx] = val) would silently OVERWRITE when the same idx repeats in
        # one batch (e.g. the same user_id appears twice in a batch of 8192
        # rows) — you'd lose one of the two updates. np.add.at instead
        # accumulates all of them, which is the mathematically correct behavior.
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W   # NOTE: L2 penalty's gradient term.
        self.t += 1
        # NOTE: Adam optimizer — instead of a plain "weight -= lr * gradient"
        # step, Adam keeps a running average of past gradients (M, "momentum" —
        # smooths out noisy updates) and a running average of past *squared*
        # gradients (Vv, "variance" — lets it take smaller steps on parameters
        # that have been changing wildly, bigger steps on ones that have been
        # stable). b1/b2 are the decay rates for those two running averages; the
        # `(1 - b1**t)` divisions are a standard correction so the averages
        # aren't biased low in early steps (when there's little history yet).
        # `eps` just avoids divide-by-zero. You don't need to derive Adam
        # yourself to use this file — treat it as a smarter, standard drop-in
        # replacement for plain gradient descent.
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        # NOTE: this return value is the batch's binary cross-entropy (logloss) —
        # a number that should trend downward while training works. It's printed
        # in run_fm()'s progress log, but it is NOT what decides when to stop
        # training — that's the VALIDATION primary score, checked separately
        # below in run_fm(). Training loss can keep improving even as the model
        # starts overfitting, which is exactly why validation is checked instead.
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def predict(self, X, bs=200_000):
        # NOTE: predicts in chunks of `bs` rows instead of all at once, purely
        # to avoid allocating one giant intermediate array in memory for large
        # splits. Doesn't change the result, just avoids running out of RAM.
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
    enc, dim = encode(splits)
    # NOTE: `dim` = total size of the embedding table, i.e. every possible value
    # across all 5 fields added together (computed in data.encode()).
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    # NOTE: "epoch" = one full pass over the entire training set. Training runs
    # for up to `epochs` passes, but usually stops early (see patience below).
    for ep in range(1, epochs + 1):
        # NOTE: shuffle the row order every epoch (rng.permutation) so the model
        # doesn't learn some spurious pattern from the fixed CSV row order, then
        # chop the shuffled data into "mini-batches" of size `bs` and take one
        # gradient step per batch. Batching is a memory/speed tradeoff — you
        # *could* update on 1 row at a time (slow, noisy) or the whole dataset
        # at once (memory-heavy, less frequent updates); 8192-row batches are
        # the practical middle ground.
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        # NOTE: "early stopping" — after every epoch, check the score on the
        # VALIDATION set (data the model never trains on). If it's a new best,
        # save a snapshot of the weights (`best_state`) and reset the "bad
        # epochs" counter. If it doesn't improve for `patience` epochs in a row,
        # stop and roll back to that best snapshot. This is how the model avoids
        # overfitting: training loss can keep dropping while validation score
        # gets worse (the model starts memorizing train instead of
        # generalizing) — early stopping catches that and keeps the good version.
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state   # NOTE: restore the best-validation-score weights,
    # discarding any later epochs that made things worse.
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}

if __name__ == '__main__':
    # NOTE: argparse = reads command-line flags like `--model fm --k 32` and
    # turns them into a Python object (a.model, a.k, ...). Standard CLI
    # plumbing, nothing ML-specific.
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='fm', choices=['pop', 'fm', 'random'])
    ap.add_argument('--k', type=int, default=16)        # NOTE: embedding vector length (see FM.__init__).
    ap.add_argument('--lr', type=float, default=0.001)   # NOTE: "learning rate" — step size per update.
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)       # NOTE: fixes randomness for reproducibility;
    # changing only this (everything else equal) is how baseline_scores.json's
    # "std across 5 seeds" noise numbers were measured — run with a few
    # different seeds to tell if a change you made is a real improvement or luck.
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    # NOTE: this dict-of-functions + `[a.model](splits)` is just "pick which
    # function to call based on a string" (a dispatch table) — equivalent to an
    # if/elif/else chain, just more compact. Nothing ML-specific about it.
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
