"""Pairwise Factorization Machine for within-user ranking on KuaiRand-Pure.

The official FM uses pointwise log loss.  This model samples a positive and a
negative impression from the same user and optimises BPR loss, which directly
encourages the positive item to rank higher.  It accepts train/validation-only
splits so research runs do not score the test split.
"""
import argparse
import collections
import time

import numpy as np

from data import FIELDS, encode, load
from evaluate import evaluate


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class IntraUserPairSampler:
    """Draw positive/negative training indices from the same user."""

    def __init__(self, labels, users, seed=0):
        positives = collections.defaultdict(list)
        negatives = collections.defaultdict(list)
        for index, (label, user) in enumerate(zip(labels, users)):
            (positives if label > 0.5 else negatives)[user].append(index)

        self.users = [user for user in positives if user in negatives]
        self.positives = {
            user: np.asarray(positives[user], dtype=np.int32) for user in self.users
        }
        self.negatives = {
            user: np.asarray(negatives[user], dtype=np.int32) for user in self.users
        }
        self.rng = np.random.default_rng(seed)

    @property
    def positives_per_epoch(self):
        return sum(len(indices) for indices in self.positives.values())

    def sample_pairs(self, negative_ratio=1):
        positive_parts, negative_parts = [], []
        for user in self.users:
            pos = self.positives[user]
            neg = self.negatives[user]
            positive_parts.append(np.repeat(pos, negative_ratio))
            negative_parts.append(
                self.rng.choice(neg, size=len(pos) * negative_ratio, replace=True)
            )
        positive = np.concatenate(positive_parts)
        negative = np.concatenate(negative_parts)
        order = self.rng.permutation(len(positive))
        return positive[order], negative[order]


class BPRFM:
    """FM with Adam updates for the pairwise BPR objective."""

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        embeddings = self.V[X]
        summed = embeddings.sum(axis=1)
        interactions = 0.5 * ((summed ** 2).sum(axis=1) - (embeddings ** 2).sum(axis=(1, 2)))
        return self.W[X].sum(axis=1) + interactions, embeddings, summed

    def step_pair(self, X_pos, X_neg):
        batch_size = len(X_pos)
        pos_score, pos_embeddings, pos_summed = self.logits(X_pos)
        neg_score, neg_embeddings, neg_summed = self.logits(X_neg)
        probability = sigmoid(pos_score - neg_score)
        pos_gradient = ((probability - 1.0) / batch_size).astype(np.float32)
        neg_gradient = -pos_gradient

        grad_v = np.zeros_like(self.V)
        grad_w = np.zeros_like(self.W)
        np.add.at(grad_w, X_pos, pos_gradient[:, None])
        np.add.at(grad_w, X_neg, neg_gradient[:, None])
        np.add.at(grad_v, X_pos,
                  pos_gradient[:, None, None] * (pos_summed[:, None, :] - pos_embeddings))
        np.add.at(grad_v, X_neg,
                  neg_gradient[:, None, None] * (neg_summed[:, None, :] - neg_embeddings))
        grad_v += self.l2 * self.V
        grad_w += self.l2 * self.W

        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, gradient, first, second in (
            (self.V, grad_v, self.mV, self.vV),
            (self.W, grad_w, self.mW, self.vW),
        ):
            first *= beta1
            first += (1 - beta1) * gradient
            second *= beta2
            second += (1 - beta2) * (gradient * gradient)
            parameter -= self.lr * (first / (1 - beta1 ** self.t)) / (
                np.sqrt(second / (1 - beta2 ** self.t)) + epsilon
            )
        return float(np.mean(-np.log(probability + 1e-9)))

    def predict(self, X, batch_size=200_000):
        return np.concatenate([
            self.logits(X[start:start + batch_size])[0]
            for start in range(0, len(X), batch_size)
        ])


def run_bpr_fm(splits, k=16, lr=0.001, l2=1e-6, epochs=30, batch_size=8192,
               negative_ratio=1, patience=6, seed=0, verbose=True,
               return_predictions=False):
    encoded, dim = encode(splits)
    X_train, y_train, users_train = encoded['train']
    X_valid, y_valid, users_valid = encoded['valid']
    sampler = IntraUserPairSampler(y_train, users_train, seed=seed)
    if not sampler.users:
        raise ValueError('no users have both positive and negative training impressions')

    if verbose:
        pair_count = sampler.positives_per_epoch * negative_ratio
        print(f'pairwise users {len(sampler.users):,} | pairs/epoch {pair_count:,}')

    model = BPRFM(dim, k=k, lr=lr, l2=l2, seed=seed)
    best_primary, best_state, stale = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        started = time.time()
        pos_idx, neg_idx = sampler.sample_pairs(negative_ratio=negative_ratio)
        losses = []
        for start in range(0, len(pos_idx), batch_size):
            stop = start + batch_size
            losses.append(model.step_pair(X_train[pos_idx[start:stop]], X_train[neg_idx[start:stop]]))
        metrics = evaluate(users_valid, y_valid, model.predict(X_valid))
        if verbose:
            print(f"  epoch {epoch:2d} | loss {np.mean(losses):.4f} | valid GAUC {metrics['GAUC']:.4f} "
                  f"nDCG@5 {metrics['nDCG@5']:.4f} primary {metrics['primary']:.4f} "
                  f"| {time.time() - started:.1f}s")
        if metrics['primary'] > best_primary + 1e-5:
            best_primary = metrics['primary']
            best_state = (model.V.copy(), model.W.copy())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f'  early stop at epoch {epoch}')
                break

    model.V, model.W = best_state
    valid_scores = model.predict(X_valid)
    result = {'valid': evaluate(users_valid, y_valid, valid_scores)}
    if 'test' in encoded:
        X_test, y_test, users_test = encoded['test']
        result['test'] = evaluate(users_test, y_test, model.predict(X_test))
    if return_predictions:
        return result, valid_scores
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--negative_ratio', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--valid_only', action='store_true')
    args = parser.parse_args()

    splits = load(args.data_dir, requested_splits=('train', 'valid') if args.valid_only else None)
    print({name: len(rows) for name, rows in splits.items()}, f'fields={FIELDS}')
    results = run_bpr_fm(
        splits, k=args.k, lr=args.lr, l2=args.l2, epochs=args.epochs,
        negative_ratio=args.negative_ratio, seed=args.seed,
    )
    print(f'\n=== BPR-FM (seed={args.seed}) ===')
    for split in ('valid', 'test'):
        if split in results:
            metrics = results[split]
            print(f"  {split:5s}  GAUC {metrics['GAUC']:.4f} | nDCG@5 {metrics['nDCG@5']:.4f} "
                  f"| primary {metrics['primary']:.4f}")
