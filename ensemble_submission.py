"""Seed-ensemble models/fm_bpr.py: train the same model at N seeds and average their scores.

Why this is a different proposition from the popularity blend that failed. That blend added a
SECOND SOURCE of information the FM had already extracted (a per-video propensity it learns
anyway through W[video_id]), so it could only add redundancy. This averages N runs of the SAME
model over the same information, which cancels the part of each run's error that comes from
initialization and minibatch order rather than from the data. Different mechanism, and the one
place where "combine several models" is nearly free.

Scores are standardized WITHIN each user before averaging. That matters: seeds land on different
absolute score scales, so a plain mean would let whichever seed happens to have the widest spread
dominate. Standardizing per user is also exactly the right normalization for a metric that only
ever compares rows inside one user (see evaluate.py).

    python ensemble_submission.py --seeds 5 --split test --out submission_ens_test.csv

Prints the valid score after each additional seed, so the curve shows where it stops paying.

THE TEST SPLIT: like make_submission.py, this is a human-run script and the only kind of code
allowed to touch it. Early stopping inside each run still selects on valid alone.
"""
import argparse
import time

import numpy as np

from data import load
from evaluate import evaluate
from make_submission import train_bpr_keep_model
from models.fm_bpr import DEFAULTS
from submit import write_submission


def _per_user_z(user_codes, n_users, scores):
    """(s - mean_u) / std_u, vectorized via bincount. Order-preserving within each user, so it
    cannot change any single seed's own ranking — it only puts the seeds on a common scale."""
    counts = np.maximum(np.bincount(user_codes, minlength=n_users).astype(np.float64), 1.0)
    s1 = np.bincount(user_codes, weights=scores, minlength=n_users)
    s2 = np.bincount(user_codes, weights=scores * scores, minlength=n_users)
    mean = s1 / counts
    std = np.sqrt(np.maximum(s2 / counts - mean * mean, 0.0))
    std[std < 1e-9] = 1.0
    return (scores - mean[user_codes]) / std[user_codes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--split', default='test', choices=['valid', 'test'])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or f"submission_ens_{a.split}.csv"

    t0 = time.time()
    splits = load(a.data_dir)
    print("loaded: " + ", ".join(f"{k}={len(v):,d}" for k, v in splits.items()))

    codes, n_users, acc = {}, {}, {}
    for name in ('valid', 'test'):
        uniq, c = np.unique(np.asarray([x[1] for x in splits[name]]), return_inverse=True)
        codes[name], n_users[name] = c, len(uniq)
        acc[name] = np.zeros(len(splits[name]), dtype=np.float64)

    for s in range(a.seeds):
        cfg = {**DEFAULTS, 'seed': s}
        print(f"\n--- seed {s} ---")
        m, enc = train_bpr_keep_model(splits, cfg, verbose=False)
        for name in ('valid', 'test'):
            X, y, u = enc[name]
            raw = np.asarray(m.predict(X), dtype=np.float64)
            single = evaluate(u, y, raw)
            acc[name] += _per_user_z(codes[name], n_users[name], raw)
            if name == 'valid':
                print(f"  seed {s} alone : valid primary {single['primary']:.5f}")
        # Report the running ensemble after each seed, so the curve is visible.
        Xv, yv, uv = enc['valid']
        ens = evaluate(uv, yv, acc['valid'])
        print(f"  ensemble of {s + 1}: valid primary {ens['primary']:.5f} "
              f"| GAUC {ens['GAUC']:.5f} | nDCG@5 {ens['nDCG@5']:.5f}")

    print("\n=== final ensemble ===")
    for name in ('valid', 'test'):
        rows = splits[name]
        r = evaluate([x[1] for x in rows], [x[6] for x in rows], acc[name])
        mark = '  <- submission split' if name == a.split else ''
        print(f"  {name:5s}  primary {r['primary']:.4f} | GAUC {r['GAUC']:.4f} | "
              f"nDCG@5 {r['nDCG@5']:.4f}{mark}")

    write_submission(out, splits[a.split], acc[a.split])
    print(f"wrote {out}: {len(splits[a.split]):,d} rows "
          f"({a.seeds} seeds, {time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
