"""Write a submission CSV from any models/ variant that exposes the training internals.

Why this exists rather than `submit.py --make`: that path hardcodes the official FM baseline
inline, and models/*.py's train() returns metrics only — it deliberately does not hand back the
fitted model, because models/base.py's contract is about scoring splits, not about exporting
predictions. So this driver re-runs the SAME training loop as models/fm_bpr.train, importing that
module's own _build_pair_index / _sample_pairs / _step_pair rather than copying them, and keeps
the model object so it can emit per-row scores.

    python make_submission.py --split test  --out submission_test.csv
    python make_submission.py --split valid --out submission_valid.csv
    python make_submission.py --model fm_v1 --split test --out submission_fm_v1.csv

THE TEST SPLIT. This script is the one place in the repo that is allowed to touch it, and only
because a human runs it deliberately — the same status submit.py has. The automated agent loop
loads data through agent/data_guard.load_train_valid(), which physically removes the 'test' key
before any agent-facing code can see it, and nothing here is importable from that path.

Early stopping still selects on VALID only (fm_bpr.train's `primary_split` logic prefers 'valid'
whenever it is present), so including 'test' in the splits dict changes which rows get SCORED, not
which epoch gets chosen. That distinction is the whole reason it is safe to encode all three
splits together: data.encode() fits its vocabularies and bucket edges on train alone.
"""
import argparse
import time

import numpy as np

import baseline as B
from data import load, encode
from evaluate import evaluate
from submit import write_submission


def train_bpr_keep_model(splits, cfg, verbose=True):
    """models/fm_bpr.train's loop, but returns (model, enc) instead of metrics."""
    from models.fm_bpr import _build_pair_index, _sample_pairs, _step_pair

    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']

    pos_idx, neg_flat, neg_start, neg_counts, codes = _build_pair_index(utr, ytr)
    n_pairs = max(1, int(len(pos_idx) * cfg['pairs_per_pos']))
    if verbose:
        print(f"  [fm_bpr] {len(pos_idx):,d} eligible positives, {n_pairs:,d} pairs/epoch, dim={dim}")

    m = B.FM(dim, k=cfg['k'], lr=cfg['lr'], l2=cfg['l2'], seed=cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    bs = cfg['batch_size']

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, cfg['epochs'] + 1):
        p, n = _sample_pairs(rng, n_pairs, pos_idx, neg_flat, neg_start, neg_counts, codes)
        for i in range(0, n_pairs, bs):
            _step_pair(m, Xtr[p[i:i + bs]], Xtr[n[i:i + bs]])
        cur = evaluate(uva, yva, m.predict(Xva))   # VALID only — never the submission split
        if verbose:
            print(f"  epoch {ep:2d} | valid primary {cur['primary']:.4f}")
        if cur['primary'] > best + 1e-5:
            best, bad = cur['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= cfg['patience']:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return m, enc


def train_pointwise_keep_model(splits, cfg, verbose=True):
    """The models/fm_v1.py loop (pointwise logloss, extra_fields=[]), same deal — kept so the
    two models can be compared on test under identical conditions rather than against the
    numbers recorded in baseline_scores.json from some other machine."""
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']

    m = B.FM(dim, k=cfg['k'], lr=cfg['lr'], l2=cfg['l2'], seed=cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    bs = cfg['batch_size']

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, cfg['epochs'] + 1):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), bs):
            m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
        cur = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | valid primary {cur['primary']:.4f}")
        if cur['primary'] > best + 1e-5:
            best, bad = cur['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= cfg['patience']:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return m, enc


BUILDERS = {'fm_bpr': train_bpr_keep_model, 'fm_v1': train_pointwise_keep_model}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='test', choices=['valid', 'test'])
    ap.add_argument('--model', default='fm_bpr', choices=sorted(BUILDERS))
    ap.add_argument('--out', default=None, help='submission CSV path (default: submission_<model>_<split>.csv)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    out = a.out or f"submission_{a.model}_{a.split}.csv"

    if a.model == 'fm_bpr':
        from models.fm_bpr import DEFAULTS
    else:
        from models.fm_v1 import DEFAULT_CONFIG as DEFAULTS
    cfg = {**DEFAULTS, 'seed': a.seed}

    t0 = time.time()
    splits = load(a.data_dir)
    print(f"loaded: " + ", ".join(f"{k}={len(v):,d}" for k, v in splits.items()))

    m, enc = BUILDERS[a.model](splits, cfg)

    # Report BOTH splits so the valid-to-test drop is visible in one place.
    for name in ('valid', 'test'):
        X, y, u = enc[name]
        r = evaluate(u, y, m.predict(X))
        mark = '  <- submission split' if name == a.split else ''
        print(f"  {name:5s}  primary {r['primary']:.4f} | GAUC {r['GAUC']:.4f} | "
              f"nDCG@5 {r['nDCG@5']:.4f}{mark}")

    X, _, _ = enc[a.split]
    write_submission(out, splits[a.split], m.predict(X))
    print(f"wrote {out}: {len(splits[a.split]):,d} rows "
          f"(model={a.model}, split={a.split}, seed={a.seed}, {time.time() - t0:.0f}s)")
    print(f"verify with: python submit.py --score {out} --split {a.split}")


if __name__ == '__main__':
    main()
