"""Adapter: wraps baseline.FM (the model class — forward pass, gradients, Adam optimizer, all
reused unmodified) behind the models/ convention (models/base.py). This is "current-best @
iteration 0" for the agent loop — decisions #10 in AGENT_STRATEGY.md: wrap, don't reimplement.

Why not just call baseline.run_fm() directly? run_fm() hardcodes `enc['test']` — fine for a human
running `python baseline.py --model fm` with the real, full splits dict, but the agent loop never
has a 'test' key at all (agent/data_guard.py strips it before anything here ever runs), so that
would KeyError. The epoch/batch/early-stopping DRIVER loop below duplicates run_fm()'s *shape* only
to the extent needed to generalize over "whichever non-train splits are present" instead of
assuming 'test' exists — the actual model math (forward pass, gradients, Adam) all still comes from
baseline.FM, reused unmodified. baseline.py itself is never touched — it stays the pristine
reference used by submit.py, ablation_features.py, and this repo's recorded baseline_scores.json.
"""
import time

import numpy as np

import baseline as B
from data import encode
from evaluate import evaluate
from models.base import non_train_splits

DEFAULT_CONFIG = {
    'k': 16, 'lr': 0.001, 'l2': 1e-6,
    'epochs': 40, 'patience': 4, 'batch_size': 8192,
    'seed': 0,
}


def train(splits, config=None, verbose=False):
    """splits: {'train': [...], 'valid': [...], ...} — see models/base.py's contract.
    config: overrides merged on top of DEFAULT_CONFIG.
    Returns {<split_name>: evaluate(...) dict} for every non-'train' key in splits.
    Raises on a genuinely broken run (e.g. every epoch produced a non-finite score) rather than
    silently returning nonsense — agent/debug_run.py and the orchestrator are expected to catch
    that and route to error_recovery, not treat a raised exception as unexpected.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    eval_names = non_train_splits(splits)
    if 'train' not in splits or not eval_names:
        raise ValueError("fm_v0.train needs a 'train' key and at least one other split to evaluate against")

    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']
    eval_enc = {name: enc[name] for name in eval_names}
    # Early stopping is tracked against 'valid' by convention (same rule baseline.run_fm uses:
    # stop on validation primary, never on training loss) — fall back to whichever split is first
    # if 'valid' isn't present (shouldn't happen in normal use, but don't crash over it).
    primary_split = 'valid' if 'valid' in eval_enc else eval_names[0]

    m = B.FM(dim, k=cfg['k'], lr=cfg['lr'], l2=cfg['l2'], seed=cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    bs = cfg['batch_size']

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, cfg['epochs'] + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        for i in range(0, len(idx), bs):
            m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
        Xp, yp, up = eval_enc[primary_split]
        cur = evaluate(up, yp, m.predict(Xp))
        if verbose:
            print(f"  epoch {ep:2d} | {primary_split} primary {cur['primary']:.4f} "
                  f"| {time.time() - t0:.1f}s")
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
        raise RuntimeError("fm_v0.train: no epoch produced a finite primary score — likely NaN "
                            "inputs or a broken config")
    m.V, m.W, m.b = best_state

    return {name: evaluate(u, y, m.predict(X)) for name, (X, y, u) in eval_enc.items()}
