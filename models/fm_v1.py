"""Adapter, like models/fm_v0.py (wraps baseline.FM — Adam/forward/backward reused unmodified),
but supports the agent's `toggle_field` action: trains on FIELDS + zero-or-more extra CWM fields
(data.EXTRA_FIELDS — music_id, video_type, upload_type, follow_user_num_range, register_days_range,
fans_user_num_range, friend_user_num_range, user_active_degree), via data.encode_with_extra_fields()
(additive to data.py — doesn't touch FIELDS/encode()/load()).

config adds two keys beyond fm_v0.DEFAULT_CONFIG:
  'extra_fields': list of field names, subset of data.EXTRA_FIELDS, default [] — with [], this
                  behaves IDENTICALLY to fm_v0 (same field list, same everything).
  'data_dir':     needed only to load the two side-info CSVs the extra fields come from
                  (video_features_basic_pure.csv, user_features_pure.csv) — these carry no
                  interaction/label data, so this does not violate models/base.py's "never load
                  interaction-log data yourself" rule; `splits` (the actual labeled rows, subject
                  to the hidden-test guard) still always comes from the caller.

This is now the default model for agent/orchestrator.py and agent/cli.py — it's a strict superset
of fm_v0's behavior, not a separate thing to choose between, so there's no need for the
not-yet-executable `swap_model_variant` action type to pick between them.
"""
import time

import numpy as np

import baseline as B
from data import encode_with_extra_fields
from evaluate import evaluate
from models.base import non_train_splits

DEFAULT_CONFIG = {
    'k': 16, 'lr': 0.001, 'l2': 1e-6,
    'epochs': 40, 'patience': 4, 'batch_size': 8192,
    'seed': 0,
    'extra_fields': [],
    'data_dir': './KuaiRand-Pure/data',
}


def train(splits, config=None, verbose=False):
    """splits: {'train': [...], 'valid': [...], ...} — see models/base.py's contract.
    config: overrides merged on top of DEFAULT_CONFIG.
    Returns {<split_name>: evaluate(...) dict} for every non-'train' key in splits."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    eval_names = non_train_splits(splits)
    if 'train' not in splits or not eval_names:
        raise ValueError("fm_v1.train needs a 'train' key and at least one other split to evaluate against")

    enc, dim, field_list = encode_with_extra_fields(splits, cfg['data_dir'], cfg['extra_fields'])
    if verbose and cfg['extra_fields']:
        print(f"  [fm_v1] fields: {field_list}")
    Xtr, ytr, _ = enc['train']
    eval_enc = {name: enc[name] for name in eval_names}
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
        raise RuntimeError("fm_v1.train: no epoch produced a finite primary score — likely NaN "
                            "inputs or a broken config")
    m.V, m.W, m.b = best_state

    return {name: evaluate(u, y, m.predict(X)) for name, (X, y, u) in eval_enc.items()}
