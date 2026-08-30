"""Lever D -- LightGBM LambdaRank on engineered item/author features.

A tree-based learner with a complementary inductive bias to the embedding FM. The features
that matter for *within-user* ranking are the ones that VARY across a user's impressions:
item long_view rate, author rate, duration, tab, and global item statistics. (Pure user-side
features are constant within a user and carry no ranking signal -- the organizer insight.)

Leakage-safe: all aggregates are computed from the TRAIN split only. No current-row play_time
(that is the label's source).
"""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

SPLITS = ("train", "valid", "test")
PRIOR = 20.0
N_VSTAT = 16          # first N numeric columns of video_features_statistic_pure.csv


def build_features(data_dir, cache_dir, force=False, splits=None):
    fc = Path(cache_dir) / "gbm"
    if (fc / "meta.json").exists() and not force:
        return json.loads((fc / "meta.json").read_text())
    fc.mkdir(parents=True, exist_ok=True)
    if splits is None:
        from data import load
        splits = load(data_dir)

    ipos, iimp = defaultdict(float), defaultdict(float)
    apos, aimp = defaultdict(float), defaultdict(float)
    for x in splits["train"]:
        vid, aid, lab = x[2], x[3], x[6]
        iimp[vid] += 1; ipos[vid] += lab
        aimp[aid] += 1; apos[aid] += lab
    gmean = sum(ipos.values()) / max(1.0, sum(iimp.values()))

    def irate(v): return (ipos[v] + PRIOR * gmean) / (iimp[v] + PRIOR) if iimp[v] else gmean
    def arate(a): return (apos[a] + PRIOR * gmean) / (aimp[a] + PRIOR) if aimp[a] else gmean

    # global item statistics (popularity/engagement), z-scored on train presence
    vstat, cols = {}, []
    stat_path = Path(data_dir) / "video_features_statistic_pure.csv"
    if stat_path.exists():
        with open(stat_path) as fh:
            rd = csv.DictReader(fh)
            cols = [c for c in (rd.fieldnames or []) if c != "video_id"][:N_VSTAT]
            for r in rd:
                try:
                    vstat[r["video_id"]] = [float(r[c] or 0.0) for c in cols]
                except Exception:
                    vstat[r["video_id"]] = [0.0] * len(cols)
    zero = [0.0] * len(cols)

    sizes = {}
    for name in SPLITS:
        rows = splits[name]
        feats = np.empty((len(rows), 6 + len(cols)), dtype=np.float32)
        y = np.empty(len(rows), np.float32)
        u = np.empty(len(rows), np.int64)
        for i, x in enumerate(rows):
            vid, aid = x[2], x[3]
            base = [irate(vid), np.log1p(iimp[vid]), arate(aid), np.log1p(aimp[aid]),
                    np.log1p(x[5]), float(x[4])]
            feats[i] = base + vstat.get(vid, zero)
            y[i] = x[6]; u[i] = int(x[1])
        # z-score the stat columns using train mean/std
        if name == "train":
            mu = feats[:, 6:].mean(0); sd = feats[:, 6:].std(0) + 1e-6
        feats[:, 6:] = (feats[:, 6:] - mu) / sd
        np.save(fc / f"{name}_X.npy", feats)
        np.save(fc / f"{name}_y.npy", y)
        np.save(fc / f"{name}_u.npy", u)
        sizes[name] = len(rows)

    meta = {"n_features": 6 + len(cols), "stat_cols": cols, "sizes": sizes}
    (fc / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _load(fc, name):
    return (np.load(fc / f"{name}_X.npy"), np.load(fc / f"{name}_y.npy"),
            np.load(fc / f"{name}_u.npy"))


def load_features(cache_dir):
    """Return (X, y, users) dicts keyed by split, for a features block."""
    fc = Path(cache_dir) / "gbm"
    X, y, u = {}, {}, {}
    for name in SPLITS:
        X[name], y[name], u[name] = _load(fc, name)
    return X, y, u


def _groups(u_sorted):
    _, counts = np.unique(u_sorted, return_counts=True)
    return counts


def train_ranker(cache_dir, cfg=None):
    import lightgbm as lgb
    fc = Path(cache_dir) / "gbm"
    Xtr, ytr, utr = _load(fc, "train")
    Xva, yva, uva = _load(fc, "valid")
    o = np.argsort(utr, kind="stable"); Xtr, ytr, utr = Xtr[o], ytr[o], utr[o]
    ov = np.argsort(uva, kind="stable"); Xva2, yva2, uva2 = Xva[ov], yva[ov], uva[ov]
    dtr = lgb.Dataset(Xtr, label=ytr, group=_groups(utr))
    dva = lgb.Dataset(Xva2, label=yva2, group=_groups(uva2), reference=dtr)
    params = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5],
              "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "label_gain": [0, 1], "verbose": -1, "seed": (cfg.seed if cfg else 0)}
    model = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
    return model


def predict(model, cache_dir, split):
    fc = Path(cache_dir) / "gbm"
    X, _, _ = _load(fc, split)
    return model.predict(X, num_iteration=model.best_iteration).astype(np.float32)
