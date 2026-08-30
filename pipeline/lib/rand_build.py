"""Lever E data -- the random-exposure log as an UNBIASED validation set.

`log_random_*.csv` is a randomized-exposure log (public, NOT the hidden test split), so a model that
scores well on the biased `valid` but poorly here is exploiting exposure-policy bias. We read it into
data.load()'s 7-tuple shape and encode it with the SAME vocab/offsets as train (via the frozen
data.encode), so the ids align with the base cache's index space.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from data import encode                 # frozen: fits vocab on splits['train'], encodes every split given

RAND_LOG = "log_random_4_22_to_5_08_pure.csv"


def _read_rand_rows(data_dir):
    vid2author = {}
    with open(Path(data_dir) / "video_features_basic_pure.csv") as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
    rows = []
    with open(Path(data_dir) / RAND_LOG) as fh:                    # same 7-tuple shape as data.load()
        for r in csv.DictReader(fh):
            rows.append((int(r["date"]), r["user_id"], r["video_id"],
                         vid2author.get(r["video_id"], "UNK"), r["tab"],
                         float(r["duration_ms"]), 1 if r["long_view"] != "0" else 0))
    return rows


def build(data_dir, cache_dir, train_rows, force=False):
    fc = Path(cache_dir) / "rand"
    if (fc / "meta.json").exists() and not force:
        return json.loads((fc / "meta.json").read_text())
    fc.mkdir(parents=True, exist_ok=True)
    rand_rows = _read_rand_rows(data_dir)
    enc, _ = encode({"train": train_rows, "rand": rand_rows})      # rand encoded with train's vocab
    X, y, users = enc["rand"]
    np.save(fc / "rand_X.npy", np.asarray(X, np.int32))
    np.save(fc / "rand_y.npy", np.asarray(y, np.float32))
    np.save(fc / "rand_u.npy", np.array([int(v) for v in users], dtype=np.int64))
    meta = {"size": int(len(y))}
    (fc / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_rand(cache_dir):
    fc = Path(cache_dir) / "rand"
    return (np.load(fc / "rand_u.npy"), np.load(fc / "rand_y.npy"))
