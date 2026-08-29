"""Lever B data -- per-user behavior sequences, cached.

For every impression, the history is that user's PRIOR interactions (across all splits, in
chronological order), truncated to the last L. Temporal safety is structural: we process rows
in global time order (train=file1 < valid < test) and append the current item to a user's
history only AFTER snapshotting -- so a row never sees its own outcome or any future row.

Chronology proxy: KuaiRand's logs are time-ordered within each file, and data.load concatenates
file1 (train dates) before file2 (valid then test dates), so row order approximates time.
"""
from __future__ import annotations
import json
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

SPLITS = ("train", "valid", "test")


def build(data_dir, cache_dir, L=30, force=False, splits=None):
    fc = Path(cache_dir) / "seq"
    meta_p = fc / "meta.json"
    if meta_p.exists() and not force:
        m = json.loads(meta_p.read_text())
        if m.get("L") == L:
            return m
    fc.mkdir(parents=True, exist_ok=True)
    if splits is None:
        from data import load
        splits = load(data_dir)

    # video vocab from train: 1..V ; 0 = PAD ; V+1 = UNK (unseen in valid/test)
    vv = {}
    for x in splits["train"]:
        if x[2] not in vv:
            vv[x[2]] = len(vv) + 1
    V = len(vv)
    UNK = V + 1

    hist = defaultdict(lambda: deque(maxlen=L))
    sizes = {}
    for name in SPLITS:
        rows = splits[name]
        N = len(rows)
        seq = np.zeros((N, L), np.int32)
        slen = np.zeros(N, np.int32)
        tgt = np.zeros(N, np.int32)
        for i, x in enumerate(rows):
            h = hist[x[1]]
            if h:
                a = np.fromiter(h, np.int32)
                seq[i, L - len(a):] = a                 # left-pad
                slen[i] = len(a)
            tgt[i] = vv.get(x[2], UNK)
            h.append(vv.get(x[2], UNK))                 # append AFTER snapshot (no leakage)
        np.save(fc / f"{name}_seq.npy", seq)
        np.save(fc / f"{name}_slen.npy", slen)
        np.save(fc / f"{name}_tgt.npy", tgt)
        sizes[name] = N

    meta = {"V": V, "UNK": UNK, "L": L, "sizes": sizes}
    meta_p.write_text(json.dumps(meta, indent=2))
    return meta


def load_split(cache_dir, name):
    fc = Path(cache_dir) / "seq"
    return (np.load(fc / f"{name}_tgt.npy"), np.load(fc / f"{name}_seq.npy"),
            np.load(fc / f"{name}_slen.npy"))


def meta(cache_dir):
    return json.loads((Path(cache_dir) / "seq" / "meta.json").read_text())
