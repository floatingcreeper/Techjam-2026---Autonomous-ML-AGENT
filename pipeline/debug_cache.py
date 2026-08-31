"""Build a small, row-consistent subsample of runs/_cache for a debug/smoke run.

Every cache array is per-row and aligned across the base, gbm, seq (and aux) caches, so a single
shared row-index subset per split stays coherent everywhere. Global metas are copied with their
`sizes` patched. Used by executor.debug_gate() to fast-fail a candidate before a full run.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PER_ROW = {                       # subdir ("" = base) -> per-row array stems
    "":    ["X", "y", "u", "vid"],
    "gbm": ["X", "y", "u"],
    "seq": ["seq", "fb", "slen", "tgt"],
    "aux": ["aux", "vid"],        # present only after Feature 1
}


def build(cache_dir, out_dir, n_train=20_000, n_other=10_000, seed=0):
    src, dst = Path(cache_dir), Path(out_dir)
    meta = json.loads((src / "meta.json").read_text())
    rng = np.random.default_rng(seed)
    idx = {}
    for split, size in meta["sizes"].items():
        n = n_train if split == "train" else n_other
        idx[split] = np.arange(size) if size <= n else rng.choice(size, n, replace=False)
    new_sizes = {k: int(len(idx[k])) for k in idx}
    for sub, stems in PER_ROW.items():
        s = (src / sub) if sub else src
        d = (dst / sub) if sub else dst
        if not s.exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        for split in meta["sizes"]:
            for stem in stems:
                p = s / f"{split}_{stem}.npy"
                if p.exists():
                    np.save(d / f"{split}_{stem}.npy",
                            np.asarray(np.load(p, mmap_mode="r"))[idx[split]])
        mp = s / "meta.json"                       # copy sibling meta, patch sizes only
        if mp.exists():
            mm = json.loads(mp.read_text())
            if "sizes" in mm:
                # Only claim sizes for splits this sub-cache actually holds. Since v7 the aux cache
                # holds train only (the valid/test slices are label-derived holdout data, docs/EN/SYSTEM.md §8), so a
                # blanket `sizes = new_sizes` would advertise arrays that do not exist.
                mm["sizes"] = {k: v for k, v in new_sizes.items()
                               if (d / f"{k}_{stems[0]}.npy").exists()}
            (d / "meta.json").write_text(json.dumps(mm))
    return str(dst)
