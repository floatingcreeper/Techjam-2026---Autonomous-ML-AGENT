"""Lever C data -- per-row auxiliary labels (is_click/like/follow/comment/forward), cached.

The frozen data.load() exposes only long_view; these auxiliary targets live in the raw logs. We
reproduce data.load()'s file order + date filter EXACTLY so the arrays are row-aligned to the base
cache, then agent.datced._assert_aux_aligned() verifies that alignment against the cached video ids
before any training trusts this cache.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from data import SPLITS                      # frozen: date ranges only, safe to read

AUX_COLUMNS = {                              # task name -> raw CSV column (all binary in v1)
    "click": "is_click", "like": "is_like", "follow": "is_follow",
    "comment": "is_comment", "forward": "is_forward",
}
LOG_FILES = ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")


def build(data_dir, cache_dir, force=False):
    fc = Path(cache_dir) / "aux"
    if (fc / "meta.json").exists() and not force:
        return json.loads((fc / "meta.json").read_text())
    fc.mkdir(parents=True, exist_ok=True)
    tasks = list(AUX_COLUMNS)
    rows = {name: {"aux": [], "vid": []} for name in SPLITS}
    for fname in LOG_FILES:
        with open(Path(data_dir) / fname, newline="") as fh:
            for r in csv.DictReader(fh):
                date = int(r["date"])
                for name, (lo, hi) in SPLITS.items():
                    if lo <= date <= hi:                       # disjoint ranges -> exactly one split
                        rows[name]["aux"].append([1.0 if r[AUX_COLUMNS[t]] != "0" else 0.0
                                                  for t in tasks])
                        rows[name]["vid"].append(int(r["video_id"]))
                        break
    sizes = {}
    for name in SPLITS:
        A = np.asarray(rows[name]["aux"], np.float32).reshape(-1, len(tasks))
        np.save(fc / f"{name}_aux.npy", A)
        np.save(fc / f"{name}_vid.npy", np.asarray(rows[name]["vid"], np.int64))
        sizes[name] = len(A)
    meta = {"tasks": tasks, "sizes": sizes}
    (fc / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_aux(cache_dir, name):
    fc = Path(cache_dir) / "aux"
    meta = json.loads((fc / "meta.json").read_text())
    A = np.load(fc / f"{name}_aux.npy", mmap_mode="r")
    return {t: A[:, j] for j, t in enumerate(meta["tasks"])}
