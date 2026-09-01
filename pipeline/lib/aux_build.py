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


CACHE_SPLITS = ("train",)     # the only slice a block may see -- fit_din reads aux["train"] only
HOLDOUT_SPLITS = ("valid", "test")


def build(data_dir, cache_dir, holdout_dir=None, force=False):
    """Build per-row auxiliary labels.

    docs/EN/RESEARCH.md §9: `is_click` correlates 0.75 with `long_view` and
    P(long_view=1 | is_click=0) = 0.002, so the valid/test aux slices are near-oracle proxies for the
    labels being scored -- ranking valid by `is_click` alone reaches primary 0.7466 against an FM
    baseline of 0.6015 and an oracle ceiling of 0.8484. Only the TRAIN slice is written into the
    block-visible cache; valid/test go to `holdout_dir` (outside `bundle.cache_dir`) purely so offline
    analysis remains possible for a human.
    """
    fc = Path(cache_dir) / "aux"
    hd = Path(holdout_dir) / "aux" if holdout_dir else None
    if (fc / "meta.json").exists() and not force:
        return json.loads((fc / "meta.json").read_text())
    fc.mkdir(parents=True, exist_ok=True)
    if hd:
        hd.mkdir(parents=True, exist_ok=True)
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
        vid = np.asarray(rows[name]["vid"], np.int64)
        dst = fc if name in CACHE_SPLITS else hd
        if dst is None:
            continue                                    # no holdout dir given -> simply do not emit
        np.save(dst / f"{name}_aux.npy", A)
        np.save(dst / f"{name}_vid.npy", vid)
        sizes[name] = len(A)
    # Remove any pre-v7 valid/test aux left in the block-visible cache by an older build.
    for name in HOLDOUT_SPLITS:
        for stem in ("aux", "vid"):
            stale = fc / f"{name}_{stem}.npy"
            if stale.exists():
                stale.unlink()
    meta = {"tasks": tasks, "sizes": sizes, "cache_splits": list(CACHE_SPLITS)}
    (fc / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_aux(cache_dir, name):
    """Auxiliary labels for a split. Only 'train' is available from the block-visible cache (docs/EN/SYSTEM.md §8)."""
    fc = Path(cache_dir) / "aux"
    meta = json.loads((fc / "meta.json").read_text())
    allowed = tuple(meta.get("cache_splits", CACHE_SPLITS))
    if name not in allowed:
        raise KeyError(
            f"auxiliary labels for split {name!r} are not available to pipeline blocks: they are "
            f"label-derived holdout data (is_click is a near-oracle proxy for long_view). "
            f"Available: {list(allowed)}.")
    A = np.load(fc / f"{name}_aux.npy", mmap_mode="r")
    return {t: A[:, j] for j, t in enumerate(meta["tasks"])}
