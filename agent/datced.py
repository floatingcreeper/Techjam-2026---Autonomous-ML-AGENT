"""DataBundle builder + cache.

Encodes the fixed harness's data.load()/data.encode() output ONCE into memory-mapped
.npy files under runs/_cache, so every node loads it in well under a second instead of
re-reading 106 MB of CSV. This is what keeps a 50-iteration run inside the wall-clock budget.

M0 scope: base 5-field encoded arrays for train/valid/test. Sequences, negative-sampling
index, aux labels, and the random-exposure log are added in later milestones (they extend
this cache without changing the base layout).
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

SPLITS = ("train", "valid", "test")
CACHE_VERSION = 10          # bump when the cached array layout changes (forces a rebuild)
SEQ_L = 30                 # max user-history length for Lever B (DIN)

# docs/SYSTEM.md §8 -- label-derived data that must NOT sit inside the block-visible cache.
# History: v6 wrote runs/_cache/test_y.npy and runs/_cache/aux/{valid,test}_aux.npy. Every block
# receives `bundle.cache_dir` and numpy is on the executor allowlist, so `is_click` for test rows was
# reachable in one line -- and scoring by is_click alone reaches primary 0.7466 on valid (58.8% of the
# entire oracle headroom above FM). v7 keeps those arrays in a sibling HOLDOUT directory that is never
# handed to a block. Nothing in the pipeline needs them: run_node evaluates valid only, fit_din reads
# aux["train"] only, and `submit.py --score` reads test labels straight from the raw CSVs.
HOLDOUT_DIRNAME = "_holdout"


def holdout_dir(cache_dir: str) -> str:
    """Sibling of the cache, never passed to a block. See docs/SYSTEM.md §8."""
    return str(Path(cache_dir).parent / HOLDOUT_DIRNAME)


@dataclass
class Bundle:
    X: dict          # split -> int32 (N,F)  (mmap)
    y: dict          # split -> float32 (N,)
    users: dict      # split -> int64 (N,)
    dim: int
    field_dims: list | None
    n_fields: int
    cache_dir: str = ""   # so blocks can load sibling caches (e.g. gbm features)


def build_or_load(data_dir: str, cache_dir: str, force: bool = False) -> dict:
    """Build the cache if missing; return its meta dict. Idempotent."""
    cache = Path(cache_dir)
    meta_p = cache / "meta.json"
    if meta_p.exists() and not force:
        meta = json.loads(meta_p.read_text())
        if meta.get("cache_version") == CACHE_VERSION:
            return meta                                   # up-to-date cache

    cache.mkdir(parents=True, exist_ok=True)
    hold = Path(holdout_dir(cache_dir))
    hold.mkdir(parents=True, exist_ok=True)
    from data import load, encode, FIELDS          # fixed harness
    splits = load(data_dir)
    enc, dim = encode(splits)

    sizes = {}
    for name in SPLITS:
        X, y, users = enc[name]
        u = np.array([int(v) for v in users], dtype=np.int64)   # user_id codes for grouping
        # raw ids in data.load() row order, for building --check-valid submissions at finalize
        vid = np.array([int(r[2]) for r in splits[name]], dtype=np.int64)
        np.save(cache / f"{name}_X.npy", np.asarray(X, dtype=np.int32))
        np.save(cache / f"{name}_u.npy", u)
        np.save(cache / f"{name}_vid.npy", vid)
        # docs/SYSTEM.md §8: hidden-test labels go to the holdout dir, never into the block-visible cache.
        ydst = (hold if name == "test" else cache) / f"{name}_y.npy"
        np.save(ydst, np.asarray(y, dtype=np.float32))
        sizes[name] = int(len(y))
    # A stale v6 cache leaves test_y.npy behind; remove it so an upgrade actually closes the hole.
    for stale in (cache / "test_y.npy",):
        if stale.exists():
            stale.unlink()

    # Lever D features (LightGBM) live alongside, reusing the already-loaded splits
    from pipeline.lib import gbm
    gbm.build_features(data_dir, str(cache), force=True, splits=splits)
    # Lever B sequences (DIN)
    from pipeline.lib import seq_build
    seq_build.build(data_dir, str(cache), L=SEQ_L, force=True, splits=splits)
    # Lever C auxiliary labels (re-reads raw logs for the aux columns data.load() drops).
    # docs/SYSTEM.md §8: only the TRAIN slice lands in the cache -- fit_din reads aux["train"] and nothing else.
    # valid/test aux (is_click correlates 0.75 with long_view) go to the holdout dir.
    from pipeline.lib import aux_build
    aux_build.build(data_dir, str(cache), str(hold), force=True)
    _assert_aux_aligned(str(cache))
    # Lever E: the random-exposure log as an unbiased validation set (public; train-vocab encoded)
    from pipeline.lib import rand_build
    rand_build.build(data_dir, str(cache), splits["train"], force=True)

    meta = {"cache_version": CACHE_VERSION, "dim": int(dim), "n_fields": len(FIELDS),
            "fields": list(FIELDS), "field_dims": None, "sizes": sizes}
    meta_p.write_text(json.dumps(meta, indent=2))
    return meta


def _assert_aux_aligned(cache_dir: str) -> None:
    """Hard guard: aux rows must match base rows (aux_build re-derives row order independently, so a
    silent drift would misattribute every aux label). Compare against the cached {split}_vid.npy the
    base builder wrote in data.load() order.

    Since v7 only the TRAIN aux slice exists in the cache, so this also asserts that the valid/test
    aux arrays -- the `is_click` proxy for the labels being scored -- are absent from the
    block-visible interface (docs/SYSTEM.md §8)."""
    cache = Path(cache_dir)
    base_vid = np.load(cache / "train_vid.npy")
    aux_vid = np.load(cache / "aux" / "train_vid.npy")
    if base_vid.shape != aux_vid.shape or not np.array_equal(base_vid, aux_vid):
        raise RuntimeError(
            f"aux cache misaligned with base cache on split 'train' "
            f"(aux {aux_vid.shape} vs base {base_vid.shape}) -- aux_build's read order "
            f"diverged from data.load(); refusing to train on this cache.")
    leaked = [p.name for name in ("valid", "test")
              for p in (cache / "aux" / f"{name}_aux.npy",) if p.exists()]
    leaked += [p.name for p in (cache / "test_y.npy",) if p.exists()]
    if leaked:
        raise RuntimeError(
            f"holdout leakage: {leaked} present in the block-visible cache {cache_dir!r}. "
            f"These are label-derived arrays for splits the agent must not see; they belong in "
            f"{holdout_dir(cache_dir)!r}. Refusing to run.")


def load_bundle(cache_dir: str) -> Bundle:
    cache = Path(cache_dir)
    meta = json.loads((cache / "meta.json").read_text())
    X, y, users = {}, {}, {}
    for name in SPLITS:
        X[name] = np.load(cache / f"{name}_X.npy", mmap_mode="r")
        users[name] = np.load(cache / f"{name}_u.npy", mmap_mode="r")
        if name != "test":                 # F6 guard: never expose hidden-test labels to agent blocks
            y[name] = np.load(cache / f"{name}_y.npy", mmap_mode="r")
    rand_dir = cache / "rand"              # Lever E: unbiased-exposure split (public labels -> kept)
    if (rand_dir / "rand_X.npy").exists():
        X["rand"] = np.load(rand_dir / "rand_X.npy", mmap_mode="r")
        y["rand"] = np.load(rand_dir / "rand_y.npy", mmap_mode="r")
        users["rand"] = np.load(rand_dir / "rand_u.npy", mmap_mode="r")
    return Bundle(X=X, y=y, users=users, dim=meta["dim"],
                  field_dims=meta.get("field_dims"), n_fields=meta["n_fields"],
                  cache_dir=str(cache))
