"""Save score arrays for splits beyond `run_node`'s single `--extra-split`.

`pipeline/run_node.py` is frozen and can infer exactly one extra split per invocation. Two splits are
wanted from the SAME trained model:

  * `test` -- so the submitted predictions come from the same instance whose validation predictions
    drove model and portfolio selection. docs/SYSTEM.md §18 (model identity at finalization) measured why this matters: re-training a
    DIN produces a genuinely different model (rank correlation 0.926, primary differing by 0.00042),
    so the old finalize-time retrain broke the best-checkpoint invariant for torch families.
  * `rand` -- the random-exposure surface (Lever E / docs/RESEARCH.md §15), reported as a SECOND scientific surface
    and never as the competition target.

`--extra-split test` is passed on every node run, and the infer block writes the rand array itself as
a side effect. The node directory is recoverable from the block module's `__file__` because
`run_node._load_block` loads blocks with `importlib.util.spec_from_file_location`.

This routes AROUND the frozen runner rather than through it -- the same pattern as F5 (subsampled
cache) and F6 (load-time label withholding).
"""
from __future__ import annotations

import os

import numpy as np


def node_dir(block_file: str) -> str:
    """`<node_dir>` given `<node_dir>/blocks/<block>.py`."""
    return os.path.dirname(os.path.dirname(os.path.abspath(block_file)))


def save(block_file: str, split: str, scores) -> str | None:
    """Write `<node_dir>/<split>_scores.npy`. Never raises -- a failed side effect must not fail a
    training run that has already succeeded."""
    try:
        p = os.path.join(node_dir(block_file), f"{split}_scores.npy")
        np.save(p, np.asarray(scores, np.float32))
        return p
    except Exception:
        return None


def can_infer_rand(feats) -> bool:
    """Whether this feature set actually carries the random-exposure split.

    True for the FM family (`load_bundle` puts X/y/users["rand"] in the bundle). False for DIN and
    LightGBM until their sibling rand caches exist (`seq`/`gbm` have no rand slice) -- the documented
    v1 scope limit, surfaced here as a value rather than an exception.
    """
    try:
        return "rand" in feats.X and feats.seq is None and getattr(feats, "vstat", None) is None
    except Exception:
        return False
