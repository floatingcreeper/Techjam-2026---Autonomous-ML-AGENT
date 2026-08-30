"""Multi-seed confirmation (F5 from Jon's archive): re-run a node's blocks under extra seeds and
decide on the seed-MEAN rather than a single lucky seed. Our adoption thresholds sit below the
per-seed noise floor (std ~0.0008), and tree.best() is a max over ~50 draws, so the single-best
pick is upward-biased; this guards the submission against that selection bias.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from agent import executor


def seed_scores(block_dir, cfg, cache_dir, timeout_s, out_root, seeds):
    """Re-run block_dir under each seed; return the valid primaries that ran cleanly."""
    out = []
    for s in seeds:
        c = cfg.replace(seed=int(s))
        od = Path(out_root) / f"seed{s}"; od.mkdir(parents=True, exist_ok=True)
        cp = od / "cfg.json"; c.to_json(cp)
        res, _ = executor.run_node(block_dir, str(od), str(cp), cache_dir, timeout_s)
        if not isinstance(res, executor.Failure):
            out.append(float(res["primary_valid"]))
    return out


def confirm(block_dir, cfg, orig_primary, current_best, cache_dir, timeout_s, out_root,
            extra_seeds=(1, 2), eps=0.0002):
    """Return (accept, seed_mean, per_seed_primaries). Short-circuits (no extra seeds) if the
    single-seed run can't even beat current_best -- no realistic seed-mean would either."""
    if orig_primary <= current_best:
        return False, orig_primary, [orig_primary]
    prims = [orig_primary] + seed_scores(block_dir, cfg, cache_dir, timeout_s, out_root, extra_seeds)
    mean = float(np.mean(prims))
    return mean > current_best + eps, mean, prims
