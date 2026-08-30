"""Cross-run champion (F3): a durable snapshot of the best validated single node, so a later run
resumes from the high-water mark instead of starting cold. Stored as a block snapshot + cfg + a
small metadata file, mirroring the run-node layout so it can be re-validated by the normal runner.
"""
from __future__ import annotations
import json, shutil, time
from pathlib import Path

BLOCKS = ("features", "model", "loss", "train", "infer", "ensemble")


def load(champion_dir):
    meta = Path(champion_dir) / "champion.json"
    if not meta.exists():
        return None
    try:
        return json.loads(meta.read_text())
    except (json.JSONDecodeError, OSError):
        return None                     # corrupt meta -> treat as no champion, never crash a run


def save(champion_dir, block_dir, cfg, primary_valid, cache_version, run_id, node_id):
    d = Path(champion_dir); (d / "blocks").mkdir(parents=True, exist_ok=True)
    for b in BLOCKS:
        shutil.copy(Path(block_dir) / f"{b}.py", d / "blocks" / f"{b}.py")
    cfg.to_json(d / "cfg.json")
    (d / "champion.json").write_text(json.dumps({
        "primary_valid": float(primary_valid), "cache_version": cache_version,
        "run_id": run_id, "node_id": node_id, "cfg_hash": cfg.hash(),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
