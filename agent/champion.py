"""Cross-run champion: a durable snapshot of the best validated single node, so a later run
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


def load_ext(champion_dir):
    """The champion's extension sidecar (may be absent for a pre-v7 champion)."""
    p = Path(champion_dir) / "cfg_ext.json"
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save(champion_dir, block_dir, cfg, primary_valid, cache_version, run_id, node_id,
         ext=None, code_state=""):
    d = Path(champion_dir); (d / "blocks").mkdir(parents=True, exist_ok=True)
    for b in BLOCKS:
        shutil.copy(Path(block_dir) / f"{b}.py", d / "blocks" / f"{b}.py")
    cfg.to_json(d / "cfg.json")
    # The extension sidecar is part of the node's identity, so a champion without it would
    # re-validate as a different experiment (see agent/provenance.content_signature).
    (d / "cfg_ext.json").write_text(json.dumps(dict(ext or {}), indent=2, sort_keys=True,
                                               default=list))
    (d / "champion.json").write_text(json.dumps({
        "primary_valid": float(primary_valid), "cache_version": cache_version,
        "run_id": run_id, "node_id": node_id, "cfg_hash": cfg.hash(),
        "code_state": code_state,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
