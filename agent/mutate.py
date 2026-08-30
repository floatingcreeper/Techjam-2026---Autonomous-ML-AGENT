"""Node materialisation.

A node is a snapshot of the six block sources + a cfg. materialize_child copies the parent's
blocks, applies the config delta, and (for a block mutation) overwrites exactly one block --
computing the unified diff for the run-log.
"""
from __future__ import annotations
import shutil, json, difflib, hashlib
from dataclasses import asdict
from pathlib import Path
from pipeline.contracts import Cfg

BLOCKS = ["features", "model", "loss", "train", "infer", "ensemble"]


def _snapshot(src_dir, dst_dir):
    dst = Path(dst_dir); dst.mkdir(parents=True, exist_ok=True)
    for b in BLOCKS:
        shutil.copy(Path(src_dir) / f"{b}.py", dst / f"{b}.py")


def apply_delta(cfg: Cfg, delta_json: str) -> Cfg:
    d = json.loads(delta_json or "{}")
    base = asdict(cfg); base.update(d)
    return Cfg.from_dict(base)


def materialize_root(run_dir, src_blocks="pipeline/baseline_blocks", seed=0):
    node_dir = Path(run_dir) / "nodes" / "root"
    blocks = node_dir / "blocks"
    _snapshot(src_blocks, blocks)
    cfg = Cfg(seed=seed)
    cfg.to_json(node_dir / "cfg.json")
    return str(node_dir), str(blocks), cfg


def materialize_named(run_dir, node_id, src_blocks, cfg):
    """Snapshot an external block set (e.g. a saved cross-run champion) into nodes/<id>/ with its
    cfg, so the fixed runner can re-validate it exactly like any other node. Returns
    (node_dir, blocks_dir, cfg)."""
    node_dir = Path(run_dir) / "nodes" / node_id
    blocks = node_dir / "blocks"
    _snapshot(src_blocks, blocks)
    cfg.to_json(node_dir / "cfg.json")
    return str(node_dir), str(blocks), cfg


def materialize_child(run_dir, node_id, parent, hypothesis, block_edit):
    """Returns (node_dir, blocks_dir, cfg, diff)."""
    node_dir = Path(run_dir) / "nodes" / node_id
    blocks = node_dir / "blocks"
    adopt = getattr(hypothesis, "adopt_blockset", None)
    src = f"pipeline/lib/{adopt}_blocks" if adopt else parent.block_dir
    _snapshot(src, blocks)
    cfg = apply_delta(parent.cfg, hypothesis.config_delta_json)
    if adopt:
        cfg = cfg.replace(model_type=adopt)   # family label follows the adopted blocks
    cfg.to_json(node_dir / "cfg.json")

    diff = ""
    if adopt:
        diff = f"# adopted block set: {adopt}\n"
    elif hypothesis.mutation_kind == "block" and block_edit is not None:
        tgt = block_edit.target_block
        old = (Path(parent.block_dir) / f"{tgt}.py").read_text(encoding="utf-8")
        new = block_edit.new_source
        (blocks / f"{tgt}.py").write_text(new, encoding="utf-8")
        diff = "".join(difflib.unified_diff(
            old.splitlines(True), new.splitlines(True),
            fromfile=f"a/{tgt}.py", tofile=f"b/{tgt}.py"))
    return str(node_dir), str(blocks), cfg, diff


def signature(cfg: Cfg, diff: str) -> str:
    return cfg.hash() + ":" + hashlib.sha256(diff.encode()).hexdigest()[:12]
