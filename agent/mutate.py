"""Node materialisation.

A node is a snapshot of the six block sources + a `Cfg` + an optional `cfg_ext.json` sidecar.
`materialize_child` copies the parent's blocks (or an adopted block set), applies the **validated**
config delta, and returns a `Provenance` record proving what will actually execute.

Two docs/EN/RESEARCH.md defects are closed here:
  docs/EN/SYSTEM.md §11  a delta is now filtered through `blockspec.validate_delta`, so knobs the mounted block set
        cannot read never reach `cfg.json` and never masquerade as an experiment.
  docs/EN/SYSTEM.md §12  node identity is the hash of the node's CONTENT (cfg + ext + six block sources), not of the
        unified diff against its parent, so a duplicate is detected however it was reached.
"""
from __future__ import annotations

import difflib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from agent import blockspec, provenance
from pipeline.contracts import Cfg
from pipeline.lib import ext as ext_mod

BLOCKS = ["features", "model", "loss", "train", "infer", "ensemble"]


def _snapshot(src_dir, dst_dir):
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for b in BLOCKS:
        shutil.copy(Path(src_dir) / f"{b}.py", dst / f"{b}.py")


def apply_delta(cfg: Cfg, delta_json) -> Cfg:
    """Apply a raw delta (JSON string or dict) with no validation -- used by the Reflector's
    `degrade` recovery path, where the harness (not the Proposer) chooses the keys."""
    d = json.loads(delta_json or "{}") if isinstance(delta_json, (str, bytes)) else dict(delta_json or {})
    base = asdict(cfg)
    base.update(d)
    return Cfg.from_dict(base)


def _apply_validated(cfg: Cfg, effective: dict) -> Cfg:
    base = asdict(cfg)
    base.update(effective)
    return Cfg.from_dict(base)


def blockset_of(cfg, adopt=None) -> str:
    """Which block set will execute: the adopted one if adopting, else the node's family label."""
    return adopt or getattr(cfg, "model_type", "fm") or "fm"


def materialize_root(run_dir, src_blocks="pipeline/baseline_blocks", seed=0):
    node_dir = Path(run_dir) / "nodes" / "root"
    blocks = node_dir / "blocks"
    _snapshot(src_blocks, blocks)
    cfg = Cfg(seed=seed)
    cfg.to_json(node_dir / "cfg.json")
    ext_mod.dump(str(node_dir), {})
    return str(node_dir), str(blocks), cfg


def materialize_named(run_dir, node_id, src_blocks, cfg, ext=None):
    """Snapshot an external block set (e.g. a saved cross-run champion) into nodes/<id>/."""
    node_dir = Path(run_dir) / "nodes" / node_id
    blocks = node_dir / "blocks"
    _snapshot(src_blocks, blocks)
    cfg.to_json(node_dir / "cfg.json")
    ext_mod.dump(str(node_dir), dict(ext or {}))
    return str(node_dir), str(blocks), cfg


def validate_hypothesis(parent, hypothesis, parent_ext=None) -> tuple:
    """Check a hypothesis against the block set that would execute it -- BEFORE any file is written.

    Returns (validation, blockset, intended_delta). A hypothesis whose `validation.has_effect` is
    False and which carries no block edit is a STRUCTURAL no-op: it provably cannot change execution,
    so the orchestrator re-proposes instead of training it (docs/EN/SYSTEM.md §12).
    """
    raw = hypothesis.config_delta_json or "{}"
    try:
        intended = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw or {})
        if not isinstance(intended, dict):
            intended = {}
    except ValueError:
        intended = {}
    adopt = getattr(hypothesis, "adopt_blockset", None)
    bs = blockset_of(parent.cfg, adopt)
    val = blockspec.validate_delta(bs, parent.cfg, intended, ext_current=parent_ext or {})
    return val, bs, intended


def materialize_child(run_dir, node_id, parent, hypothesis, block_edit,
                      validation=None, parent_ext=None, cache_version=0):
    """Write a child node to disk. Returns (node_dir, blocks_dir, cfg, diff, ext, prov)."""
    node_dir = Path(run_dir) / "nodes" / node_id
    blocks = node_dir / "blocks"
    adopt = getattr(hypothesis, "adopt_blockset", None)
    src = f"pipeline/lib/{adopt}_blocks" if adopt else parent.block_dir
    _snapshot(src, blocks)

    if validation is None:
        validation, _bs, _intended = validate_hypothesis(parent, hypothesis, parent_ext)
    _val, bs, intended = validation, blockset_of(parent.cfg, adopt), \
        json.loads(hypothesis.config_delta_json or "{}") if hypothesis.config_delta_json else {}

    # Only the VALIDATED effective keys reach the config the runner will read.
    cfg = _apply_validated(parent.cfg, _val.effective)
    if adopt:
        # The family label must follow the mounted blocks, else `assemble` groups every node into one
        # family and no ensemble forms (docs/EN/SYSTEM.md §6 (model_type must follow the blocks)).
        cfg = cfg.replace(model_type=adopt)
    cfg.to_json(node_dir / "cfg.json")

    ext = dict(parent_ext or {})
    ext.update(_val.ext_effective)
    ext_mod.dump(str(node_dir), ext)

    diff = ""
    target_block = None
    if adopt:
        diff = f"# adopted block set: {adopt}\n"
    elif hypothesis.mutation_kind == "block" and block_edit is not None:
        target_block = block_edit.target_block
        old = (Path(parent.block_dir) / f"{target_block}.py").read_text(encoding="utf-8")
        new = block_edit.new_source
        (blocks / f"{target_block}.py").write_text(new, encoding="utf-8")
        diff = "".join(difflib.unified_diff(
            old.splitlines(True), new.splitlines(True),
            fromfile=f"a/{target_block}.py", tofile=f"b/{target_block}.py"))

    prov = provenance.build(intended, _val, cfg, str(blocks), ext, bs,
                            adopt_blockset=adopt, target_block=target_block,
                            cache_version=cache_version)
    (node_dir / "provenance.json").write_text(
        json.dumps(prov.to_dict(), indent=2, default=list), encoding="utf-8")
    return str(node_dir), str(blocks), cfg, diff, ext, prov


def signature(cfg: Cfg, blocks_dir: str, ext: dict | None = None) -> str:
    """Content-based node identity (docs/EN/SYSTEM.md §12). Path-independent, unlike the old cfg+diff signature."""
    return provenance.content_signature(cfg, blocks_dir, ext)
