"""Extension config sidecar -- knobs that do not fit the FROZEN `Cfg` dataclass.

`pipeline/contracts.py` is frozen, so `Cfg` can never gain a field, and `Cfg.from_dict` silently drops
unknown keys. Levers added after the contract was frozen (behavior-aware history, LightGBM
hyper-parameters, negative-sampling strategy) therefore keep their knobs in a sidecar
`cfg_ext.json` written next to `cfg.json` in the node directory by `agent.mutate`.

How a block reaches it
----------------------
`pipeline/run_node.py` loads each block with `importlib.util.spec_from_file_location`, so a block
module's `__file__` is `<node_dir>/blocks/<name>.py` and the sidecar is two levels up:

    from pipeline.lib import ext
    e = ext.load(__file__)
    if e.get("use_fb"): ...

No frozen file changes and no new import surface for agent-written blocks (`pipeline` is already on
`agent.executor.ALLOWLIST`).

Integrity: the sidecar is folded into the node's content signature and provenance hash by
`agent.mutate`, so it cannot be used to smuggle a change past deduplication or the audit trail.
"""
from __future__ import annotations

import json
import os

FILENAME = "cfg_ext.json"


def path_for(block_file: str) -> str:
    """`<node_dir>/cfg_ext.json` given `<node_dir>/blocks/<block>.py`."""
    blocks_dir = os.path.dirname(os.path.abspath(block_file))
    return os.path.join(os.path.dirname(blocks_dir), FILENAME)


def load(block_file: str, defaults: dict | None = None) -> dict:
    """Extension knobs for the node that owns `block_file`. Missing sidecar -> defaults only.

    Never raises: a node without a sidecar (e.g. the root, or a champion snapshot from an older run)
    must still run.
    """
    out = dict(defaults or {})
    try:
        with open(path_for(block_file), "r", encoding="utf-8") as fh:
            out.update(json.load(fh))
    except (OSError, ValueError):
        pass
    return out


def dump(node_dir: str, data: dict) -> str:
    """Write the sidecar for a node. Returns the path written."""
    p = os.path.join(node_dir, FILENAME)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True, default=list)
    return p
