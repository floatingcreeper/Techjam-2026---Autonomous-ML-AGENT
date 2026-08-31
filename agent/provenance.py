"""Effective-intervention provenance -- docs/SYSTEM.md §11 (config-effectiveness validation).

The silent-no-op bug proved that

    intended intervention  !=  executed intervention

can hold silently for an entire run: the agent asked for BPR, the harness trained BCE, and the run log
recorded "BPR -> 0.6015" as a scientific result. Recording only the *hypothesis* is therefore not an
audit trail. This module records what was actually executed, so any node can be replayed and any claim
can be traced to the code and configuration that produced it.

A provenance record answers, for one node:
  * what was asked for            (intended_delta / adopt_blockset / target_block)
  * what survived validation      (effective_delta, ext_effective, and every rejected key + reason)
  * what actually ran             (effective cfg hash, per-block source hashes, content signature)
  * in what environment           (cache version, code state, seed, interpreter)
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

BLOCKS = ["features", "model", "loss", "train", "infer", "ensemble"]

# Source tree whose contents define the "code state" of an experiment. Two nodes with the same cfg and
# the same block sources are still not comparable if the libs those blocks delegate to changed -- which
# is exactly what makes the pooled `din/bce` rows in docs/RESEARCH.md §8 (auxiliary tasks) uninterpretable.
_CODE_ROOTS = ("pipeline", "agent")
_FROZEN = ("data.py", "evaluate.py", "submit.py")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_hash(path) -> str:
    try:
        return _sha(Path(path).read_bytes())[:16]
    except OSError:
        return ""


def block_hashes(blocks_dir) -> dict:
    """Per-block source hash. The identity of the code that will actually execute."""
    d = Path(blocks_dir)
    return {b: file_hash(d / f"{b}.py") for b in BLOCKS}


def blocks_digest(blocks_dir) -> str:
    """One hash over all six block sources, order-stable."""
    h = block_hashes(blocks_dir)
    return _sha(json.dumps(h, sort_keys=True).encode())[:16]


_code_state_cache: dict = {}


def code_state(root: str = ".") -> str:
    """Hash of every tracked .py under pipeline/ and agent/, plus the frozen harness.

    Preferred over a git SHA because it is correct with uncommitted changes, which is the normal state
    during development. Cached per root: the tree does not change inside one run.
    """
    if root in _code_state_cache:
        return _code_state_cache[root]
    rp = Path(root)
    parts = []
    for sub in _CODE_ROOTS:
        for p in sorted((rp / sub).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            parts.append(f"{p.relative_to(rp).as_posix()}:{file_hash(p)}")
    for f in _FROZEN:
        parts.append(f"{f}:{file_hash(rp / f)}")
    out = _sha("\n".join(parts).encode())[:16]
    _code_state_cache[root] = out
    return out


@dataclass
class Provenance:
    """Everything needed to prove the intended intervention was the executed intervention."""
    intended_delta: dict = field(default_factory=dict)
    effective_delta: dict = field(default_factory=dict)
    ext_effective: dict = field(default_factory=dict)
    rejected_not_honoured: dict = field(default_factory=dict)
    rejected_invalid: dict = field(default_factory=dict)
    rejected_ineffective: dict = field(default_factory=dict)
    rejection_reasons: list = field(default_factory=list)
    adopt_blockset: str | None = None
    target_block: str | None = None
    blockset: str = "fm"
    effective_cfg_hash: str = ""
    block_hashes: dict = field(default_factory=dict)
    blocks_digest: str = ""
    content_signature: str = ""
    cache_version: int = 0
    code_state: str = ""
    seed: int = 0
    python: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def intervention_matched(self) -> bool:
        """True when every key the hypothesis asked for actually reached execution.

        False is not an error -- it is the honest signal that the executed experiment is narrower than
        the proposed one, which is precisely what went unrecorded before.
        """
        return (not self.rejected_not_honoured and not self.rejected_invalid
                and not self.rejected_ineffective)


def build(intended_delta, validation, cfg, blocks_dir, ext, blockset,
          adopt_blockset=None, target_block=None, cache_version=0, root=".") -> Provenance:
    bh = block_hashes(blocks_dir)
    bd = _sha(json.dumps(bh, sort_keys=True).encode())[:16]
    return Provenance(
        intended_delta=dict(intended_delta or {}),
        effective_delta=dict(validation.effective),
        ext_effective=dict(validation.ext_effective),
        rejected_not_honoured=dict(validation.not_honoured),
        rejected_invalid=dict(validation.invalid),
        rejected_ineffective=dict(validation.ineffective),
        rejection_reasons=list(validation.reasons),
        adopt_blockset=adopt_blockset,
        target_block=target_block,
        blockset=blockset,
        effective_cfg_hash=cfg.hash(),
        block_hashes=bh,
        blocks_digest=bd,
        content_signature=content_signature(cfg, blocks_dir, ext),
        cache_version=int(cache_version),
        code_state=code_state(root),
        seed=int(getattr(cfg, "seed", 0)),
        python=f"{sys.version_info.major}.{sys.version_info.minor}",
    )


def content_signature(cfg, blocks_dir, ext=None) -> str:
    """CONTENT-based node identity -- docs/SYSTEM.md §12 (deduplication and no-op handling).

    The old `signature(cfg, diff)` hashed the unified diff **against the parent**, so two nodes with
    identical configs and identical block sources got different signatures when reached from different
    parents. That is how `run_20260831_000142/n6` -- a re-run of `n3` whose config dict differed by
    nothing at all -- escaped deduplication and was recorded as the run's Lever-E finding.

    Hashing the node's actual content (cfg + extension sidecar + the six block sources) makes identity
    path-independent, which is the only definition under which "duplicate" means anything.
    """
    from dataclasses import asdict as _asdict
    payload = {
        "cfg": _asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(cfg),
        "ext": dict(ext or {}),
        "blocks": block_hashes(blocks_dir),
    }
    return _sha(json.dumps(payload, sort_keys=True, default=list).encode())[:24]
