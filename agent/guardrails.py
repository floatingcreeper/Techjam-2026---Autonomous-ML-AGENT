"""Guardrails.

Immutability enforcement. The fixed harness + runner + contracts are hashed into
agent/frozen.lock; any later mismatch aborts the run, so the agent can never (accidentally
or otherwise) edit the files the score depends on. The import allowlist and holdout read
guard live in agent/executor.py; the temporal guard in pipeline/lib/seq_build.py; the
submission check in the frozen submit.py. See docs/EN/SYSTEM.md §3 and §8.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

FROZEN = [
    "data.py",
    "evaluate.py",
    "submit.py",
    "pipeline/run_node.py",
    "pipeline/contracts.py",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ensure_frozen(root: str = ".", lock: str = "agent/frozen.lock", create: bool = True):
    root = Path(root)
    lp = root / lock
    cur = {f: _sha(root / f) for f in FROZEN}
    if not lp.exists():
        if not create:
            raise SystemExit("frozen.lock missing (run with create=True once to pin the harness)")
        lp.write_text(json.dumps(cur, indent=2))
        return "created"
    old = json.loads(lp.read_text())
    bad = [f for f in FROZEN if old.get(f) != cur.get(f)]
    if bad:
        raise SystemExit(
            "FROZEN FILE MODIFIED -- refusing to run. The score depends on these staying "
            f"immutable: {bad}. Restore them or re-pin deliberately (delete {lock})."
        )
    return "verified"
