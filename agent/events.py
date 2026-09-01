"""Structured append-only research event stream -- docs/EN/SYSTEM.md §19 (Research Console).

`run_log.jsonl` records one line per NODE, which is the right granularity for the research ledger but
too coarse to show the research *loop*: a viewer cannot see the agent observe, hypothesise, write code,
have that code guarded, train, evaluate, compare against a control, and decide. `events.jsonl` is that
finer stream. It is additive -- `run_log.jsonl` keeps its existing schema so the old dashboard and any
downstream tooling continue to work.

Hard rule (docs/EN/SYSTEM.md §19): events carry OBSERVABLE, AUDITABLE artifacts only -- the hypothesis the model
actually emitted, the effective intervention, guard verdicts, metrics, statistics, decisions. Private
chain-of-thought is never logged, reconstructed, or invented.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

# The research loop, in narrative order. The console renders these as its primary lane.
RUN_START = "RUN_START"
OBSERVE = "OBSERVE"           # what the agent noticed in its current state
HYPOTHESIZE = "HYPOTHESIZE"   # the Hypothesis object the Proposer returned
PLAN = "PLAN"                 # the intended intervention, before validation
CODE = "CODE"                 # a block was rewritten / a block set adopted
GUARD = "GUARD"               # import allowlist, holdout guard, config validation, dedup, tripwire
DEBUG = "DEBUG"               # the subsampled fast-fail gate
TRAIN = "TRAIN"               # a node started/finished training
EVALUATE = "EVALUATE"         # metrics for a node
COMPARE = "COMPARE"           # paired bootstrap of a node against its control
REFLECT = "REFLECT"           # failure diagnosis
RECOVER = "RECOVER"           # recovery action taken
ENSEMBLE = "ENSEMBLE"         # portfolio statistics / assembly
DECIDE = "DECIDE"             # adoption / retention / rejection decision
CONVERGENCE = "CONVERGENCE"   # official eps/N window state
FINALIZE = "FINALIZE"         # submission + report
RUN_END = "RUN_END"

FILENAME = "events.jsonl"


class EventLog:
    """Append-only JSONL. One line per event, flushed immediately so a live UI can tail it."""

    def __init__(self, run_dir: str, enabled: bool = True):
        self.path = Path(run_dir) / FILENAME
        self.enabled = enabled
        self._seq = 0
        self._lock = threading.Lock()
        self.events: list[dict] = []
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, type: str, summary: str = "", node_id=None, phase=None, **data) -> dict:
        """Record one event. Never raises: logging must not be able to kill a research run."""
        with self._lock:
            self._seq += 1
            rec = {
                "seq": self._seq,
                "ts": round(time.time(), 3),
                "type": type,
                "node_id": node_id,
                "phase": phase,
                "summary": summary,
                "data": _safe(data),
            }
            self.events.append(rec)
            if self.enabled:
                try:
                    with self.path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, default=_fallback) + "\n")
                        fh.flush()
                except OSError:
                    pass
            return rec


def _fallback(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def _safe(d):
    """Keep event payloads small and JSON-clean; long strings are truncated, not dropped."""
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, str) and len(v) > 4000:
            v = v[:4000] + f"...[truncated {len(v) - 4000} chars]"
        out[k] = v
    return out


class NullEventLog(EventLog):
    """Used by unit tests and by any path that must not write to disk."""

    def __init__(self):
        super().__init__(run_dir=".", enabled=False)


def load(path) -> list[dict]:
    """Read a completed run's event stream (for replay). Tolerates a truncated final line."""
    out = []
    p = Path(path)
    if p.is_dir():
        p = p / FILENAME
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue                    # a run killed mid-write leaves one partial line
    return out
