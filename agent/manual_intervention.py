"""Manual-intervention counter (AGENT_STRATEGY.md hard requirement) — human-incremented ONLY.
Nothing in the automated loop (orchestrator, hypothesis_agent, error_recovery, ...) calls
record() on its own; it's exposed for a human to call explicitly (see agent/cli.py's
`note-intervention` command) whenever they step in and touch something themselves — restarting a
stuck run, editing a config by hand, overriding a rejected candidate, etc. This is intentionally
never inferred automatically: an inferred count would just be a guess at what "counts" as an
intervention, and the whole point of this metric is that a human is asserting it happened.
"""
import json
import os
import time

LOG_PATH = os.path.join('runs', 'manual_interventions.jsonl')


def record(reason, *, path=LOG_PATH):
    """Appends one intervention event. `reason` should be a short human-written note — this is
    read by a person later, not parsed by anything."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps({'timestamp': time.time(), 'reason': reason}) + '\n')


def count(path=LOG_PATH):
    if not os.path.exists(path):
        return 0
    with open(path, encoding='utf-8') as fh:
        return sum(1 for line in fh if line.strip())


def all_events(path=LOG_PATH):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]
