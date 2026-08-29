"""Crash-safe resume (AGENT_STRATEGY.md hard requirement). runs/state.json holds only
{'last_completed_iteration', 'current_best'} — NOT the history list, which lives in
runs/experiment_log.jsonl already (agent/archivist.py appends there every iteration) and is
re-derived on resume by reading that file, so there's exactly one source of truth for history, not
two that can drift apart.

Written atomically (temp file + os.replace) so a crash mid-write never leaves state.json
half-written or corrupt — the load on next startup either sees the old state or the fully-new one,
never something in between.
"""
import json
import os

from agent.json_utils import json_default

STATE_PATH = os.path.join('runs', 'state.json')


def load_state(path=STATE_PATH):
    """Returns {'last_completed_iteration': int, 'current_best': dict|None}, or the same shape
    freshly initialized (iteration 0, no current-best) if no state file exists yet."""
    if not os.path.exists(path):
        return {'last_completed_iteration': 0, 'current_best': None}
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def save_state(state, path=STATE_PATH):
    """Atomic write: write to a temp file in the same directory, then os.replace() — os.replace
    is atomic on both POSIX and Windows (unlike a plain rename on Windows, which fails if the
    target exists), so a process killed mid-save can never leave a corrupt state.json."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as fh:
        json.dump(state, fh, indent=2, default=json_default)  # numpy-safe, see agent/json_utils.py
    os.replace(tmp_path, path)
