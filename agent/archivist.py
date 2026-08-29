"""Writes the one thing every iteration must produce (AGENT_STRATEGY.md hard requirement):
one line in runs/experiment_log.jsonl per iteration, via agent/logging_schema.py's record shape.
Then updates runs/state.json (agent/resume.py) LAST — so a crash between the log append and the
state update just means resume replays this iteration's bookkeeping, never skips or double-counts
it (the log append is what actually matters and happens first).

Dashboard regeneration (Phase 6) is wired in as best-effort: if agent/viewer.py doesn't exist yet
(it doesn't, as of Phase 5), this silently no-ops rather than failing the whole archive step over
a UI file that isn't the source of truth.
"""
import json
import os

from agent.resume import save_state

LOG_PATH = os.path.join('runs', 'experiment_log.jsonl')


def append_record(record, *, log_path=LOG_PATH):
    d = os.path.dirname(log_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(record) + '\n')


def _regenerate_dashboard():
    try:
        from agent import viewer  # Phase 6 — not built yet as of Phase 5
    except ImportError:
        return
    viewer.regenerate()


def archive(record, *, current_best, log_path=LOG_PATH, state_path=None):
    """record: one entry from agent.logging_schema.new_record().
    current_best: the (possibly just-updated) current-best dict to persist in state.json — the
    caller (agent/orchestrator.py) decides accept/reject, this function just persists the result.
    """
    append_record(record, log_path=log_path)
    state = {'last_completed_iteration': record['iteration'], 'current_best': current_best}
    if state_path is not None:
        save_state(state, path=state_path)
    else:
        save_state(state)
    _regenerate_dashboard()
