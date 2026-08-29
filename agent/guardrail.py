"""The Guardrail step in the loop diagram (AGENT_STRATEGY.md) — the last static check before an
action is actually applied and run.

Deliberately thin, and that's a real design property, not laziness: with a free-form-code-diff
action space, Guardrail would need real static analysis (AST-scan for an evaluate.py edit, a
data.load() bypass, etc.). With v0's constrained action space (agent/action_space.py), NONE of
that is possible in the first place — a `set_hyperparam` action is a dict mutating one number in a
config, it cannot write to any file, cannot import evaluate.py, cannot call data.load() directly
(agent/data_guard.py is the only data entrypoint every models/*.py variant is allowed to use, and
no action here ever touches how data gets loaded). So Guardrail's real job right now is just:
re-validate the action (defense in depth — coding_agent.py already validates, but never trust a
single checkpoint) and reject anything not in EXECUTABLE_ACTION_TYPES outright, with a clear reason
rather than a confusing downstream crash.
"""
from agent.action_space import EXECUTABLE_ACTION_TYPES, validate_action


def check(action):
    """Returns (ok: bool, reason: str). ok=False means: do not call apply_action() on this —
    log it as a rejected iteration and move to the next hypothesis, no repair attempt (this is a
    static rejection of the PROPOSAL, not a runtime failure of code that ran — error_recovery.py's
    repair() is for the latter, not this)."""
    err = validate_action(action)
    if err is not None:
        return False, f"Guardrail: {err}"
    if action['type'] not in EXECUTABLE_ACTION_TYPES:
        return False, (f"Guardrail: action.type={action['type']!r} is not executable in this "
                        f"repo's v0 action space (see agent/action_space.py)")
    return True, "Guardrail: OK"
