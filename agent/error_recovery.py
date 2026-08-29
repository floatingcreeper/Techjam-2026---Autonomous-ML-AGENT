"""Error/recovery half of strategy 4 (AGENT_STRATEGY.md), using the repair prompt sourced from
06-Master-Prompts_1.md's §2 ("cheap/fast-tier model ... doesn't need deep reasoning, just 'here's
an error, fix this line'" — same qwen2.5-coder:7b via the one wrapped client, just a smaller
prompt and lower stakes per call than the propose prompt).

Scope, stated explicitly: `repair()` is ONE LLM call — diagnose this error, propose a fix. It does
NOT re-execute anything and does NOT loop. Re-applying `corrected_code_diff`, re-running it
through Guardrail -> debug_run, and calling `repair()` again with the NEW error and
attempt_number+1 if it still fails (up to MAX_REPAIR_ATTEMPTS, then rolling back to current-best)
is the orchestrator's job (Phase 5, not yet built) — a fake "loop" living here that can't actually
re-execute the candidate would be misleading, not useful.
"""
import os

from agent import llm_client
from agent.prompt_utils import fill_template, load_template, parse_json

PROMPT_PATH = os.path.join(os.path.dirname(__file__), 'prompts', 'repair.md')
MAX_REPAIR_ATTEMPTS = 3


def _validate(obj):
    if not isinstance(obj, dict):
        return "response is not a JSON object"
    for field in ('diagnosis', 'fixable', 'fix_description'):
        if field not in obj:
            return f"missing '{field}'"
    if not isinstance(obj['fixable'], bool):
        return "'fixable' must be a JSON boolean"
    if 'corrected_code_diff' not in obj:
        return "missing 'corrected_code_diff'"
    if obj['fixable'] and not obj['corrected_code_diff']:
        return "'fixable' is true but 'corrected_code_diff' is empty/null"
    return None


class RepairResult:
    """ok=True means "got a well-formed repair response" — that response can still legitimately
    say fixable=False (the hypothesis itself wasn't implementable), which is a successful
    diagnosis, not a failure of this call."""

    def __init__(self, ok, data=None, error=None, usage=None):
        self.ok = ok
        self.data = data
        self.error = error
        self.usage = usage or {'input_tokens': 0, 'output_tokens': 0}

    def __repr__(self):
        if not self.ok:
            return f"RepairResult(ok=False, error={self.error!r})"
        return (f"RepairResult(ok=True, fixable={self.data['fixable']}, "
                f"diagnosis={self.data['diagnosis']!r})")


def repair(*, hypothesis_statement, code_diff, error_message, attempt_number,
           max_attempts=MAX_REPAIR_ATTEMPTS, llm_call=llm_client.call, template_path=PROMPT_PATH,
           caller='error_recovery'):
    """One repair attempt. Unlike hypothesis_agent.propose(), this does NOT retry-on-bad-JSON
    internally — a malformed repair response is itself surfaced as ok=False and counted as one of
    the caller's MAX_REPAIR_ATTEMPTS, since the whole point of this prompt is cheap-and-fast, not
    another reasoning cycle."""
    template = load_template(template_path)
    system = fill_template(template, {
        'hypothesis_statement': hypothesis_statement,
        'attempt_number': attempt_number,
        'max_attempts': max_attempts,
        'code_diff': code_diff,
        'error_message': error_message,
    }, source_name='repair.md')
    messages = [{'role': 'user', 'content': 'Diagnose and fix this failure now.'}]
    text, usage = llm_call(system=system, messages=messages, json_mode=True, caller=caller)

    obj, err = parse_json(text)
    if err is None:
        err = _validate(obj)
    if err is not None:
        return RepairResult(ok=False, error=err, usage=usage)
    return RepairResult(ok=True, data=obj, usage=usage)


def error_event(attempt, error_message, result, *, repaired=False):
    """Builds one entry for the iteration record's error_events list (agent/logging_schema.py)
    from a repair() call's RepairResult.

    `repaired`: whether the proposed fix was actually applied AND re-verified to work. This
    function has no way to know that on its own — it only has a diagnosis — so it defaults to
    False; a caller must explicitly pass True, and only after confirming a re-run succeeded.
    Found in code review: this used to default to `bool(d['fixable'])`, conflating "the model
    believes this is fixable in principle" with "this was fixed" — a real bug, since
    agent/orchestrator.py never re-applies or re-verifies a fix in v0 (see its own docstring),
    so every call site here was logging `repaired: true` for candidates that were actually just
    discarded like any other failure. `fixable` (the model's diagnosis-level claim) is unaffected
    and still comes straight from the repair response."""
    if not result.ok:
        return {'attempt': attempt, 'error_text': error_message, 'diagnosis': None,
                'fixable': None, 'fix_description': f"repair response itself invalid: {result.error}",
                'repaired': False}
    d = result.data
    return {'attempt': attempt, 'error_text': error_message, 'diagnosis': d['diagnosis'],
            'fixable': d['fixable'], 'fix_description': d['fix_description'],
            'repaired': repaired}
