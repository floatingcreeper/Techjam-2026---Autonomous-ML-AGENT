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
import json
import os
import re

from agent import llm_client

PROMPT_PATH = os.path.join(os.path.dirname(__file__), 'prompts', 'repair.md')
MAX_REPAIR_ATTEMPTS = 3

_PLACEHOLDER_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def _load_template(path=PROMPT_PATH):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def _fill_template(template, mapping):
    missing = []

    def _sub(m):
        key = m.group(1)
        if key not in mapping:
            missing.append(key)
            return m.group(0)
        return str(mapping[key])

    filled = _PLACEHOLDER_RE.sub(_sub, template)
    if missing:
        raise KeyError(f"repair.md referenced placeholders not supplied: {sorted(set(missing))}")
    return filled


def _strip_fences(text):
    t = text.strip()
    if t.startswith('```'):
        t = re.sub(r'^```(?:json)?\s*', '', t)
        t = re.sub(r'```\s*$', '', t)
    return t.strip()


def _parse(text):
    try:
        return json.loads(_strip_fences(text)), None
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"


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
    template = _load_template(template_path)
    system = _fill_template(template, {
        'hypothesis_statement': hypothesis_statement,
        'attempt_number': attempt_number,
        'max_attempts': max_attempts,
        'code_diff': code_diff,
        'error_message': error_message,
    })
    messages = [{'role': 'user', 'content': 'Diagnose and fix this failure now.'}]
    text, usage = llm_call(system=system, messages=messages, json_mode=True, caller=caller)

    obj, err = _parse(text)
    if err is None:
        err = _validate(obj)
    if err is not None:
        return RepairResult(ok=False, error=err, usage=usage)
    return RepairResult(ok=True, data=obj, usage=usage)


def error_event(attempt, error_message, result):
    """Builds one entry for the iteration record's error_events list (agent/logging_schema.py)
    from a repair() call's RepairResult."""
    if not result.ok:
        return {'attempt': attempt, 'error_text': error_message, 'diagnosis': None,
                'fixable': None, 'fix_description': f"repair response itself invalid: {result.error}",
                'repaired': False}
    d = result.data
    return {'attempt': attempt, 'error_text': error_message, 'diagnosis': d['diagnosis'],
            'fixable': d['fixable'], 'fix_description': d['fix_description'],
            'repaired': bool(d['fixable'])}
