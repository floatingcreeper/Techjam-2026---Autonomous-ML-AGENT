"""The "Coding" step in the loop diagram (AGENT_STRATEGY.md) — turns a validated hypothesis into
one action from agent/action_space.py's constrained vocabulary. NOT sourced from
06-Master-Prompts_1.md — that file only specifies the propose and repair prompts; this one didn't
exist anywhere until Phase 5, written fresh here following the same conventions (JSON-only output,
schema validation, one retry) as the rest of the codebase for consistency.

Same reasoning as hypothesis_agent.py's docstring: this can honestly report `implementable: false`
rather than force a hyperparameter change that doesn't really implement what was proposed — that's
a first-class, expected outcome here, not a failure mode to be hidden or retried away.
"""
import json
import os
import re

from agent import llm_client
from agent.action_space import HYPERPARAM_BOUNDS, validate_action

PROMPT_PATH = os.path.join(os.path.dirname(__file__), 'prompts', 'code.md')

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
        raise KeyError(f"code.md referenced placeholders not supplied: {sorted(set(missing))}")
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
    if 'implementable' not in obj or not isinstance(obj['implementable'], bool):
        return "missing/non-boolean 'implementable'"
    if not obj.get('reason'):
        return "missing/empty 'reason'"
    if obj['implementable']:
        action = obj.get('action')
        if not isinstance(action, dict):
            return "'implementable' is true but 'action' is missing/malformed"
        err = validate_action(action)
        if err is not None:
            return f"'action' failed validation: {err}"
    return None


class CodingResult:
    def __init__(self, ok, data=None, error=None, attempts=0, total_usage=None):
        self.ok = ok
        self.data = data
        self.error = error
        self.attempts = attempts
        self.total_usage = total_usage or {'input_tokens': 0, 'output_tokens': 0}

    def __repr__(self):
        if not self.ok:
            return f"CodingResult(ok=False, error={self.error!r}, attempts={self.attempts})"
        return f"CodingResult(ok=True, implementable={self.data['implementable']})"


def propose_action(hypothesis, current_config, *, llm_call=llm_client.call, max_retries=1,
                    template_path=PROMPT_PATH, caller='coding_agent'):
    """hypothesis: the `hypothesis` sub-object from hypothesis_agent.propose()'s result
    ({statement, target_stage, reasoning, expected_effect}).
    current_config: the config dict the candidate would start from (current-best's config, or a
    variant's DEFAULT_CONFIG at iteration 0).
    Returns a CodingResult. Same retry-once-on-invalid-response pattern as hypothesis_agent.py."""
    template = _load_template(template_path)
    system = _fill_template(template, {
        'target_stage': hypothesis['target_stage'],
        'statement': hypothesis['statement'],
        'reasoning': hypothesis['reasoning'],
        'implementation_sketch': hypothesis.get('implementation_sketch', ''),
        'current_config': json.dumps({k: current_config.get(k) for k in HYPERPARAM_BOUNDS}),
    })
    messages = [{'role': 'user', 'content': 'Produce the action for this hypothesis now.'}]
    total_usage = {'input_tokens': 0, 'output_tokens': 0}

    attempts = 0
    err = None
    obj = None
    while True:
        attempts += 1
        text, usage = llm_call(system=system, messages=messages, json_mode=True, caller=caller)
        total_usage['input_tokens'] += usage.get('input_tokens', 0)
        total_usage['output_tokens'] += usage.get('output_tokens', 0)

        obj, err = _parse(text)
        if err is None:
            err = _validate(obj)
        if err is None:
            return CodingResult(ok=True, data=obj, attempts=attempts, total_usage=total_usage)
        if attempts > max_retries:
            break
        messages.append({'role': 'assistant', 'content': text})
        messages.append({'role': 'user', 'content':
            f"Your last response was invalid: {err}. Return ONLY the corrected JSON object, "
            f"matching the schema exactly — no other text."})

    return CodingResult(ok=False, error=err, attempts=attempts, total_usage=total_usage)
