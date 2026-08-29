"""v0 action space (AGENT_STRATEGY.md Phase 0 decision #2): a constrained, config-driven set of
changes an iteration can make — NOT free-form code generation. This is what makes the Guardrail
step (agent/guardrail.py) genuinely lightweight rather than needing real static analysis: every
action here is a validated dict mutation (a hyperparameter value, or which pre-built CWM fields
to include), nothing here can ever touch evaluate.py, data.py's train-only-fitting invariant, or
the hidden-test guard, because no action type writes to a file or reads interaction-log data
itself — that structural property held even once `toggle_field` became real (v0.11): it still only
picks from a fixed, pre-built list of fields (data.EXTRA_FIELDS) via models/fm_v1.py, it doesn't
generate new code.

Two action types are executable: `set_hyperparam` (everything models/fm_v1.py's DEFAULT_CONFIG
exposes) and `toggle_field` (add/remove one of data.EXTRA_FIELDS from the resulting config's
`extra_fields` list). `swap_model_variant` is still named (so the vocabulary has somewhere to
point if a genuinely different architecture ever gets built) but not executable — fm_v1 already
subsumes fm_v0's behavior (extra_fields=[] is identical), so there's currently nothing to swap
between. A hypothesis that maps to `swap_model_variant` raises ActionNotExecutable, which the
Coding step (agent/coding_agent.py) is expected to catch and report as `implementable: false`.
"""
from data import EXTRA_FIELDS

# (type, min, max) — inclusive bounds. Chosen to keep training numerically sane (e.g. lr's upper
# bound is deliberately well below 1.0 — Adam on this FM's embeddings realistically diverges/NaNs
# well before that) while still leaving real room for a hypothesis to move a knob meaningfully.
HYPERPARAM_BOUNDS = {
    'k': (int, 4, 256),
    'lr': (float, 1e-5, 0.5),
    'l2': (float, 0.0, 1e-2),
    'epochs': (int, 1, 200),
    'patience': (int, 1, 20),
    'batch_size': (int, 256, 65536),
}

EXECUTABLE_ACTION_TYPES = {'set_hyperparam', 'toggle_field'}
KNOWN_BUT_NOT_EXECUTABLE_ACTION_TYPES = {'swap_model_variant'}
ALL_ACTION_TYPES = EXECUTABLE_ACTION_TYPES | KNOWN_BUT_NOT_EXECUTABLE_ACTION_TYPES


class ActionNotExecutable(Exception):
    """Raised by apply_action() for a structurally valid action this repo's v0 action space just
    can't run yet (toggle_field / swap_model_variant) — a scope limit, not a malformed action."""


def validate_action(action):
    """Returns None if `action` is well-formed and (if executable) within bounds; otherwise a
    human-readable error string. Does NOT distinguish "malformed" from "known but not executable
    yet" — that's apply_action()'s job (raises ActionNotExecutable specifically) — this function
    only answers "is this a legitimate action at all"."""
    if not isinstance(action, dict):
        return "action is not a JSON object"
    action_type = action.get('type')
    if action_type not in ALL_ACTION_TYPES:
        return f"action.type={action_type!r} not one of {sorted(ALL_ACTION_TYPES)}"

    if action_type == 'set_hyperparam':
        param = action.get('param')
        if param not in HYPERPARAM_BOUNDS:
            return f"action.param={param!r} not one of {sorted(HYPERPARAM_BOUNDS)}"
        pytype, lo, hi = HYPERPARAM_BOUNDS[param]
        value = action.get('value')
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"action.value={value!r} is not a number"
        # Found in review: apply_action's pytype(value) silently TRUNCATES a fractional value for
        # an int-typed param (e.g. k=64.9 -> 64) with no warning — the logged pseudo-diff then
        # misrepresents what was actually proposed. Reject it here instead, so coding_agent's
        # existing retry-once-on-invalid-response loop asks for a whole number instead.
        if pytype is int and float(value) != int(value):
            return (f"action.value={value!r} for {param!r} must be a whole number ({param!r} is "
                     f"integer-typed) — a fractional value would be silently truncated")
        if not (lo <= value <= hi):
            return f"action.value={value!r} for {param!r} outside allowed range [{lo}, {hi}]"
        return None

    if action_type == 'toggle_field':
        field = action.get('field')
        op = action.get('op')
        if field not in EXTRA_FIELDS:
            return f"action.field={field!r} not one of {EXTRA_FIELDS}"
        if op not in ('add', 'remove'):
            return f"action.op={op!r} must be 'add' or 'remove'"
        return None

    # swap_model_variant: only check the action names a real thing, not whether it's executable
    # (that's apply_action's job) — a Guardrail check for this should always end in
    # ActionNotExecutable, never a silent no-op.
    if action_type == 'swap_model_variant':
        return None if action.get('variant') else "action.variant missing"
    return None  # unreachable given the ALL_ACTION_TYPES check above


def apply_action(config, action):
    """Returns a NEW config dict with the action applied — never mutates `config` in place.
    Raises ActionNotExecutable for a structurally valid but not-yet-supported action type (only
    swap_model_variant, currently); the caller is expected to have already run validate_action()
    first (this doesn't re-validate bounds — call validate_action() before apply_action(), same
    two-step pattern as everywhere else in this codebase)."""
    action_type = action['type']
    if action_type == 'set_hyperparam':
        pytype, _, _ = HYPERPARAM_BOUNDS[action['param']]
        new_config = dict(config)
        new_config[action['param']] = pytype(action['value'])
        return new_config
    if action_type == 'toggle_field':
        new_config = dict(config)
        current = list(config.get('extra_fields', []))
        field, op = action['field'], action['op']
        if op == 'add' and field not in current:
            current.append(field)
        elif op == 'remove' and field in current:
            current.remove(field)
        # op == 'add' on an already-present field, or 'remove' on an absent one, is a no-op by
        # design — not an error, just nothing to do (the duplicate-config guard in
        # agent/orchestrator.py will catch it as a duplicate of whatever config already has this
        # field state, since resulting_config would be unchanged from a prior attempt).
        new_config['extra_fields'] = current
        return new_config
    raise ActionNotExecutable(
        f"action.type={action_type!r} is a known target_stage but this repo's v0 action space "
        f"can't execute it yet (would need a genuinely different model architecture — fm_v1 "
        f"already subsumes fm_v0, so there's nothing to swap between right now)."
    )
