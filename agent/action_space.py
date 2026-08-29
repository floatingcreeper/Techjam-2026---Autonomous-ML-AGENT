"""v0 action space (AGENT_STRATEGY.md Phase 0 decision #2): a constrained, config-driven set of
changes an iteration can make — NOT free-form code generation. This is what makes the Guardrail
step (agent/guardrail.py) genuinely lightweight rather than needing real static analysis: every
action here is a validated dict mutating a hyperparameter within known-sane bounds, nothing here
can ever touch evaluate.py, data.py's train-only-fitting invariant, or the hidden-test guard,
because no action type writes to a file or reads data at all.

Only ONE action type is actually executable in v0: `set_hyperparam` — everything models/fm_v0.py's
DEFAULT_CONFIG already exposes. `toggle_field` and `swap_model_variant` are defined here (so the
Coding step and propose prompt's `target_stage` vocabulary have somewhere to point) but are NOT
executable yet: toggling a field would need data.encode() to support extra fields per-run, and
swapping variants needs more than one models/*.py module to exist — neither exists today. A
hypothesis whose implementation_sketch maps to one of these raises ActionNotExecutable, which the
Coding step (agent/coding_agent.py) is expected to catch and report as `implementable: false`
rather than force a fake `set_hyperparam` action that doesn't match what was actually proposed.
"""

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

EXECUTABLE_ACTION_TYPES = {'set_hyperparam'}
KNOWN_BUT_NOT_EXECUTABLE_ACTION_TYPES = {'toggle_field', 'swap_model_variant'}
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
        if not (lo <= value <= hi):
            return f"action.value={value!r} for {param!r} outside allowed range [{lo}, {hi}]"
        return None

    # toggle_field / swap_model_variant: only check the action names a real thing, not whether
    # it's executable (that's apply_action's job) — a Guardrail check for these should always end
    # in ActionNotExecutable, never a silent no-op.
    if action_type == 'toggle_field':
        return None if action.get('field') else "action.field missing"
    if action_type == 'swap_model_variant':
        return None if action.get('variant') else "action.variant missing"
    return None  # unreachable given the ALL_ACTION_TYPES check above


def apply_action(config, action):
    """Returns a NEW config dict with the action applied — never mutates `config` in place.
    Raises ActionNotExecutable for a structurally valid but not-yet-supported action type; the
    caller is expected to have already run validate_action() first (this doesn't re-validate
    bounds — call validate_action() before apply_action(), same two-step pattern as everywhere
    else in this codebase)."""
    action_type = action['type']
    if action_type == 'set_hyperparam':
        pytype, _, _ = HYPERPARAM_BOUNDS[action['param']]
        new_config = dict(config)
        new_config[action['param']] = pytype(action['value'])
        return new_config
    raise ActionNotExecutable(
        f"action.type={action_type!r} is a known target_stage but this repo's v0 action space "
        f"can't execute it yet (needs a models/fm_v1_*.py variant / data.encode() extension that "
        f"doesn't exist — see AGENT_STRATEGY.md decision #9 and models/base.py's docstring)."
    )
