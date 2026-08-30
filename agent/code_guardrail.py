"""Static analysis of LLM-GENERATED code, before it is ever written to disk or executed.

Why this file exists and why it is strict: every earlier action type (`set_hyperparam`,
`toggle_field`) was a validated dict mutating a config — it structurally could not touch a file,
read the test split, or damage the repo, which is exactly why agent/guardrail.py is three lines of
bounds-checking. Free-form generated code has none of that safety for free, so it has to be earned
back here with real AST analysis. This was flagged as a prerequisite in AGENT_STRATEGY.md v0.11
("would need a real Guardrail overhaul — static analysis of generated code, sandboxing — before
it's safe") and this module is that overhaul.

The central rule: **generated code performs NO file or network I/O, at all.** No `open()`, no `os`,
no `subprocess`, no sockets, no dynamic `exec`/`eval`/`__import__`. That one rule does most of the
work:
  - It makes the hidden-test guard airtight for generated code. A generated model cannot read
    KuaiRand's CSVs itself, so the only data it can ever see is the `splits` dict handed to it —
    which agent/data_guard.py already stripped 'test' from. It cannot leak what it cannot reach.
  - It makes the repo safe. Generated code cannot overwrite evaluate.py (the fixed scoring
    contract), cannot rewrite data.py's train-only-fitting logic, cannot delete anything.
  - It keeps failures cheap and local: a bad generated model raises inside train(), which
    agent/debug_run.py already catches and routes to the repair/debug path.

Everything a legitimate model variant needs (numpy math, the shared FM primitives in baseline.py,
evaluate.evaluate, the encoders in data.py) is still available via the import allowlist below.
"""
import ast

# Modules generated code may import. Deliberately minimal: numpy + stdlib math/data-structure
# helpers + this repo's own already-audited modules. `os`/`sys`/`subprocess`/`shutil`/`socket`/
# `urllib`/`requests`/`pickle`/`importlib` are all absent on purpose — see the module docstring.
ALLOWED_IMPORTS = {
    'numpy', 'math', 'collections', 'itertools', 'functools', 'time', 'random',
    'baseline', 'data', 'evaluate', 'models', 'models.base',
}

# Callable names that defeat static analysis or reach the filesystem/interpreter internals.
FORBIDDEN_CALLS = {
    'open', 'exec', 'eval', 'compile', '__import__', 'globals', 'locals', 'vars',
    'setattr', 'delattr', 'input', 'breakpoint', 'memoryview',
}

# Attribute/name tokens that indicate an attempt to reach around the sandbox even without a
# forbidden import (e.g. `().__class__.__bases__[0].__subclasses__()` style escapes).
FORBIDDEN_ATTRIBUTES = {
    '__subclasses__', '__bases__', '__mro__', '__globals__', '__code__', '__closure__',
    '__builtins__', '__loader__', '__spec__', '__dict__', '__getattribute__', '__reduce__',
}

REQUIRED_FUNCTION = 'train'

# --- API-contract checks (correctness, not safety) -------------------------------------------
# Everything above this line stops generated code from doing damage. This block stops it from
# being WRONG in the one specific way a small model gets wrong over and over: calling this repo's
# functions with the wrong signature. Found live — 19 of 20 iterations in one run died on the
# identical `TypeError: list indices must be integers or slices, not str`, from
# `encode(splits['train'])` (encode takes the whole splits DICT) followed by `enc['train']`. The
# safety analysis passed all 19, because nothing about them was unsafe — just broken.
#
# Catching it here rather than at runtime matters because codegen_agent.py's retry loop feeds
# these reasons straight back to the model, so a signature slip costs one cheap retry instead of a
# whole iteration plus a debug-branch chase.
#
# Rule of thumb for adding to this: only encode a contract that is FIXED by this repo (evaluate.py
# is explicitly do-not-modify; data.encode's shape is depended on everywhere). Never encode a
# preference about how a model should be written — that belongs in the prompt, not in a hard
# rejection.
ENCODE_FUNCTIONS = {
    # name -> (must be handed the whole splits dict?, how many values it returns)
    'encode': (True, 2),
    'encode_with_extra_fields': (True, 3),
}
# Keyword arguments each encoder actually accepts. Found live: 11 of 89 candidates in one real run
# died on `encode_with_extra_fields() got an unexpected keyword argument 'extra'` — the parameter
# is `extra_fields`. A wrong kwarg name is invisible to the arity/subscript checks above and only
# shows up as a TypeError after the module is on disk and running.
ENCODE_KWARGS = {
    'encode': {'splits'},
    'encode_with_extra_fields': {'splits', 'data_dir', 'extra_fields'},
}
# evaluate(user_ids, labels, scores, k=5) — the three positional args are mandatory.
EVALUATE_MIN_ARGS = 3


def _called_name(fn):
    """Last name in a call target: `evaluate` for both `evaluate(...)` and `ev.evaluate(...)`."""
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return None


class CodeRejected(Exception):
    """Generated code failed static analysis. Carries every reason, not just the first."""

    def __init__(self, reasons):
        self.reasons = reasons
        super().__init__('; '.join(reasons))


def _root_module(name):
    return (name or '').split('.')[0]


def check_source(source):
    """Returns a list of human-readable rejection reasons — empty list means the code passed.

    Returning reasons (rather than raising) so the caller can feed them straight back to the model
    as repair context, which is how agent/codegen_agent.py's retry loop uses them.
    """
    reasons = []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        # A syntax error is a legitimate, common LLM failure — report it as a normal rejection
        # reason so the repair loop can fix it, not as an exception from the checker itself.
        return [f"SyntaxError: {e.msg} (line {e.lineno})"]

    for node in ast.walk(tree):
        # --- imports ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _root_module(alias.name) not in ALLOWED_IMPORTS:
                    reasons.append(f"import of {alias.name!r} is not allowed "
                                    f"(allowed: {sorted(ALLOWED_IMPORTS)})")
        elif isinstance(node, ast.ImportFrom):
            # `from . import x` / `from .. import x` — relative imports have module=None
            if node.level and not node.module:
                reasons.append("relative imports are not allowed")
            elif _root_module(node.module) not in ALLOWED_IMPORTS:
                reasons.append(f"import from {node.module!r} is not allowed "
                                f"(allowed: {sorted(ALLOWED_IMPORTS)})")

        # --- dangerous calls ---
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in FORBIDDEN_CALLS:
                reasons.append(f"call to {fn.id}() is not allowed "
                                f"(generated code must not do file I/O or dynamic execution)")
            elif isinstance(fn, ast.Attribute) and fn.attr in FORBIDDEN_CALLS:
                reasons.append(f"call to .{fn.attr}() is not allowed")

        # --- sandbox-escape attribute access ---
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTES:
                reasons.append(f"access to {node.attr!r} is not allowed")

    # --- API-contract checks (see the ENCODE_FUNCTIONS block above for why these live here) ---
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node.func)

        if name in ENCODE_FUNCTIONS and node.args:
            wants_dict, _ = ENCODE_FUNCTIONS[name]
            first = node.args[0]
            # `encode(splits['train'])` / `encode(splits["valid"])` — the classic mistake. Any
            # subscript of the splits dict is wrong; encode() slices the splits up itself and
            # fits its vocabularies on splits['train'] internally.
            if wants_dict and isinstance(first, ast.Subscript):
                reasons.append(
                    f"{name}() must be given the WHOLE splits dict, not one split: write "
                    f"`{name}(splits)` and then read `enc['train']` / `enc[name]` from its "
                    f"result. Passing a single split's row list makes {name}() try to index a "
                    f"list with a string (TypeError: list indices must be integers ...).")

        if name in ENCODE_KWARGS:
            allowed = ENCODE_KWARGS[name]
            for kw in node.keywords:
                if kw.arg is not None and kw.arg not in allowed:
                    reasons.append(
                        f"{name}() has no keyword argument {kw.arg!r}. Its signature is "
                        f"encode(splits) / encode_with_extra_fields(splits, data_dir, "
                        f"extra_fields) - the third parameter is `extra_fields`, not "
                        f"{kw.arg!r}. Prefer positional arguments.")

        if name == 'evaluate':
            positional = [a for a in node.args if not isinstance(a, ast.Starred)]
            has_star = any(isinstance(a, ast.Starred) for a in node.args)
            kwargs = {kw.arg for kw in node.keywords if kw.arg}
            supplied = len(positional) + len(kwargs & {'user_ids', 'labels', 'scores'})
            if not has_star and not any(kw.arg is None for kw in node.keywords) \
                    and supplied < EVALUATE_MIN_ARGS:
                reasons.append(
                    f"evaluate() takes three positional arguments - "
                    f"evaluate(user_ids, labels, scores) - but was called with {supplied}. "
                    f"Unpack the encoded split first: `X, y, users = enc[name]`, then "
                    f"`evaluate(users, y, model.predict(X))`. It does NOT take a list of rows.")

    # --- return-arity of the encoders (wrong unpacking is a runtime ValueError otherwise) ---
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        name = _called_name(node.value.func)
        if name not in ENCODE_FUNCTIONS:
            continue
        _, returns = ENCODE_FUNCTIONS[name]
        for target in node.targets:
            if isinstance(target, ast.Tuple) and not any(
                    isinstance(el, ast.Starred) for el in target.elts):
                if len(target.elts) != returns:
                    reasons.append(
                        f"{name}() returns {returns} values, but its result is being unpacked "
                        f"into {len(target.elts)}. encode(splits) -> (enc, dim); "
                        f"encode_with_extra_fields(splits, data_dir, extra) -> "
                        f"(enc, dim, field_list).")

    # --- the test split must never be named in generated code ---
    # data_guard already makes 'test' unreachable, but a generated module asking for it by name is
    # a strong signal of intent worth refusing outright rather than silently KeyError-ing later.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'test':
            reasons.append("the string 'test' appears in generated code - the test split is "
                            "off-limits; evaluate against whichever non-train splits are present "
                            "(see models/base.py's non_train_splits())")
            break

    # --- required entrypoint ---
    top_level_fns = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if REQUIRED_FUNCTION not in top_level_fns:
        reasons.append(f"must define a top-level {REQUIRED_FUNCTION}(splits, config=None, "
                        f"verbose=False) function (see models/base.py's contract)")

    return reasons


def assert_safe(source):
    """check_source(), but raises CodeRejected instead of returning reasons."""
    reasons = check_source(source)
    if reasons:
        raise CodeRejected(reasons)
