"""models/ — pluggable model variants behind one convention (AGENT_STRATEGY.md Phase 0).

Every variant module (e.g. models/fm_v0.py) exposes a module-level function

    train(splits, config=None, verbose=False) -> results

with this exact shape — no base class needed, the convention IS the interface:

    splits:  {'train': [...], 'valid': [...], ...} — row tuples, same shape data.load() returns.
             Evaluate against whichever non-'train' keys are actually present; NEVER hardcode
             'valid' or 'test' by name, and never assume 'test' exists (see the isolation note
             below for why this matters, not just style).
    config:  dict of variant-specific hyperparameters (validated upstream by the not-yet-built
             agent/action_space.py once it exists).
    returns: {<split_name>: {'GAUC':…, 'nDCG@5':…, 'primary':…, 'users':…, 'rows':…}, ...}
             — evaluate.evaluate()'s own return shape, one entry per non-'train' key in splits.

Hidden-test isolation (AGENT_STRATEGY.md hard requirement): the automated loop always loads splits
through agent/data_guard.load_train_valid(), which physically removes the 'test' key before any
agent-facing code sees it. A variant that only ever evaluates "whichever non-train splits are
present" (via non_train_splits() below) therefore can't leak a test score even by accident — it
never has to know the isolation rule exists, the guard does the work upstream. Variants MUST NOT
call data.load() or accept a data_dir themselves; always take already-loaded `splits` from the
caller.
"""


def non_train_splits(splits):
    """Every key in `splits` except 'train' — the splits a variant's train() must evaluate
    against. Use this instead of hardcoding 'valid'/'test' anywhere, so a variant automatically
    respects whatever agent/data_guard.py did or didn't strip out."""
    return [k for k in splits if k != 'train']
