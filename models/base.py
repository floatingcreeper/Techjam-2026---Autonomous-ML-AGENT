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
call data.load() themselves, or accept a config that lets them load interaction-log rows on their
own — `splits` (the labeled interaction data) always comes pre-loaded from the caller.

Exception: a variant MAY load static, label-free side-info CSVs itself (e.g. models/fm_v1.py loads
video_features_basic_pure.csv/user_features_pure.csv for extra CWM fields, via a 'data_dir' config
key) — those carry no interaction/label data at all, so there is no hidden-test-isolation risk
regardless of which split is being encoded. The rule is specifically about interaction-log rows
(the ones with `long_view` labels and date ranges), not every file in the data directory.
"""


def non_train_splits(splits):
    """Every key in `splits` except 'train' — the splits a variant's train() must evaluate
    against. Use this instead of hardcoding 'valid'/'test' anywhere, so a variant automatically
    respects whatever agent/data_guard.py did or didn't strip out."""
    return [k for k in splits if k != 'train']
