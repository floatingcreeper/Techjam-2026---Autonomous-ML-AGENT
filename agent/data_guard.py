"""Hidden-test isolation (AGENT_STRATEGY.md hard requirement).

Every agent-executed code path must load data through `load_train_valid()`, never through
`data.load()` directly. It physically drops the 'test' key before returning — no code downstream
(debug_run, a model's train(), encode()) ever sees test rows, so nothing has to remember not to
look at them. `encode()` in data.py only builds vocab/entries for keys actually present in the
splits dict it's given, so a splits dict missing 'test' produces an encoded dict missing 'test' too
— the isolation holds all the way through, not just at this one call site.

`submit.py --score --split test` remains the one sanctioned, human-run exception, and stays
completely outside this module and outside the orchestrator's reach.
"""
import data as _data


def load_train_valid(data_dir):
    """Same shape as data.load(splits dict), minus the 'test' key.
    This is the ONLY data-loading entrypoint any agent/* or models/* code may call."""
    splits = _data.load(data_dir)
    splits.pop('test', None)
    return splits
