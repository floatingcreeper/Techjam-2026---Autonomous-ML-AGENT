# pipeline/

This is a seeded copy of the organizer's kuairand-starter-kit (data.py,
evaluate.py, baseline.py, submit.py, ablation_features.py,
baseline_scores.json), plus one harness-owned file the controller relies on.

## What the agent is and isn't allowed to touch

The agent (agent/controller.py) edits **only two files** here, one full-file
rewrite per iteration:

  - `data.py`      — feature engineering (add fields, change encoding,
                      add sequence features, etc.)
  - `baseline.py`  — model + training strategy (loss function, model
                      architecture, hyperparameters, etc.)

It must NEVER edit:

  - `evaluate.py`       — official scoring code, fixed by the organizers.
  - `_run_iteration.py` — harness-owned, see below.
  - `submit.py`, `baseline_scores.json`, `ablation_features.py` — not part
    of the iterating loop (submit.py is used once at the end, by
    agent/finalize.py).

## The interface contract that must never break

`_run_iteration.py` is what the controller actually runs each iteration. It
calls exactly two functions, and if the agent's rewritten data.py/baseline.py
stop providing them with this shape, the iteration counts as a failure (and
gets retried / rolled back by the controller — see agent/sandbox.py):

    data.load(data_dir) -> {'train': [...], 'valid': [...], 'test': [...]}
    baseline.run_fm(splits, verbose=False) -> {
        'valid': {'GAUC': float, 'nDCG@5': float, 'primary': float, ...},
        'test':  {'GAUC': float, 'nDCG@5': float, 'primary': float, ...},
    }

Everything else inside data.py and baseline.py is free to change completely
— new features, a different model class entirely, a different loss, as long
as calling `run_fm(load(data_dir))` still returns that shape.

There is a second, smaller contract used only ONCE, at the very end, by
`agent/finalize.py` (not by `_run_iteration.py`, and not every iteration):

    baseline.train_and_predict(splits, predict_split='test') -> (metrics, scores)

where `metrics` is the same `{'valid':..., 'test':...}` shape as `run_fm`,
and `scores` is a 1-D array of per-row prediction scores for
`predict_split`, from the same best-validation checkpoint reflected in
`metrics`. This is what lets finalize.py produce the actual submission file
without knowing anything about your model's internal class or weights —
keep this function's signature stable too, even if you replace FM entirely.
