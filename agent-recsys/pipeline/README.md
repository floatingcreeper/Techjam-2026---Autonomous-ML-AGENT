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
    }

**Hidden-test compliance:** `data.load()`'s `'test'` split IS the challenge's
hidden test set — its official baseline numbers match this codebase's
`baseline_scores.json` exactly. Per the challenge rules it must never be
touched during development, only scored once at the very end. So
`_run_iteration.py` deliberately pops `'test'` out of `splits` before calling
`run_fm()` — `run_fm()` never receives it, not merely "isn't asked to score
it." This is why `run_fm()`'s contract above only requires a `'valid'` key: a
rewritten `run_fm()` that expects a `'test'` split will get a `KeyError`, by
design (the controller treats that like any other recoverable error and
retries with the traceback fed back — `agent/context.py`'s prompt already
tells the agent not to reach for `splits['test']` inside `run_fm()`).

`agent/sandbox.py` independently raises `HiddenTestViolation` if an
iteration's output carries a `'test'` key at all, which stops the run
outright. That redundancy is deliberate and was earned the hard way: the
stripping line in `_run_iteration.py` was silently lost once to a `git
checkout` restoring a pre-fix commit, and nothing detected it until a manual
audit — by which point several runs had scored the hidden test set every
iteration. The structural fix lives in a file that something outside this
codebase can revert; the behavioural check catches the consequence on the
very next run regardless.

Everything else inside data.py and baseline.py is free to change completely
— new features, a different model class entirely, a different loss, as long
as calling `run_fm()` still returns that shape.

There is a second, smaller contract used only ONCE, at the very end, by
`agent/finalize.py` (not by `_run_iteration.py`, and not every iteration) —
this is the one and only place allowed to touch the hidden test set:

    baseline.train_and_predict(splits, predict_split='test') -> (metrics, scores)

where `metrics` is the same `{'valid':..., 'test':...}` shape (this function
gets the FULL `splits`, test split included — `finalize.py` calls
`data.load()` directly, not through `_run_iteration.py`'s stripped-down
path), and `scores` is a 1-D array of per-row prediction scores for
`predict_split`, from the same best-validation checkpoint reflected in
`metrics`. This is what lets finalize.py produce the actual submission file
without knowing anything about your model's internal class or weights —
keep this function's signature stable too, even if you replace FM entirely.
