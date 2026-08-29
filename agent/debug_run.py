"""Debug-first coding workflow (AGENT_STRATEGY.md strategy 1, highest priority).

Before any full-scale training run, validate the candidate model+config on a small, fast sample
first. Only if that sample run executes cleanly and produces a plausible metric do we commit to
the (much more expensive) full run — a failing sample routes straight to error_recovery.py instead
of ever burning full-run compute on code that doesn't even work.

Sampling: fixed small N, not "10% of train" — KuaiRand-Pure's train split is large enough that 10%
is still a slow, heavy sample for what's meant to be a quick smoke test. Trade-off (see
AGENT_STRATEGY.md §1): a small fixed sample is cheap and predictable in wall-clock, but may
under-exercise vocab-size/UNK-rate-dependent bugs that only show up at real scale — the full run is
still the first real signal on that front. This gate only answers "does it crash / is the number
sane", not "is it good".
"""
import time

import numpy as np

from models.base import non_train_splits

DEBUG_TRAIN_N = 20_000
DEBUG_VALID_N = 10_000
DEBUG_EPOCHS = 2
DEBUG_PATIENCE = 1


class DebugResult:
    """ok=False means: don't proceed to a full run. `reason` is a human-readable string, always
    present. `sample_metrics` is populated whenever train_fn returned something at all (even an
    implausible one) — useful for diagnosing *why* it failed, not just that it did."""

    def __init__(self, ok, reason, estimated_full_runtime_s=None, sample_metrics=None, elapsed_s=None):
        self.ok = ok
        self.reason = reason
        self.estimated_full_runtime_s = estimated_full_runtime_s
        self.sample_metrics = sample_metrics
        self.elapsed_s = elapsed_s

    def __repr__(self):
        return (f"DebugResult(ok={self.ok}, reason={self.reason!r}, "
                f"estimated_full_runtime_s={self.estimated_full_runtime_s}, "
                f"elapsed_s={self.elapsed_s})")


def _sample_rows(rows, n, seed):
    if len(rows) <= n:
        return rows
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=n, replace=False)
    return [rows[i] for i in idx]


def _is_plausible(metrics_by_split):
    """No NaN/inf, primary (and its two components) in [0,1], every expected split present.
    This is deliberately shallow — it's a crash/sanity gate, not a quality gate."""
    if not isinstance(metrics_by_split, dict) or not metrics_by_split:
        return False, f"train_fn returned {metrics_by_split!r}, expected a non-empty dict of split -> metrics"
    for name, m in metrics_by_split.items():
        if not isinstance(m, dict):
            return False, f"{name}'s result is {m!r}, expected a metrics dict"
        for key in ('GAUC', 'nDCG@5', 'primary'):
            if key not in m:
                return False, f"{name}.{key} missing from result"
            v = m[key]
            if v != v or v in (float('inf'), float('-inf')):  # v != v catches NaN
                return False, f"{name}.{key} is NaN/Inf ({v})"
            if not (0.0 <= v <= 1.0):
                return False, f"{name}.{key}={v} outside the valid [0,1] range"
    return True, "sample run OK"


def debug_run(train_fn, splits, config, *, seed=0,
              debug_train_n=DEBUG_TRAIN_N, debug_valid_n=DEBUG_VALID_N,
              debug_epochs=DEBUG_EPOCHS, debug_patience=DEBUG_PATIENCE):
    """train_fn: a models/*.py-style train(splits, config) function (see models/base.py).
    splits: the full-scale splits dict from agent.data_guard.load_train_valid() (already test-free
            — debug_run doesn't re-check that; it trusts its caller, per the isolation design).
    config: the candidate's real config — debug_run only overrides epochs/patience for speed, every
            other hyperparameter runs exactly as the full run would use it, so a bug that only shows
            up with a specific k or lr is still caught here, not just structural/import errors.

    Returns a DebugResult. Never lets an exception from train_fn propagate — it's caught and
    reported as ok=False so the caller can route straight to error_recovery.py rather than the
    whole orchestrator loop dying on a bad candidate.
    """
    debug_splits = {'train': _sample_rows(splits['train'], debug_train_n, seed)}
    for name in non_train_splits(splits):
        debug_splits[name] = _sample_rows(splits[name], debug_valid_n, seed)

    debug_config = {**config, 'epochs': debug_epochs, 'patience': debug_patience}

    t0 = time.time()
    try:
        sample_metrics = train_fn(debug_splits, debug_config)
    except Exception as e:  # noqa: BLE001 — intentionally broad: ANY candidate-code failure
        # (shape mismatch, KeyError from a bad field name, OOM, ...) must become a DebugResult,
        # not a crash. The Guardrail step (not yet built) is the first line of defense against
        # obviously-bad diffs; this is the last line of defense for what it missed.
        elapsed = time.time() - t0
        return DebugResult(ok=False, reason=f"{type(e).__name__}: {e}", elapsed_s=elapsed)
    elapsed = time.time() - t0

    ok, reason = _is_plausible(sample_metrics)
    if not ok:
        return DebugResult(ok=False, reason=reason, sample_metrics=sample_metrics, elapsed_s=elapsed)

    # Rough LINEAR extrapolation (AGENT_STRATEGY.md §1: "documented as a rough estimate, not
    # precise"). Scales by row-count ratio and epoch-count ratio; ignores early stopping possibly
    # cutting the full run shorter than `epochs` — so this tends to over-estimate, not under-.
    row_ratio = len(splits['train']) / max(len(debug_splits['train']), 1)
    epoch_ratio = config.get('epochs', debug_epochs) / max(debug_epochs, 1)
    estimated_full_runtime_s = elapsed * row_ratio * epoch_ratio

    return DebugResult(ok=True, reason=reason, sample_metrics=sample_metrics,
                        estimated_full_runtime_s=estimated_full_runtime_s, elapsed_s=elapsed)
