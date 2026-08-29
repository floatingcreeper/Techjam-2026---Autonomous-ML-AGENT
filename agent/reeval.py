"""Simplified aggregated evaluation (AGENT_STRATEGY.md strategy 4b). Before accepting a new
current-best, optionally re-run with 1-2 additional seeds and require the SEED-MEAN (not a single
seed) to beat current-best by CONVERGENCE_EPSILON. Toggleable via RECHECK_TOP_CANDIDATE, default
ON: fm_official's own std across seeds (~0.0008, baseline_scores.json) is close enough to
CONVERGENCE_EPSILON (0.002) that trusting a single seed is genuinely likely to accept noise as a
real improvement.

This is this project's (much cheaper) substitute for R&D-Agent's multi-trace merge and AIRA's
"robust final-node-selection" finding (a 9-13% generalization gap from trusting one validation
pass) — see AGENT_STRATEGY.md's search-strategy section for the full reasoning.
"""
import numpy as np

from agent.config import CONVERGENCE_EPSILON

RECHECK_TOP_CANDIDATE = True
RECHECK_EXTRA_SEEDS = (1, 2)  # in addition to the candidate's own original seed


def recheck(train_fn, splits, config, *, original_primary, current_best_primary,
            extra_seeds=RECHECK_EXTRA_SEEDS, epsilon=CONVERGENCE_EPSILON,
            enabled=RECHECK_TOP_CANDIDATE):
    """train_fn/splits/config: same shape as models/*.py's train() and its inputs — re-run here
    with different seeds only, every other hyperparameter held fixed.
    original_primary: the valid `primary` score from the run that triggered this recheck.
    current_best_primary: what it needs to beat by more than `epsilon` to actually be accepted.

    Returns (accept: bool, mean_primary: float, seed_primaries: dict[seed, primary]).
    If `enabled` is False, skips the extra seeds entirely and just applies the epsilon check to
    `original_primary` alone — RECHECK_TOP_CANDIDATE must stay a toggle, not a hard requirement,
    since the extra seeds cost roughly as much compute as the original run each.
    """
    seed_primaries = {config.get('seed', 0): original_primary}
    if enabled:
        for s in extra_seeds:
            seed_cfg = {**config, 'seed': s}
            result = train_fn(splits, seed_cfg)
            seed_primaries[s] = result['valid']['primary']

    mean_primary = float(np.mean(list(seed_primaries.values())))
    accept = mean_primary > current_best_primary + epsilon
    return accept, mean_primary, seed_primaries
