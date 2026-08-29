"""The one structured per-iteration log record (AGENT_STRATEGY.md hard requirement) —
runs/experiment_log.jsonl, one JSON object per line, written by agent/archivist.py (Phase 5, not
yet built). Built now because Phase 4's error_recovery.py needs error_events' exact shape and
Phase 2's hypothesis output needs a home to slot into unchanged, not reshaped.

Plain dict-building functions, not a dataclass/pydantic model — matches this repo's
minimal-abstraction, no-extra-dependency style (see CLAUDE.md).
"""
import time


def new_record(*, iteration, proposal, code_diff, metrics, error_events, token_cost,
                wall_clock_s, accepted):
    """proposal: the dict from agent.hypothesis_agent.propose()'s `.data` — stored verbatim
        ({problem_identified, hypothesis: {...}, implementation_sketch}), no reshaping, per
        06-Master-Prompts_1.md's own design goal. None if hypothesis generation itself failed.
    code_diff: str or None (None if the run never reached the Coding step).
    metrics: {'valid': {'GAUC':…, 'nDCG@5':…, 'primary':…, ...}} or None if never reached.
    error_events: list of {attempt, error_text, diagnosis, fixable, fix_description, repaired}
        (see agent.error_recovery.error_event()) — empty list if nothing failed this iteration.
    token_cost: {'input_tokens': int, 'output_tokens': int} — summed across every LLM call this
        iteration (hypothesis generation + any repair attempts).
    wall_clock_s: total time this iteration took (debug_run + full_run + any repair cycles).
    accepted: whether this candidate became the new current-best.
    """
    return {
        'iteration': iteration,
        'timestamp': time.time(),
        'problem_identified': proposal.get('problem_identified') if proposal else None,
        'hypothesis': proposal.get('hypothesis') if proposal else None,
        'implementation_sketch': proposal.get('implementation_sketch') if proposal else None,
        'code_diff': code_diff,
        'metrics': metrics,
        'error_events': error_events or [],
        'token_cost': token_cost or {'input_tokens': 0, 'output_tokens': 0},
        # NOTE: always 0.0 — this repo is CPU-only (numpy, no torch/CUDA). Logged explicitly as a
        # real field rather than omitted, so the cost report has something to point at, not a
        # silent gap, if that ever changes.
        'gpu_time_s': 0.0,
        'wall_clock_s': wall_clock_s,
        'accepted': accepted,
    }


def to_history_entry(record):
    """Projects a full record down to the minimal shape agent.hypothesis_agent.format_history()
    expects — the bridge between what Phase 5's archivist writes and what Phase 2's propose
    prompt reads back in as {{ history_block }}."""
    metrics = (record.get('metrics') or {}).get('valid') or {}
    hyp = record.get('hypothesis') or {}
    return {
        'iteration': record['iteration'],
        'target_stage': hyp.get('target_stage', 'unknown'),
        'statement': hyp.get('statement', '(no hypothesis — generation failed this iteration)'),
        'gauc': metrics.get('GAUC', 0.0),
        'ndcg5': metrics.get('nDCG@5', 0.0),
        'primary': metrics.get('primary', 0.0),
        'kept': bool(record.get('accepted', False)),
    }
