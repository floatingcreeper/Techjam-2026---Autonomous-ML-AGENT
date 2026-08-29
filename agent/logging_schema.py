"""The one structured per-iteration log record (AGENT_STRATEGY.md hard requirement) —
runs/experiment_log.jsonl, one JSON object per line, written by agent/archivist.py (Phase 5, not
yet built). Built now because Phase 4's error_recovery.py needs error_events' exact shape and
Phase 2's hypothesis output needs a home to slot into unchanged, not reshaped.

Plain dict-building functions, not a dataclass/pydantic model — matches this repo's
minimal-abstraction, no-extra-dependency style (see CLAUDE.md).
"""
import time


def new_record(*, iteration, proposal, code_diff, metrics, error_events, token_cost,
                wall_clock_s, accepted, resulting_config=None, decision=None, node_id=None):
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
    resulting_config: the FULL config dict (base config + the proposed action applied) once
        Coding+Guardrail succeed, else None. Stored so agent/orchestrator.py's duplicate-action
        guard can compare full configs, not just the raw action — found necessary live: the
        pipeline is fully deterministic given (config, seed), so an identical resulting_config
        WILL reproduce a bit-identical result, and re-running it is pure wasted compute (observed
        directly: 3 consecutive real iterations all proposed the same lr=0.01 change and got the
        exact same primary to the last float32 digit).
    """
    return {
        'iteration': iteration,
        'timestamp': time.time(),
        'problem_identified': proposal.get('problem_identified') if proposal else None,
        'hypothesis': proposal.get('hypothesis') if proposal else None,
        'implementation_sketch': proposal.get('implementation_sketch') if proposal else None,
        'code_diff': code_diff,
        'resulting_config': resulting_config,
        # The commit/revert decision tree's verdict + the exact branch path it took
        # (agent/decision.py) — this is what makes "why was this kept/thrown away" auditable in
        # the log rather than something a human has to reconstruct from surrounding fields.
        'decision': decision,
        # Which agent/solution_tree.py node this iteration produced, so a log line can be traced
        # back to its position in the search tree.
        'node_id': node_id,
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


def _rejection_reason(record):
    """The most recent error_events entry's fix_description (covers both a real repair diagnosis
    AND coding/guardrail's "not implementable"/"rejected" reasons — error_event() is reused for
    both, see agent/error_recovery.py and agent/orchestrator.py's _reject())."""
    events = record.get('error_events') or []
    if events:
        return events[-1].get('fix_description') or events[-1].get('error_text')
    return None


def to_history_entry(record):
    """Projects a full record down to the minimal shape agent.hypothesis_agent.format_history()
    expects — the bridge between what Phase 5's archivist writes and what Phase 2's propose
    prompt reads back in as {{ history_block }}.

    Includes WHY a discarded iteration was discarded (found necessary on the first real
    end-to-end Phase 5 run: without it, the model has no way to learn "feature hypotheses aren't
    executable yet" from its own history and just keeps proposing them every iteration)."""
    metrics = (record.get('metrics') or {}).get('valid') or {}
    hyp = record.get('hypothesis') or {}
    accepted = bool(record.get('accepted', False))
    return {
        'iteration': record['iteration'],
        'target_stage': hyp.get('target_stage', 'unknown'),
        'statement': hyp.get('statement', '(no hypothesis — generation failed this iteration)'),
        'gauc': metrics.get('GAUC', 0.0),
        'ndcg5': metrics.get('nDCG@5', 0.0),
        'primary': metrics.get('primary', 0.0),
        'kept': accepted,
        'reason': None if accepted else _rejection_reason(record),
        'resulting_config': record.get('resulting_config'),
        # Whether this iteration actually completed a real training run (has metrics) — a
        # hypothesis rejected before ever training (not-implementable, guardrail-rejected,
        # duplicate-skipped, hypothesis-generation-failed) provides NO evidence the search has
        # stalled and must not count the same as a real executed attempt that failed to improve.
        # Found live: without this distinction, a short run of pre-training rejections falsely
        # triggered convergence after essentially no real search — see agent/orchestrator.py's
        # _converged() and agent/hypothesis_agent.py's _stale_count(), both of which filter on it.
        'executed': bool(record.get('metrics')),
    }
