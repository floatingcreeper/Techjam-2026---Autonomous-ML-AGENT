"""The full loop (AGENT_STRATEGY.md Phase 5) — supersedes the Phase 1 skeleton (which only wired
debug_run -> full_run with no hypothesis/coding/guardrail/repair/archiving/resume). Same
debug-first gate behavior is preserved here, just embedded in the complete sequence:

    Hypothesis -> Coding -> Guardrail -> debug_run -> [repair on failure] -> full_run
    -> reeval -> accept/reject -> Archivist (log + resume state) -> convergence check -> loop

Decisions/design notes worth knowing before reading run_iteration():
- v0's action space (agent/action_space.py) only executes `set_hyperparam` actions. If Coding
  reports `implementable: false` (a real, expected outcome — see agent/coding_agent.py), or
  Guardrail rejects the action, the iteration is logged as rejected and the loop moves on WITHOUT
  ever calling debug_run — there's nothing runnable to test.
- On a debug_run failure, this calls error_recovery.repair() exactly ONCE per iteration for
  diagnosis + logging (satisfies "must recover from failures instead of crashing" and the
  error_events log requirement) but does NOT attempt automated re-parameterization — v0 has no
  mechanism to ask Coding for a different value mid-iteration. The iteration is simply rejected and
  the loop moves to the NEXT hypothesis. `MAX_REPAIR_ATTEMPTS` stays defined in error_recovery.py
  for when a smarter retry (re-invoking coding_agent with the error as context) gets built.
- reeval.recheck() decides accept/reject using the multi-seed MEAN primary, not the single
  original run's — current_best's stored `gauc`/`ndcg5` are still the single original run's
  values (reeval doesn't break those out per seed, by design — see reeval.py), only `primary`
  reflects the recheck mean. A known, accepted simplification, not an oversight.
- Every accept/reject decision uses **valid** only (`primary` from splits['valid']). Test is never
  read by this file — agent.data_guard.load_train_valid() structurally cannot return it.
"""
import json
import os
import time

from agent import archivist, error_recovery, guardrail, logging_schema, reeval, resume
from agent.action_space import apply_action
from agent.coding_agent import propose_action
from agent.config import CONVERGENCE_EPSILON, CONVERGENCE_N
from agent.data_guard import load_train_valid
from agent.debug_run import debug_run
from agent.hypothesis_agent import build_state, propose
from agent.llm_client import LLMError
from models import fm_v0

# Backoff schedule for a transient LLMError (Ollama unreachable/timed out) before giving up and
# letting crash-safe resume (agent/resume.py) be the final safety net. Found live: a real Ollama
# timeout after a long run of back-to-back heavy calls crashed the whole process — a network
# hiccup is not the same class of failure as "the process died", and treating it that bluntly
# means every transient timeout needs a human to notice and restart, which the "recovers from
# failures instead of crashing" requirement is explicitly about avoiding.
LLM_RETRY_DELAYS_S = (5, 15, 30)

# How many recent REJECTED iterations to check for an exact duplicate resulting_config before
# spending debug_run + full_run + reeval compute on it again. Found live: with a fixed seed and a
# static current-best, this pipeline is fully deterministic — 3 consecutive real iterations
# proposed the identical lr=0.01 change and got a bit-identical primary each time (see
# AGENT_STRATEGY.md's Changelog). Re-running a config we already have the exact answer for is pure
# waste, not a second data point.
DUPLICATE_CHECK_WINDOW = 5


def _add_usage(a, b):
    return {'input_tokens': a['input_tokens'] + b['input_tokens'],
            'output_tokens': a['output_tokens'] + b['output_tokens']}


def _find_duplicate_config(config, history, window=DUPLICATE_CHECK_WINDOW):
    """Returns the iteration number of the most recent REJECTED iteration whose resulting_config
    exactly matches `config`, or None. Deliberately compares the FULL resulting config (not just
    the raw action) — the same action applied on top of a DIFFERENT current-best would be a
    genuinely different, worth-testing config, not a duplicate."""
    for h in reversed(history[-window:]):
        if not h.get('kept') and h.get('resulting_config') == config:
            return h['iteration']
    return None


def run_iteration(data_dir, *, iteration_number, expected_total_iterations, current_best,
                   history, model=fm_v0, seed=0, verbose=True):
    """One full iteration. Returns (record, new_current_best) — new_current_best is `current_best`
    unchanged if this iteration was rejected at any stage, or a new dict if accepted."""
    splits = load_train_valid(data_dir)  # 'test' never present past this line
    t0 = time.time()
    token_cost = {'input_tokens': 0, 'output_tokens': 0}
    base_config = dict(current_best['config']) if current_best else dict(model.DEFAULT_CONFIG)

    def _reject(proposal, code_diff, error_events, resulting_config=None):
        record = logging_schema.new_record(
            iteration=iteration_number, proposal=proposal, code_diff=code_diff, metrics=None,
            error_events=error_events, token_cost=token_cost, wall_clock_s=time.time() - t0,
            accepted=False, resulting_config=resulting_config)
        return record, current_best

    # ---- Hypothesis ----
    state = build_state(splits, current_best=current_best, iteration_number=iteration_number,
                         expected_total_iterations=expected_total_iterations, history=history)
    proposal_result = propose(state)
    token_cost = _add_usage(token_cost, proposal_result.total_usage)
    if not proposal_result.ok:
        if verbose:
            print(f"  [hypothesis] FAILED: {proposal_result.error}")
        return _reject(None, None, [{'attempt': proposal_result.attempts, 'error_text': None,
                                      'diagnosis': None, 'fixable': False,
                                      'fix_description': f"hypothesis_generation_failed: "
                                                          f"{proposal_result.error}",
                                      'repaired': False}])
    proposal = proposal_result.data
    hyp = proposal['hypothesis']
    if verbose:
        print(f"  [hypothesis] {hyp['target_stage']}: {hyp['statement']}")

    # ---- Coding ----
    coding_result = propose_action(hyp, base_config)
    token_cost = _add_usage(token_cost, coding_result.total_usage)
    if not coding_result.ok:
        if verbose:
            print(f"  [coding] FAILED: {coding_result.error}")
        return _reject(proposal, None, [{'attempt': coding_result.attempts, 'error_text': None,
                                          'diagnosis': None, 'fixable': False,
                                          'fix_description': f"coding_failed: {coding_result.error}",
                                          'repaired': False}])
    if not coding_result.data['implementable']:
        if verbose:
            print(f"  [coding] not implementable: {coding_result.data['reason']}")
        return _reject(proposal, None, [{'attempt': 1, 'error_text': None, 'diagnosis': None,
                                          'fixable': False,
                                          'fix_description': coding_result.data['reason'],
                                          'repaired': False}])
    action = coding_result.data['action']

    # ---- Guardrail ----
    ok, reason = guardrail.check(action)
    if not ok:
        if verbose:
            print(f"  [guardrail] REJECTED: {reason}")
        return _reject(proposal, None, [{'attempt': 1, 'error_text': None, 'diagnosis': None,
                                          'fixable': False, 'fix_description': reason,
                                          'repaired': False}])

    new_config = apply_action(base_config, action)
    pseudo_diff = (f"set_hyperparam {action['param']} = {action['value']} "
                   f"(was {base_config.get(action['param'])})")
    if verbose:
        print(f"  [coding] {pseudo_diff}")

    # ---- duplicate-config guard ----
    # Found live: with a static current-best, the SAME action reproduces the SAME resulting
    # config every time, and this pipeline is deterministic — re-running it wastes a full
    # debug_run + full_run + reeval cycle to rediscover a bit-identical result we already have.
    # Checked BEFORE debug_run, not after, so the skip is actually cheap.
    dup_iter = _find_duplicate_config(new_config, history)
    if dup_iter is not None:
        reason = (f"Identical resulting config already tried and rejected in iteration "
                  f"{dup_iter} — skipped without re-running training (this pipeline is "
                  f"deterministic given a fixed seed, so re-running would reproduce the exact "
                  f"same result, not new information).")
        if verbose:
            print(f"  [duplicate-guard] {reason}")
        return _reject(proposal, pseudo_diff, [{'attempt': 1, 'error_text': None,
                                                  'diagnosis': None, 'fixable': False,
                                                  'fix_description': reason, 'repaired': False}],
                       resulting_config=new_config)

    # ---- debug_run gate ----
    dbg = debug_run(model.train, splits, new_config, seed=seed)
    if verbose:
        print(f"  [debug_run] {dbg}")
    if not dbg.ok:
        rr = error_recovery.repair(hypothesis_statement=hyp['statement'], code_diff=pseudo_diff,
                                    error_message=dbg.reason, attempt_number=1)
        token_cost = _add_usage(token_cost, rr.usage)
        ev = error_recovery.error_event(1, dbg.reason, rr)
        if verbose:
            print(f"  [error_recovery] {ev}")
        return _reject(proposal, pseudo_diff, [ev], resulting_config=new_config)

    # ---- full run (+ reeval) ----
    if verbose:
        print(f"  [full_run] debug sample OK (~{dbg.estimated_full_runtime_s:.0f}s estimated), "
              f"starting ...")
    try:
        full_metrics = model.train(splits, new_config, verbose=verbose)
        orig_primary = full_metrics['valid']['primary']
        cb_primary = current_best['primary'] if current_best else -1.0
        accept, mean_primary, seed_primaries = reeval.recheck(
            model.train, splits, new_config, original_primary=orig_primary,
            current_best_primary=cb_primary)
    except Exception as e:  # noqa: BLE001 — found in code review: debug_run passing is NOT a
        # full guarantee at real scale (its own docstring says so explicitly), and neither the
        # full run nor reeval's extra-seed passes were wrapped, so a crash here used to propagate
        # all the way out of run_loop and kill the whole process — directly violating "recovers
        # from failures instead of crashing". Same repair-and-reject handling as a debug_run
        # failure, just triggered by a real-scale-only failure instead of a sample-scale one.
        error_message = f"{type(e).__name__}: {e}"
        rr = error_recovery.repair(hypothesis_statement=hyp['statement'], code_diff=pseudo_diff,
                                    error_message=error_message, attempt_number=1)
        token_cost = _add_usage(token_cost, rr.usage)
        ev = error_recovery.error_event(1, error_message, rr)
        if verbose:
            print(f"  [full_run] CRASHED: {error_message}")
            print(f"  [error_recovery] {ev}")
        return _reject(proposal, pseudo_diff, [ev], resulting_config=new_config)
    if verbose:
        print(f"  [reeval] seed_primaries={seed_primaries} mean={mean_primary:.4f} "
              f"accept={accept}")

    record = logging_schema.new_record(
        iteration=iteration_number, proposal=proposal, code_diff=pseudo_diff,
        metrics=full_metrics, error_events=[], token_cost=token_cost,
        wall_clock_s=time.time() - t0, accepted=accept, resulting_config=new_config)

    if accept:
        new_best = {
            'gauc': float(full_metrics['valid']['GAUC']),
            'ndcg5': float(full_metrics['valid']['nDCG@5']),
            'primary': mean_primary,
            'summary': f"iter {iteration_number}: {hyp['statement']}",
            'config': new_config,
        }
        return record, new_best
    return record, current_best


def _run_iteration_with_retry(data_dir, *, verbose=True, **kwargs):
    """Wraps run_iteration() with a short backoff-retry specifically for agent.llm_client.LLMError
    — a transient infrastructure hiccup (Ollama unreachable/timed out), not a hypothesis-quality
    problem, so it must not be logged as a rejected iteration or count against the model. If every
    retry is also exhausted, re-raises — crash-safe resume is still the final fallback, just no
    longer the first line of defense for something this recoverable."""
    last_err = None
    for attempt, delay in enumerate((0,) + LLM_RETRY_DELAYS_S, start=1):
        if delay:
            if verbose:
                print(f"  [llm_client] transient failure ({last_err}), retrying in {delay}s "
                      f"(attempt {attempt}/{len(LLM_RETRY_DELAYS_S) + 1}) ...")
            time.sleep(delay)
        try:
            return run_iteration(data_dir, verbose=verbose, **kwargs)
        except LLMError as e:
            last_err = e
    raise last_err


def _load_history(log_path=archivist.LOG_PATH):
    if not os.path.exists(log_path):
        return []
    history = []
    with open(log_path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                history.append(logging_schema.to_history_entry(json.loads(line)))
    return history


def _converged(history, n=CONVERGENCE_N, epsilon=CONVERGENCE_EPSILON):
    """Our interpretation of the convergence spec ("validation score hasn't improved by more than
    epsilon over the last N iterations"): the best primary in the last N EXECUTED iterations must
    exceed the best primary seen before that window by more than epsilon, or this is converged.

    Filters to executed iterations only (h['executed'] — a real training run happened, metrics
    exist) BEFORE applying the N/epsilon window — found live, this matters a lot: a hypothesis
    rejected before training (not-implementable, guardrail-rejected, duplicate-skipped) is not
    evidence the search is exhausted, and counting it the same as a real failed attempt caused
    false convergence after essentially no real search (observed directly: converged at iteration
    6 with only ONE iteration since the last accept — iter 2 — that had actually finished
    training; the other three were rejected before ever reaching debug_run)."""
    executed = [h for h in history if h.get('executed')]
    if len(executed) < n:
        return False
    best_before = max((h['primary'] for h in executed[:-n]), default=-1.0)
    recent_best = max(h['primary'] for h in executed[-n:])
    return recent_best <= best_before + epsilon


def run_loop(data_dir, *, expected_total_iterations, max_iterations=None, model=fm_v0, seed=0,
             log_path=archivist.LOG_PATH, state_path=resume.STATE_PATH, verbose=True):
    """Drives the full loop, resuming from `state_path` if it exists (crash-safe resume hard
    requirement). `max_iterations` caps how many iterations run regardless of convergence —
    resolves AGENT_STRATEGY.md's open "iteration cap" question: yes, always capped, defaults to
    `expected_total_iterations` so a run is never accidentally unbounded."""
    state = resume.load_state(state_path)
    current_best = state['current_best']
    start_iter = state['last_completed_iteration'] + 1
    cap = max_iterations if max_iterations is not None else expected_total_iterations
    history = _load_history(log_path)

    if verbose and start_iter > 1:
        print(f"[resume] continuing from iteration {start_iter} "
              f"(current-best primary={current_best['primary']:.4f})" if current_best else
              f"[resume] continuing from iteration {start_iter} (no current-best yet)")

    for it in range(start_iter, cap + 1):
        if verbose:
            print(f"=== iteration {it}/{cap} ===")
        record, current_best = _run_iteration_with_retry(
            data_dir, iteration_number=it, expected_total_iterations=expected_total_iterations,
            current_best=current_best, history=history, model=model, seed=seed, verbose=verbose)
        archivist.archive(record, current_best=current_best, log_path=log_path,
                           state_path=state_path)
        history.append(logging_schema.to_history_entry(record))
        if verbose:
            print(f"  -> {'ACCEPTED' if record['accepted'] else 'rejected'}")
        if _converged(history):
            if verbose:
                print(f"  converged: last {CONVERGENCE_N} iterations improved primary by "
                      f"<= {CONVERGENCE_EPSILON}")
            break

    return current_best, history
