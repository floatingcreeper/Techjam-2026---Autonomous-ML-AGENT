"""The full loop (AGENT_STRATEGY.md Phase 5) — supersedes the Phase 1 skeleton (which only wired
debug_run -> full_run with no hypothesis/coding/guardrail/repair/archiving/resume). Same
debug-first gate behavior is preserved here, just embedded in the complete sequence:

    Hypothesis -> Coding -> Guardrail -> debug_run -> [repair on failure] -> full_run
    -> reeval -> accept/reject -> Archivist (log + resume state) -> convergence check -> loop

Decisions/design notes worth knowing before reading run_iteration():
- The action space (agent/action_space.py) executes `set_hyperparam` and, as of v0.11,
  `toggle_field` (add/remove a real CWM signal — see models/fm_v1.py, data.EXTRA_FIELDS). If
  Coding reports `implementable: false` (a real, expected outcome — see agent/coding_agent.py), or
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

from agent import (archivist, codegen_agent, decision, error_recovery, guardrail, logging_schema,
                    reeval, resume)
from agent.action_space import apply_action
from agent.coding_agent import propose_action
from agent.config import CONVERGENCE_EPSILON, CONVERGENCE_N
from agent.data_guard import load_train_valid
from agent.debug_run import check_coverage, debug_run
from agent.hypothesis_agent import build_state, propose
from agent.llm_client import LLMError
from agent.solution_tree import SolutionTree
from models import fm_v1

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

# Hard ceiling on how many epochs a GENERATED module may ask for. Generated code chooses its own
# hyperparameters (that's the point — see _config_for_generated), but "epochs" is the one knob
# that translates directly into wall-clock, and an autonomous loop must not let one candidate
# quietly ask for 500 epochs and eat the whole budget. Matches fm_v1.DEFAULT_CONFIG's 40.
MAX_GENERATED_EPOCHS = 40

# Smallest batch a generated module may actually run with. Found live on the first real run after
# generated hyperparameters started taking effect: the model hypothesised "increase the batch size
# to 32" (it meant well - smaller batches, more updates) and debug_run estimated the resulting full
# run at 9,601s - 2.7 HOURS for one iteration, because 1.1M rows at 32 per step is ~35,000 numpy
# calls per epoch. Below a few hundred rows, per-step Python/numpy overhead dominates the actual
# math, so a tiny batch is not a modelling choice with a real trade-off, it's just waste. Floored
# rather than rejected so the underlying idea (smaller batches) still gets tested at a size that
# can finish; the resulting_config logged for the iteration records the floored value.
MIN_GENERATED_BATCH_SIZE = 256

# Reject a candidate whose estimated full-run time exceeds this. debug_run has always computed
# `estimated_full_runtime_s` and NOTHING ever read it - so there was no bound at all on how long a
# single candidate could take, and a generated module could quietly consume the entire run. The
# estimate is a deliberate over-estimate (linear extrapolation, ignores early stopping), so this
# can be generous: the reference model's real full run is ~55s, making this ~10x headroom for a
# legitimately heavier model while still ruling out the 2.7-hour case above.
MAX_ESTIMATED_FULL_RUN_S = 600.0

# Keys the HARNESS owns on the codegen path, no matter what a generated module's DEFAULTS say:
# `seed` because reproducibility and reeval's multi-seed recheck depend on the caller setting it,
# `data_dir` because the interaction logs and any side-info CSVs must come from the same place.
HARNESS_OWNED_CONFIG_KEYS = ('seed', 'data_dir')


def _module_defaults(module):
    """A generated module's own hyperparameter dict. Accepts either name: prompts/write_model.md
    asks for `DEFAULTS`, while the repo's hand-written variants (models/fm_v1.py) use
    `DEFAULT_CONFIG`, and a generated module often imitates whichever it last saw."""
    for attr in ('DEFAULTS', 'DEFAULT_CONFIG'):
        d = getattr(module, attr, None)
        if isinstance(d, dict):
            return dict(d)
    return {}


def _config_for_generated(module, base_config, *, data_dir, seed):
    """The config a generated module is actually run with: its own declared defaults win over the
    incumbent's config (they encode the hypothesis), except for the harness-owned keys and the
    epoch ceiling. Keys the module never declared fall back to `base_config` so a module that
    omits, say, batch_size still gets a sane value rather than relying on an internal literal."""
    cfg = {**base_config, **_module_defaults(module)}
    for key in HARNESS_OWNED_CONFIG_KEYS:
        if key == 'data_dir':
            if 'data_dir' in cfg:
                cfg['data_dir'] = data_dir
        else:
            cfg[key] = base_config.get(key, seed)
    try:
        cfg['epochs'] = max(1, min(int(cfg.get('epochs', MAX_GENERATED_EPOCHS)),
                                    MAX_GENERATED_EPOCHS))
    except (TypeError, ValueError):
        # A module declaring a non-numeric epochs is broken, but that's debug_run's verdict to
        # deliver against real code, not something to crash the orchestrator over here.
        cfg['epochs'] = MAX_GENERATED_EPOCHS
    if 'batch_size' in cfg:
        try:
            cfg['batch_size'] = max(MIN_GENERATED_BATCH_SIZE, int(cfg['batch_size']))
        except (TypeError, ValueError):
            cfg['batch_size'] = base_config.get('batch_size', MIN_GENERATED_BATCH_SIZE)
    return cfg


def _error_with_diagnosis(error_message, repair_result):
    """Fold error_recovery.repair()'s diagnosis into the text stored as a node's `error`, so the
    next DEBUG attempt on that node sees the analysis and not just the traceback line. Degrades
    to the bare error whenever repair() itself failed or came back empty — a node's error must
    always still be readable on its own."""
    data = repair_result.data if repair_result.ok else None
    if not data:
        return error_message
    parts = [error_message]
    if data.get('diagnosis'):
        parts.append(f"Diagnosis (from the repair agent): {data['diagnosis']}")
    if data.get('fix_description'):
        parts.append(f"Suggested fix: {data['fix_description']}")
    if data.get('corrected_code_diff'):
        parts.append(f"Suggested correction:\n{data['corrected_code_diff']}")
    if data.get('fixable') is False:
        parts.append("The repair agent judged this NOT fixable as written - if the same error "
                      "recurs, change the approach rather than patching the same line again.")
    return '\n'.join(parts)


def _add_usage(a, b):
    return {'input_tokens': a['input_tokens'] + b['input_tokens'],
            'output_tokens': a['output_tokens'] + b['output_tokens']}


def _format_action(action, base_config):
    """Human-readable pseudo-diff for logging/repair-prompt context — generic over action TYPE,
    not hardcoded to set_hyperparam's shape. Found live: this used to assume action['param']/
    ['value'] unconditionally and crashed with KeyError the first time the agent actually chose a
    toggle_field action (v0.11) — exactly the kind of action-type-specific hardcoding this
    function now exists to prevent from recurring for any future action type too."""
    if action['type'] == 'set_hyperparam':
        return (f"set_hyperparam {action['param']} = {action['value']} "
                f"(was {base_config.get(action['param'])})")
    if action['type'] == 'toggle_field':
        return f"toggle_field {action['op']} {action['field']}"
    return f"{action['type']} {action}"  # fallback for any action type this doesn't know yet


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
                   history, tree=None, model=fm_v1, seed=0, verbose=True):
    """One full iteration. Returns (record, new_current_best) — new_current_best is `current_best`
    unchanged unless the decision tree returned COMMIT.

    `tree` is the AIDE-style SolutionTree (agent/solution_tree.py). Its select() decides what this
    iteration DOES — debug a broken node, draft a new solution, or improve the best working one —
    and which node the work hangs off. Passing tree=None creates a fresh in-memory tree, which is
    only really useful for tests."""
    splits = load_train_valid(data_dir)  # 'test' never present past this line
    t0 = time.time()
    token_cost = {'input_tokens': 0, 'output_tokens': 0}
    tree = tree if tree is not None else SolutionTree()

    # ---- solution-tree selection: what kind of work is this iteration doing? ----
    tree_op, parent_node = tree.select()
    if verbose:
        where = f" on node #{parent_node.id} (primary={parent_node.primary})" if parent_node else ""
        print(f"  [tree] operation={tree_op}{where}")

    # A node's own config is the right base when we're building on it; otherwise fall back to
    # current-best's config, then to the default model's.
    if parent_node and parent_node.config:
        base_config = dict(parent_node.config)
    elif current_best:
        base_config = dict(current_best['config'])
    else:
        base_config = dict(model.DEFAULT_CONFIG)
    # A model variant's DEFAULT_CONFIG may hardcode its own 'data_dir' (fm_v1 does, for loading
    # extra-field side-info CSVs) — always override it with the actual data_dir this run was
    # invoked with, so the interaction logs and the side-info files can never silently come from
    # two different places if a caller ever passes a non-default --data_dir.
    if 'data_dir' in base_config:
        base_config['data_dir'] = data_dir

    def _reject(proposal, code_diff, error_events, resulting_config=None, decision=None,
                 node_id=None):
        record = logging_schema.new_record(
            iteration=iteration_number, proposal=proposal, code_diff=code_diff, metrics=None,
            error_events=error_events, token_cost=token_cost, wall_clock_s=time.time() - t0,
            accepted=False, resulting_config=resulting_config,
            decision=decision.to_dict() if decision is not None else None, node_id=node_id)
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

    # ---- Coding: which path implements this iteration? ----
    # Routed by the tree operation, AIDE-style:
    #   DRAFT   -> always generate code. "Write a new solution from scratch" is inherently a code
    #              operation; expressing it as a hyperparameter tweak would not be a new solution.
    #   DEBUG   -> always generate code. There is broken code to fix, and no hyperparameter tweak
    #              can fix broken code.
    #   IMPROVE -> try the cheap config action FIRST (set_hyperparam/toggle_field: one small LLM
    #              call, no code risk), and only fall through to code generation when the idea
    #              genuinely can't be expressed that way. This is where most of the loop's
    #              iterations land, so keeping the cheap path primary here is what stops
    #              codegen's cost from dominating the run.
    # This also retires the old "not implementable -> wasted iteration" dead end entirely: an idea
    # the config space can't express now becomes a generated module instead of a rejection.
    action = None
    generated = None
    if tree_op == 'improve':
        coding_result = propose_action(hyp, base_config)
        token_cost = _add_usage(token_cost, coding_result.total_usage)
        if coding_result.ok and coding_result.data['implementable']:
            action = coding_result.data['action']
        elif verbose:
            why = coding_result.error if not coding_result.ok else coding_result.data['reason']
            print(f"  [coding] config action unavailable ({why}) -> generating code")

    if action is not None:
        ok, reason = guardrail.check(action)
        if not ok:
            if verbose:
                print(f"  [guardrail] REJECTED: {reason}")
            return _reject(proposal, None, [{'attempt': 1, 'error_text': None, 'diagnosis': None,
                                              'fixable': False, 'fix_description': reason,
                                              'repaired': False}])
        new_config = apply_action(base_config, action)
        pseudo_diff = _format_action(action, base_config)
        run_model = model
        code_path = parent_node.code_path if parent_node else None
        if code_path:
            # Config tweak applied on top of a generated parent: keep running the parent's code.
            run_model = codegen_agent.load_module(code_path)
        if verbose:
            print(f"  [coding] {pseudo_diff}")
    else:
        # ---- code generation path ----
        parent_source = None
        if parent_node and parent_node.code_path and os.path.exists(parent_node.code_path):
            with open(parent_node.code_path, encoding='utf-8') as fh:
                parent_source = fh.read()
        gen_op = tree_op if tree_op in ('draft', 'debug', 'improve') else 'draft'
        if gen_op != 'draft' and parent_source is None:
            gen_op = 'draft'   # nothing to debug/improve from; fall back to a fresh draft
        if verbose:
            print(f"  [codegen] operation={gen_op}"
                  + (f" from node #{parent_node.id}" if parent_node and parent_source else ""))
        generated = codegen_agent.generate(
            gen_op, hyp, iteration=iteration_number, parent_source=parent_source,
            parent_primary=parent_node.primary if parent_node else None,
            error=parent_node.error if parent_node else None)
        token_cost = _add_usage(token_cost, generated.total_usage)
        if not generated.ok:
            # Static analysis rejected every attempt — this never ran, so per the decision tree
            # it's REJECT_UNSAFE: nothing to revert, and no BUGGY node (there's no code on disk
            # worth spending a DEBUG child on).
            d = decision.decide(code_safe=False, code_reasons=generated.reasons)
            if verbose:
                print(f"  [code_guardrail] {d}")
            return _reject(proposal, None, [{'attempt': generated.attempts, 'error_text': None,
                                              'diagnosis': None, 'fixable': False,
                                              'fix_description': d.reason, 'repaired': False}],
                           decision=d)
        code_path = generated.module_path
        pseudo_diff = f"{gen_op} generated module: {code_path}"
        new_config = dict(base_config)
        if verbose:
            print(f"  [codegen] wrote {code_path} (attempts={generated.attempts})")
        try:
            run_model = codegen_agent.load_module(code_path)
        except Exception as e:  # noqa: BLE001 — an import-time failure (bad indentation, a
            # NameError at module scope) is exactly the "broken but promising" case the tree's
            # DEBUG operation exists for, so this becomes a BUGGY node rather than a dead end.
            err = f"import failed: {type(e).__name__}: {e}"
            d = decision.decide(debug_ok=False, debug_reason=err)
            node = tree.add(parent_id=parent_node.id if parent_node else None, operation=gen_op,
                             summary=hyp['statement'], config=new_config, code_path=code_path,
                             iteration=iteration_number)
            tree.mark_buggy(node.id, err)
            tree.save()
            if verbose:
                print(f"  [codegen] {err}")
            return _reject(proposal, pseudo_diff, [{'attempt': 1, 'error_text': err,
                                                     'diagnosis': None, 'fixable': True,
                                                     'fix_description': d.reason,
                                                     'repaired': False}],
                           resulting_config=new_config, decision=d, node_id=node.id)
        # A generated module declares its OWN hyperparameters (see prompts/write_model.md) — they
        # are how a "lower the learning rate" hypothesis actually takes effect on this path. Read
        # them and run with them.
        #
        # Found live, and it silently neutered every codegen iteration of a whole run: this used
        # to be `new_config = dict(base_config)`, and since every generated module merges
        # `cfg = {**DEFAULTS, **(config or {})}`, the config the harness passed WON. So a module
        # written with lr=0.0001 was run at the incumbent's lr=0.001, and all 20 log records
        # carried a byte-identical resulting_config despite 20 different hypotheses.
        new_config = _config_for_generated(run_model, base_config, data_dir=data_dir, seed=seed)

    # ---- register this candidate as a node in the solution tree ----
    node = tree.add(parent_id=parent_node.id if parent_node else None,
                     operation=(tree_op if generated else 'config'),
                     summary=hyp['statement'], config=new_config, code_path=code_path,
                     iteration=iteration_number)

    # ---- duplicate-config guard (config-action path only) ----
    # Only meaningful for config-only candidates: two generated modules with the same config are
    # different programs, so an identical config says nothing about whether the result repeats.
    if generated is None:
        dup_iter = _find_duplicate_config(new_config, history)
        if dup_iter is not None:
            reason = (f"Identical resulting config already tried and rejected in iteration "
                      f"{dup_iter} — skipped without re-running training (this pipeline is "
                      f"deterministic given a fixed seed, so re-running would reproduce the exact "
                      f"same result, not new information).")
            tree.mark_buggy(node.id, reason)
            tree.save()
            if verbose:
                print(f"  [duplicate-guard] {reason}")
            return _reject(proposal, pseudo_diff, [{'attempt': 1, 'error_text': None,
                                                      'diagnosis': None, 'fixable': False,
                                                      'fix_description': reason, 'repaired': False}],
                           resulting_config=new_config, node_id=node.id)

    # ---- debug_run gate ----
    dbg = debug_run(run_model.train, splits, new_config, seed=seed)
    if verbose:
        print(f"  [debug_run] {dbg}")
    if not dbg.ok:
        d = decision.decide(debug_ok=False, debug_reason=dbg.reason)
        rr = error_recovery.repair(hypothesis_statement=hyp['statement'], code_diff=pseudo_diff,
                                    error_message=dbg.reason, attempt_number=1)
        token_cost = _add_usage(token_cost, rr.usage)
        # repair()'s diagnosis goes ON THE NODE, not just into the log — that node's `error` is
        # exactly what codegen_agent._context_block() shows the model on the next DEBUG attempt.
        # Found live: repair() was called every failed iteration and its output was written to
        # experiment_log.jsonl and read by nobody, while the debug prompt got only the raw
        # one-line traceback. That's a whole LLM call per failure thrown away.
        tree.mark_buggy(node.id, _error_with_diagnosis(dbg.reason, rr))
        tree.save()
        ev = error_recovery.error_event(1, dbg.reason, rr)
        if verbose:
            print(f"  [decision] {d}")
        return _reject(proposal, pseudo_diff, [ev], resulting_config=new_config, decision=d,
                       node_id=node.id)

    # ---- runtime budget gate ----
    # debug_run's estimate is the only advance warning that a candidate is unaffordable, and until
    # now nothing consumed it. Rejecting here (rather than after discovering it live) is the whole
    # point of the debug-first design: the sample already told us what the full run would cost.
    est = dbg.estimated_full_runtime_s
    if est is not None and est > MAX_ESTIMATED_FULL_RUN_S:
        reason = (f"estimated full-run time {est:.0f}s exceeds the {MAX_ESTIMATED_FULL_RUN_S:.0f}s "
                  f"budget for one candidate (config: batch_size="
                  f"{new_config.get('batch_size')}, epochs={new_config.get('epochs')}) - skipped "
                  f"without running it. Use a larger batch_size or fewer epochs.")
        d = decision.decide(debug_ok=False, debug_reason=reason)
        tree.mark_buggy(node.id, reason)
        tree.save()
        if verbose:
            print(f"  [budget] {reason}")
        return _reject(proposal, pseudo_diff, [{'attempt': 1, 'error_text': None,
                                                 'diagnosis': None, 'fixable': True,
                                                 'fix_description': reason, 'repaired': False}],
                       resulting_config=new_config, decision=d, node_id=node.id)

    # ---- full run (+ reeval) ----
    if verbose:
        print(f"  [full_run] debug sample OK (~{dbg.estimated_full_runtime_s:.0f}s estimated), "
              f"starting ...")
    cb_primary = current_best['primary'] if current_best else None
    try:
        full_metrics = run_model.train(splits, new_config, verbose=verbose)
        # The debug sample passing does not prove the full run scored every row — and a partially
        # scored split still yields a plausible [0,1] primary (see debug_run.check_coverage).
        # This is the gate that stops such a candidate from being crowned current-best.
        covered, coverage_reason = check_coverage(full_metrics, splits)
        if not covered:
            raise ValueError(coverage_reason)
        orig_primary = full_metrics['valid']['primary']
        _, mean_primary, seed_primaries = reeval.recheck(
            run_model.train, splits, new_config, original_primary=orig_primary,
            current_best_primary=cb_primary if cb_primary is not None else -1.0)
    except Exception as e:  # noqa: BLE001 — debug_run passing is NOT a guarantee at real scale
        # (its own docstring says so), so the full run and reeval's extra seeds both need this.
        # Becomes a BUGGY node: the tree can spend a DEBUG child fixing it.
        error_message = f"{type(e).__name__}: {e}"
        d = decision.decide(run_ok=False, run_error=error_message)
        rr = error_recovery.repair(hypothesis_statement=hyp['statement'], code_diff=pseudo_diff,
                                    error_message=error_message, attempt_number=1)
        token_cost = _add_usage(token_cost, rr.usage)
        tree.mark_buggy(node.id, _error_with_diagnosis(error_message, rr))
        tree.save()
        ev = error_recovery.error_event(1, error_message, rr)
        if verbose:
            print(f"  [full_run] CRASHED: {error_message}")
            print(f"  [decision] {d}")
        return _reject(proposal, pseudo_diff, [ev], resulting_config=new_config, decision=d,
                       node_id=node.id)

    # ---- the commit/revert decision tree ----
    d = decision.decide(metrics=full_metrics.get('valid'), incumbent_primary=cb_primary,
                        reeval_mean=mean_primary)
    if verbose:
        print(f"  [reeval] seed_primaries={seed_primaries} mean={mean_primary:.4f}")
        print(f"  [decision] {d.outcome}: {d.reason}")
        print(f"             path: {' > '.join(d.path)}")

    if d.is_buggy:
        tree.mark_buggy(node.id, d.reason)
    else:
        tree.mark_working(node.id, full_metrics, full_metrics['valid']['primary'])
    tree.save()

    record = logging_schema.new_record(
        iteration=iteration_number, proposal=proposal, code_diff=pseudo_diff,
        metrics=full_metrics, error_events=[], token_cost=token_cost,
        wall_clock_s=time.time() - t0, accepted=d.accepted, resulting_config=new_config,
        decision=d.to_dict(), node_id=node.id)

    if d.accepted:
        sha = decision.snapshot_to_git(
            f"agent iter {iteration_number}: {hyp['statement'][:60]} (primary={mean_primary:.4f})")
        if sha and verbose:
            print(f"  [git] snapshotted as {sha[:8]}")
        new_best = {
            'gauc': float(full_metrics['valid']['GAUC']),
            'ndcg5': float(full_metrics['valid']['nDCG@5']),
            'primary': mean_primary,
            'summary': f"iter {iteration_number}: {hyp['statement']}",
            'config': new_config,
            'code_path': code_path,
            'node_id': node.id,
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


def seed_baseline(data_dir, tree, *, model=fm_v1, seed=0, verbose=True):
    """Train the repo's own reference model once and install it as the tree's root WORKING node
    and the starting current-best. Returns the current_best dict, or None if the run failed.

    Why this exists: decision.decide() commits the first working candidate unconditionally
    (`incumbent=none` -> COMMIT), because with nothing to beat, anything that runs is by
    definition the best so far. Starting from an EMPTY incumbent made that a trap — found live, a
    first-iteration generated module scoring primary=0.5228 became current-best even though
    baseline_scores.json records fm_official at 0.5993 and the hypothesis prompt was quoting that
    same number back at the model. The whole rest of the run then measured itself against a
    baseline the repo already beat.

    Seeding a REAL run (rather than reading baseline_scores.json) costs one training pass and buys
    two things a recorded number can't: it is measured through this exact pipeline on this exact
    data_dir, and it gives the solution tree a working root node to IMPROVE from instead of
    forcing it to draft blind. It runs once — afterwards it lives in state.json/solution_tree.json
    and resume picks it up like anything else.
    """
    splits = load_train_valid(data_dir)   # 'test' never present past this line
    config = dict(model.DEFAULT_CONFIG)
    if 'data_dir' in config:
        config['data_dir'] = data_dir
    config['seed'] = seed
    if verbose:
        print(f"[baseline] no incumbent yet - training {model.__name__} once to set the bar ...")
    try:
        metrics = model.train(splits, config, verbose=verbose)
    except Exception as e:  # noqa: BLE001 — a broken baseline must not stop the loop from
        # running; it just means iteration 1 starts with no incumbent, exactly as it used to.
        if verbose:
            print(f"[baseline] FAILED ({type(e).__name__}: {e}) - continuing with no incumbent")
        return None

    valid = metrics['valid']
    node = tree.add(parent_id=None, operation='baseline',
                     summary=f"{model.__name__} reference baseline", config=config,
                     code_path=None, iteration=0)
    tree.mark_working(node.id, metrics, valid['primary'])
    tree.save()
    if verbose:
        print(f"[baseline] {model.__name__} primary={valid['primary']:.4f} "
              f"(GAUC={valid['GAUC']:.4f}, nDCG@5={valid['nDCG@5']:.4f}) - this is the bar to beat")
    return {
        'gauc': float(valid['GAUC']),
        'ndcg5': float(valid['nDCG@5']),
        'primary': float(valid['primary']),
        'summary': f"baseline: {model.__name__} at its default config",
        'config': config,
        'code_path': None,
        'node_id': node.id,
    }


def run_loop(data_dir, *, expected_total_iterations, max_iterations=None, model=fm_v1, seed=0,
             log_path=archivist.LOG_PATH, state_path=resume.STATE_PATH, verbose=True,
             seed_with_baseline=True, max_hours=None, ignore_convergence=False):
    """Drives the full loop, resuming from `state_path` if it exists (crash-safe resume hard
    requirement). `max_iterations` caps how many iterations run regardless of convergence —
    resolves AGENT_STRATEGY.md's open "iteration cap" question: yes, always capped, defaults to
    `expected_total_iterations` so a run is never accidentally unbounded.

    `seed_with_baseline` trains the reference model once at the very start of a FRESH run to set
    the incumbent — see seed_baseline(). Skipped entirely on resume (the incumbent is already in
    state.json) and switchable off for tests that don't want a real training run.

    `max_hours` is a WALL-CLOCK budget, checked between iterations: the loop stops once the
    deadline passes rather than when an iteration count runs out. It never interrupts an iteration
    mid-flight — a half-finished training run archives nothing, so killing one would just discard
    work. Pair it with a large `max_iterations` when you want "run for N hours" semantics.

    `ignore_convergence` disables the early stop. The convergence rule exists to avoid burning
    compute once the search has stalled, so leave it on for normal use; turn it off only when the
    goal is explicitly to fill a time budget with attempts."""
    state = resume.load_state(state_path)
    current_best = state['current_best']
    start_iter = state['last_completed_iteration'] + 1
    cap = max_iterations if max_iterations is not None else expected_total_iterations
    history = _load_history(log_path)
    tree = SolutionTree.load()   # persisted separately; survives crash/resume like state.json

    if seed_with_baseline and current_best is None and start_iter == 1:
        current_best = seed_baseline(data_dir, tree, model=model, seed=seed, verbose=verbose)
        if current_best is not None:
            # Persist immediately: a crash between here and iteration 1 finishing must not throw
            # away a training run, and must not silently drop the incumbent on resume.
            resume.save_state({'last_completed_iteration': 0, 'current_best': current_best},
                               path=state_path)

    if verbose and start_iter > 1:
        print(f"[resume] continuing from iteration {start_iter} "
              f"(current-best primary={current_best['primary']:.4f})" if current_best else
              f"[resume] continuing from iteration {start_iter} (no current-best yet)")

    deadline = (time.time() + max_hours * 3600.0) if max_hours else None

    for it in range(start_iter, cap + 1):
        if deadline is not None and time.time() >= deadline:
            if verbose:
                print(f"[budget] wall-clock budget of {max_hours}h reached - stopping before "
                      f"iteration {it}. Re-run the same command to continue from here.")
            break
        if verbose:
            remaining = f" | {(deadline - time.time()) / 60:.0f} min left" if deadline else ""
            print(f"=== iteration {it}/{cap}{remaining} ===")
        record, current_best = _run_iteration_with_retry(
            data_dir, iteration_number=it, expected_total_iterations=expected_total_iterations,
            current_best=current_best, history=history, tree=tree, model=model, seed=seed,
            verbose=verbose)
        archivist.archive(record, current_best=current_best, log_path=log_path,
                           state_path=state_path)
        history.append(logging_schema.to_history_entry(record))
        if verbose:
            outcome = (record.get('decision') or {}).get('outcome')
            print(f"  -> {outcome or ('ACCEPTED' if record['accepted'] else 'rejected')}")
            print(f"  [tree]\n{tree.render()}")
        if not ignore_convergence and _converged(history):
            if verbose:
                print(f"  converged: last {CONVERGENCE_N} iterations improved primary by "
                      f"<= {CONVERGENCE_EPSILON}")
            break

    return current_best, history
