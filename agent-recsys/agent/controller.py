"""
The outer loop. Deterministic Python owns iteration count, wall-clock, file
promotion, and logging -- the LLM is only ever called to produce one
Proposal per candidate (see llm_client.py), never given direct control over
the real pipeline directory.

One pass through the loop:
  0. check the cost and consecutive-failure kill switches before spending
     anything this iteration
  1. build context from the best-known pipeline code + iteration history
  2. ask the LLM for `candidates_per_iteration` independent proposals for
     this same iteration slot (each: one hypothesis + one full-file rewrite
     + its own predicted improvement); each candidate sees what earlier
     candidates THIS iteration scored, so it can refine a near-miss instead
     of only ever pivoting to something unrelated
  3. apply each candidate to its own throwaway scratch copy, run it under a
     timeout, and keep whichever successful candidate scored highest
  4. on success: log it, print expected vs. actual improvement, and only
     promote the winning candidate's code -> current if it matches or beats
     the best score seen so far (across ALL runs, not just this one -- see
     "cross-run best" below) -- a regression is logged honestly but never
     becomes the base the next iteration builds on, so mistakes can't
     compound
  5. on a recoverable failure (bad LLM output, sandbox/training error) for a
     given candidate: feed the traceback back to the LLM and retry the SAME
     hypothesis up to max_repair_attempts times; if still failing, that one
     candidate is dropped and the others are still considered
  6. on a FATAL failure (the API call itself didn't complete: network,
     auth, rate limit, quota, ...): stop the whole run immediately --
     retrying with no backoff can't fix any of these
  7. check the convergence monitor and the consecutive-failure counter;
     stop if converged / budget exhausted / stuck erroring

Cross-run best: `best/` also carries a small `_best_meta.json` recording the
score of whatever snapshot is saved there. A fresh `run()` call reads it (if
present) and resumes from it -- both the score to beat AND the actual code
in `pipeline_dir` are restored from `best/` before iteration 0. Without
this, a second run starting after the first would only know about whatever
was last left in `pipeline_dir`, which is not necessarily the best code ever
found (e.g. if an earlier run predates the compounding-regression fix, or
was interrupted) -- `best/` is the one place a validated high-water mark is
guaranteed to live.

Hidden-test compliance: per the challenge brief, "the agent... never sees
the hidden test set" during development -- only agent/finalize.py's ONE-TIME
call after the run has stopped is allowed to touch it. Every iteration here
is scored on `metrics['valid']` only; `pipeline/_run_iteration.py` never
even hands the test split's rows to run_fm(), so this isn't just a policy
this module follows, it's structurally enforced upstream. See that file's
HIDDEN-TEST COMPLIANCE note for the actual enforcement point.

Baseline reproduction: before iteration 0 of a genuinely fresh run (no prior
`best/` to resume from), `_check_baseline_reproduction()` runs the untouched
seeded pipeline once -- no LLM call, no code change -- and compares the
result against `baseline_scores.json`'s official validation number, writing
`logs/baseline_reproduction.json`. This is the brief's explicit step 1
("confirm it reaches the official baseline's reported validation score")
as its own auditable artifact, not just an assumption that iteration 0
happens to start from unmodified code.

Run log: each record includes a `code_diff` (unified diff of the winning
candidate's target_file against the pre-change version) alongside the
hypothesis, metrics, and error/recovery fields already logged -- the run-log
schema the challenge brief's deliverables ask for.
"""
import difflib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from agent import context as ctx
from agent import sandbox
from agent.convergence import ConvergenceMonitor
from agent.llm_client import LLMFatalError

MAX_REPAIR_ATTEMPTS = 3
BEST_META_FILENAME = "_best_meta.json"
MAX_DIFF_CHARS = 20000  # generous relative to a ~30KB full-file rewrite; caps pathological cases only


def _load_best_meta(best_dir: Path) -> dict | None:
    meta_path = best_dir / BEST_META_FILENAME
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None  # corrupt/unreadable meta -- treat as "no prior best" rather than crash


def _write_best_meta(best_dir: Path, valid_primary: float, iteration: int, hypothesis: str,
                      test_primary: float | None = None):
    # test_primary is None from every call inside this module -- the
    # hidden-test set is never scored during iteration (see this module's
    # docstring), so there is no test score to record until
    # agent/finalize.py's one-time run computes one. finalize.py patches it
    # in afterward via this same function, loading the existing meta first
    # so it doesn't clobber valid_primary/iteration/hypothesis.
    meta = {
        # float(...) here is deliberate, not decorative: every caller inside
        # this module already passes a native float (run_result.metrics
        # round-tripped through JSON via sandbox.run_scratch()), but
        # finalize.py's one-time patch of test_primary calls
        # baseline.train_and_predict() directly, in-process -- no JSON
        # round-trip in between -- so a numpy.float32/float64 score reaches
        # here unconverted and json.dump() below would raise
        # "Object of type float32 is not JSON serializable". float() is a
        # no-op on an already-native float, so this is safe for every
        # caller, not just finalize.py's.
        "valid_primary": float(valid_primary),
        "test_primary": float(test_primary) if test_primary is not None else None,
        "iteration": iteration,
        "hypothesis": hypothesis,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(best_dir / BEST_META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _compute_diff(pipeline_dir: Path, target_file: str, new_content: str) -> str:
    """Unified diff of the proposed new_content against target_file's
    current (pre-change) content in pipeline_dir -- i.e. against the
    best-known state this candidate is proposing to change. Required by the
    challenge brief's run-log deliverable ("the code diff applied")."""
    before_path = pipeline_dir / target_file
    before = before_path.read_text(encoding="utf-8") if before_path.exists() else ""
    diff_lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"{target_file} (before)",
        tofile=f"{target_file} (after)",
    )
    diff = "".join(diff_lines)
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + f"\n... [diff truncated at {MAX_DIFF_CHARS} chars]"
    return diff


def _check_baseline_reproduction(pipeline_dir: Path, baseline_ref: dict, data_dir: str, log_path: Path):
    """Runs the untouched, seeded pipeline exactly once -- no LLM call, no
    code change -- and compares it against the official validation score
    from baseline_scores.json. Writes logs/baseline_reproduction.json as an
    explicit, auditable artifact for the brief's step 1 ("Reproduce the
    official baseline... confirm it reaches the official baseline's
    reported validation score"). Only called for a genuinely fresh run (see
    run()) -- a resumed run already has this evidence from whichever run
    first produced the best/ snapshot it's resuming from.

    Best-effort: a failure here is reported but never blocks iteration 0
    from starting -- if the seeded pipeline genuinely can't run, iteration
    0's own attempt will surface the identical error through the normal
    repair-attempt path anyway.

    Returns the measured valid primary (used by run() to seed
    best_valid_primary -- see the call site) or None if the check failed to
    run at all."""
    official = baseline_ref.get("scores", {}).get("fm_official", {}).get("valid", {}).get("primary")
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "official_valid_primary": official}
    scratch = sandbox.make_scratch_copy(pipeline_dir)
    try:
        run_result = sandbox.run_scratch(scratch, data_dir)
    finally:
        sandbox.cleanup(scratch)

    if not run_result.ok:
        result.update({"ok": False, "error": (run_result.error or "")[:2000]})
        print(f"baseline reproduction check: FAILED to run the seeded pipeline -- {(run_result.error or '')[:300]}")
    else:
        valid_primary = run_result.metrics["valid"]["primary"]
        diff = (valid_primary - official) if official is not None else None
        result.update({"ok": True, "measured_valid_primary": valid_primary, "delta_vs_official": diff})
        if official is not None:
            print(
                f"baseline reproduction check: measured valid primary {valid_primary:.4f} "
                f"vs. official {official:.4f} (delta {'+' if diff >= 0 else ''}{diff:.4f})"
            )
        else:
            print(f"baseline reproduction check: measured valid primary {valid_primary:.4f} "
                  f"(no official number found in baseline_scores.json to compare against)")

    out_path = log_path.parent / "baseline_reproduction.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return run_result.metrics["valid"]["primary"] if run_result.ok else None


def _propose_and_run_one_candidate(
    pipeline_dir: Path, baseline_ref: dict, history: list, reference_primary: float | None,
    llm_client, iteration: int, candidate_idx: int, candidates_per_iteration: int,
    sibling_candidates: list, data_dir: str,
) -> dict:
    """Runs propose -> apply -> run_scratch for ONE candidate slot within
    `iteration`, retrying up to MAX_REPAIR_ATTEMPTS times on a recoverable
    failure. Returns a plain dict describing the outcome (never raises,
    except that a fatal LLM/API error is surfaced via the "fatal" key for
    the caller to act on, not retried here)."""
    last_error = None
    proposal = None
    run_result = None
    scratch = None
    fatal = None
    input_tokens = 0
    output_tokens = 0
    repair_attempts = 0
    diff = ""

    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        prompt = ctx.build_prompt(
            pipeline_dir, baseline_ref, history, last_error, reference_primary,
            candidate_idx=candidate_idx, candidates_per_iteration=candidates_per_iteration,
            sibling_candidates=sibling_candidates,
        )
        # iteration=iteration (not attempt) is what a stateful client should
        # key off of: a real LLM sees the traceback via `last_error` in the
        # prompt and can propose a genuinely different fix each retry;
        # DryRunClient uses `iteration` to pick from its fixed plan so
        # retries-within-an-iteration don't burn through the whole plan.
        try:
            proposal = llm_client.propose(prompt, iteration=iteration)
        except LLMFatalError as e:
            # The API call itself never completed -- retrying instantly with
            # no backoff cannot fix a network outage, a bad key, a rate
            # limit, or a dead model name. Surface it so the caller aborts
            # the whole run instead of burning further attempts/candidates
            # on a failure that will just repeat identically.
            fatal = f"{type(e).__name__}: {e}"
            proposal = None
            run_result = sandbox.RunResult(ok=False, error=fatal)
            break
        except Exception as e:  # noqa: BLE001 - a recoverable LLM-side content
            # failure (truncated/malformed JSON): feed it back as last_error
            # and let the model try again next attempt.
            proposal = None
            last_error = f"{type(e).__name__}: {e}"
            run_result = sandbox.RunResult(ok=False, error=last_error)
            repair_attempts = attempt + 1
            continue

        input_tokens += proposal.input_tokens
        output_tokens += proposal.output_tokens
        # Diffed against pipeline_dir's CURRENT content, i.e. the
        # best-known state as of the start of this candidate's attempt --
        # computed fresh each retry, since a repair attempt may propose a
        # different new_content than the one that just failed.
        diff = _compute_diff(pipeline_dir, proposal.target_file, proposal.new_content)

        label = f"[iter {iteration}]" if candidates_per_iteration == 1 else f"[iter {iteration} cand {candidate_idx + 1}/{candidates_per_iteration}]"
        print(
            f"{label} proposing: \"{proposal.hypothesis[:90]}\" "
            f"(expected valid primary {'+' if proposal.expected_delta >= 0 else ''}{proposal.expected_delta:.4f})"
        )

        scratch = sandbox.make_scratch_copy(pipeline_dir)
        try:
            sandbox.apply_change(scratch, proposal.target_file, proposal.new_content)
            run_result = sandbox.run_scratch(scratch, data_dir)
        except sandbox.HiddenTestViolation:
            # Deliberately NOT folded into the recoverable path below: the
            # hidden test set was scored, which no alternative proposal
            # could fix and which must not cost the agent repair attempts
            # while it keeps happening. Stop the whole run instead -- see
            # HiddenTestViolation's docstring.
            sandbox.cleanup(scratch)
            raise
        except Exception as e:  # noqa: BLE001 - any sandbox-side failure is a recoverable iteration error
            run_result = sandbox.RunResult(ok=False, error=f"{type(e).__name__}: {e}")

        if run_result.ok:
            break
        last_error = run_result.error
        sandbox.cleanup(scratch)
        scratch = None
        repair_attempts = attempt + 1

    return {
        "proposal": proposal, "run_result": run_result, "scratch": scratch,
        "fatal": fatal, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "repair_attempts": repair_attempts, "diff": diff,
    }


def run(
    pipeline_dir: Path,
    log_path: Path,
    best_dir: Path,
    llm_client,
    data_dir: str,
    max_iterations: int = 50,
    max_wall_clock_hours: float = 6.0,
    max_cost_usd: float | None = None,
    max_consecutive_failures: int = 3,
    candidates_per_iteration: int = 2,
):
    # Resolve to absolute up front: run_scratch() executes with cwd set to a
    # throwaway scratch directory, so a relative data_dir would silently
    # resolve against the WRONG directory and fail with a confusing
    # FileNotFoundError every single iteration.
    data_dir = str(Path(data_dir).resolve())

    if candidates_per_iteration < 1:
        raise ValueError(f"candidates_per_iteration must be >= 1, got {candidates_per_iteration}")

    baseline_ref = ctx.load_baseline_reference(pipeline_dir)
    epsilon = baseline_ref["convergence_rule"]["epsilon"]
    n_iter = baseline_ref["convergence_rule"]["N"]

    monitor = ConvergenceMonitor(
        epsilon=epsilon, n_iterations=n_iter,
        max_iterations=max_iterations, max_wall_clock_hours=max_wall_clock_hours,
    )
    history = []
    best_valid_primary = -1.0
    total_cost_usd = 0.0
    consecutive_failed_iterations = 0
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from a prior run's best, if one was ever recorded. Both the
    # score to beat AND the actual code get restored from best/, because
    # whatever is currently sitting in pipeline_dir is not guaranteed to be
    # the best code ever found (see this function's docstring).
    prior_best = _load_best_meta(best_dir)
    if prior_best is not None:
        best_valid_primary = prior_best["valid_primary"]
        sandbox.promote(best_dir, pipeline_dir)  # promote() is a generic "copy every file src -> dst"
        # promote() has no notion of BEST_META_FILENAME -- it just copied
        # everything, meta file included. pipeline_dir (and every scratch
        # copy made from it) has no business carrying that bookkeeping file
        # around, so drop the stray copy here rather than teaching the
        # generic sandbox.promote() about a controller-specific filename.
        stray_meta = pipeline_dir / BEST_META_FILENAME
        if stray_meta.exists():
            stray_meta.unlink()
        print(
            f"resuming from a prior best: valid primary {best_valid_primary:.4f} "
            f"(iteration {prior_best.get('iteration', '?')} of an earlier run, saved {prior_best.get('saved_at', '?')}) "
            f"-- pipeline/ restored to that state"
        )
    else:
        # Only for a genuinely fresh run -- a resumed run already has this
        # evidence from whichever earlier run first produced the best/
        # snapshot it just restored above.
        baseline_measured = _check_baseline_reproduction(pipeline_dir, baseline_ref, data_dir, log_path)
        if baseline_measured is not None:
            # Without this, best_valid_primary stays at -1.0 through
            # iteration 0, so reference_primary is None and iteration 0 is
            # adopted UNCONDITIONALLY (see the adoption rule below) -- even
            # if its hypothesis makes things worse than doing nothing at
            # all. Seeding from the just-measured baseline means iteration
            # 0 is held to the same "must match or beat" bar as every
            # iteration after it.
            best_valid_primary = baseline_measured
            # Also snapshot the unmodified baseline itself into best_dir as
            # a floor: if no iteration this run ever beats it, best/ still
            # holds a valid, submission-ready state (finalize.py and any
            # future resume both depend on best/ being non-empty) instead
            # of staying empty because nothing "new" ever got promoted.
            sandbox.snapshot_to(pipeline_dir, best_dir)
            _write_best_meta(
                best_dir, best_valid_primary, -1,
                "(unmodified seeded baseline -- no LLM change yet)",
            )
            print(
                f"iteration 0 will be held to the reproduced baseline: "
                f"valid primary {best_valid_primary:.4f} must be matched or "
                f"beaten to be adopted"
            )

    if max_cost_usd:
        print(f"cost kill switch armed: stopping if estimated spend reaches ${max_cost_usd:.2f}")
    print(f"consecutive-failure kill switch armed: stopping after {max_consecutive_failures} failed iteration(s) in a row")
    if candidates_per_iteration > 1:
        print(f"trying {candidates_per_iteration} candidate proposals per iteration, keeping whichever scores best")

    for i in range(max_iterations):
        # --- kill switches, checked before spending anything this iteration ---
        if max_cost_usd and total_cost_usd >= max_cost_usd:
            print(f"\nstopped before iteration {i}: estimated spend ${total_cost_usd:.2f} reached the ${max_cost_usd:.2f} cap")
            break
        if consecutive_failed_iterations >= max_consecutive_failures:
            print(f"\nstopped before iteration {i}: {consecutive_failed_iterations} failed iterations in a row")
            break

        iter_start = time.time()
        record = {"iteration": i, "timestamp": datetime.now(timezone.utc).isoformat()}
        reference_primary = best_valid_primary if best_valid_primary >= 0 else None

        candidate_outcomes = []
        sibling_candidates = []  # brief summaries shown to the NEXT candidate this same iteration
        fatal = None
        iter_input_tokens = 0
        iter_output_tokens = 0

        for c in range(candidates_per_iteration):
            outcome = _propose_and_run_one_candidate(
                pipeline_dir, baseline_ref, history, reference_primary, llm_client, i,
                candidate_idx=c, candidates_per_iteration=candidates_per_iteration,
                sibling_candidates=sibling_candidates, data_dir=data_dir,
            )
            iter_input_tokens += outcome["input_tokens"]
            iter_output_tokens += outcome["output_tokens"]
            candidate_outcomes.append(outcome)

            if outcome["proposal"] is not None:
                rr = outcome["run_result"]
                sibling_candidates.append({
                    "hypothesis": outcome["proposal"].hypothesis,
                    "status": "ok" if rr.ok else "failed",
                    "valid_primary": rr.metrics["valid"]["primary"] if rr.ok else None,
                    "error": (rr.error or "")[:300] if not rr.ok else None,
                })

            if outcome["fatal"]:
                fatal = outcome["fatal"]
                break  # don't spend further candidates on an infra-level failure

        total_cost_usd += llm_client.estimate_cost(iter_input_tokens, iter_output_tokens)

        # Pick whichever candidate actually succeeded and scored highest.
        successful = [o for o in candidate_outcomes if o["run_result"] is not None and o["run_result"].ok]
        winner = max(successful, key=lambda o: o["run_result"].metrics["valid"]["primary"]) if successful else None
        loser_outcomes = [o for o in candidate_outcomes if o is not winner]
        for o in loser_outcomes:
            if o["scratch"] is not None:
                sandbox.cleanup(o["scratch"])

        if winner is not None:
            proposal, run_result, scratch, diff = winner["proposal"], winner["run_result"], winner["scratch"], winner["diff"]
        else:
            # nothing succeeded -- report the LAST candidate tried, same as
            # single-candidate behavior before this feature existed
            last = candidate_outcomes[-1]
            proposal, run_result, scratch, diff = last["proposal"], last["run_result"], last["scratch"], last["diff"]

        total_repair_attempts = sum(o["repair_attempts"] for o in candidate_outcomes)
        if total_repair_attempts:
            record["repair_attempts"] = total_repair_attempts
        if len(candidate_outcomes) > 1:
            record["candidates_tried"] = len(candidate_outcomes)
            record["candidate_scores"] = [
                (o["run_result"].metrics["valid"]["primary"] if o["run_result"] is not None and o["run_result"].ok else None)
                for o in candidate_outcomes
            ]

        if proposal is not None:
            record.update({
                "hypothesis": proposal.hypothesis,
                "target_stage": proposal.target_stage,
                "target_file": proposal.target_file,
                "expected_delta": proposal.expected_delta,
                "code_diff": diff,
                "resource_usage": {
                    "llm_input_tokens": iter_input_tokens,
                    "llm_output_tokens": iter_output_tokens,
                },
                "manual_intervention": False,
            })
        else:
            record.update({
                "hypothesis": (
                    f"(fatal LLM/API error -- {fatal[:200]})" if fatal
                    else "(no proposal produced -- every LLM call this iteration failed)"
                ),
                "target_stage": "n/a",
                "target_file": "n/a",
                "expected_delta": None,
                "resource_usage": {"llm_input_tokens": iter_input_tokens, "llm_output_tokens": iter_output_tokens},
                "manual_intervention": False,
            })

        if run_result is not None and run_result.ok:
            record["status"] = "ok"
            record["metrics"] = run_result.metrics
            record["wall_clock_s"] = round(time.time() - iter_start, 2)
            valid_primary = run_result.metrics["valid"]["primary"]
            actual_delta = valid_primary - reference_primary if reference_primary is not None else None
            record["reference_primary"] = reference_primary
            record["actual_delta"] = actual_delta

            # Only adopt this change as the new "current" if it matches or
            # beats the best score seen so far (across all runs -- see the
            # cross-run-best restore at the top of this function). This is
            # the fix for compounding regressions: previously EVERY
            # successful iteration got promoted regardless of whether it
            # helped, so iteration N+1 would build its hypothesis on top of
            # iteration N's regression, then N+2 would build on N+1's, etc.
            # -- three iterations in a row could each individually "work"
            # (the harness runs fine) while the actual score drifts steadily
            # downward. Now a regression is logged honestly but discarded:
            # the next iteration still reasons from the best-known code, not
            # from a worse state.
            adopted = reference_primary is None or valid_primary >= reference_primary
            record["adopted"] = adopted
            if adopted:
                sandbox.promote(scratch, pipeline_dir)
            if valid_primary > best_valid_primary:
                best_valid_primary = valid_primary
                sandbox.snapshot_to(scratch, best_dir)
                # test_primary is intentionally omitted -- the hidden test
                # set is never scored during iteration (see this module's
                # docstring); agent/finalize.py fills it in once, later.
                _write_best_meta(best_dir, valid_primary, i, proposal.hypothesis)
                record["new_best"] = True
            sandbox.cleanup(scratch)

            consecutive_failed_iterations = 0
            monitor.record(i, valid_primary, iter_input_tokens, iter_output_tokens)

            delta_str = "n/a (first iteration)" if actual_delta is None else f"{'+' if actual_delta >= 0 else ''}{actual_delta:.4f}"
            expected_str = f"{'+' if proposal.expected_delta >= 0 else ''}{proposal.expected_delta:.4f}"
            candidate_note = f"  (won over {len(candidate_outcomes) - 1} other candidate(s))" if len(candidate_outcomes) > 1 else ""
            print(
                f"[iter {i}] OK  \"{proposal.hypothesis[:70]}\" "
                f"-> valid primary {valid_primary:.4f}  "
                f"(expected {expected_str}, actual {delta_str})"
                f"{'  (new best)' if record.get('new_best') else ''}"
                f"{'' if adopted else '  [NOT ADOPTED -- regressed vs best, current left unchanged]'}"
                f"{candidate_note}"
            )
        else:
            record["status"] = "failed"
            record["final_error"] = (run_result.error if run_result is not None else "") or ""
            record["final_error"] = record["final_error"][:2000]
            record["wall_clock_s"] = round(time.time() - iter_start, 2)
            record["actual_delta"] = None
            record["reference_primary"] = reference_primary
            # current pipeline is left untouched -- next iteration proposes
            # a fresh hypothesis from the last known-good state
            consecutive_failed_iterations += 1
            # Record the current best (whatever it is -- including a score
            # resumed from a prior run) as this iteration's score: a failed
            # iteration is "no change", never a regression to -1.0. Before
            # cross-run resume existed this was equivalent to just always
            # passing best_valid_primary (an empty `history` implied no
            # success yet, which meant best_valid_primary was already -1.0)
            # -- with resume, `history` can be empty on iteration 0 while
            # best_valid_primary is a real resumed score, so branching on
            # `history` would wrongly feed -1.0 into the convergence window.
            monitor.record(i, best_valid_primary, iter_input_tokens, iter_output_tokens)
            hyp_preview = proposal.hypothesis[:70] if proposal is not None else "(no proposal -- LLM call kept failing)"
            if fatal:
                print(f"[iter {i}] FATAL: {fatal[:200]}")
            else:
                print(
                    f"[iter {i}] FAIL \"{hyp_preview}\" "
                    f"({len(candidate_outcomes)} candidate(s) tried, none succeeded)"
                )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        history.append(record)

        if fatal:
            print(f"\nstopped at iteration {i}: fatal LLM/API error -- {fatal[:300]}")
            print(f"estimated spend so far: ${total_cost_usd:.2f}")
            break

        stop, reason = monitor.should_stop(i)
        if stop:
            print(f"\nstopped at iteration {i}: {reason}")
            print("resource summary:", json.dumps(monitor.resource_summary(), indent=2))
            break

    # Deliverable 3 explicitly asks for "a short summary reporting the
    # number of manual interventions during the run." manual_intervention
    # is per-record (see pipeline/README.md's Autonomy note: hardcoded
    # False throughout an unattended run, flip it by hand for any iteration
    # where a human actually stepped in); this is the aggregate a human
    # would otherwise have to compute themselves from the raw log.
    manual_interventions = sum(1 for r in history if r.get("manual_intervention"))

    return {
        "history": history,
        "resource_summary": monitor.resource_summary(),
        "best_valid_primary": best_valid_primary,
        "total_cost_usd": total_cost_usd,
        "manual_interventions": manual_interventions,
    }
