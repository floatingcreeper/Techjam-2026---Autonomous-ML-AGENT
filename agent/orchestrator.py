"""Orchestrator -- hypothesis-driven best-first tree search.

The control loop is deterministic policy; the LLM roles are the operators.

THREE STOPPING CONCEPTS, kept strictly separate (docs/SYSTEM.md §16):

  1. OFFICIAL   `eps=0.002, N=3` over the best-so-far series of EXECUTED experiments, plus the hard
                `max_iter=50` cap and the 6 h wall-clock backstop. Read verbatim from
                `baseline_scores.json -> convergence_rule`. This is the only thing that ends a healthy
                run, and nothing below may postpone it.
  2. RESEARCH   `research_stall` / plateau escalation. Changes WHAT is proposed. Never terminates.
  3. LIVENESS   `proposal_guard`. Aborts only the pathological case where the Proposer cannot emit
                anything executable at all. Not a competing convergence definition.

Compute is deliberately not the binding constraint: under the official rule a run converges after
roughly 4-6 executed experiments. The objective is therefore NOT to spend iterations, it is to make
every pre-convergence experiment a real one -- which is what the duplicate/no-op/validation machinery
below is for.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from agent import blockspec, champion, datced, events, executor, guardrails, ledger, mutate, portfolio, reeval
from agent.llm.schemas import BlockEdit, Hypothesis, RecoveryAction
from agent.memory import Memory
from agent.roles import coder, proposer, reflector
from agent.stats import Evaluator, classify_evidence, paired_bootstrap, per_user_rank
from agent.tree import EXACT_NOOP, NEAR_NOOP, STRUCTURAL_NOOP, Node, SearchTree
from evaluate import evaluate                      # fixed harness (used by assembly)
from pipeline.contracts import Cfg

BLOCK_SRC = "pipeline/baseline_blocks"
FM_VALID = 0.6015

# Read the OFFICIAL convergence rule from the organizer's artifact so the code cannot drift from it.
try:
    _BS = json.loads(Path("baseline_scores.json").read_text())
    OFFICIAL_EPS = float(_BS["convergence_rule"]["epsilon"])
    OFFICIAL_N = int(_BS["convergence_rule"]["N"])
except Exception:                                   # never let a missing file stop a run
    OFFICIAL_EPS, OFFICIAL_N = 0.002, 3


# ------------------------------------------------------------------ phases
def phase_of(it, ph):
    if it == 0:
        return 0
    if it <= ph.breadth_until:
        return 1
    if it <= ph.depth_until:
        return 2
    return 3


def _converged(best_series, eps, N):
    """The OFFICIAL rule: the best-so-far has not improved by more than `eps` over the last `N`
    accepted iterations. `best_series` is appended ONLY for executed experiments -- a duplicate, a
    structural no-op or a rejected proposal never trained a model and is not an iteration."""
    return len(best_series) > N and (best_series[-1] - best_series[-1 - N]) <= eps


# ------------------------------------------------------------------ context
def _budget_tier(executed, maxit):
    frac = executed / max(1, maxit)
    if frac < 0.40:
        return "EARLY: prefer cheap, fast, NOVEL mechanisms -- maximise breadth of evidence."
    if frac < 0.75:
        return "MID: refine and COMBINE the strongest directions found so far."
    return "LATE: squeeze remaining gains; composition/ensembling is encouraged."


def build_proposer_context(tree, mem, phase, executed, maxit, wall_used, wall_limit,
                           tokens, plateau, feedback=None):
    """What the Proposer sees. Budget figures are informational -- they shape WHICH experiment is
    chosen before convergence. They are explicitly NOT a quota to consume (docs/SYSTEM.md §16)."""
    best = tree.best()
    t = mem.research_table()
    lines = [
        f"Phase {phase}. Executed experiments: {executed}/{maxit} (HARD CAP -- not a target). "
        f"Wall-clock {wall_used / 60:.1f}min / {wall_limit / 3600:.1f}h. LLM tokens {tokens}.",
        f"STRATEGY: {_budget_tier(executed, maxit)}",
        "NOTE: the benchmark stops at the official convergence rule "
        f"(eps={OFFICIAL_EPS}, N={OFFICIAL_N}); do not try to use up the remaining iterations.",
        f"FM baseline valid primary = {FM_VALID}.",
    ]
    ch = t.get("champion")
    if ch:
        lines.append(f"CURRENT BEST: {ch['node_id']} lever={ch['lever']} model={ch['model_type']} "
                     f"primary={ch['primary_valid']:.5f} (GAUC {ch['GAUC']:.5f}, "
                     f"nDCG@5 {ch['nDCG@5']:.5f})")
    lines.append(f"Best of {best.cfg.model_type} family currently mounted on the selected parent.")

    def blk(title, items, empty="(none yet)"):
        lines.append(title)
        lines.extend([f"  {x}" for x in items] or [f"  {empty}"])

    blk("CONFIRMED (P(delta>0)>=0.90 -- build on these):", t["confirmed"])
    blk("PROMISING (0.60<=P<0.90 -- worth a sharper test):", t["promising"])
    blk("INCONCLUSIVE (effect below what this validation set can resolve; "
        "a repeat alone will NOT settle it -- change the mechanism or the design):", t["inconclusive"])
    blk("REJECTED (P(delta<0)>=0.90 -- do not repeat; consider the opposite):", t["rejected"])
    blk("NO EFFECT (the intervention never reached execution -- NOT scientific evidence):",
        t["no_effect"], "(none)")
    blk("UNSUPPORTED CAPABILITIES (asked for, not available in this harness):",
        t["unsupported_capability"], "(none)")
    blk("DIVERSE PORTFOLIO CANDIDATES (weak standalone, valuable in the blend -- "
        "do NOT treat these as failures):", t["diverse_portfolio_candidates"], "(none yet)")
    lines.append(f"BEST PER LEVER: {t['best_per_lever']}")
    lines.append(f"UNTRIED LEVERS: {', '.join(t['untried_levers']) or 'none'}")

    lines.append("CONFIG KNOBS ACTUALLY HONOURED BY EACH BLOCK SET "
                 "(anything else is rejected before training):")
    lines.append(blockspec.honoured_summary())
    lines.append(f"loss_type must be one of {sorted(blockspec.LOSS_TYPES)}; "
                 f"mtl_arch must be one of {sorted(blockspec.MTL_ARCHS)}; "
                 f"aux_tasks from {sorted(blockspec.AUX_TASK_NAMES)}.")
    if plateau:
        lines.append("RESEARCH PLATEAU: recent experiments produced neither performance nor new "
                     "information. Change the mechanism -- an untried lever, an untested interaction "
                     "between two confirmed levers, or a deliberately DECORRELATED model for the "
                     "portfolio. Do not re-run a variation of the current best.")
    if feedback:
        lines.append(f"YOUR PREVIOUS PROPOSAL WAS NOT EXECUTABLE: {feedback} "
                     f"Propose a materially DIFFERENT experiment.")
    lines.append("Propose the next high-expected-value change.")
    return "\n".join(lines)


def build_coder_context(parent, hyp):
    tgt = hyp.target_block
    src = (Path(parent.block_dir) / f"{tgt}.py").read_text(encoding="utf-8")
    return (f"Hypothesis: {hyp.statement}\nRationale: {hyp.rationale}\n"
            f"Target block: {tgt}\nconfig_delta: {hyp.config_delta_json}\n\n"
            f"Current source of {tgt}.py:\n```python\n{src}\n```\n"
            f"Rewrite {tgt}.py to implement the hypothesis. Keep the block's function signature.")


# ------------------------------------------------------------------ run state
class RunState:
    """Counters that keep the benchmark contract auditable (docs/SYSTEM.md §16 accounting table)."""

    def __init__(self, cfg, maxit):
        self.experiments_executed = 0        # trained + produced metrics; compared against max_iter
        self.proposal_attempts = 0           # every Proposer call, including rejected ones
        self.proposals_rejected = 0
        self.research_stall = 0              # informational; never terminates
        self.proposal_guard = 0              # liveness only
        self.maxit = maxit
        self.t0 = time.time()
        self.cfg = cfg

    @property
    def wall(self):
        return time.time() - self.t0

    def snapshot(self, tree, best_series):
        b = tree.best()
        return {
            "experiments_executed": self.experiments_executed,
            "max_iter": self.maxit,
            "proposal_attempts": self.proposal_attempts,
            "proposals_rejected": self.proposals_rejected,
            "wall_clock_s": round(self.wall, 1),
            "wall_clock_limit_s": self.cfg.budget.wall_clock_hours * 3600,
            "research_stall": self.research_stall,
            "proposal_guard": self.proposal_guard,
            "best_primary": None if b is None else b.score(),
            "official_eps": OFFICIAL_EPS, "official_N": OFFICIAL_N,
            "official_converged": _converged(best_series, OFFICIAL_EPS, OFFICIAL_N),
            "best_series": [round(float(x), 6) for x in best_series],
        }


# ------------------------------------------------------------------ main loop
def run(cfg, driver, run_id=None, max_iter=None, verbose=True):
    guardrails.ensure_frozen(create=True)
    datced.build_or_load(cfg.data_dir, cfg.cache_dir)

    run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = str(Path(cfg.runs_dir) / run_id)
    mem = Memory(run_dir)
    tree = SearchTree()
    ev_log = events.EventLog(run_dir, enabled=getattr(cfg, "events", True))
    rng = np.random.default_rng(cfg.seed)
    maxit = max_iter or cfg.budget.max_iter
    st = RunState(cfg, maxit)

    bundle = datced.load_bundle(cfg.cache_dir)
    uva = np.asarray(bundle.users["valid"])
    yva = np.asarray(bundle.y["valid"])
    ev = Evaluator(uva, yva)                     # per-user grouping, built once (cheap analysis)
    register_evaluator_users(ev, uva)            # rank vectors need the user column

    def log(msg):
        if verbose:
            print(msg, flush=True)

    ev_log.emit(events.RUN_START,
                f"Run {run_id}: KuaiRand-Pure within-user ranking. FM baseline {FM_VALID}. "
                f"Official convergence eps={OFFICIAL_EPS}, N={OFFICIAL_N}; hard cap {maxit} "
                f"experiments; {cfg.budget.wall_clock_hours}h backstop.",
                run_id=run_id, cache_version=datced.CACHE_VERSION,
                official_eps=OFFICIAL_EPS, official_N=OFFICIAL_N, max_iter=maxit)
    ev_log.emit(events.GUARD, "Frozen harness verified (SHA-256 of the 5 pinned files).",
                guard="frozen_boundary", ok=True)
    ev_log.emit(events.GUARD,
                "Holdout isolation: hidden-test labels and the is_click proxy are outside the "
                "block-visible cache.", guard="holdout_isolation", ok=True,
                holdout_dir=datced.holdout_dir(cfg.cache_dir))

    # ---- root: reproduce FM ----
    node_dir, blocks, rcfg = mutate.materialize_root(run_dir, BLOCK_SRC, seed=cfg.seed)
    ev_log.emit(events.TRAIN, "Root node: reproduce the official FM baseline.", node_id="root")
    res, wc = executor.run_node(blocks, node_dir, Path(node_dir) / "cfg.json",
                                cfg.cache_dir, cfg.budget.per_iter_timeout_s, extra_split="test")
    if isinstance(res, executor.Failure):
        raise SystemExit(f"root node failed: {res}")
    root = Node(id="root", parent=None, phase=0, cfg=rcfg, block_dir=blocks,
                lever="-", hypothesis="reproduce FM baseline",
                metrics={"GAUC": res["GAUC"], "nDCG@5": res["nDCG@5"],
                         "primary_valid": res["primary_valid"], "primary_unbiased": None},
                status="root")
    tree.add(root)
    mem.append(_record(0, root, diff="", events=[], cost={"wall_clock_s": wc}, signature=None))
    ev_log.emit(events.EVALUATE, f"Root reproduces FM: primary_valid={root.score():.5f} "
                                 f"(GAUC {root.gauc():.5f}, nDCG@5 {root.ndcg():.5f}).",
                node_id="root", **root.metrics)
    log(f"[root] primary_valid={root.score():.4f} ({wc:.0f}s)")

    _maybe_resume_champion(cfg, run_dir, tree, mem, ev_log, log)

    best_series = [tree.best().score()]
    hist = ledger.load(cfg.ledger_path) if getattr(cfg, "use_ledger", True) else []
    if hist:
        ev_log.emit(events.OBSERVE,
                    f"Cross-run evidence ledger loaded: {len(hist)} prior experiments.",
                    ledger_entries=len(hist))

    stop_reason = None
    while True:
        # ---- OFFICIAL stop conditions, evaluated first and never overridden -------------------
        if _converged(best_series, OFFICIAL_EPS, OFFICIAL_N):
            stop_reason = "official_convergence"
            break
        if st.experiments_executed >= maxit:
            stop_reason = "max_iter_hard_cap"
            break
        if st.wall >= cfg.budget.wall_clock_hours * 3600:
            stop_reason = "wall_clock_backstop"
            break
        # ---- liveness guard (NOT a convergence rule) -----------------------------------------
        if st.proposal_guard >= cfg.research.proposal_guard_limit:
            stop_reason = "proposal_guard"
            break

        it = st.experiments_executed + 1
        phase = phase_of(it, cfg.phases)
        try:
            _iterate(cfg, driver, run_dir, tree, mem, ev, ev_log, rng, it, phase, st,
                     best_series, hist, log)
        except Exception as e:
            log(f"[it {it}] orchestrator error: {e!r} -> continue")
            ev_log.emit(events.RECOVER, f"Orchestrator error, iteration abandoned: {e!r}")
            st.proposal_guard += 1

    ev_log.emit(events.CONVERGENCE,
                _stop_sentence(stop_reason, best_series, st, maxit, cfg),
                stop_reason=stop_reason, **st.snapshot(tree, best_series))
    log(f"[stop] reason={stop_reason} executed={st.experiments_executed} "
        f"proposals={st.proposal_attempts} best={tree.best().score():.4f}")

    final_valid = finalize(cfg, run_dir, tree, mem, ev, ev_log, st, best_series, stop_reason, log)
    ev_log.emit(events.RUN_END, f"Run complete. Final validation estimate {final_valid:.5f}.",
                final_valid=final_valid, stop_reason=stop_reason)
    return run_dir, tree.best(), final_valid


def _stop_sentence(reason, best_series, st, maxit, cfg):
    if reason == "official_convergence":
        return (f"OFFICIAL CONVERGENCE: best-so-far improved by "
                f"{best_series[-1] - best_series[-1 - OFFICIAL_N]:+.5f} over the last "
                f"{OFFICIAL_N} accepted iterations, which is <= eps={OFFICIAL_EPS}. "
                f"Stopping as the benchmark specifies, after {st.experiments_executed} executed "
                f"experiments of a permitted {maxit}.")
    if reason == "max_iter_hard_cap":
        return f"HARD CAP: {st.experiments_executed}/{maxit} executed experiments."
    if reason == "wall_clock_backstop":
        return f"WALL-CLOCK BACKSTOP: {st.wall / 3600:.2f}h of {cfg.budget.wall_clock_hours}h."
    return ("PROPOSAL GUARD: the Proposer could not produce an executable experiment "
            f"{cfg.research.proposal_guard_limit} times. This is a liveness guard, not convergence.")


def _maybe_resume_champion(cfg, run_dir, tree, mem, ev_log, log):
    champ = champion.load(cfg.champion_dir) if getattr(cfg, "resume", False) else None
    if not champ:
        return
    try:
        ccfg = Cfg.from_json(Path(cfg.champion_dir) / "cfg.json")
        cext = champion.load_ext(cfg.champion_dir)
        cdir, cblocks, _ = mutate.materialize_named(
            run_dir, "champion", str(Path(cfg.champion_dir) / "blocks"), ccfg, cext)
        cres, cwc = executor.run_node(cblocks, cdir, Path(cdir) / "cfg.json",
                                      cfg.cache_dir, cfg.budget.per_iter_timeout_s,
                                      extra_split="test")
    except Exception as e:
        cres, cwc = executor.Failure("code", repr(e)), 0.0
    if isinstance(cres, executor.Failure):
        log(f"[champion] revalidation failed ({cres.kind}) -> ignoring stale champion")
        ev_log.emit(events.GUARD, "Stale cross-run champion failed re-validation; ignored.",
                    guard="champion_revalidation", ok=False)
        return
    cnode = Node(id="champion", parent="root", phase=0, cfg=ccfg, block_dir=cblocks,
                 lever="resume", hypothesis="resumed cross-run champion", ext=cext,
                 metrics={"GAUC": cres["GAUC"], "nDCG@5": cres["nDCG@5"],
                          "primary_valid": cres["primary_valid"], "primary_unbiased": None},
                 status="improved")
    tree.add(cnode)
    mem.append(_record(0, cnode, diff="# resumed champion",
                       events=[{"class": "resume", "stored": champ.get("primary_valid"),
                                "cache_version": champ.get("cache_version")}],
                       cost={"wall_clock_s": cwc}, signature=None))
    log(f"[champion] revalidated primary_valid={cnode.score():.4f} "
        f"(stored {champ.get('primary_valid')}, cache_v {champ.get('cache_version')})")


# ------------------------------------------------------------------ one iteration
def _propose_executable(cfg, driver, run_dir, tree, mem, ev_log, rng, it, phase, st, log):
    """Obtain ONE executable experiment, re-proposing past duplicates / no-ops / invalid configs.

    Returns (parent, hyp, node_dir, blocks, ncfg, diff, ext, prov, sig, cost) or None.

    A rejected proposal never trains a model, never appends to `best_series`, and never advances
    `experiments_executed` (docs/SYSTEM.md §16 accounting). It is logged and counted as a proposal attempt so the
    claim stays auditable -- it is NOT presented as a free benchmark iteration.
    """
    feedback = None
    plateau = st.research_stall >= cfg.research.plateau_after
    explore_p = cfg.research.explore_p_escalated if plateau else cfg.phases.explore_p
    if plateau:
        ev_log.emit(events.OBSERVE,
                    f"Research plateau: {st.research_stall} consecutive experiments produced neither "
                    f"performance nor new information. Escalating proposal policy "
                    f"(exploration {cfg.phases.explore_p:.2f} -> {explore_p:.2f}, preferring a "
                    f"decorrelated parent). This does NOT postpone official convergence.",
                    research_stall=st.research_stall, explore_p=explore_p)

    cost = {"input_tokens": 0, "output_tokens": 0, "wall_clock_s": 0.0}
    last_fingerprint = None
    for attempt in range(cfg.research.max_reproposals + 1):
        parent = tree.select(explore_p, rng, prefer_diverse=plateau)
        ctx = build_proposer_context(tree, mem, phase, st.experiments_executed, st.maxit,
                                     st.wall, cfg.budget.wall_clock_hours * 3600,
                                     cost["input_tokens"] + cost["output_tokens"], plateau, feedback)
        hyp, u = proposer.propose(driver, ctx, cfg.llm.proposer, cfg.llm.temperature)
        st.proposal_attempts += 1
        cost["input_tokens"] += u.input_tokens
        cost["output_tokens"] += u.output_tokens

        # If the Proposer repeats a proposal we just rejected, it is not responding to the feedback.
        # Retrying costs real tokens for a guaranteed-identical outcome, so stop immediately.
        fp = (hyp.lever, hyp.statement, hyp.config_delta_json,
              getattr(hyp, "adopt_blockset", None), hyp.target_block)
        if fp == last_fingerprint:
            ev_log.emit(events.DECIDE,
                        "Proposer repeated a proposal that was just rejected; it is not responding "
                        "to the feedback. Abandoning this iteration rather than spending more tokens.",
                        node_id=f"n{it}", decision="abandon_iteration", reason="repeated_proposal")
            log(f"[it {it}] Proposer repeated the rejected proposal -> abandon iteration")
            break
        last_fingerprint = fp

        ev_log.emit(events.OBSERVE, hyp.problem_identified, node_id=f"n{it}", phase=phase,
                    attempt=attempt + 1)
        ev_log.emit(events.HYPOTHESIZE,
                    f"[Lever {hyp.lever}] {hyp.statement}", node_id=f"n{it}", phase=phase,
                    lever=hyp.lever, rationale=hyp.rationale,
                    expected_metric=hyp.expected_metric, expected_gain=hyp.expected_gain,
                    mutation_kind=hyp.mutation_kind, attempt=attempt + 1)

        adopt = getattr(hyp, "adopt_blockset", None)
        is_block = hyp.mutation_kind == "block" and hyp.target_block and not adopt
        parent_ext = dict(parent.ext or {})
        val, bs, intended = mutate.validate_hypothesis(parent, hyp, parent_ext)
        ev_log.emit(events.PLAN,
                    f"Intended intervention on block set '{bs}': "
                    f"{intended or ('adopt ' + adopt if adopt else 'block edit')}.",
                    node_id=f"n{it}", blockset=bs, intended_delta=intended,
                    adopt_blockset=adopt, target_block=hyp.target_block)

        # ---- structural validation, BEFORE any training ---------------------------------------
        if not intended and not is_block and not adopt:
            feedback = "It changed nothing at all (no config delta, no block edit, no adoption)."
            _reject(st, mem, ev_log, it, phase, parent, hyp, feedback, "empty_proposal", cost, log)
            continue
        if val.invalid or val.not_honoured:
            ev_log.emit(events.GUARD,
                        f"Config validation rejected {sorted(set(val.invalid) | set(val.not_honoured))}: "
                        + " ".join(val.reasons[:2]),
                        node_id=f"n{it}", guard="effective_config", ok=False,
                        invalid=val.invalid, not_honoured=val.not_honoured)
        if not val.has_effect and not is_block and not adopt:
            feedback = val.feedback(bs)
            _reject(st, mem, ev_log, it, phase, parent, hyp, feedback, STRUCTURAL_NOOP, cost, log)
            continue

        block_edit = None
        if is_block:
            be, u2 = coder.code(driver, build_coder_context(parent, hyp),
                                cfg.llm.coder, cfg.llm.temperature)
            cost["input_tokens"] += u2.input_tokens
            cost["output_tokens"] += u2.output_tokens
            if not getattr(be, "implementable", True):
                # 1D honest rejection: a capability the harness genuinely lacks. Research information
                # (it constrains the search space), not a failure and not a scientific result.
                ev_log.emit(events.GUARD,
                            f"Coder declined honestly: {be.reason}", node_id=f"n{it}",
                            guard="honest_rejection", ok=True, capability=hyp.statement)
                node = Node(id=f"n{it}", parent=parent.id, phase=phase, cfg=parent.cfg,
                            block_dir=parent.block_dir, lever=hyp.lever, hypothesis=hyp.statement,
                            problem=hyp.problem_identified, metrics=None, status="abandoned")
                mem.append(_record(it, node, diff="",
                                   events=[{"class": "not_implementable", "detail": be.reason}],
                                   cost=cost, signature=None))
                st.proposals_rejected += 1
                feedback = (f"That capability is not available: {be.reason} "
                            f"Propose something the harness can actually execute.")
                log(f"[it {it}] {hyp.lever} not implementable ({be.reason}) -> re-propose")
                continue
            ok, msg = executor.check_imports(be.new_source)
            ev_log.emit(events.GUARD,
                        ("Block source passed the import allowlist and holdout access guard."
                         if ok else f"Block source REJECTED: {msg}"),
                        node_id=f"n{it}", guard="check_imports", ok=ok, detail="" if ok else msg)
            if not ok:
                feedback = f"The generated code was rejected by the static guard: {msg}"
                _reject(st, mem, ev_log, it, phase, parent, hyp, feedback, "code", cost, log)
                continue
            block_edit = be
            ev_log.emit(events.CODE,
                        f"Rewrote block '{be.target_block}.py' ({len(be.new_source)} chars).",
                        node_id=f"n{it}", target_block=be.target_block,
                        imports_used=be.imports_used)

        node_dir, blocks, ncfg, diff, ext, prov = mutate.materialize_child(
            run_dir, f"n{it}", parent, hyp, block_edit, validation=val,
            parent_ext=parent_ext, cache_version=datced.CACHE_VERSION)
        if adopt:
            ev_log.emit(events.CODE, f"Adopted the '{adopt}' block set wholesale (model family "
                                     f"changes to '{adopt}').", node_id=f"n{it}", adopt_blockset=adopt)

        sig = mutate.signature(ncfg, blocks, ext)
        if mem.seen(sig):
            # CONTENT-based identity (docs/SYSTEM.md §12): catches a re-run reached from a different parent, which
            # the old cfg+diff signature missed.
            ev_log.emit(events.GUARD,
                        "Deduplication: this node's content (config + extension + all six block "
                        "sources) is identical to an experiment already run.",
                        node_id=f"n{it}", guard="dedup", ok=False, signature=sig)
            node = Node(id=f"n{it}", parent=parent.id, phase=phase, cfg=ncfg, block_dir=blocks,
                        lever=hyp.lever, hypothesis=hyp.statement, metrics=None,
                        status="duplicate", noop_class=STRUCTURAL_NOOP)
            mem.append(_record(it, node, diff=diff, events=[{"class": "duplicate"}],
                               cost=cost, signature=sig))
            st.proposals_rejected += 1
            feedback = ("That experiment is byte-identical to one already run "
                        "(same config and same block sources).")
            log(f"[it {it}] duplicate of a tried node -> re-propose")
            continue
        mem.note_seen(sig)

        if not prov.intervention_matched:
            parts = [f"effective={prov.effective_delta or '{}'}"]
            if prov.rejected_ineffective:
                parts.append(f"already-at-that-value={sorted(prov.rejected_ineffective)}")
            if prov.rejected_not_honoured:
                parts.append(f"not-honoured-by-{prov.blockset}="
                             f"{sorted(prov.rejected_not_honoured)}")
            if prov.rejected_invalid:
                parts.append(f"invalid={sorted(prov.rejected_invalid)}")
            ev_log.emit(events.GUARD,
                        "Executed intervention is NARROWER than proposed: " + "; ".join(parts) + ".",
                        node_id=f"n{it}", guard="provenance", ok=True, **prov.to_dict())
        return parent, hyp, node_dir, blocks, ncfg, diff, ext, prov, sig, cost

    st.proposal_guard += 1
    log(f"[it {it}] no executable proposal after {cfg.research.max_reproposals + 1} attempts")
    return None


def _reject(st, mem, ev_log, it, phase, parent, hyp, feedback, klass, cost, log):
    st.proposals_rejected += 1
    node = Node(id=f"n{it}", parent=parent.id, phase=phase, cfg=parent.cfg,
                block_dir=parent.block_dir, lever=hyp.lever, hypothesis=hyp.statement,
                problem=hyp.problem_identified, metrics=None, status="rejected_proposal",
                noop_class=klass if klass == STRUCTURAL_NOOP else None)
    mem.append(_record(it, node, diff="", events=[{"class": klass, "detail": feedback}],
                       cost=cost, signature=None))
    ev_log.emit(events.DECIDE, f"Proposal not executed ({klass}). Re-proposing. {feedback[:200]}",
                node_id=f"n{it}", decision="re-propose", reason=klass)
    log(f"[it {it}] proposal rejected ({klass}) -> re-propose")


def _iterate(cfg, driver, run_dir, tree, mem, ev, ev_log, rng, it, phase, st,
             best_series, hist, log):
    got = _propose_executable(cfg, driver, run_dir, tree, mem, ev_log, rng, it, phase, st, log)
    if got is None:
        return
    parent, hyp, node_dir, blocks, ncfg, diff, ext, prov, sig, cost = got

    node_events = []
    res = None
    wc = 0.0
    # F5 debug-first gate: cheap crash/sanity check on a subsample before the full run (torch nodes).
    if cfg.debug_gate and ncfg.model_type in ("din", "bst"):
        dbg = executor.debug_gate(blocks, ncfg, cfg.cache_dir, str(Path(node_dir) / "_dbg"),
                                  n_train=cfg.debug_train_n, n_other=cfg.debug_other_n,
                                  epochs=cfg.debug_epochs)
        if isinstance(dbg, executor.Failure):
            ev_log.emit(events.DEBUG, f"Debug gate FAILED ({dbg.kind}) before the full run.",
                        node_id=f"n{it}", ok=False, kind=dbg.kind)
            log(f"[it {it}] {hyp.lever} debug gate FAILED ({dbg.kind}) -> recovery")
            res, wc, node_events = _recover(cfg, driver, dbg, node_dir, blocks, ncfg,
                                            None, ev_log, it, log)
        else:
            ev_log.emit(events.DEBUG,
                        f"Debug gate passed on a subsample (primary~{dbg['primary_valid']:.3f}); "
                        f"proceeding to the full run.", node_id=f"n{it}", ok=True)
            log(f"[it {it}] {hyp.lever} debug gate ok (sample primary~{dbg['primary_valid']:.3f})")

    if res is None:
        ev_log.emit(events.TRAIN, f"Training {ncfg.model_type} node n{it} "
                                  f"(loss={ncfg.loss_type}, aux={list(ncfg.aux_tasks)}).",
                    node_id=f"n{it}", model_type=ncfg.model_type, loss_type=ncfg.loss_type)
        # `extra_split="test"` on EVERY node: the submitted predictions must come from the same
        # trained instance whose validation predictions drove selection (docs/SYSTEM.md §18).
        res, wc = executor.run_node(blocks, node_dir, Path(node_dir) / "cfg.json",
                                    cfg.cache_dir, cfg.budget.per_iter_timeout_s,
                                    extra_split="test")
        if isinstance(res, executor.Failure):
            res, wc2, node_events = _recover(cfg, driver, res, node_dir, blocks, ncfg,
                                             None, ev_log, it, log)
            wc += wc2
    cost["wall_clock_s"] = round(wc, 1)

    if isinstance(res, executor.Failure):
        node = Node(id=f"n{it}", parent=parent.id, phase=phase, cfg=ncfg, block_dir=blocks,
                    lever=hyp.lever, hypothesis=hyp.statement, problem=hyp.problem_identified,
                    metrics=None, status="abandoned", provenance=prov.to_dict(), ext=ext)
        tree.add(node)
        mem.append(_record(it, node, diff=diff, events=node_events, cost=cost, signature=sig))
        ev_log.emit(events.DECIDE, f"Experiment abandoned after failure ({res.kind}).",
                    node_id=f"n{it}", decision="abandoned")
        st.research_stall += 1
        log(f"[it {it}] {hyp.lever} failed ({res.kind}) -> abandoned")
        return

    # ---------------------------------------------------------------- executed experiment
    st.experiments_executed += 1
    pv = res["primary_valid"]
    node = Node(id=f"n{it}", parent=parent.id, phase=phase, cfg=ncfg, block_dir=blocks,
                lever=hyp.lever, hypothesis=hyp.statement, problem=hyp.problem_identified,
                metrics={"GAUC": res["GAUC"], "nDCG@5": res["nDCG@5"],
                         "primary_valid": pv, "primary_unbiased": res.get("primary_unbiased")},
                status="no_gain", provenance=prov.to_dict(), ext=ext)
    ev_log.emit(events.EVALUATE,
                f"n{it} primary_valid={pv:.5f} (GAUC {res['GAUC']:.5f}, "
                f"nDCG@5 {res['nDCG@5']:.5f}) in {wc:.0f}s.",
                node_id=f"n{it}", **node.metrics, wall_clock_s=round(wc, 1))

    # ---- integrity tripwire (docs/SYSTEM.md §8) --------------------------------------------------------
    if pv > cfg.research.leak_tripwire_primary:
        node.status = "quarantined"
        node.evidence = {"class": "rejected", "reason": "leak_tripwire"}
        tree.add(node)
        mem.append(_record(it, node, diff=diff,
                           events=node_events + [{"class": "leak_tripwire", "primary_valid": pv}],
                           cost=cost, signature=sig))
        ev_log.emit(events.GUARD,
                    f"LEAKAGE TRIPWIRE: primary_valid={pv:.4f} exceeds "
                    f"{cfg.research.leak_tripwire_primary}. Real progress here is measured in "
                    f"thousandths; a jump of this size is a bug or a label leak, never a model. "
                    f"Node quarantined and excluded from the portfolio.",
                    node_id=f"n{it}", guard="leak_tripwire", ok=False, primary_valid=pv)
        log(f"[it {it}] LEAK TRIPWIRE primary={pv:.4f} -> quarantined")
        best_series.append(tree.best().score())
        return

    # ---- post-hoc no-op classification (§ P0.4) -------------------------------------------
    sv = _val_scores(node)
    pv_parent = _val_scores(parent)
    node.noop_class = _classify_noop(sv, pv_parent, ncfg.model_type)
    if node.noop_class in (EXACT_NOOP,):
        ev_log.emit(events.GUARD,
                    "EXACT_NOOP: predictions are bit-identical to the parent's, so the intervention "
                    "never reached execution. Recorded, but NOT as scientific evidence.",
                    node_id=f"n{it}", guard="noop_detection", ok=False, noop_class=node.noop_class)
    elif node.noop_class == NEAR_NOOP:
        ev_log.emit(events.OBSERVE,
                    "NEAR_NOOP: predictions differ only marginally. This IS legitimate evidence of a "
                    "negligible effect (not a no-op).",
                    node_id=f"n{it}", noop_class=node.noop_class)

    # ---- statistical evidence vs. the control (docs/SYSTEM.md §13) --------------------------------------
    node.evidence = _evidence(cfg, ev, node, parent, tree.best(), sv, pv_parent, ev_log, it)

    # ---- portfolio valuation (docs/SYSTEM.md §15) -- cheap analysis on saved predictions -----------------
    _update_portfolio(cfg, ev, tree, node, ev_log, it)

    # ---- Lever E second surface (docs/RESEARCH.md §15) ---------------------------------------------------
    _rand_surface(cfg, node, node_dir, ev_log, it, log)

    # ---- adoption (tree shape only -- NOT convergence) ------------------------------------
    prev_best = tree.best().score()
    node.status = "improved" if pv > prev_best + cfg.budget.adopt_eps else "no_gain"
    if (cfg.recheck and node.status == "improved"
            and blockspec.is_stochastic(ncfg.model_type)):
        # Multi-seed RE-TRAINING is reserved for stochastic families: fm/lgbm have training std
        # 0.00000 at a fixed seed, so re-seeding them measures nothing the free paired bootstrap
        # does not already measure (docs/RESEARCH.md §4).
        ok, mean, seeds = reeval.confirm(blocks, ncfg, pv, prev_best, cfg.cache_dir,
                                         cfg.budget.per_iter_timeout_s,
                                         str(Path(node_dir) / "reeval"),
                                         cfg.recheck_seeds, cfg.budget.adopt_eps)
        node.metrics["primary_valid_seedmean"] = round(mean, 5)
        node.metrics["training_seed_scores"] = [round(s, 5) for s in seeds]
        if not ok:
            node.status = "no_gain"
        ev_log.emit(events.COMPARE,
                    f"Training-variance check on n{it} ({ncfg.model_type} is stochastic): "
                    f"seeds={[round(s, 4) for s in seeds]} mean={mean:.5f} -> "
                    f"{'kept' if ok else 'reverted'}. This is TRAINING stochasticity, a different "
                    f"quantity from the validation-sample uncertainty reported above.",
                    node_id=f"n{it}", seed_mean=mean, seeds=seeds, kept=ok)

    # ---- research information (docs/SYSTEM.md §16) ------------------------------------------------------
    node.informative, why = _is_informative(cfg, tree, node, hist)
    st.research_stall = 0 if (node.informative or node.status == "improved") else st.research_stall + 1

    tree.add(node)
    mem.append(_record(it, node, diff=diff, events=node_events, cost=cost, signature=sig))
    best_series.append(tree.best().score())

    ev_log.emit(events.DECIDE,
                f"n{it}: status={node.status}, evidence={node.evidence.get('class')}, "
                f"informative={node.informative} ({why}).",
                node_id=f"n{it}", decision=node.status, evidence=node.evidence.get("class"),
                informative=node.informative, reason=why,
                **st.snapshot(tree, best_series))
    ev_log.emit(events.CONVERGENCE,
                f"Official window (eps={OFFICIAL_EPS}, N={OFFICIAL_N}): "
                f"{'SATISFIED' if _converged(best_series, OFFICIAL_EPS, OFFICIAL_N) else 'not yet satisfied'}"
                f" after {st.experiments_executed} executed experiments.",
                **st.snapshot(tree, best_series))
    d = pv - parent.score()
    log(f"[it {it}] {hyp.lever} {node.status}/{node.evidence.get('class')} primary={pv:.4f} "
        f"(d{d:+.4f} vs parent) [{hyp.statement[:44]}] {wc:.0f}s")


# ------------------------------------------------------------------ analysis helpers
def _classify_noop(sv, parent_sv, model_type):
    """STRUCTURAL_NOOP is decided before execution; this classifies what the RUN produced.

    Deliberately does NOT equate 'rank correlation > 0.9999' with a structural no-op (a real
    distinction the task brief calls out): two trainings of an identical DIN config correlate 0.926,
    so near-identity in a stochastic family is informative, not proof of a missing intervention.
    """
    if sv is None or parent_sv is None or len(sv) != len(parent_sv):
        return None
    if np.array_equal(sv, parent_sv):
        return EXACT_NOOP
    from agent.stats import rank_corr
    if rank_corr(sv, parent_sv) > 0.9999:
        return NEAR_NOOP
    return None


def _evidence(cfg, ev, node, parent, champ, sv, parent_sv, ev_log, it):
    """Paired user-level bootstrap of this node against its control. Cheap: no retraining."""
    if sv is None or parent_sv is None:
        return {"class": "inconclusive", "reason": "missing predictions"}
    st_t = ev.user_stats(sv)
    st_c = ev.user_stats(parent_sv)
    r = paired_bootstrap(ev, st_t, st_c, B=cfg.research.bootstrap_B, seed=0)
    klass = classify_evidence(r["p_gt0"], hi=cfg.research.adopt_p,
                              lo_band=cfg.research.promising_p)
    if node.noop_class in (EXACT_NOOP,):
        klass = "no_effect"
    out = dict(r)
    out["class"] = klass
    out["control_id"] = parent.id
    ev_log.emit(events.COMPARE,
                f"n{it} vs {parent.id}: primary {r['delta_primary']:+.5f} "
                f"(GAUC {r['delta_GAUC']:+.5f}, nDCG@5 {r['delta_nDCG']:+.5f}), "
                f"paired-bootstrap SE {r['boot_se']:.5f}, P(delta>0)={r['p_gt0']:.2f} "
                f"-> {klass}. This is VALIDATION-SAMPLE uncertainty; it is not reduced by re-training.",
                node_id=f"n{it}", control=parent.id, **{k: v for k, v in r.items() if k != "B"},
                evidence_class=klass)
    return out


def _update_portfolio(cfg, ev, tree, node, ev_log, it):
    """Recompute portfolio statistics over the current viable set. Never trains (docs/SYSTEM.md §15)."""
    users = ev_users_of(ev)
    if users is None:
        return
    nodes = [n for n in tree._viable()] + [node]
    members = portfolio.build_members(nodes, users, _val_scores)
    if len(members) < 2:
        return
    champ = max(nodes, key=lambda n: n.score())
    vals = portfolio.valuation(ev, members, champion_id=champ.id)
    for n in nodes:
        if n.id in vals:
            n.portfolio = vals[n.id]
    if node.id in vals:
        v = vals[node.id]
        ev_log.emit(events.ENSEMBLE,
                    f"Portfolio value of n{it}: standalone {v['standalone_primary']:.5f}, "
                    f"rank_corr to champion {v['rank_corr_to_best']:.3f}, "
                    f"pair-blend gain {v['pair_blend_gain']:+.5f}, "
                    f"marginal contribution {v['emc']:+.5f}.",
                    node_id=f"n{it}", **v)


_EV_USERS = {}


def ev_users_of(ev):
    """The valid user vector the Evaluator was built on (cached; used to build rank vectors)."""
    return _EV_USERS.get(id(ev))


def _rand_surface(cfg, node, node_dir, ev_log, it, log):
    """Lever E: report the random-exposure primary as a SECOND surface, never as the target."""
    if not getattr(cfg, "unbiased_eval", False):
        return
    rp = Path(node_dir) / "rand_scores.npy"
    if not rp.exists():
        return
    try:
        from pipeline.lib import rand_build
        ru, ry = rand_build.load_rand(cfg.cache_dir)
        pr = round(float(evaluate(ru, ry, np.load(rp))["primary"]), 5)
    except Exception as e:
        log(f"[it {it}] rand-surface evaluation skipped: {e!r}")
        return
    node.metrics["primary_rand"] = pr
    gap = node.score() - pr
    ev_log.emit(events.EVALUATE,
                f"Random-exposure surface for n{it}: primary_rand={pr:.5f} vs "
                f"primary_valid={node.score():.5f} (gap {gap:+.5f}). Reported as a robustness "
                f"surface; the competition target remains primary_valid.",
                node_id=f"n{it}", primary_rand=pr, gap=gap)
    log(f"[it {it}] unbiased(rand) primary={pr:.4f} vs valid {node.score():.4f} (gap {gap:+.4f})")


def _is_informative(cfg, tree, node, hist):
    """Did this experiment yield RESEARCH INFORMATION, independent of performance? (docs/SYSTEM.md §16)

    Affects proposal policy and memory only. It can never postpone official convergence.
    """
    r = cfg.research
    if node.noop_class in (EXACT_NOOP, STRUCTURAL_NOOP):
        return False, "no intervention reached execution"
    klass = (node.evidence or {}).get("class")
    if klass == "confirmed":
        return True, "statistically supported positive evidence"
    if klass == "rejected":
        return True, "statistically supported negative evidence"
    fams = {n.cfg.model_type for n in tree._viable()}
    if node.cfg.model_type not in fams:
        return True, f"first evaluation of the '{node.cfg.model_type}' model family"
    levers = {n.lever for n in tree._viable()}
    if node.lever not in levers:
        return True, f"first evaluation of lever {node.lever}"
    emc = (node.portfolio or {}).get("emc")
    if emc is not None and emc > r.ens_eps:
        return True, f"positive ensemble marginal contribution ({emc:+.5f})"
    rc = (node.portfolio or {}).get("rank_corr_to_best")
    if rc is not None and rc < r.diversity_corr:
        return True, f"new ranking diversity (rank_corr {rc:.3f})"
    return False, "neither performance nor new information"


def _recover(cfg, driver, failure, node_dir, blocks, ncfg, block_edit, ev_log, it, log):
    """One bounded recovery attempt."""
    node_events = [{"class": failure.kind, "detail": failure.detail[:400]}]
    ev_log.emit(events.REFLECT, f"Failure classified as '{failure.kind}'. Diagnosing.",
                node_id=f"n{it}", kind=failure.kind, detail=failure.detail[-600:])
    ctx = (f"Failure class: {failure.kind}\nDetail (tail):\n{failure.detail[-1200:]}\n"
           f"cfg.loss_type={ncfg.loss_type} model={ncfg.model_type}")
    try:
        act, _ = reflector.reflect(driver, ctx, cfg.llm.reflector, cfg.llm.temperature)
    except Exception as e:
        node_events.append({"recovery": "reflector_unavailable", "detail": str(e)})
        return failure, 0.0, node_events
    node_events.append({"recovery": act.action, "explanation": act.explanation[:200]})
    ev_log.emit(events.RECOVER, f"Recovery action '{act.action}': {act.explanation[:180]}",
                node_id=f"n{it}", action=act.action)
    if act.action == "patch_retry" and act.new_source and act.patch_block:
        ok, msg = executor.check_imports(act.new_source)
        ev_log.emit(events.GUARD, f"Recovery patch {'accepted' if ok else 'rejected: ' + msg}",
                    node_id=f"n{it}", guard="check_imports", ok=ok)
        if ok:
            (Path(blocks) / f"{act.patch_block}.py").write_text(act.new_source, encoding="utf-8")
            res, wc = executor.run_node(blocks, node_dir, Path(node_dir) / "cfg.json",
                                        cfg.cache_dir, cfg.budget.per_iter_timeout_s,
                                        extra_split="test")
            return res, wc, node_events
    if act.action == "degrade":
        ncfg2 = mutate.apply_delta(ncfg, act.config_delta_json)
        ncfg2.to_json(Path(node_dir) / "cfg.json")
        res, wc = executor.run_node(blocks, node_dir, Path(node_dir) / "cfg.json",
                                    cfg.cache_dir, cfg.budget.per_iter_timeout_s,
                                    extra_split="test")
        return res, wc, node_events
    return failure, 0.0, node_events


# ------------------------------------------------------------------ records
def _record(it, node, diff, events, cost, signature):
    from dataclasses import asdict
    return {
        "iter": it, "phase": node.phase, "node_id": node.id, "parent_id": node.parent,
        "lever": node.lever, "hypothesis": node.hypothesis,
        "problem_identified": getattr(node, "problem", ""),
        "config": asdict(node.cfg), "cfg_ext": dict(node.ext or {}), "code_diff": diff,
        "metrics": node.metrics, "status": node.status,
        "evidence": node.evidence or {}, "portfolio": node.portfolio or {},
        "provenance": node.provenance or {}, "noop_class": node.noop_class,
        "informative": bool(node.informative),
        "events": events, "cost": cost, "signature": signature,
    }


def _per_user_rank(scores, users):
    """Within-user percentile rank in [0,1] -- monotone, scale-free, ideal for blending."""
    return per_user_rank(scores, users)


def _val_scores(node):
    p = Path(node.block_dir).parent / "val_scores.npy"
    return np.load(p) if p.exists() else None


def _test_scores(node):
    """Test predictions written by the SAME training pass that produced val_scores (docs/SYSTEM.md §18)."""
    p = Path(node.block_dir).parent / "test_scores.npy"
    return np.load(p) if p.exists() else None


def _ablation_summary(tree):
    out = {}
    for n in tree.nodes.values():
        m = n.metrics or {}
        if m.get("primary_valid") is not None and n.status != "quarantined":
            out[n.lever] = max(out.get(n.lever, 0.0), m["primary_valid"])
    return {lv: round(p, 5) for lv, p in sorted(out.items())}


# ------------------------------------------------------------------ finalize
def finalize(cfg, run_dir, tree, mem, ev, ev_log, st, best_series, stop_reason, log):
    best_dir = Path(run_dir) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    bundle = datced.load_bundle(cfg.cache_dir)
    uva = np.asarray(bundle.users["valid"])
    ute = np.asarray(bundle.users["test"])
    _EV_USERS[id(ev)] = uva

    viable = [n for n in tree._viable()]
    members = portfolio.build_members(viable, uva, _val_scores)
    ev_log.emit(events.ENSEMBLE,
                f"Portfolio candidates after de-duplication: "
                f"{[(m.node_id, m.model_type, round(m.primary, 5)) for m in members]}",
                candidates=[m.node_id for m in members])

    asm = None
    if len(members) >= 2:
        vals = portfolio.valuation(ev, members)
        for n in viable:
            if n.id in vals:
                n.portfolio = vals[n.id]
        ev_log.emit(events.ENSEMBLE,
                    "Marginal contributions (leave-one-out on the full-pool blend): "
                    + ", ".join(f"{k} EMC {v['emc']:+.5f} / corr {v['rank_corr_to_best']:.3f}"
                                for k, v in vals.items()),
                    valuation=vals)
        try:
            asm = portfolio.assemble_cv(ev, members, K=cfg.research.cv_folds, seed=cfg.seed,
                                        max_members=cfg.research.max_members,
                                        step=cfg.research.weight_step)
        except Exception as e:
            log(f"[assemble] errored, falling back to single best: {e!r}")
            asm = None

    best = tree.best()
    final_valid = float(best.score())
    honest = {"kind": "single_best", "estimate": final_valid, "se": None}
    ensemble_info = None
    final_test = None

    if asm and asm.get("ok"):
        ensemble_info = asm
        parts = {}
        for nid in asm["members"]:
            ts = _test_scores(tree.nodes[nid])
            parts[nid] = ts
        missing = [k for k, v in parts.items() if v is None]
        if missing:
            log(f"[finalize] missing test predictions for {missing}; falling back to single best")
            ev_log.emit(events.ENSEMBLE,
                        f"Portfolio rejected: test predictions missing for {missing}.", ok=False)
            ensemble_info = None
        else:
            R = [per_user_rank(parts[nid], ute) for nid in asm["members"]]
            final_test = sum(w * r for w, r in zip(asm["weights"], R))
            final_valid = asm["valid_primary_tuned"]
            honest = {"kind": "portfolio_cv", "estimate": asm["cv_mean"], "se": asm["cv_se"]}
            ev_log.emit(events.ENSEMBLE,
                        f"Portfolio selected: {asm['members']} "
                        f"({'+'.join(asm['member_families'])}) weights {asm['weights']}. "
                        f"Tuned-on-all-valid primary {asm['valid_primary_tuned']:.5f} "
                        f"(in-sample, optimistic). HONEST {asm['K']}-fold CV estimate "
                        f"{asm['cv_mean']:.5f} +/- {asm['cv_se']:.5f}, gain over the best single "
                        f"member {asm['cv_gain_over_best_single']:+.5f}. Test predictions come from "
                        f"the same trained instances used for selection.",
                        **asm)
            log(f"[finalize] ENSEMBLE {asm['members']} w={asm['weights']} "
                f"tuned={asm['valid_primary_tuned']:.5f} cv={asm['cv_mean']:.5f}+/-{asm['cv_se']:.5f}")

    if final_test is None:
        final_test = _test_scores(best)
        ev_log.emit(events.ENSEMBLE,
                    f"Using the best single node {best.id} ({best.score():.5f}); "
                    f"no portfolio beat it or candidates were insufficient.", ok=False)
        log(f"[finalize] using best single node {best.id} ({best.score():.4f})")

    # Training-variance report for a STOCHASTIC finalist. This is reporting only: the submitted
    # predictions stay those of the instance that was actually selected (docs/SYSTEM.md §18).
    training_variance = None
    if cfg.recheck and blockspec.is_stochastic(best.cfg.model_type):
        _, mean, seeds = reeval.confirm(best.block_dir, best.cfg, best.score(), -1.0,
                                        cfg.cache_dir, cfg.budget.per_iter_timeout_s,
                                        str(Path(run_dir) / "reeval" / best.id),
                                        cfg.recheck_seeds, cfg.budget.eps)
        training_variance = {"node": best.id, "selected_instance": round(best.score(), 5),
                             "retrain_mean": round(mean, 5),
                             "retrain_scores": [round(s, 5) for s in seeds]}
        ev_log.emit(events.COMPARE,
                    f"Training stochasticity of the finalist {best.id}: retrains give "
                    f"{[round(s, 4) for s in seeds]} (mean {mean:.5f}) vs the selected instance "
                    f"{best.score():.5f}. Reported, not substituted -- the submission must come from "
                    f"the instance whose validation predictions drove selection.",
                    **training_variance)

    sub_ok = _write_and_check(cfg, best_dir, final_test, ev_log, log)

    totals = mem.resource_totals()
    report = {
        "stop_reason": stop_reason,
        "benchmark": {
            "official_convergence_rule": {"epsilon": OFFICIAL_EPS, "N": OFFICIAL_N,
                                          "source": "baseline_scores.json"},
            "official_converged": _converged(best_series, OFFICIAL_EPS, OFFICIAL_N),
            "experiments_executed": st.experiments_executed,
            "max_iter_hard_cap": st.maxit,
            "proposal_attempts": st.proposal_attempts,
            "proposals_rejected": st.proposals_rejected,
            "wall_clock_s": round(st.wall, 1),
            "wall_clock_backstop_s": cfg.budget.wall_clock_hours * 3600,
            "best_series": [round(float(x), 6) for x in best_series],
            "research_stall_at_stop": st.research_stall,
            "proposal_guard_at_stop": st.proposal_guard,
        },
        "best_single_node": best.id,
        "best_single_valid": best.score(),
        "final_valid_tuned": final_valid,
        "final_valid_honest": honest,
        "delta_over_fm_tuned": round(final_valid - FM_VALID, 5),
        "delta_over_fm_honest": (round(honest["estimate"] - FM_VALID, 5)
                                 if honest["estimate"] is not None else None),
        "ensemble": ensemble_info,
        "portfolio_valuation": {n.id: n.portfolio for n in viable if n.portfolio},
        "training_variance": training_variance,
        "submission_valid": sub_ok,
        "ablation_best_by_lever": _ablation_summary(tree),
        "uncertainty_note": (
            "final_valid_tuned is measured on the same users used to tune the blend weights and is "
            "OPTIMISTIC (measured optimism ~+0.0007, docs/RESEARCH.md §12 (ensemble optimism)). final_valid_honest is a "
            "user-level K-fold cross-validation of the whole assembly procedure. 'training_variance' "
            "is a different quantity again: it measures re-training stochasticity, not "
            "validation-sample uncertainty."),
        "resource_totals": totals,
    }
    (Path(run_dir) / "resource_report.json").write_text(json.dumps(report, indent=2, default=float))
    _write_results_md(run_dir, report)

    if getattr(cfg, "use_ledger", True):
        try:
            ledger.append_run(cfg.ledger_path, run_dir, tree, datced.CACHE_VERSION)
        except Exception as e:
            log(f"[ledger] append skipped: {e!r}")

    if getattr(cfg, "resume", False):
        prev = champion.load(cfg.champion_dir)
        if prev is None or best.score() > prev["primary_valid"]:
            from agent import provenance as _prov
            champion.save(cfg.champion_dir, best.block_dir, best.cfg, best.score(),
                          datced.CACHE_VERSION, Path(run_dir).name, best.id,
                          ext=best.ext, code_state=_prov.code_state())
            log(f"[champion] saved {best.id} valid={best.score():.4f}")

    ev_log.emit(events.FINALIZE,
                f"Final: tuned {final_valid:.5f}; honest {honest['kind']} estimate "
                f"{honest['estimate']:.5f}"
                + (f" +/- {honest['se']:.5f}" if honest.get("se") else "")
                + f"; submission {'PASSED' if sub_ok else 'FAILED'} submit.py --check. "
                  f"Stopped because: {stop_reason}.",
                **report["benchmark"], final_valid_tuned=final_valid, honest=honest,
                submission_valid=sub_ok)
    log(f"[finalize] tuned {final_valid:.4f} | honest {honest['estimate']:.4f} "
        f"(d{honest['estimate'] - FM_VALID:+.4f} vs FM) | tokens="
        f"{totals['input_tokens'] + totals['output_tokens']} wall={totals['wall_clock_s']}s")
    return final_valid


def _write_and_check(cfg, best_dir, final_test, ev_log, log):
    try:
        if final_test is None:
            log("[finalize] no test scores to submit")
            return False
        sub = Path(best_dir) / "submission_test.csv"
        _write_submission(cfg.cache_dir, np.asarray(final_test), sub)
        chk = subprocess.run([sys.executable, "submit.py", "--check", "--split", "test", str(sub)],
                             capture_output=True, text=True,
                             encoding="utf-8", errors="replace", env=executor.utf8_env())
        ok = chk.returncode == 0
        ev_log.emit(events.GUARD,
                    f"Submission {'PASSED' if ok else 'FAILED'} the frozen submit.py --check "
                    f"(format + row alignment).", guard="submission_check", ok=ok)
        log(f"[finalize] submission {'PASSED' if ok else 'FAILED'} submit.py --check")
        if not ok:
            log("   " + (chk.stderr or chk.stdout)[-400:])
        return ok
    except Exception as e:
        log(f"[finalize] submission step errored (run still finalizes): {e!r}")
        return False


def _write_submission(cache_dir, scores, out_csv):
    uid = np.load(Path(cache_dir) / "test_u.npy")
    vid = np.load(Path(cache_dir) / "test_vid.npy")
    s = np.asarray(scores)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("row_id,user_id,video_id,score\n")
        for i in range(len(s)):
            f.write(f"{i},{int(uid[i])},{int(vid[i])},{float(s[i])}\n")


def _write_results_md(run_dir, rep):
    b = rep["benchmark"]
    h = rep["final_valid_honest"]
    ens = rep.get("ensemble")
    kind = (f"portfolio {ens['members']} (w={ens['weights']})" if ens else
            f"single best {rep['best_single_node']}")
    se = f" ± {h['se']:.5f}" if h.get("se") else ""
    md = [
        "# Results\n",
        f"**Honest estimate: {h['estimate']:.5f}{se}** ({h['kind']}, {kind})\n",
        f"Tuned/in-sample validation primary: {rep['final_valid_tuned']:.5f} "
        f"(optimistic — the blend weights were tuned on these same users)\n",
        "| | FM baseline | agent (honest) | agent (tuned) |",
        "|---|---|---|---|",
        f"| primary | {FM_VALID} | {h['estimate']:.5f} | {rep['final_valid_tuned']:.5f} |",
        f"| delta | — | {rep['delta_over_fm_honest']:+} | {rep['delta_over_fm_tuned']:+} |\n",
        "## Benchmark contract\n",
        f"- Stop reason: **{rep['stop_reason']}**",
        f"- Official convergence rule: eps={b['official_convergence_rule']['epsilon']}, "
        f"N={b['official_convergence_rule']['N']} (source: baseline_scores.json) — "
        f"satisfied: **{b['official_converged']}**",
        f"- Executed experiments: {b['experiments_executed']} / {b['max_iter_hard_cap']} (hard cap)",
        f"- Proposal attempts: {b['proposal_attempts']} ({b['proposals_rejected']} rejected before "
        f"training — duplicates, structural no-ops, invalid configs, unsupported capabilities)",
        f"- Wall-clock: {b['wall_clock_s']:.0f}s / {b['wall_clock_backstop_s']:.0f}s backstop\n",
        f"Resource usage: {rep['resource_totals']}\n",
    ]
    (Path(run_dir) / "results.md").write_text("\n".join(md), encoding="utf-8")


# `_update_portfolio` needs the valid user vector; register it when the Evaluator is created.
def register_evaluator_users(ev, users):
    _EV_USERS[id(ev)] = users
