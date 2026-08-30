"""Orchestrator -- hypothesis-driven best-first tree search.

The control loop is deterministic policy; the LLM roles are the operators. It enforces the
budget, convergence, and the best-checkpoint invariant, and recovers from node failures.
"""
from __future__ import annotations
import time, json, subprocess, sys
from pathlib import Path
import numpy as np

from agent import guardrails, datced, executor, mutate, reeval, champion
from agent.tree import Node, SearchTree
from agent.memory import Memory
from agent.roles import proposer, coder, reflector
from agent.llm.schemas import Hypothesis, BlockEdit, RecoveryAction
from pipeline.contracts import Cfg
from evaluate import evaluate                      # fixed harness (used by assembly)

BLOCK_SRC = "pipeline/baseline_blocks"
STALL_LIMIT = 6
FM_VALID = 0.6015


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
    return len(best_series) > N and (best_series[-1] - best_series[-1 - N]) <= eps


# ------------------------------------------------------------------ context
def build_proposer_context(tree, mem, phase, it, budget_left):
    best = tree.best()
    lines = [
        f"Phase {phase}, iteration {it}, {budget_left} iterations of budget left.",
        f"FM baseline valid primary = {FM_VALID}.",
        f"Current BEST: primary_valid={best.score():.4f} lever={best.lever} "
        f"cfg.loss_type={best.cfg.loss_type} model={best.cfg.model_type}",
        "Recent experiments (hypothesis -> outcome):",
    ]
    for r in mem.recall(k=8):
        m = r.get("metrics") or {}
        pv = m.get("primary_valid")
        lines.append(f"  [{r.get('status')}] {r.get('lever')} {r.get('hypothesis','')[:70]} "
                     f"-> {pv if pv is None else round(pv,4)}")
    lines.append("Propose the next high-EV change.")
    return "\n".join(lines)


def build_coder_context(parent, hyp):
    tgt = hyp.target_block
    src = (Path(parent.block_dir) / f"{tgt}.py").read_text(encoding="utf-8")
    return (f"Hypothesis: {hyp.statement}\nRationale: {hyp.rationale}\n"
            f"Target block: {tgt}\nconfig_delta: {hyp.config_delta_json}\n\n"
            f"Current source of {tgt}.py:\n```python\n{src}\n```\n"
            f"Rewrite {tgt}.py to implement the hypothesis. Keep the block's function signature.")


# ------------------------------------------------------------------ main loop
def run(cfg, driver, run_id=None, max_iter=None, verbose=True):
    guardrails.ensure_frozen(create=True)
    datced.build_or_load(cfg.data_dir, cfg.cache_dir)

    run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = str(Path(cfg.runs_dir) / run_id)
    mem = Memory(run_dir)
    tree = SearchTree()
    rng = np.random.default_rng(cfg.seed)
    maxit = max_iter or cfg.budget.max_iter
    wall_limit = cfg.budget.wall_clock_hours * 3600
    t_start = time.time()

    def log(msg):
        if verbose:
            print(msg, flush=True)

    # ---- root: reproduce FM ----
    node_dir, blocks, rcfg = mutate.materialize_root(run_dir, BLOCK_SRC, seed=cfg.seed)
    res, wc = executor.run_node(blocks, node_dir, Path(node_dir) / "cfg.json",
                                cfg.cache_dir, cfg.budget.per_iter_timeout_s)
    if isinstance(res, executor.Failure):
        raise SystemExit(f"root node failed: {res}")
    root = Node(id="root", parent=None, phase=0, cfg=rcfg, block_dir=blocks,
                lever="-", hypothesis="reproduce FM baseline",
                metrics={"GAUC": res["GAUC"], "nDCG@5": res["nDCG@5"],
                         "primary_valid": res["primary_valid"], "primary_unbiased": None},
                status="root")
    tree.add(root)
    mem.append(_record(0, root, diff="", events=[], cost={"wall_clock_s": wc}, signature=None))
    log(f"[root] primary_valid={root.score():.4f} ({wc:.0f}s)")

    # F3: seed a cross-run champion (re-validated under the CURRENT cache) as an expandable node.
    champ = champion.load(cfg.champion_dir) if getattr(cfg, "resume", False) else None
    if champ:
        try:
            ccfg = Cfg.from_json(Path(cfg.champion_dir) / "cfg.json")
            cdir, cblocks, _ = mutate.materialize_named(run_dir, "champion",
                                                        str(Path(cfg.champion_dir) / "blocks"), ccfg)
            cres, cwc = executor.run_node(cblocks, cdir, Path(cdir) / "cfg.json",
                                          cfg.cache_dir, cfg.budget.per_iter_timeout_s)
        except Exception as e:                                   # never let a bad champion kill a run
            cres, cwc = executor.Failure("code", repr(e)), 0.0
        if isinstance(cres, executor.Failure):
            log(f"[champion] revalidation failed ({cres.kind}) -> ignoring stale champion")
        else:
            cnode = Node(id="champion", parent="root", phase=0, cfg=ccfg, block_dir=cblocks,
                         lever="resume", hypothesis="resumed cross-run champion",
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

    best_series = [tree.best().score()]
    stall = 0
    it = 0
    while (it < maxit and (time.time() - t_start) < wall_limit
           and not _converged(best_series, cfg.budget.eps, cfg.budget.N)
           and stall < STALL_LIMIT):
        it += 1
        phase = phase_of(it, cfg.phases)
        try:
            stall = _iterate(cfg, driver, run_dir, tree, mem, rng, it, phase,
                             maxit - it, best_series, stall, log)
        except Exception as e:                                   # never let one iter kill the run
            log(f"[it {it}] orchestrator error: {e!r} -> continue")
            stall += 1
        # best-checkpoint invariant: a valid best always exists in memory
    reason = ("converged" if _converged(best_series, cfg.budget.eps, cfg.budget.N)
              else "stalled" if stall >= STALL_LIMIT
              else "budget" if it >= maxit else "wall-clock")
    log(f"[stop] reason={reason} iters={it} best={tree.best().score():.4f}")
    final_valid = finalize(cfg, run_dir, tree, mem, log)
    return run_dir, tree.best(), final_valid


def _iterate(cfg, driver, run_dir, tree, mem, rng, it, phase, budget_left,
             best_series, stall, log):
    parent = tree.select(cfg.phases.explore_p, rng)
    hyp, u = proposer.propose(driver, build_proposer_context(tree, mem, phase, it, budget_left),
                              cfg.llm.proposer, cfg.llm.temperature)
    cost = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens, "wall_clock_s": 0.0}

    delta = json.loads(hyp.config_delta_json or "{}")
    adopt = getattr(hyp, "adopt_blockset", None)
    is_block = hyp.mutation_kind == "block" and hyp.target_block and not adopt
    if not delta and not is_block and not adopt:
        log(f"[it {it}] no-op hypothesis -> stall"); return stall + 1

    block_edit = None
    if is_block:
        be, u2 = coder.code(driver, build_coder_context(parent, hyp), cfg.llm.coder, cfg.llm.temperature)
        cost["input_tokens"] += u2.input_tokens; cost["output_tokens"] += u2.output_tokens
        ok, msg = executor.check_imports(be.new_source)
        if not ok:
            node = Node(id=f"n{it}", parent=parent.id, phase=phase, cfg=parent.cfg,
                        block_dir=parent.block_dir, lever=hyp.lever, hypothesis=hyp.statement,
                        metrics=None, status="abandoned")
            mem.append(_record(it, node, diff="", events=[{"class": "code", "detail": msg}],
                               cost=cost, signature=None))
            log(f"[it {it}] rejected edit ({msg}) -> stall"); return stall + 1
        block_edit = be

    node_dir, blocks, ncfg, diff = mutate.materialize_child(run_dir, f"n{it}", parent, hyp, block_edit)
    sig = mutate.signature(ncfg, diff)
    if mem.seen(sig):
        node = Node(id=f"n{it}", parent=parent.id, phase=phase, cfg=ncfg, block_dir=blocks,
                    lever=hyp.lever, hypothesis=hyp.statement, metrics=None, status="duplicate")
        mem.append(_record(it, node, diff=diff, events=[{"class": "duplicate"}], cost=cost, signature=sig))
        log(f"[it {it}] duplicate of a tried node -> stall"); return stall + 1
    mem.note_seen(sig)

    events = []
    res = None
    wc = 0.0
    # F5 debug-first gate: cheap crash/sanity check on a subsample before the full run (torch nodes).
    if cfg.debug_gate and ncfg.model_type in ("din", "bst"):
        dbg = executor.debug_gate(blocks, ncfg, cfg.cache_dir, str(Path(node_dir) / "_dbg"),
                                  n_train=cfg.debug_train_n, n_other=cfg.debug_other_n,
                                  epochs=cfg.debug_epochs)
        if isinstance(dbg, executor.Failure):
            log(f"[it {it}] {hyp.lever} debug gate FAILED ({dbg.kind}) -> recovery")
            res, wc, events = _recover(cfg, driver, dbg, node_dir, blocks, ncfg, block_edit, log, it)
            # recovery re-runs the FULL node if it can patch; else res stays a Failure (abandoned below)
        else:
            log(f"[it {it}] {hyp.lever} debug gate ok (sample primary~{dbg['primary_valid']:.3f})")

    if res is None:                                 # gate passed or not applicable -> full run
        res, wc = executor.run_node(blocks, node_dir, Path(node_dir) / "cfg.json",
                                    cfg.cache_dir, cfg.budget.per_iter_timeout_s)
        if isinstance(res, executor.Failure):
            res, wc2, events = _recover(cfg, driver, res, node_dir, blocks, ncfg, block_edit, log, it)
            wc += wc2
    cost["wall_clock_s"] = round(wc, 1)

    if isinstance(res, executor.Failure):
        node = Node(id=f"n{it}", parent=parent.id, phase=phase, cfg=ncfg, block_dir=blocks,
                    lever=hyp.lever, hypothesis=hyp.statement, metrics=None, status="abandoned")
        tree.add(node)
        mem.append(_record(it, node, diff=diff, events=events, cost=cost, signature=sig))
        log(f"[it {it}] {hyp.lever} failed ({res.kind}) -> abandoned"); return stall + 1

    pv = res["primary_valid"]
    status = "improved" if pv > parent.score() + 1e-9 else "no_gain"
    node = Node(id=f"n{it}", parent=parent.id, phase=phase, cfg=ncfg, block_dir=blocks,
                lever=hyp.lever, hypothesis=hyp.statement,
                metrics={"GAUC": res["GAUC"], "nDCG@5": res["nDCG@5"],
                         "primary_valid": pv, "primary_unbiased": res.get("primary_unbiased")},
                status=status)
    tree.add(node)
    mem.append(_record(it, node, diff=diff, events=events, cost=cost, signature=sig))
    best_series.append(tree.best().score())
    d = pv - parent.score()
    log(f"[it {it}] {hyp.lever} {status} primary={pv:.4f} (d{d:+.4f} vs parent) "
        f"[{hyp.statement[:48]}] {wc:.0f}s")
    return 0 if status == "improved" else stall + 1


def _recover(cfg, driver, failure, node_dir, blocks, ncfg, block_edit, log, it):
    """One bounded recovery attempt (M5 expands this)."""
    events = [{"class": failure.kind, "detail": failure.detail[:400]}]
    ctx = (f"Failure class: {failure.kind}\nDetail (tail):\n{failure.detail[-1200:]}\n"
           f"cfg.loss_type={ncfg.loss_type} model={ncfg.model_type}")
    try:
        act, _ = reflector.reflect(driver, ctx, cfg.llm.reflector, cfg.llm.temperature)
    except Exception as e:
        events.append({"recovery": "reflector_unavailable", "detail": str(e)}); return failure, 0.0, events
    events.append({"recovery": act.action, "explanation": act.explanation[:200]})
    if act.action == "patch_retry" and act.new_source and act.patch_block:
        ok, _ = executor.check_imports(act.new_source)
        if ok:
            (Path(blocks) / f"{act.patch_block}.py").write_text(act.new_source, encoding="utf-8")
            res, wc = executor.run_node(blocks, node_dir, Path(node_dir) / "cfg.json",
                                        cfg.cache_dir, cfg.budget.per_iter_timeout_s)
            return res, wc, events
    if act.action == "degrade":
        ncfg2 = mutate.apply_delta(ncfg, act.config_delta_json)
        ncfg2.to_json(Path(node_dir) / "cfg.json")
        res, wc = executor.run_node(blocks, node_dir, Path(node_dir) / "cfg.json",
                                    cfg.cache_dir, cfg.budget.per_iter_timeout_s)
        return res, wc, events
    return failure, 0.0, events


# ------------------------------------------------------------------ records / finalize
def _record(it, node, diff, events, cost, signature):
    from dataclasses import asdict
    return {
        "iter": it, "phase": node.phase, "node_id": node.id, "parent_id": node.parent,
        "lever": node.lever, "hypothesis": node.hypothesis,
        "config": asdict(node.cfg), "code_diff": diff,
        "metrics": node.metrics, "status": node.status,
        "events": events, "cost": cost, "signature": signature,
    }


def _per_user_rank(scores, users):
    """Within-user percentile rank in [0,1] -- monotone, scale-free, ideal for blending."""
    scores = np.asarray(scores); users = np.asarray(users)
    out = np.empty(len(scores), np.float32)
    order = np.argsort(users, kind="stable")
    us = users[order]
    for grp in np.split(order, np.flatnonzero(np.diff(us)) + 1):
        s = scores[grp]
        out[grp] = np.argsort(np.argsort(s)) / max(1, len(s) - 1)
    return out


def _val_scores(node):
    p = Path(node.block_dir).parent / "val_scores.npy"
    return np.load(p) if p.exists() else None


def _ablation_summary(tree):
    out = {}
    for n in tree.nodes.values():
        m = n.metrics or {}
        if m.get("primary_valid") is not None:
            out[n.lever] = max(out.get(n.lever, 0.0), m["primary_valid"])
    return {lv: round(p, 5) for lv, p in sorted(out.items())}


def _weight_grids(n):
    if n == 2:
        return [(w, 1 - w) for w in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
    return [(a / 10, b / 10, (10 - a - b) / 10)                 # 3-member simplex, step 0.1
            for a in range(11) for b in range(11 - a)]


def assemble(cfg, tree, log):
    """Phase-3 assembly: best node of each model family, per-user rank-blend, tune weights on
    valid over up to 3 diverse learners. Returns {members, weights, valid_primary} if the blend
    beats the best single, else None."""
    viable = [n for n in tree.nodes.values()
              if n.metrics and n.status in ("root", "improved", "no_gain")]
    fam = {}
    for n in viable:                                    # keep the best node per model family
        mt = n.cfg.model_type
        if n.score() > fam.get(mt, (None, -1.0))[1]:
            fam[mt] = (n, n.score())
    ranked = sorted((v[0] for v in fam.values()), key=lambda n: -n.score())[:3]
    keep = [(m, _val_scores(m)) for m in ranked]
    keep = [(m, s) for m, s in keep if s is not None]
    if len(keep) < 2:
        return None
    members = [m for m, _ in keep]
    b = datced.load_bundle(cfg.cache_dir)
    yva, uva = np.asarray(b.y["valid"]), np.asarray(b.users["valid"])
    R = [_per_user_rank(s, uva) for _, s in keep]
    best_w, best_p = None, -1.0
    for wt in _weight_grids(len(members)):
        blend = sum(wi * Ri for wi, Ri in zip(wt, R))
        p = evaluate(uva, yva, blend)["primary"]
        if p > best_p:
            best_p, best_w = p, wt
    single = max(m.score() for m in members)
    fams = " + ".join(f"{m.cfg.model_type}({m.score():.4f})" for m in members)
    log(f"[assemble] {fams} -> blend w={tuple(round(x,2) for x in best_w)} "
        f"primary={best_p:.4f} vs best single {single:.4f}")
    if best_p <= single + 1e-6:
        return None
    return {"members": [m.id for m in members], "weights": [float(w) for w in best_w],
            "valid_primary": round(float(best_p), 5)}


def _rerun_test(cfg, node, out_dir):
    r, _ = executor.run_node(node.block_dir, out_dir, Path(node.block_dir).parent / "cfg.json",
                             cfg.cache_dir, cfg.budget.per_iter_timeout_s, extra_split="test")
    tp = Path(out_dir) / "test_scores.npy"
    return np.load(tp) if (not isinstance(r, executor.Failure) and tp.exists()) else None


def _multiseed_best(cfg, tree, run_dir, log):
    """F4: re-rank the top viable candidates by seed-MEAN valid primary, guarding the single-best
    pick against single-seed selection bias. Returns (best_node, summary_dict)."""
    ranked = sorted(tree._viable(), key=lambda n: -n.score())[: cfg.recheck_top_k]
    summary, scored = {}, []
    for n in ranked:
        _, mean, prims = reeval.confirm(n.block_dir, n.cfg, n.score(), -1.0, cfg.cache_dir,
                                        cfg.budget.per_iter_timeout_s,
                                        str(Path(run_dir) / "reeval" / n.id),
                                        cfg.recheck_seeds, cfg.budget.eps)
        summary[n.id] = {"orig": round(n.score(), 5), "seed_mean": round(mean, 5),
                         "seeds": [round(p, 5) for p in prims]}
        scored.append((mean, n))
        log(f"[reeval] {n.id} orig={n.score():.4f} seeds={[round(p, 4) for p in prims]} mean={mean:.4f}")
    if not scored:
        return tree.best(), summary
    scored.sort(key=lambda t: -t[0])
    best = scored[0][1]
    log(f"[reeval] seed-mean best = {best.id} (mean {scored[0][0]:.4f})")
    return best, summary


def finalize(cfg, run_dir, tree, mem, log):
    reeval_summary = None
    if cfg.recheck:
        best, reeval_summary = _multiseed_best(cfg, tree, run_dir, log)
    else:
        best = tree.best()
    best_dir = Path(run_dir) / "best"
    b = datced.load_bundle(cfg.cache_dir)
    ute = np.asarray(b.users["test"])

    try:
        asm = assemble(cfg, tree, log)
    except Exception as e:
        log(f"[assemble] errored, falling back to single best: {e!r}")
        asm = None
    final_valid, final_test, ensemble_info = float(best.score()), None, None

    if asm:                                             # ENSEMBLE final: re-run members on test, blend
        parts = {nid: _rerun_test(cfg, tree.nodes[nid], best_dir / nid) for nid in asm["members"]}
        if all(v is not None for v in parts.values()):
            R = [_per_user_rank(parts[nid], ute) for nid in asm["members"]]
            final_test = sum(w * r for w, r in zip(asm["weights"], R))
            final_valid = asm["valid_primary"]
            ensemble_info = asm
            log(f"[finalize] using ENSEMBLE {asm['members']} (w={asm['weights']}) valid={final_valid:.4f}")

    if final_test is None:                              # single-best final
        log(f"[finalize] using best single node {best.id} ({best.score():.4f})")
        final_test = _rerun_test(cfg, best, best_dir)

    sub_ok = False
    try:
        if final_test is not None:
            sub = best_dir / "submission_test.csv"
            _write_submission(cfg.cache_dir, np.asarray(final_test), sub)
            chk = subprocess.run([sys.executable, "submit.py", "--check", "--split", "test", str(sub)],
                                 capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", env=executor.utf8_env())
            sub_ok = chk.returncode == 0
            log(f"[finalize] submission {'PASSED' if sub_ok else 'FAILED'} submit.py --check")
            if not sub_ok:
                log("   " + (chk.stderr or chk.stdout)[-400:])
        else:
            log("[finalize] no test scores to submit")
    except Exception as e:
        log(f"[finalize] submission step errored (run still finalizes): {e!r}")

    totals = mem.resource_totals()
    report = {
        "best_single_node": best.id, "best_single_valid": best.score(),
        "final_valid": final_valid, "ensemble": ensemble_info,
        "delta_over_fm": round(final_valid - FM_VALID, 4),
        "submission_valid": sub_ok, "ablation_best_by_lever": _ablation_summary(tree),
        "reeval": reeval_summary,
        "resource_totals": totals,
    }
    (Path(run_dir) / "resource_report.json").write_text(json.dumps(report, indent=2, default=float))
    _write_results_md(run_dir, final_valid, ensemble_info, totals)
    if getattr(cfg, "resume", False):                 # F3: persist the best single node as champion
        prev = champion.load(cfg.champion_dir)
        if prev is None or best.score() > prev["primary_valid"]:
            champion.save(cfg.champion_dir, best.block_dir, best.cfg, best.score(),
                          datced.CACHE_VERSION, Path(run_dir).name, best.id)
            log(f"[champion] saved {best.id} valid={best.score():.4f} "
                f"(was {prev['primary_valid'] if prev else None})")
    log(f"[finalize] FINAL valid {final_valid:.4f} (d{final_valid-FM_VALID:+.4f} vs FM) | "
        f"tokens={totals['input_tokens']+totals['output_tokens']} wall={totals['wall_clock_s']}s")
    return final_valid


def _write_submission(cache_dir, scores, out_csv):
    uid = np.load(Path(cache_dir) / "test_u.npy")
    vid = np.load(Path(cache_dir) / "test_vid.npy")
    s = np.asarray(scores)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("row_id,user_id,video_id,score\n")
        for i in range(len(s)):
            f.write(f"{i},{int(uid[i])},{int(vid[i])},{float(s[i])}\n")


def _write_results_md(run_dir, final_valid, ensemble_info, totals):
    kind = f"ensemble {ensemble_info['members']} (w={ensemble_info['weights']})" if ensemble_info else "single best"
    md = (f"# Results\n\n"
          f"Final validation primary: **{final_valid:.4f}** ({kind})\n\n"
          f"| | FM baseline | agent final | delta |\n|---|---|---|---|\n"
          f"| primary | {FM_VALID} | {final_valid:.4f} | {round(final_valid - FM_VALID, 4):+} |\n\n"
          f"Resource usage: {totals}\n")
    (Path(run_dir) / "results.md").write_text(md, encoding="utf-8")
