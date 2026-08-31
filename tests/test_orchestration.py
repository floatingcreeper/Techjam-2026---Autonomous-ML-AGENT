"""Benchmark-contract and experimental-validity invariants.

These are the properties a live run must satisfy but that a short mock run may converge before
exercising: content-based deduplication, the official convergence semantics, no-op classification,
and the separation of adoption status from scientific evidence.

Run: cudaenv/Scripts/python.exe -m tests.test_orchestration
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import mutate, orchestrator as orch, provenance          # noqa: E402
from agent.llm.schemas import Hypothesis                            # noqa: E402
from agent.memory import Memory                                     # noqa: E402
from agent.stats import Evaluator                                   # noqa: E402
from agent.tree import EXACT_NOOP, NEAR_NOOP, Node, SearchTree      # noqa: E402
from pipeline.contracts import Cfg                                  # noqa: E402

failures: list[str] = []
SCRATCH = ROOT / "runs" / "_test_orch"


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


def _H(**kw):
    kw.setdefault("problem_identified", "test")
    kw.setdefault("rationale", "test")
    kw.setdefault("mutation_kind", "config")
    kw.setdefault("config_delta_json", "{}")
    return Hypothesis(**kw)


def _node(nid, cfg, blocks, parent=None, pv=None):
    return Node(id=nid, parent=parent, phase=1, cfg=cfg, block_dir=str(blocks), lever="A",
                hypothesis="t", metrics=None if pv is None else
                {"GAUC": 0.6, "nDCG@5": 0.5, "primary_valid": pv},
                status="root" if parent is None else "no_gain")


# ------------------------------------------------------------------ official convergence
def test_official_convergence_semantics():
    print("\n[1] official convergence rule (eps=0.002, N=3 from baseline_scores.json)")
    bs = json.loads((ROOT / "baseline_scores.json").read_text())["convergence_rule"]
    check("orchestrator reads eps from the organizer artifact",
          orch.OFFICIAL_EPS == bs["epsilon"], f"{orch.OFFICIAL_EPS} == {bs['epsilon']}")
    check("orchestrator reads N from the organizer artifact",
          orch.OFFICIAL_N == bs["N"], f"{orch.OFFICIAL_N} == {bs['N']}")
    from agent.config import Budget
    check("Budget default matches the official rule",
          Budget().eps == bs["epsilon"] and Budget().N == bs["N"])

    c = orch._converged
    eps, N = orch.OFFICIAL_EPS, orch.OFFICIAL_N
    check("not converged with too few points", not c([0.60147], eps, N))
    check(f"needs strictly more than N={N} points before it can fire",
          not c([0.6] * N, eps, N) and c([0.6] * (N + 1), eps, N))
    # a run still improving by more than eps over the window must NOT converge
    check("still improving -> NOT converged", not c([0.60147, 0.6025, 0.6030, 0.6036], eps, N),
          "delta=+0.00213 > 0.002")
    # a plateaued run MUST converge
    check("plateaued -> converged", c([0.60147, 0.60361, 0.60361, 0.60361, 0.60361], eps, N))
    # Boundary: the rule is `improvement <= eps -> converged`, applied verbatim. Values chosen so the
    # difference is exactly representable in binary float (0.602-0.600 is 0.0020000000000000018,
    # which is a float artifact 15 orders of magnitude below the 0.0009 noise floor, not a semantic
    # question -- the rule is deliberately NOT given a tolerance, as that would change benchmark
    # semantics).
    check("delta exactly eps -> converged", c([0.0, 0.0, 0.0, eps], eps, N))
    check("delta just above eps -> not converged", not c([0.0, 0.0, 0.0, eps * 1.05], eps, N))
    check("only the window endpoints matter, not the path",
          c([0.0, 0.5, 0.0, 0.0, eps], eps, N))


# ------------------------------------------------------------------ content dedup
def test_content_dedup_is_path_independent():
    print("\n[2] deduplication is CONTENT-based, not path-based (docs/EN/SYSTEM.md §12)")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    b1, b2 = SCRATCH / "b1", SCRATCH / "b2"
    for b in (b1, b2):
        b.mkdir(parents=True, exist_ok=True)
        for f in (ROOT / "pipeline/baseline_blocks").glob("*.py"):
            if f.name != "__init__.py":
                shutil.copy(f, b / f.name)
    cfg = Cfg(seed=0, loss_type="bpr")

    s1 = mutate.signature(cfg, str(b1), {})
    s2 = mutate.signature(cfg, str(b2), {})
    check("identical cfg + identical block sources -> SAME signature "
          "(this is the n3/n6 case the old cfg+diff signature missed)", s1 == s2, s1)

    s3 = mutate.signature(Cfg(seed=0, loss_type="bce"), str(b1), {})
    check("different cfg -> different signature", s1 != s3)

    (b2 / "loss.py").write_text((b2 / "loss.py").read_text() + "\n# a real code change\n")
    s4 = mutate.signature(cfg, str(b2), {})
    check("different block source -> different signature", s1 != s4)

    s5 = mutate.signature(cfg, str(b1), {"use_fb": True})
    check("extension sidecar participates in identity (cannot smuggle a change)", s1 != s5)


# ------------------------------------------------------------------ no-op classification
def test_noop_classification():
    print("\n[3] no-op classification distinguishes the three cases")
    a = np.array([0.1, 0.5, 0.3, 0.9], np.float32)
    check("bit-identical predictions -> EXACT_NOOP",
          orch._classify_noop(a, a.copy(), "fm") == EXACT_NOOP)

    rng = np.random.default_rng(0)
    base = rng.normal(size=4000).astype(np.float32)
    check("genuinely different predictions -> not a no-op",
          orch._classify_noop(base + rng.normal(scale=1.0, size=4000), base, "fm") is None)
    check("microscopically perturbed predictions -> NEAR_NOOP (legitimate evidence of a "
          "negligible effect, NOT a missing intervention)",
          orch._classify_noop(base + 1e-7 * rng.normal(size=4000), base, "fm") == NEAR_NOOP)

    # The decisive distinction the task brief calls out: two trainings of one stochastic config
    # correlate ~0.926 (measured), which must NOT be read as a structural no-op.
    r = ROOT / "runs" / "run_20260831_000142" / "nodes"
    if (r / "n3" / "val_scores.npy").exists() and (r / "n6" / "val_scores.npy").exists():
        n3 = np.load(r / "n3" / "val_scores.npy")
        n6 = np.load(r / "n6" / "val_scores.npy")
        check("two trainings of an IDENTICAL DIN config are not classified as a no-op",
              orch._classify_noop(n6, n3, "din") is None,
              f"class={orch._classify_noop(n6, n3, 'din')}")
    else:
        print("  SKIP  historical DIN re-run pair not present")


# ------------------------------------------------------------------ evidence vs adoption
def test_evidence_is_not_adoption_status():
    print("\n[4] scientific evidence is independent of tree adoption status (docs/EN/SYSTEM.md §14)")
    mem = Memory(str(SCRATCH / "mem"))
    # A node that is the CHAMPION but was never labelled "improved" -- the exact reference-run case.
    mem.records = [
        {"iter": 0, "node_id": "root", "lever": "-", "status": "root", "hypothesis": "baseline",
         "metrics": {"primary_valid": 0.60147, "GAUC": 0.667, "nDCG@5": 0.536},
         "evidence": {}, "portfolio": {}, "config": {"model_type": "fm"}},
        {"iter": 1, "node_id": "n1", "lever": "C", "status": "no_gain",
         "hypothesis": "multi-task DIN", "config": {"model_type": "din"},
         "metrics": {"primary_valid": 0.60265, "GAUC": 0.6686, "nDCG@5": 0.5367},
         "evidence": {"class": "inconclusive", "delta_primary": 0.00082, "p_gt0": 0.85,
                      "control_id": "root"},
         "portfolio": {"emc": 0.0012, "rank_corr_to_best": 0.845}},
        {"iter": 2, "node_id": "n2", "lever": "D", "status": "no_gain",
         "hypothesis": "LightGBM", "config": {"model_type": "lgbm"},
         "metrics": {"primary_valid": 0.60205, "GAUC": 0.6673, "nDCG@5": 0.5368},
         "evidence": {"class": "rejected", "delta_primary": -0.0006, "p_gt0": 0.05,
                      "control_id": "n1"},
         "portfolio": {"emc": 0.00121, "rank_corr_to_best": 0.845}},
        {"iter": 3, "node_id": "n3", "lever": "E", "status": "no_gain",
         "hypothesis": "raise neg_ratio", "config": {"model_type": "din"},
         "metrics": {"primary_valid": 0.60223}, "noop_class": "STRUCTURAL_NOOP",
         "evidence": {}, "portfolio": {}},
    ]
    t = mem.research_table()
    joined = " ".join(t["rejected"])
    check("the CHAMPION is never listed as rejected", "multi-task DIN" not in joined,
          joined[:80] or "(rejected bucket empty)")
    check("the champion is surfaced explicitly",
          t["champion"] and t["champion"]["node_id"] == "n1", str(t["champion"]))
    check("a below-champion effect is 'inconclusive', not 'rejected'",
          any("multi-task" in x for x in t["inconclusive"]))
    check("a structural no-op is quarantined as no_effect, not scientific evidence",
          any("STRUCTURAL_NOOP" in x for x in t["no_effect"])
          and not any("neg_ratio" in x for x in t["rejected"]))
    check("a standalone-rejected model with high EMC is surfaced as a portfolio asset",
          any("n2" in x for x in t["diverse_portfolio_candidates"]),
          str(t["diverse_portfolio_candidates"]))
    check("GAUC and nDCG are carried separately for the champion",
          t["champion"].get("GAUC") is not None and t["champion"].get("nDCG@5") is not None)


# ------------------------------------------------------------------ provenance
def test_provenance_records_executed_not_intended():
    print("\n[5] provenance distinguishes intended from executed intervention (docs/EN/SYSTEM.md §11)")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    blocks = SCRATCH / "pb"
    blocks.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "pipeline/baseline_blocks").glob("*.py"):
        if f.name != "__init__.py":
            shutil.copy(f, blocks / f.name)
    parent = _node("root", Cfg(seed=0), blocks, pv=0.60147)

    # A delta mixing an effective knob, a not-honoured knob and an unknown key.
    hyp = _H(lever="A", statement="mixed",
             config_delta_json='{"loss_type":"bpr","aux_tasks":["click"],"num_leaves":63}')
    val, bs, intended = mutate.validate_hypothesis(parent, hyp, {})
    nd, bd, cfg, diff, ext, prov = mutate.materialize_child(
        str(SCRATCH / "run"), "n1", parent, hyp, None, validation=val, parent_ext={},
        cache_version=7)
    check("effective delta contains only the honoured, valid, changing key",
          prov.effective_delta == {"loss_type": "bpr"}, str(prov.effective_delta))
    check("not-honoured key recorded separately", "aux_tasks" in prov.rejected_not_honoured)
    check("unknown key recorded as invalid", "num_leaves" in prov.rejected_invalid)
    check("intervention_matched is False when the executed change is narrower",
          not prov.intervention_matched)
    check("the written cfg.json actually has loss_type=bpr",
          Cfg.from_json(Path(nd) / "cfg.json").loss_type == "bpr")
    check("the written cfg.json did NOT absorb the unknown key",
          not hasattr(Cfg.from_json(Path(nd) / "cfg.json"), "num_leaves"))
    check("block hashes recorded for all six blocks", len(prov.block_hashes) == 6)
    check("code_state recorded", bool(prov.code_state))
    check("cache_version recorded", prov.cache_version == 7)
    check("provenance.json written next to the node", (Path(nd) / "provenance.json").exists())

    # An all-ignored delta is a structural no-op -> must not be executable.
    hyp2 = _H(lever="C", statement="ignored", config_delta_json='{"aux_tasks":["click"]}')
    val2, _, _ = mutate.validate_hypothesis(parent, hyp2, {})
    check("a delta of only not-honoured keys has NO effect (never trains)", not val2.has_effect)


def test_accounting_separates_proposals_from_experiments():
    print("\n[6] proposal attempts and executed experiments are separate counters (docs/EN/SYSTEM.md §16)")
    from agent.config import Config
    st = orch.RunState(Config(), 50)
    st.proposal_attempts += 3
    st.proposals_rejected += 2
    st.experiments_executed += 1
    tree = SearchTree()
    tree.add(_node("root", Cfg(), SCRATCH, pv=0.6))
    snap = st.snapshot(tree, [0.6])
    check("executed experiments counted separately", snap["experiments_executed"] == 1)
    check("proposal attempts counted separately", snap["proposal_attempts"] == 3)
    check("a rejected proposal is NOT an executed experiment",
          snap["proposal_attempts"] > snap["experiments_executed"])
    check("official rule surfaced in the snapshot",
          snap["official_eps"] == orch.OFFICIAL_EPS and snap["official_N"] == orch.OFFICIAL_N)
    check("max_iter surfaced as the hard cap", snap["max_iter"] == 50)


if __name__ == "__main__":
    try:
        test_official_convergence_semantics()
        test_content_dedup_is_path_independent()
        test_noop_classification()
        test_evidence_is_not_adoption_status()
        test_provenance_records_executed_not_intended()
        test_accounting_separates_proposals_from_experiments()
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
    sys.exit(1 if failures else 0)
