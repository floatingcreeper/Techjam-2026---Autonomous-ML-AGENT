"""agent.stats must reproduce the FROZEN evaluator's semantics exactly enough to decide on.

Run: cudaenv/Scripts/python.exe -m tests.test_stats
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.stats import Evaluator, classify_evidence, paired_bootstrap, per_user_rank, rank_corr
from evaluate import evaluate                                  # FROZEN reference

TOL = 1e-5          # two orders of magnitude below the smallest effect we can resolve (~1e-3)
failures: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


def test_matches_frozen_evaluator_synthetic():
    """Random data, including heavy score ties and degenerate users -- the tie/exclusion paths."""
    print("\n[1] agent.stats.Evaluator vs frozen evaluate.evaluate (synthetic)")
    rng = np.random.default_rng(7)
    for trial, (nu, tie) in enumerate([(200, False), (200, True), (50, True), (500, False)]):
        users, labels, scores = [], [], []
        for u in range(nu):
            n = int(rng.integers(1, 12))
            users += [u] * n
            labels += list(rng.integers(0, 2, n).astype(float))
            # `tie` collapses scores onto few distinct values to exercise average-rank tie handling
            scores += list(rng.integers(0, 3, n).astype(float) if tie else rng.normal(size=n))
        u_, y_, s_ = np.array(users), np.array(labels, np.float32), np.array(scores, np.float32)
        ref = evaluate(u_, y_, s_)
        ev = Evaluator(u_, y_)
        p, g, n = ev.primary(s_)
        ok = (abs(p - float(ref["primary"])) < TOL and abs(g - float(ref["GAUC"])) < TOL
              and abs(n - float(ref["nDCG@5"])) < TOL)
        check(f"trial {trial} (users={nu}, ties={tie})", ok,
              f"primary {p:.7f} vs {float(ref['primary']):.7f}")


def test_matches_frozen_evaluator_real_nodes():
    """Real saved node predictions -- the vectors the agent actually decides on."""
    print("\n[2] agent.stats.Evaluator vs frozen evaluate.evaluate (real val_scores.npy)")
    cache = Path("runs/_cache")
    if not (cache / "valid_y.npy").exists():
        print("  SKIP  no cache present")
        return
    y = np.load(cache / "valid_y.npy")
    u = np.load(cache / "valid_u.npy")
    ev = Evaluator(u, y)
    found = 0
    for sp in sorted(Path("runs").glob("run_*/nodes/*/val_scores.npy"))[:12]:
        s = np.load(sp)
        if len(s) != len(y):
            continue
        ref = evaluate(u, y, s)
        p, g, n = ev.primary(s)
        ok = abs(p - float(ref["primary"])) < TOL
        check(f"{sp.parts[-3]}/{sp.parts[-2]}", ok,
              f"|d|={abs(p - float(ref['primary'])):.2e}")
        found += 1
    if not found:
        print("  SKIP  no node score files found")


def test_paired_bootstrap_properties():
    print("\n[3] paired_bootstrap")
    rng = np.random.default_rng(3)
    nu = 800
    users, labels = [], []
    for uu in range(nu):
        n = int(rng.integers(2, 9))
        users += [uu] * n
        labels += list(rng.integers(0, 2, n).astype(float))
    u_, y_ = np.array(users), np.array(labels, np.float64)
    ev = Evaluator(u_, y_)
    base = rng.normal(size=len(y_))

    # identical predictions -> delta exactly 0, SE exactly 0
    st = ev.user_stats(base)
    r = paired_bootstrap(ev, st, st, B=200, seed=0)
    check("identical models -> delta==0", abs(r["delta_primary"]) < 1e-12)
    check("identical models -> SE==0", r["boot_se"] < 1e-12)

    # a strictly better model (scores nudged toward the labels) -> P(delta>0) high
    better = base + 3.0 * y_
    r2 = paired_bootstrap(ev, ev.user_stats(better), st, B=300, seed=1)
    check("clearly better model -> delta>0", r2["delta_primary"] > 0, f"d={r2['delta_primary']:+.4f}")
    check("clearly better model -> P(d>0) high", r2["p_gt0"] > 0.95, f"P={r2['p_gt0']:.3f}")
    check("SE is positive and finite", 0 < r2["boot_se"] < 1, f"SE={r2['boot_se']:.5f}")
    check("CI brackets the point estimate",
          r2["ci_lo"] <= r2["delta_primary"] <= r2["ci_hi"])

    # antisymmetry: swapping treatment/control flips the sign
    r3 = paired_bootstrap(ev, st, ev.user_stats(better), B=300, seed=1)
    check("antisymmetric delta", abs(r3["delta_primary"] + r2["delta_primary"]) < 1e-12)

    # determinism under a fixed seed
    ra = paired_bootstrap(ev, ev.user_stats(better), st, B=100, seed=42)
    rb = paired_bootstrap(ev, ev.user_stats(better), st, B=100, seed=42)
    check("seeded bootstrap is deterministic", ra == rb)


def test_evidence_classes():
    print("\n[4] classify_evidence")
    check("0.95 -> confirmed", classify_evidence(0.95) == "confirmed")
    check("0.05 -> rejected", classify_evidence(0.05) == "rejected")
    check("0.75 -> promising", classify_evidence(0.75) == "promising")
    check("0.50 -> inconclusive", classify_evidence(0.50) == "inconclusive")
    check("0.90 boundary -> confirmed", classify_evidence(0.90) == "confirmed")
    check("0.10 boundary -> rejected", classify_evidence(0.10) == "rejected")
    check("NaN -> inconclusive", classify_evidence(float("nan")) == "inconclusive")


def test_rank_helpers():
    print("\n[5] per_user_rank / rank_corr")
    u = np.array([0, 0, 0, 1, 1])
    s = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    r = per_user_rank(s, u)
    check("percentile ranks are within-user and in [0,1]",
          np.allclose(r, [0.0, 1.0, 0.5, 1.0, 0.0]), str(np.round(r, 3)))
    check("identical vectors -> corr 1", abs(rank_corr(r, r) - 1.0) < 1e-9)
    check("constant vector -> corr 1 (no divide-by-zero)",
          rank_corr(np.zeros(5), np.arange(5)) == 1.0)

    # per_user_rank must agree with the orchestrator's existing implementation
    from agent.orchestrator import _per_user_rank as orch_rank
    rng = np.random.default_rng(11)
    uu = np.repeat(np.arange(50), 4)
    ss = rng.normal(size=len(uu))
    check("matches orchestrator._per_user_rank",
          np.allclose(per_user_rank(ss, uu), orch_rank(ss, uu)))


if __name__ == "__main__":
    test_matches_frozen_evaluator_synthetic()
    test_matches_frozen_evaluator_real_nodes()
    test_paired_bootstrap_properties()
    test_evidence_classes()
    test_rank_helpers()
    print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
    sys.exit(1 if failures else 0)
