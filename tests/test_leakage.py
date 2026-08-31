"""Holdout/leakage integrity -- docs/EN/RESEARCH.md §9 / docs/EN/SYSTEM.md §8 (integrity).

The defect: v6 wrote `runs/_cache/test_y.npy` and `runs/_cache/aux/{valid,test}_aux.npy`. Every block
receives `bundle.cache_dir` and numpy is allowlisted, so the hidden-test labels and the near-oracle
`is_click` proxy were one `np.load` away. Ranking valid by `is_click` alone scores primary 0.7466
against FM 0.6015 and an oracle ceiling of 0.8484 -- 58.8% of the whole headroom, no training needed.

These tests assert the three layers that now stand:
  1. data interface -- label-derived arrays are not in the block-visible cache at all
  2. loader        -- load_bundle withholds y["test"]; load_aux refuses non-train splits
  3. static guard  -- check_imports rejects blocks that try to reach around the interface

Run: cudaenv/Scripts/python.exe -m tests.test_leakage
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import datced, executor                     # noqa: E402
from pipeline.lib import aux_build                     # noqa: E402

CACHE = ROOT / "runs" / "_cache"
failures: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


def test_cache_contains_no_label_derived_holdout():
    print("\n[1] block-visible cache holds no label-derived holdout arrays")
    for rel in ("test_y.npy", "aux/test_aux.npy", "aux/valid_aux.npy"):
        check(f"runs/_cache/{rel} absent", not (CACHE / rel).exists())
    hold = Path(datced.holdout_dir(str(CACHE)))
    check("holdout dir is a SIBLING of the cache, not inside it",
          hold.resolve() != CACHE.resolve() and CACHE.resolve() not in hold.resolve().parents,
          str(hold))
    check("holdout dir actually holds the withheld test labels", (hold / "test_y.npy").exists())


def test_loader_withholds():
    print("\n[2] loader interface")
    b = datced.load_bundle(str(CACHE))
    check("bundle.y has no 'test' key", "test" not in b.y, f"keys={sorted(b.y)}")
    check("bundle.X still has 'test' (inference must work)", "test" in b.X)
    check("bundle.users still has 'test'", "test" in b.users)
    for split in ("valid", "test"):
        try:
            aux_build.load_aux(str(CACHE), split)
            check(f"load_aux({split!r}) refuses", False, "returned data instead of raising")
        except KeyError as e:
            check(f"load_aux({split!r}) refuses", True, f"KeyError: {str(e)[:60]}...")
    try:
        a = aux_build.load_aux(str(CACHE), "train")
        check("load_aux('train') still works (fit_din needs it)", "click" in a)
    except Exception as e:
        check("load_aux('train') still works", False, repr(e))


def test_alignment_guard_detects_planted_leak():
    print("\n[3] _assert_aux_aligned rejects a re-planted holdout array")
    planted = CACHE / "aux" / "valid_aux.npy"
    try:
        np.save(planted, np.zeros((4, 5), np.float32))
        try:
            datced._assert_aux_aligned(str(CACHE))
            check("planted valid_aux.npy is detected", False, "guard did not raise")
        except RuntimeError as e:
            check("planted valid_aux.npy is detected", "leakage" in str(e).lower(),
                  str(e)[:70] + "...")
    finally:
        planted.unlink(missing_ok=True)
    datced._assert_aux_aligned(str(CACHE))          # must pass again once removed
    print("  PASS  guard passes again after removal")


HOSTILE = [
    ("hidden-test labels via cache_dir",
     'import numpy as np\ndef build_features(bundle, cfg):\n'
     '    y = np.load(bundle.cache_dir + "/test_y.npy")\n    return y\n'),
    ("aux click proxy",
     'import numpy as np\ndef build_features(bundle, cfg):\n'
     '    return np.load(bundle.cache_dir + "/aux/valid_aux.npy")\n'),
    ("holdout directory",
     'import numpy as np\ndef build_features(bundle, cfg):\n'
     '    return np.load("runs/_holdout/test_y.npy")\n'),
    ("raw log files",
     'def build_features(bundle, cfg):\n'
     '    return open("KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv").read()\n'),
    ("open() builtin",
     'def build_features(bundle, cfg):\n    return open("anything.csv").read()\n'),
    ("eval() escape hatch",
     'def build_features(bundle, cfg):\n    return eval("__import__(\'os\')")\n'),
]

LEGITIMATE = [
    ("baseline features block", (ROOT / "pipeline/baseline_blocks/features.py").read_text()),
    ("baseline loss block", (ROOT / "pipeline/baseline_blocks/loss.py").read_text()),
    ("din features block", (ROOT / "pipeline/lib/din_blocks/features.py").read_text()),
    ("lgbm train block", (ROOT / "pipeline/lib/lgbm_blocks/train.py").read_text()),
    ("plausible agent edit",
     'import numpy as np\nfrom pipeline.lib.losses import make_loss\n'
     'def build_loss(cfg):\n    return make_loss(cfg)\n'),
]


def test_static_guard():
    print("\n[4] check_imports blocks holdout access")
    for name, src in HOSTILE:
        ok, why = executor.check_imports(src)
        check(f"REJECTS: {name}", not ok, why[:70] if not ok else "ACCEPTED (bad!)")
    print("\n[5] check_imports still accepts legitimate blocks")
    for name, src in LEGITIMATE:
        ok, why = executor.check_imports(src)
        check(f"ACCEPTS: {name}", ok, "" if ok else f"rejected: {why[:70]}")


def test_leak_severity_is_real():
    print("\n[6] severity: what the closed hole was worth")
    hold = Path(datced.holdout_dir(str(CACHE)))
    va = hold / "aux" / "valid_aux.npy"
    if not va.exists():
        print("  SKIP  holdout aux not built")
        return
    from evaluate import evaluate
    y = np.load(CACHE / "valid_y.npy")
    u = np.load(CACHE / "valid_u.npy")
    click = np.asarray(np.load(va)[:, 0], np.float32)
    r = evaluate(u, y, click)
    p = float(r["primary"])
    frac = (p - 0.6015) / (0.8484 - 0.6015)
    check("ranking valid by is_click alone beats FM by a huge margin", p > 0.70,
          f"primary={p:.4f} vs FM 0.6015, oracle 0.8484 -> {frac:.1%} of headroom")
    check("that array is NOT reachable from the block-visible cache",
          not (CACHE / "aux" / "valid_aux.npy").exists())


if __name__ == "__main__":
    test_cache_contains_no_label_derived_holdout()
    test_loader_withholds()
    test_alignment_guard_detects_planted_leak()
    test_static_guard()
    test_leak_severity_is_real()
    print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
    sys.exit(1 if failures else 0)
