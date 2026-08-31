"""The honoured-config contract must match what the code ACTUALLY executes.

Two independent checks, because a hand-written declaration drifts:

  [static]  every declared field must appear as `cfg.<field>` somewhere in the modules the block set
            can reach. Catches typos and fields nothing anywhere reads.
  [runtime] the decisive one. On the subsampled debug cache, flip a knob and compare predictions:
            a HONOURED knob must change them; a NOT-HONOURED knob must leave them bit-identical.
            This is what docs/EN/RESEARCH.md §7 (BPR) / docs/EN/SYSTEM.md §11 was actually about.

Run: cudaenv/Scripts/python.exe -m tests.test_blockspec
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import blockspec, executor                       # noqa: E402
from pipeline.contracts import Cfg                          # noqa: E402

failures: list[str] = []
SCRATCH = ROOT / "runs" / "_test_blockspec"


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


# ------------------------------------------------------------------ static
# Modules each block set can reach (blocks + the libs they delegate to).
REACHABLE = {
    "fm": ["pipeline/baseline_blocks", "pipeline/lib/fm.py", "pipeline/lib/losses.py",
           "pipeline/lib/train_np.py"],
    "din": ["pipeline/lib/din_blocks", "pipeline/lib/din.py", "pipeline/lib/seq_build.py",
            "pipeline/lib/aux_build.py", "pipeline/lib/train_np.py"],
    "lgbm": ["pipeline/lib/lgbm_blocks", "pipeline/lib/gbm.py"],
}
PAT = re.compile(r'cfg\.([A-Za-z_][A-Za-z0-9_]*)|getattr\(\s*cfg\s*,\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']')


def _reads(paths):
    found = set()
    for rel in paths:
        p = ROOT / rel
        files = sorted(p.glob("*.py")) if p.is_dir() else [p]
        for f in files:
            for m in PAT.finditer(f.read_text(encoding="utf-8")):
                found.add(m.group(1) or m.group(2))
    return found


def test_static_declarations():
    print("\n[1] declared honoured fields appear in reachable source")
    for name, spec in blockspec.SPECS.items():
        reads = _reads(REACHABLE[name])
        missing = sorted(spec.honoured - reads)
        check(f"{name}: every declared field is read somewhere reachable", not missing,
              f"undeclared-but-declared: {missing}" if missing else f"({len(spec.honoured)} fields)")


# ------------------------------------------------------------------ runtime
def _debug_cache():
    from pipeline import debug_cache
    return debug_cache.build(str(ROOT / "runs" / "_cache"), str(SCRATCH / "cache"),
                             n_train=20000, n_other=10000, seed=0)


def _run(blocks_src: str, cfg: Cfg, tag: str, cache: str):
    """Run one node on the debug cache; return its valid scores (or None on failure)."""
    bdir = SCRATCH / tag / "blocks"
    bdir.mkdir(parents=True, exist_ok=True)
    for f in Path(ROOT / blocks_src).glob("*.py"):
        if f.name != "__init__.py":
            shutil.copy(f, bdir / f.name)
    cfgp = SCRATCH / tag / "cfg.json"
    cfg.to_json(cfgp)
    res, _ = executor.run_node(str(bdir), str(SCRATCH / tag / "out"), str(cfgp), cache, timeout_s=600)
    if isinstance(res, executor.Failure):
        print(f"      (run failed: {res.kind}: {res.detail[:160]})")
        return None
    p = SCRATCH / tag / "out" / "val_scores.npy"
    return np.load(p) if p.exists() else None


# (block_src, base_cfg_kwargs, honoured_change, not_honoured_change)
CASES = [
    ("pipeline/baseline_blocks", dict(epochs=2, patience=1),
     ("loss_type", "bpr"), ("aux_tasks", ("click",))),
    ("pipeline/lib/lgbm_blocks", dict(epochs=2, patience=1),
     None, ("lr", 0.5)),
]


def test_runtime_effect():
    print("\n[2] runtime: honoured knobs change predictions, not-honoured knobs do not")
    if not (ROOT / "runs" / "_cache" / "meta.json").exists():
        print("  SKIP  no cache present")
        return
    cache = _debug_cache()
    for blocks_src, base_kw, hon, nothon in CASES:
        fam = "fm" if "baseline" in blocks_src else blocks_src.split("/")[-1].replace("_blocks", "")
        base = _run(blocks_src, Cfg(seed=0, **base_kw), f"{fam}_base", cache)
        if base is None:
            check(f"{fam}: baseline run", False, "baseline node failed")
            continue

        if hon:
            k, v = hon
            check(f"{fam}: {k} is declared honoured", k in blockspec.SPECS[fam].honoured)
            got = _run(blocks_src, Cfg(seed=0, **{**base_kw, k: v}), f"{fam}_hon", cache)
            check(f"{fam}: honoured {k}={v!r} CHANGES predictions",
                  got is not None and not np.array_equal(base, got))

        k, v = nothon
        check(f"{fam}: {k} is declared NOT honoured", k not in blockspec.SPECS[fam].honoured)
        got = _run(blocks_src, Cfg(seed=0, **{**base_kw, k: v}), f"{fam}_not", cache)
        check(f"{fam}: not-honoured {k}={v!r} leaves predictions IDENTICAL",
              got is not None and np.array_equal(base, got))


# ------------------------------------------------------------------ validation logic
def test_validate_delta():
    print("\n[3] validate_delta classification")
    cur = Cfg(seed=0, loss_type="bce", k=16, aux_tasks=())

    v = blockspec.validate_delta("fm", cur, {"loss_type": "bpr"})
    check("fm: loss_type=bpr is effective", v.effective == {"loss_type": "bpr"} and v.has_effect)

    v = blockspec.validate_delta("fm", cur, {"loss_type": "softmax"})
    check("fm: loss_type='softmax' (a real live-agent emission) is INVALID",
          "loss_type" in v.invalid and not v.has_effect)

    v = blockspec.validate_delta("fm", cur, {"aux_tasks": ["click"]})
    check("fm: aux_tasks is not honoured by the fm block set",
          "aux_tasks" in v.not_honoured and not v.has_effect)

    v = blockspec.validate_delta("din", cur, {"aux_tasks": ["click", "like"]})
    check("din: aux_tasks IS honoured", v.effective and v.has_effect)

    v = blockspec.validate_delta("din", cur, {"aux_tasks": ["clik"]})
    check("din: misspelled aux task is invalid", "aux_tasks" in v.invalid)

    v = blockspec.validate_delta("fm", cur, {"loss_type": "bce"})
    check("value already equal -> ineffective, not effective",
          "loss_type" in v.ineffective and not v.has_effect)

    v = blockspec.validate_delta("fm", cur, {"num_leaves": 63})
    check("unknown key (would be silently dropped) is invalid", "num_leaves" in v.invalid)

    v = blockspec.validate_delta("fm", cur, {"model_type": "din"})
    check("model_type is managed by the harness", "model_type" in v.invalid)

    v = blockspec.validate_delta("din", cur, {"mtl_arch": "mmoe"})
    check("mtl_arch=mmoe rejected before a wasted training launch", "mtl_arch" in v.invalid)

    v = blockspec.validate_delta("lgbm", cur, {"lr": 0.05})
    check("lgbm: lr is not honoured (gbm.train_ranker hardcodes it)", "lr" in v.not_honoured)

    v = blockspec.validate_delta("fm", cur, {"lr": 99.0})
    check("out-of-range lr is invalid", "lr" in v.invalid)

    v = blockspec.validate_delta("fm", cur, {"neg_ratio": 8, "aux_tasks": ["click"]})
    check("mixed delta: effective part survives, ignored part reported",
          v.effective == {"neg_ratio": 8} and "aux_tasks" in v.not_honoured and v.has_effect)

    check("feedback text names the honoured knobs",
          "neg_ratio" in blockspec.validate_delta("fm", cur, {"aux_tasks": ["click"]}).feedback("fm"))


def test_stochastic_declaration():
    print("\n[4] stochastic-family declaration matches the measured evidence")
    check("din is stochastic (sigma ~0.00025 measured)", blockspec.is_stochastic("din"))
    check("fm is deterministic (std 0.00000 over 24 nodes)", not blockspec.is_stochastic("fm"))
    check("lgbm is deterministic (std 0.00000 over 14 nodes)", not blockspec.is_stochastic("lgbm"))


if __name__ == "__main__":
    try:
        test_static_declarations()
        test_validate_delta()
        test_stochastic_declaration()
        test_runtime_effect()
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
    sys.exit(1 if failures else 0)
