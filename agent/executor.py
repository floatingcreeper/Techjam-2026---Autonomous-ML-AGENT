"""Sandboxed node runner + import allowlist + failure classification."""
from __future__ import annotations
import subprocess, sys, os, json, time, ast
from pathlib import Path


def utf8_env():
    """Force child processes to use UTF-8 stdio (Windows pipes default to cp1252)."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env

ALLOWLIST = {
    "numpy", "np", "math", "torch", "lightgbm", "scipy", "sklearn",
    "pipeline", "data", "evaluate", "json", "time", "collections",
    "itertools", "random", "dataclasses", "typing", "functools",
}


class Failure:
    def __init__(self, kind: str, detail: str):
        self.kind = kind          # code | timeout | numerical
        self.detail = detail

    def __repr__(self):
        return f"Failure({self.kind}: {self.detail[:80]!r})"


# docs/EN/SYSTEM.md §8 -- holdout access guard.
# The data interface already withholds hidden-test labels and the near-oracle `is_click` proxy
# (agent/datced.py v7 keeps them under runs/_holdout/). This is the second layer: an agent-written
# block must not reach around the interface to the holdout directory or the raw logs. Matching is on
# string literals, which is what a generated block would realistically use.
FORBIDDEN_LITERALS = (
    "_holdout",            # the holdout directory itself
    "test_y", "test_aux",  # hidden-test labels and their aux proxy
    "valid_aux",           # is_click for the split every decision is made on
    "KuaiRand-Pure",       # the raw logs, which contain every label
    "log_standard", "log_random",
)
# Builtins that would let a block read a file outside the numpy/cache interface.
FORBIDDEN_CALLS = ("open", "eval", "exec", "compile", "__import__", "globals", "getattr_static")


def check_imports(source: str):
    """Fail-fast static gate on an agent-written block: syntax, import allowlist, holdout access.

    Returns (ok, reason). Runs BEFORE the block is executed, so a rejected edit costs no training.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] not in ALLOWLIST:
                    bad.append(a.name)
        elif isinstance(n, ast.ImportFrom):
            root = (n.module or "").split(".")[0]
            if root and root not in ALLOWLIST:
                bad.append(n.module)
    if bad:
        return False, f"disallowed imports: {bad}"

    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            for lit in FORBIDDEN_LITERALS:
                if lit in n.value:
                    hits.append(f"string literal {n.value[:60]!r} references holdout data ({lit})")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id in FORBIDDEN_CALLS:
                hits.append(f"call to {n.func.id}() is not permitted in a pipeline block")
    if hits:
        return False, ("holdout/access guard: " + "; ".join(sorted(set(hits))[:3]))
    return True, ""


def run_node(blocks_dir, out_dir, cfg_path, cache_dir="runs/_cache",
             timeout_s=900, extra_split=None):
    """Run a node in an isolated subprocess. Returns (metrics_dict | Failure, wall_clock_s)."""
    cmd = [sys.executable, "-m", "pipeline.run_node",
           "--blocks", str(blocks_dir), "--out", str(out_dir),
           "--cfg", str(cfg_path), "--cache", cache_dir]
    if extra_split:
        cmd += ["--extra-split", extra_split]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                           encoding="utf-8", errors="replace", env=utf8_env())
    except subprocess.TimeoutExpired:
        return Failure("timeout", f">{timeout_s}s"), time.time() - t0
    wc = time.time() - t0
    if r.returncode != 0:
        return Failure("code", (r.stderr or r.stdout or "")[-1800:]), wc
    mp = Path(out_dir) / "metrics.json"
    if not mp.exists():
        return Failure("code", "no metrics.json; stdout=" + (r.stdout or "")[-600:]), wc
    m = json.loads(mp.read_text())
    pv = m.get("primary_valid")
    if pv is None or (isinstance(pv, float) and pv != pv):     # None / NaN
        return Failure("numerical", "primary_valid is None/NaN"), wc
    return m, wc


def debug_gate(blocks_dir, cfg, cache_dir, scratch_dir, n_train=20000, n_other=10000, epochs=2):
    """Fast-fail sample run on a subsampled cache via the existing frozen runner. Returns a
    metrics dict on success or a Failure the caller routes to recovery -- never runs the full pipeline."""
    from pipeline import debug_cache
    dbg_cache = debug_cache.build(cache_dir, str(Path(scratch_dir) / "cache"), n_train, n_other, seed=cfg.seed)
    c = cfg.replace(epochs=min(int(cfg.epochs), epochs), patience=1)
    cp = Path(scratch_dir) / "cfg.json"; c.to_json(cp)
    res, _ = run_node(blocks_dir, str(Path(scratch_dir) / "out"), cp, dbg_cache, timeout_s=180)
    if isinstance(res, Failure):
        return res
    pv = res.get("primary_valid")
    if pv is None or not (0.0 <= float(pv) <= 1.0):            # sanity gate, not a quality gate
        return Failure("numerical", f"debug sample primary_valid out of range: {pv}")
    return res
