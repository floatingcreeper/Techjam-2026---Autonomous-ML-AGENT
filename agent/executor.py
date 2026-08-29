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


def check_imports(source: str):
    """Fail-fast static gate on an agent-written block (syntax + import allowlist)."""
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
