"""Agent CLI entrypoint.

    python -m agent.run --smoke     # smoke gate: build cache + reproduce the FM baseline
    python -m agent.run --mock      # the full loop offline via the scripted MockDriver
    python -m agent.run --faults    # fault injection: crash, recover, still finalize
    python -m agent.run             # LIVE (needs GEMINI_API_KEY or .env.local)
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import os

try:                                    # Windows consoles default to cp1252; keep logs UTF-8-safe
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Auto-load GEMINI_API_KEY from .env.local if it exists
if Path(".env.local").exists():
    with open(".env.local", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                if key.strip() == "GEMINI_API_KEY":
                    os.environ["GEMINI_API_KEY"] = val.strip().strip("'\"")

from agent import guardrails, datced
from agent.config import Config

FM_VALID_PRIMARY = 0.6015     # seed-0 reference measured on this machine
TOL = 0.003


def run_root_node(cache_dir: str, out: str, blocks: str = "pipeline/baseline_blocks",
                  cfg_path: str | None = None, extra_split: str | None = None):
    cmd = [sys.executable, "-m", "pipeline.run_node",
           "--blocks", blocks, "--out", out, "--cache", cache_dir]
    if cfg_path:
        cmd += ["--cfg", cfg_path]
    if extra_split:
        cmd += ["--extra-split", extra_split]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r


def smoke(cfg: Config):
    print("[M0] verifying frozen harness ...")
    print("   ", guardrails.ensure_frozen(create=True))

    print("[M0] building / loading DataBundle cache ...")
    t0 = time.time()
    meta = datced.build_or_load(cfg.data_dir, cfg.cache_dir)
    print("    sizes:", meta["sizes"], f"dim={meta['dim']}", f"({time.time()-t0:.1f}s)")

    out = str(Path(cfg.runs_dir) / "smoke_root")
    print("[M0] running root node (baseline blocks) ...")
    r = run_root_node(cfg.cache_dir, out)
    if r.returncode != 0:
        print(r.stdout[-800:]); print("STDERR:", r.stderr[-2000:])
        sys.exit(1)

    m = json.loads((Path(out) / "metrics.json").read_text())
    pv = m["primary_valid"]
    print(f"    root: GAUC {m['GAUC']:.4f} | nDCG@5 {m['nDCG@5']:.4f} | "
          f"primary_valid {pv:.4f} | best_epoch {m['best_epoch']} | {m['seconds']}s")
    ok = abs(pv - FM_VALID_PRIMARY) < TOL
    print(f"[M0] GATE: reproduce FM {FM_VALID_PRIMARY} +/- {TOL}  ->  "
          f"{'PASS' if ok else 'FAIL'} (got {pv:.4f})")
    sys.exit(0 if ok else 2)


def autorun(cfg: Config, mock: bool, max_iter):
    from agent import orchestrator
    if mock:
        from tests.mock_moves import build_moves, build_fault_moves
        from agent.llm.driver import MockDriver
        driver = MockDriver(build_fault_moves() if mock == "faults" else build_moves())
        cfg.phases.explore_p = 0.0          # deterministic parent selection for the scripted demo
    else:
        from agent.llm.gemini import GeminiDriver
        driver = GeminiDriver(max_retries=cfg.llm.max_retries)
    run_dir, best, final_valid = orchestrator.run(cfg, driver, max_iter=max_iter)
    beat = final_valid - 0.6015
    print(f"\n=== run complete ===\n  dir: {run_dir}\n  final valid primary: {final_valid:.4f} "
          f"(best single {best.score():.4f}) (d{beat:+.4f} vs FM)  ->  "
          f"{'BEATS BASELINE' if beat > 0 else 'no gain'}")


def main():
    ap = argparse.ArgumentParser(prog="agent.run")
    ap.add_argument("--config", default="agent/config.yaml")
    ap.add_argument("--smoke", action="store_true", help="M0 gate: reproduce the FM baseline")
    ap.add_argument("--mock", action="store_true", help="run the loop with the scripted MockDriver (no API)")
    ap.add_argument("--faults", action="store_true", help="run the fault-injection robustness script")
    ap.add_argument("--max-iter", type=int, default=None, help="override the iteration cap")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    if a.smoke:
        smoke(cfg)
    else:
        autorun(cfg, mock=("faults" if a.faults else a.mock), max_iter=a.max_iter)


if __name__ == "__main__":
    main()
