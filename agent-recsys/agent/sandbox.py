"""
Applies one proposed file change to a throwaway copy of the pipeline and
runs it under a timeout. Never touches the real "current" pipeline directory
directly -- controller.py only promotes a scratch copy to current AFTER this
reports success, which is what makes a broken iteration unable to corrupt
anything.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

TIMEOUT_S = 600  # per-iteration training timeout; generous for this numpy/CPU pipeline (~40s normally)


class HiddenTestViolation(RuntimeError):
    """The per-iteration harness scored the hidden test set.

    This is a compliance failure, NOT a recoverable iteration error, so it
    is raised rather than returned as RunResult(ok=False): there is nothing
    the agent could propose that would fix it, and continuing would keep
    appending log records that violate the challenge rules. The controller
    deliberately does not swallow this into its generic recoverable-error
    path -- see agent/controller.py's _propose_and_run_one_candidate().

    Reaching this means pipeline/_run_iteration.py is no longer stripping
    'test' out of splits before calling run_fm() (its HIDDEN-TEST
    COMPLIANCE note explains why it must). That has already happened once,
    via a git checkout restoring a pre-fix commit, and nothing caught it
    until a manual audit -- which is exactly why this second, BEHAVIOURAL
    check exists alongside the structural one. The structural fix lives in
    a file that something outside this codebase can revert; the resulting
    violation still surfaces here, immediately and loudly, on the very
    next run."""
    pass


@dataclass
class RunResult:
    ok: bool
    metrics: dict | None = None       # {'valid': {...}, 'test': {...}, 'wall_clock_s': float}
    error: str | None = None          # combined stderr/stdout on failure, truncated


def make_scratch_copy(pipeline_dir: Path) -> Path:
    scratch = Path(tempfile.mkdtemp(prefix="kuairand_iter_"))
    for f in pipeline_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, scratch / f.name)
    return scratch


def apply_change(scratch_dir: Path, target_file: str, new_content: str):
    if target_file not in ("data.py", "baseline.py"):
        raise ValueError(
            f"refusing to write to '{target_file}' -- the agent may only "
            f"edit data.py or baseline.py (see pipeline/README.md)"
        )
    # explicit encoding="utf-8": on Windows, write_text() defaults to the
    # OS locale codec (cp1252) which cannot represent an LLM proposal's
    # non-ASCII characters (or the organizer's original Chinese comments,
    # if the agent's rewrite preserves any of them) -- would corrupt the
    # file silently or raise on the NEXT read, not this write.
    (scratch_dir / target_file).write_text(new_content, encoding="utf-8")


def run_scratch(scratch_dir: Path, data_dir: str, timeout_s: int = TIMEOUT_S) -> RunResult:
    try:
        proc = subprocess.run(
            [sys.executable, "_run_iteration.py", data_dir],
            cwd=str(scratch_dir),
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return RunResult(ok=False, error=f"timed out after {timeout_s}s")

    if proc.returncode != 0:
        combined = (proc.stderr or "") + "\n" + (proc.stdout or "")
        return RunResult(ok=False, error=combined.strip()[-4000:])

    # _run_iteration.py prints exactly one JSON line -- take the last
    # non-empty line defensively, in case the agent's code left stray prints
    last_line = ""
    for line in proc.stdout.splitlines():
        if line.strip():
            last_line = line.strip()
    try:
        metrics = json.loads(last_line)
    except json.JSONDecodeError:
        return RunResult(
            ok=False,
            error=(
                "iteration ran without crashing, but did not produce the "
                "required single-line JSON output -- got:\n" + proc.stdout[-2000:]
            ),
        )
    # A 'test' key here means run_fm() scored the hidden test set, which it
    # can only do if it RECEIVED the test split -- i.e. that
    # pipeline/_run_iteration.py is no longer stripping it (see that file's
    # HIDDEN-TEST COMPLIANCE note, and HiddenTestViolation's docstring
    # above for why this redundant check earns its place). Raise rather
    # than return ok=False: it is not something a different proposal could
    # fix, so it must stop the run, not cost the agent a repair attempt.
    if "test" in metrics:
        raise HiddenTestViolation(
            "run_fm() returned test-split metrics during iteration, which "
            "means it was handed the hidden test set. Check that "
            "pipeline/_run_iteration.py still pops 'test' out of splits "
            "before calling run_fm() -- if it doesn't, restore that line "
            "before running again (see its HIDDEN-TEST COMPLIANCE note). "
            f"Offending metrics keys: {sorted(metrics)}"
        )
    # Only 'valid' is required per-iteration: 'test' is never computed
    # during iteration (above), so it must not be required here either.
    if "valid" not in metrics or "primary" not in metrics["valid"]:
        return RunResult(ok=False, error=f"output JSON missing metrics['valid']['primary']: {metrics}")
    return RunResult(ok=True, metrics=metrics)


def promote(scratch_dir: Path, pipeline_dir: Path):
    """Overwrite the current pipeline with the scratch copy's contents.
    Only ever called after run_scratch() reported ok=True."""
    for f in scratch_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, pipeline_dir / f.name)


def snapshot_to(scratch_dir: Path, dest_dir: Path):
    """Used to save the best-so-far pipeline into best/, independent of
    whatever 'current' drifts to afterward."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in scratch_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, dest_dir / f.name)


def cleanup(scratch_dir: Path):
    shutil.rmtree(scratch_dir, ignore_errors=True)
