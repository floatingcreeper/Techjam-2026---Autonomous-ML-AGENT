"""Snapshot the live run artifacts in runs/ into deliverables/evidence/.

Why a snapshot rather than moving the real files: the agent loop WRITES to runs/ continuously —
experiment_log.jsonl and solution_tree.json are appended after every iteration, and state.json is
what crash-safe resume reads. Moving or symlinking them would break a running loop. So
deliverables/evidence/ holds point-in-time COPIES, and this script refreshes them.

Run it whenever you want the deliverables to reflect the latest run — and always immediately
before packaging or recording the demo:

    python refresh_deliverables.py

It is safe to run while a loop is in progress; you just get a mid-run snapshot, and SNAPSHOT.md
records that fact so nobody mistakes a partial log for a finished one.
"""
import datetime
import io
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, 'runs')
DEST = os.path.join(ROOT, 'deliverables', 'evidence')

# (filename in runs/, what it is) — the required per-iteration log first, since it is the one
# named deliverable rather than supporting evidence.
FILES = [
    ('experiment_log.jsonl', 'REQUIRED DELIVERABLE: one record per iteration - hypothesis, '
                              'code_diff, metrics, error_events, decision, token_cost'),
    ('solution_tree.json', 'the search tree: every node, its status, score and parent'),
    ('token_ledger.jsonl', 'every LLM call with token counts and latency'),
    ('manual_interventions.jsonl', 'human interventions recorded during autonomous operation'),
    ('state.json', 'crash-safe resume state: last completed iteration + current best'),
    ('dashboard.html', 'human-readable run viewer (regenerate with: python -m agent.viewer)'),
]


def _loop_running():
    """Best-effort check for an agent loop in flight, so the snapshot can say so."""
    try:
        if sys.platform == 'win32':
            out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                                  capture_output=True, text=True, timeout=15).stdout
            return out.lower().count('python.exe') > 0
        out = subprocess.run(['pgrep', '-f', 'agent.cli'],
                              capture_output=True, text=True, timeout=15).stdout
        return bool(out.strip())
    except Exception:      # noqa: BLE001 — a failed check must not stop the snapshot
        return None


def main():
    os.makedirs(DEST, exist_ok=True)
    copied, missing = [], []

    for name, _desc in FILES:
        src = os.path.join(RUNS, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(DEST, name))
            copied.append((name, os.path.getsize(src)))
        else:
            missing.append(name)

    # Cost report is generated, not copied — it is a rendering of token_ledger.jsonl.
    cost_path = os.path.join(DEST, 'cost_report.txt')
    try:
        from agent import cost_report
        io.open(cost_path, 'w', encoding='utf-8').write(cost_report.format_report())
        copied.append(('cost_report.txt', os.path.getsize(cost_path)))
    except Exception as e:   # noqa: BLE001
        io.open(cost_path, 'w', encoding='utf-8').write(f"cost report unavailable: {e}\n")

    # Iteration count, so the snapshot header states the run's size without opening the log.
    n_iters = 0
    log = os.path.join(DEST, 'experiment_log.jsonl')
    if os.path.exists(log):
        with io.open(log, encoding='utf-8') as fh:
            n_iters = sum(1 for line in fh if line.strip())

    running = _loop_running()
    warn = ''
    if running:
        warn = ('\n> **A loop appears to have been running when this snapshot was taken.** These '
                 'files are therefore a mid-run copy, not a finished run. Re-run '
                 '`python refresh_deliverables.py` once it stops.\n')
    elif running is None:
        warn = '\n> Could not determine whether a loop was running when this was taken.\n'

    lines = [
        '# Evidence snapshot',
        '',
        f'Taken: **{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}**  ',
        f'Iteration records in this snapshot: **{n_iters}**',
        warn,
        'Copies of the live run artifacts from `runs/`. The originals stay in `runs/` because a',
        'running loop writes to them; regenerate these with `python refresh_deliverables.py`.',
        '',
        '| File | What it is |',
        '|---|---|',
    ]
    for name, desc in FILES:
        mark = '' if name not in missing else ' *(not present at snapshot time)*'
        lines.append(f'| `{name}` | {desc}{mark} |')
    lines.append('| `cost_report.txt` | LLM calls, tokens in/out, latency, GPU-hours, by caller |')
    lines.append('')

    io.open(os.path.join(DEST, 'SNAPSHOT.md'), 'w', encoding='utf-8').write('\n'.join(lines))

    print(f"refreshed deliverables/evidence/  ({n_iters} iteration records)")
    for name, size in copied:
        print(f"  {size:>9,d}  {name}")
    for name in missing:
        print(f"  {'--':>9}  {name} (missing)")
    if running:
        print("  NOTE: a loop looks like it is running - this is a mid-run snapshot.")


if __name__ == '__main__':
    main()
