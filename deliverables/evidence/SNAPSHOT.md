# Evidence snapshot

Taken: **2026-08-30 14:26:22**  
Iteration records in this snapshot: **55**

> **A loop appears to have been running when this snapshot was taken.** These files are therefore a mid-run copy, not a finished run. Re-run `python refresh_deliverables.py` once it stops.

Copies of the live run artifacts from `runs/`. The originals stay in `runs/` because a
running loop writes to them; regenerate these with `python refresh_deliverables.py`.

| File | What it is |
|---|---|
| `experiment_log.jsonl` | REQUIRED DELIVERABLE: one record per iteration - hypothesis, code_diff, metrics, error_events, decision, token_cost |
| `solution_tree.json` | the search tree: every node, its status, score and parent |
| `token_ledger.jsonl` | every LLM call with token counts and latency |
| `manual_interventions.jsonl` | human interventions recorded during autonomous operation |
| `state.json` | crash-safe resume state: last completed iteration + current best |
| `dashboard.html` | human-readable run viewer (regenerate with: python -m agent.viewer) |
| `cost_report.txt` | LLM calls, tokens in/out, latency, GPU-hours, by caller |
