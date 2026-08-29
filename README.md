# Techjam-2026 — Autonomous ML Agent

An autonomous ML research agent that iterates on the KuaiRand-Pure
recommendation challenge on its own — proposing a hypothesis, rewriting
code, running it, scoring it, and deciding whether to keep the change —
for a bounded number of iterations, with no human in the loop.

## Repo layout

```
TECHJAM/
  kuairand-starter-kit/   the organizer's starter kit, unmodified
    data.py, baseline.py, evaluate.py, submit.py, ablation_features.py
    baseline_scores.json    official baseline numbers + convergence rule
    KuaiRand-Pure/           the dataset (not tracked by git — see below)

  agent-recsys/           the agent itself — everything you actually run
    run_agent.py             CLI entrypoint
    agent/                   controller, LLM client, sandbox, convergence
    pipeline/                a seeded copy of the starter kit; the two
                              files the agent is allowed to rewrite
                              (data.py, baseline.py) live here
    logs/                    written at runtime — iteration_log.jsonl,
                              baseline_reproduction.json,
                              resource_summary.json
    best/                    validation-best snapshot, written at runtime
    hypothesis-ledger.html   drag-and-drop log viewer — see "Reading the
                              logs" below
    README.md                the full reference for run_agent.py's flags,
                              prerequisites for each LLM backend, and the
                              hidden-test-set compliance guarantees
```

`agent-recsys/README.md` is the detailed reference (every CLI flag, exact
prerequisites for the Anthropic API vs. a local Ollama model vs.
`--dry-run`, resuming across runs, and so on). This file is the shorter
starting point: how the pieces fit together, and how to read what a run
produced.

## Starting guide

1. **Get the dataset.** `kuairand-starter-kit/KuaiRand-Pure/` isn't
   tracked in git (it's tens of MB of CSVs) — download it separately and
   place it so the layout above holds, i.e. `kuairand-starter-kit/` and
   `agent-recsys/` sit side by side under `TECHJAM/`. `run_agent.py`'s
   `--data_dir` defaults to `../kuairand-starter-kit/KuaiRand-Pure/data`
   relative to `agent-recsys/`, so with that layout you don't need to pass
   it at all.

2. **Install the common dependency.** From inside `agent-recsys/`:
   ```
   pip install numpy
   ```
   (numpy is the only third-party package the pipeline itself needs —
   everything else below is specific to which LLM backend you pick.)

3. **Pick one LLM backend** and install just its extra piece:

   | Backend | Extra install | Extra setup | Cost |
   |---|---|---|---|
   | `--dry-run` | none | none | free — no LLM is called, fixed hypothesis list |
   | Anthropic API (default) | `pip install anthropic` | `export ANTHROPIC_API_KEY=sk-...` | real per-token spend, capped by `--max_cost_usd` |
   | Local Ollama (`--local_model`) | `pip install requests` | `ollama serve` running + a model pulled, e.g. `ollama pull qwen2.5-coder:7b` | free, but much slower and a weaker coder — see `agent-recsys/README.md`'s Ollama section before an unattended run |

4. **Run it.** From inside `agent-recsys/`:
   ```
   python run_agent.py --dry-run --max_iterations 3        # sanity check first
   python run_agent.py --max_iterations 50                 # real run, Anthropic API
   python run_agent.py --local_model qwen2.5-coder:7b --candidates_per_iteration 1   # real run, local Ollama
   ```
   Before iteration 0 of a fresh run, it reproduces the official baseline
   once (no LLM call) and prints/writes the comparison to
   `logs/baseline_reproduction.json` — this is the "confirm it matches the
   official numbers" checkpoint, not something you run separately. The run
   then stops on its own once the organizer's convergence rule triggers
   (validation primary not improving by more than 0.002 over 3
   iterations), the iteration cap, or the wall-clock cap — whichever comes
   first — and automatically finalizes: it scores the hidden test set
   exactly once, from the validation-best snapshot, and writes
   `submission.csv`.

## Architecture

The agent is a deterministic Python controller wrapped *around* the
starter kit, not a rewrite of it — the starter kit's own `data.py`,
`baseline.py`, `evaluate.py` are the seed, and the loop below only ever
touches copies of the first two.

```
                 ┌─────────────────────────────────────────┐
                 │              run_agent.py                │
                 │         (CLI flags, kill switches)        │
                 └───────────────────┬───────────────────────┘
                                      ▼
                 ┌─────────────────────────────────────────┐
   iteration N    │            agent/controller.py            │   one loop,
   ┌───────────►  │  the outer loop: propose → apply → run →  │   repeated
   │              │  score → adopt-or-discard → log            │   until stop
   │              └──┬──────────┬───────────┬──────────┬──────┘
   │                 ▼          ▼           ▼          ▼
   │        context.py   llm_client.py  sandbox.py  convergence.py
   │        builds the   Anthropic /    snapshot →   epsilon/N rule,
   │        prompt: data  Ollama /      apply the     iteration cap,
   │        profile, dead DryRun client  one-file      wall-clock cap,
   │        ends, ranked  → Proposal    rewrite →      cost/token cap
   │        headroom,     (hypothesis,  run under a
   │        bounded       new file      timeout →
   │        history       content)      promote only
   │                                    if it ran clean
   └── worse result discarded, logged with "adopted": false,
       next iteration still reasons from the best-known code ──┘
                                      │
                                      ▼ (once the loop stops)
                          agent/finalize.py — runs ONCE
                     best/ snapshot → real predictions on the
                     hidden test set → submission.csv, validated
                     exactly like the starter kit's `submit.py --check`
```

Two things are structural, not just documented behavior:

- **Only `data.py` and `baseline.py` are ever rewritten**, one full-file
  rewrite per iteration. `evaluate.py` (official scoring) is never
  touched. See `pipeline/README.md` for the exact
  `data.load()` / `baseline.run_fm()` contract that keeps the controller
  decoupled from whatever model the agent ends up with.
- **The hidden test set is never scored during iteration.**
  `pipeline/_run_iteration.py` pops the `'test'` split out of the data
  *before* calling `run_fm()`, so a rewritten `baseline.py` can't compute
  a test score mid-run even if it tries to — the rows simply aren't
  there. `agent/sandbox.py` independently raises `HiddenTestViolation` and
  stops the run outright if any iteration's output carries a `'test'` key
  at all. The only place the hidden test set is ever touched is the
  one-time `finalize.py` step above, after the loop has already stopped.

Every iteration's result is logged whether or not it was adopted — a
worse-scoring hypothesis is kept in the log as a record but the *code*
reverts, so the next iteration always builds on the best-known state
rather than compounding a regression.

## Reading the logs

A run writes three files into `agent-recsys/logs/`:

- **`iteration_log.jsonl`** — one JSON object per line, one line per
  iteration. This is the actual run history. Key fields per record:

  | Field | Meaning |
  |---|---|
  | `iteration` | slot number within this invocation (resets to 0 each time you run `run_agent.py`, unless you're resuming — see below) |
  | `hypothesis` | the LLM's own stated reasoning for this change |
  | `target_file` | `data.py` or `baseline.py` |
  | `status` | `"ok"` or `"failed"` |
  | `metrics.valid.primary` (+ `GAUC`, `nDCG@5`) | the validation score this iteration actually got |
  | `expected_delta` / `actual_delta` | what the model predicted it would gain vs. what it actually got |
  | `reference_primary` | the best score this iteration was judged against |
  | `adopted` | whether it beat `reference_primary` and was promoted into the live pipeline |
  | `code_diff` | a unified diff of the code change applied — this is the actual deliverable the challenge brief asks for |
  | `resource_usage` | LLM input/output tokens for this iteration |
  | `manual_intervention` | `false` unless you stepped in by hand |
  | `final_error` | present only on `"failed"` — the traceback fed back to the model for its next repair attempt |

  A regression is still a real line in this file with `"adopted": false`
  — it's evidence of what was tried and ruled out, not noise to filter
  out.

- **`baseline_reproduction.json`** — the one-time check, before iteration
  0 of a fresh run, that the untouched seeded pipeline reproduces the
  organizer's official validation number. `"ok": false` here (with a
  traceback) means the harness itself failed to even run the baseline —
  worth fixing before trusting anything downstream in that run.

- **`resource_summary.json`** — the run's totals: `llm_input_tokens`,
  `llm_output_tokens`, `wall_clock_hours`, `iterations_run`. This is the
  Feasibility/Practicality number set, at a glance.

### Visualizing the log: `hypothesis-ledger.html`

`agent-recsys/hypothesis-ledger.html` is a single self-contained HTML
file — no server, no build step. To use it:

1. Open it directly in a browser (double-click it, or right-click →
   *Open with* → your browser).
2. Drag `agent-recsys/logs/iteration_log.jsonl` onto the page (or use the
   file picker it shows) — it never uploads the file anywhere, it's read
   entirely client-side with `FileReader`.

It then renders:

- a validation-primary-by-iteration line chart, so you can see the score
  actually climbing (or not) across the run,
- a ranking-quality-components chart (GAUC / nDCG@5 side by side),
- spend-per-iteration and wall-clock-per-iteration charts,
- a sortable metrics table, and
- the full iteration log (hypothesis, diff, adopted/rejected, errors) in
  one scrollable view instead of raw JSON lines.

It needs one-time internet access to load its charting library
(Chart.js, from cdnjs) and a Google Font — if that's blocked, the charts
won't render but the metrics table and iteration log below them still
work fully offline.

## Resuming across runs

`best/_best_meta.json` records the score, iteration, hypothesis, and
timestamp of whatever's currently saved in `best/`. The next time you run
`run_agent.py` — a new process, possibly hours or days later — it reads
that file and resumes from it, restoring both the score-to-beat and the
actual `data.py`/`baseline.py` content before iteration 0. See
`agent-recsys/README.md`'s "Resuming across runs" section for the full
detail.
