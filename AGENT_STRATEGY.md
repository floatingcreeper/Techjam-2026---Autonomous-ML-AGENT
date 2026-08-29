# Agentic Research Loop — Strategy Outline

Status: **draft, v0.2** — planning only, no implementation yet. Living document; refined as
studies/prompts are fed in. Each refinement appends to the Changelog rather than rewriting history.

Design reference for this revision: Yang et al., "R&D-Agent: An LLM-Agent Framework Towards
Autonomous Data Science," Microsoft Research/GenAI, 2025, arXiv:2505.14738. Adopting 4 of their 6
components per their own ablations (most impactful for a resource-constrained setting). Explicitly
**not** adopting: (5) the probabilistic cross-branch memory kernel, (6) full MCTS/parallel branch
search — both more engineering than this timeline justifies. In place of (5): a flat, append-only
history fed into the propose prompt's context. In place of (6): a simple greedy loop that always
extends current-best (no branching structure exists in the repo today, so there's nothing to
preserve).

## Goal (unchanged from v0.1)

Autonomously improve `primary = mean(GAUC, nDCG@5)` on KuaiRand-Pure beyond the `fm_official`
reference, via a repeated loop, deciding only on **valid**, never touching `evaluate.py`, never
fitting anything on valid/test. `oracle_ceiling` in `baseline_scores.json` is the real ceiling.

## Repo state as of this revision

- **Built and stable**: data loading + official date-range split + categorical encoding (`data.py`);
  baselines random/pop/FM (`baseline.py`) with recorded reference scores (`baseline_scores.json`);
  fixed scoring contract (`evaluate.py`); submission make/check/score (`submit.py`); a secondary
  standalone ablation harness (`ablation_features.py`) that duplicates a chunk of `data.py`'s
  raw()/vocab logic inline to test CWM's extra 13 fields.
- **Missing entirely**: any agent loop code, any LLM client/wiring (grep confirms zero hits for
  `openai`/`anthropic`/API-key patterns anywhere in the repo), any config/action-space abstraction
  (today, "changing the model" means editing `baseline.py` or writing a new copy-pasted script the
  way `ablation_features.py` does), any logging module, any notebooks.
- **Compute**: numpy-only, CPU-only. No torch/CUDA dependency exists today.
- **Test-split fact**: the local `'test'` split (20220429–20220508) has real, unmasked `long_view`
  labels sitting in the committed CSV right now — `data.load()` returns them like any other split.
  `submit.py`'s own docstring self-describes test scoring as local-practice-only, implying the real
  graded evaluation is external — but that boundary is not enforced as code anywhere yet. See open
  question Q2.

## The four adopted strategies → this repo

### 1. Debug-first coding workflow (highest priority, implemented first)
Gate in front of every full training run: sample small, run the full pipeline
(features → train → eval) at reduced epochs, time it, sanity-check the output (no NaN/inf,
`primary` in `[0,1]`, clean exit), extrapolate full-run duration, and only then commit to the full
run. A failed debug run routes straight to the error/repair loop (strategy 4) instead of ever
reaching a full-scale run.

Slots into the loop as a **gate in front of the existing Execution step**, not a new top-level
phase: `Execution Agent` becomes `debug_run()` → (if ok) `full_run()`.

Sampling: **fixed small N (proposed 20,000 rows)**, not "10% of train" — train is large enough that
10% is still a heavy, slow sample. Trade-off: fixed-N is cheaper and more predictable in wall-clock,
but a small fixed sample may under-exercise vocab-size/UNK-rate-dependent bugs that only show up at
real scale. Flagged as a decision (§ Decisions made unilaterally).

### 2. Structured reasoning/hypothesis pipeline (implemented second)
Every propose step is forced through 4 explicit fields, not a free-form "improve the model" prompt:
`{hypothesis, target_stage, reasoning, expected_effect}` — problem/bottleneck identification folds
into `reasoning`. This structured output **is** the run-log's hypothesis content, not a separate
artifact (ties directly into the hard requirement for one structured log entry per iteration).
LLM output is parsed and validated (all 4 fields present, non-empty); missing fields → retry with
the validation error appended to the prompt, capped at 2 retries, then logged as a failure event.

`target_stage` controlled vocabulary: `feature | architecture | hyperparam | regularization |
sampling | loss | other` (renames v0.1's `change_type` — same concept, paper's/user's term adopted).

### 3. Time-aware dynamic planning (implemented third)
Single threshold, iteration-based by default (wall-clock budget currently unconfirmed — see Q3):
`budget_tier(iteration, expected_total_iterations)` → `"explore"` below the threshold, `"exploit"`
above it. Explore: cheap/fast/novel/single-change ideas only, ensembling and expensive sweeps
explicitly forbidden in the prompt. Exploit: refine current-best, larger search allowed.

Implemented as **prompt-variant selection** (explore vs. exploit prompt files), not a numeric
budget-tier parameter threaded everywhere — simpler, matches "keep it to one threshold." A hook for
also constraining `action_space.py` by tier is left in place so a numeric constraint can be added
later without restructuring.

### 4. Simplified aggregated evaluation + error/recovery (alongside the loop)
**Recovery**: any failure (guardrail rejection, debug-run not-ok, full-run crash/NaN) → feed the
concrete error back to the LLM, ask for a fix, cap at 3 repair attempts, then roll back to
current-best (a no-op, since current-best is never overwritten until a clean accept) and move to
the *next* iteration with a fresh hypothesis rather than retrying the same one indefinitely. Logged
as `error_events` on the iteration record.

**Aggregated eval, simplified**: no multi-branch merge — but before accepting a new current-best,
optionally re-run with 1–2 additional seeds and require the seed-mean (not a single seed) to beat
current-best by `epsilon`. Toggleable (`RECHECK_TOP_CANDIDATE`), **default on**: `fm_official`'s own
std across seeds (~0.0008) is close enough to the convergence `epsilon` (0.002) that a single-seed
accept is genuinely likely to be noise.

## Explicitly out of scope
No cross-branch/multi-trace memory (flat append-only log instead). No tree search/MCTS/parallel
branches (plain greedy loop — nothing in the repo today has branching to preserve). No RAG/external
knowledge base (confirmed nothing RAG-like exists in the repo currently).

## Hard requirements (ours, not from the paper) and where they land
- **One wrapped LLM client, logs input/output tokens** → `agent/llm_client.py`; every Hypothesis /
  Analysis / Coding / Repair call goes through it; running ledger at `runs/token_ledger.jsonl`.
- **One structured JSONL log entry per iteration** (hypothesis, code diff, metrics, error/recovery
  events, token cost, GPU time) → `runs/experiment_log.jsonl`, written by `agent/archivist.py`. This
  **supersedes v0.1's three-file-per-run schema** (`hypothesis.json`/`metrics.json`/`analysis.json`)
  — those are consolidated into one record; raw stdout/code snapshots remain as supplementary,
  non-"log" artifacts under `runs/<NNNN>/`. GPU time: repo is CPU-only today, so this field is
  logged as `0.0` / wall-clock CPU time with a note, not a real GPU metric, unless that changes.
- **Manual-intervention counter, human-incremented only** → `agent/manual_intervention.py`,
  `runs/manual_interventions.jsonl`; nothing in the automated loop calls it.
- **Hidden test split physically unreachable by agent code** → `agent/data_guard.py`: wraps
  `data.load()` and **drops the `'test'` key entirely** before returning splits to any agent-facing
  code path. Every loop entrypoint (debug_run, full_run) imports data only through this wrapper,
  never through `data.load()` directly. `submit.py`'s `--score --split test` remains a manual,
  human-run command outside the orchestrator's reach. Whether this loader-level boundary is
  sufficient depends on Q2 below.
- **Crash-safe resume** → `runs/state.json` (`{last_completed_iteration, current_best_run_id}`),
  written atomically (temp file + rename) as the last step of `agent/archivist.py`, so a mid-run
  crash before archiving completes replays that iteration on restart rather than skipping or
  double-counting it.

## Phased implementation plan (file paths)

**Phase 0 — Foundation** (blocks every later phase)
- `agent/__init__.py`
- `agent/llm_client.py` — the one wrapped client; `call(system, messages, **kw) -> (text, usage)`;
  appends to `runs/token_ledger.jsonl`.
- `agent/action_space.py` — v0 action types: toggle a field in/out from a **fixed known list**
  (reuse, don't re-derive — see reconciliation note below), tweak FM hyperparams (`k`, `lr`, `l2`,
  `epochs`, `patience`), swap among a small fixed set of pre-built model variants. Not free-form
  code-gen yet — keeps Guardrail checks and Coding Agent blast radius tractable. Flagged decision.
- `models/base.py` — common interface `train(splits, config) -> {'valid':…, 'test':…}`.
- `models/fm_v0.py` — thin adapter wrapping `baseline.FM`/`baseline.run_fm` (current-best @
  iteration 0). Wrap, don't reimplement the Adam/early-stopping loop.
- `agent/data_guard.py` — `load_train_valid(data_dir)`; drops `'test'` before returning.
- `agent/logging_schema.py` — the merged per-iteration record (see Hard requirements above).
- `agent/resume.py` — `runs/state.json` read/write, atomic.
- `agent/manual_intervention.py` — human-triggered append-only counter.
- `.gitignore` — add `runs/` (keep `runs/.gitkeep`).

**Phase 1 — Debug-first workflow** (strategy 1)
- `agent/debug_run.py` — fixed-N sample, reduced epochs, timed, output-sanity-checked, extrapolates
  full-run duration; returns `DebugResult{ok, reason, estimated_full_runtime_s, sample_metrics}`.
- `agent/orchestrator.py` (skeleton) — wires debug_run as the gate in front of full_run.

**Phase 2 — Structured hypothesis pipeline** (strategy 2)
- `agent/prompts/hypothesize.md` — 4-stage forced-JSON template.
- `agent/hypothesis_agent.py` — calls `llm_client`, parses/validates the 4 fields, retries (cap 2)
  on validation failure, logs `hypothesis_generation_failed` on exhaustion.
- `agent/logging_schema.py` update — hypothesis sub-object = literally these 4 fields.

**Phase 3 — Time-aware planning** (strategy 3)
- `agent/budget.py` — `budget_tier(iteration, expected_total_iterations)`, single threshold
  constant at top of file (proposed default `0.6`); `wall_clock_tier(...)` stubbed for later.
- `agent/prompts/hypothesize_explore.md`, `agent/prompts/hypothesize_exploit.md`.
- `hypothesis_agent.py` takes `tier`, selects prompt variant, passes a cost hint into
  `action_space.py` validation.

**Phase 4 — Error/recovery + simplified aggregated eval** (strategy 4)
- `agent/error_recovery.py` — repair-prompt loop, cap `MAX_REPAIR_ATTEMPTS = 3`, rollback + advance
  to next iteration on exhaustion.
- `agent/prompts/repair.md`.
- `agent/logging_schema.py` update — `error_events: list[{attempt, error_text, repaired}]`.
- `agent/reeval.py` — optional multi-seed recheck before accept, `RECHECK_TOP_CANDIDATE` flag
  (default on).

**Phase 5 — Orchestrator wiring**
- `agent/orchestrator.py` (complete) — full loop per the diagram below, reads `resume.py` state.
- `agent/archivist.py` — writes `runs/<NNNN>_<slug>/{stdout/, code_snapshot/}` (supplementary) +
  appends the merged record to `runs/experiment_log.jsonl` + updates `runs/state.json` (last step).
- `agent/cli.py` — `python -m agent.cli run [--iterations N | --budget-seconds N]`,
  `python -m agent.cli note-intervention "..."`, `python -m agent.cli status`.

## Updated loop diagram (supersedes v0.1's step 4)

```
Orchestrator → Hypothesis (2-stage prompt via budget tier, §2/§3)
             → Coding (constrained action space, §Phase 0)
             → Guardrail (static: evaluate.py untouched? train-only fitting? positional contract?
                           reads data only via data_guard.load_train_valid?)
             → debug_run() [NEW, §1]  → not ok → error_recovery() (repair loop, §4) → back to Guardrail
                                       → ok, with estimated_full_runtime_s logged
             → full_run() (multi-seed)
             → (optional) reeval() multi-seed recheck before accept [§4]
             → Logging → merged iteration record
             → Orchestrator accept/reject (valid primary vs. epsilon; test logged, never decisive)
             → Archivist (runs/<NNNN>/, experiment_log.jsonl, state.json)
             → convergence check (epsilon=0.002, N=3 accepted) → loop or stop
```

## Reconciliation with existing code (conflicts / duplication)

- **`ablation_features.py` already contains a field-joining action space** (music_id, video_type,
  upload_type, user-side CWM fields) inline. `action_space.py`'s field-toggle action must reuse
  this list, not re-derive a third copy (CLAUDE.md already flags `raw()`/`FIELDS` positional drift
  as a real risk with two copies; a third would compound it). Proposed: extract the CWM-13-fields
  joining logic out of `ablation_features.py` into a shared `data.py` helper both it and
  `action_space.py` call. This touches a file CLAUDE.md currently documents as *deliberately*
  standalone — flagged as a decision needing explicit sign-off, not done unilaterally.
- **v0.1's three-run-schema-files design is superseded** by the new hard requirement for one
  structured log entry — see Hard requirements above. Not a real conflict, just an explicit
  supersession worth calling out since v0.1 is still the only prior written plan.
- **v0.1's open question "Mapping to Claude Code mechanics" is resolved** by this revision: hybrid —
  deterministic Python modules for Execution/Logging/Guardrail/Archivist/Resume/data-guard, routed
  through the one wrapped LLM client for Hypothesis/Analysis/Coding/Repair. Chosen because the
  "wrapped client that logs tokens" requirement is most directly satisfiable as a literal client
  object with an HTTP/SDK call boundary, and it gives crash-safe resume + unattended-run behavior
  more directly than driving everything through interactive subagent calls.
- No RAG, cross-branch memory, or MCTS/branching code exists anywhere in the repo today — nothing
  to remove or reconcile there.

## Clarifying questions (blocking concrete parameter/architecture choices)

- **Q1 — RESOLVED, implemented**: `agent/llm_client.py` is built and smoke-tested against a local
  Ollama server running `qwen2.5-coder:7b` (32k context, confirmed reachable at
  `http://localhost:11434`). Uses Ollama's *native* `/api/chat` (not the OpenAI-compat shim) because
  it returns exact `prompt_eval_count`/`eval_count` token counts on every response — no manual
  tokenizer bookkeeping needed. `call(system, messages, *, json_mode, temperature, caller, timeout)
  -> (text, usage)`; every call, success or failure, is appended to `runs/token_ledger.jsonl`. No
  new dependency added — stdlib `urllib`, keeping the repo's numpy-only footprint (`requests` would
  have been simpler but breaks that). `json_mode=True` (Ollama's `format: "json"`) verified working
  — this is what strategy 2's hypothesis pipeline will lean on, since a 7B model is not reliable at
  freeform "return exactly these fields" instructions without it. Host/model overridable via
  `OLLAMA_HOST` / `AGENT_LLM_MODEL` env vars, defaults match the confirmed local setup. Verify with
  `python -m agent.llm_client`.
- **Q2 — What "hidden test split" actually refers to**: is the local `'test'` date-range split in
  `data.py` (real labels already sitting in the committed CSV) the thing that must be walled off —
  i.e. is it itself the graded hackathon set, so the isolation must be genuinely enforced — or is it
  a local-practice-only split, with the real graded/hidden set being a separate, not-yet-present
  external file or service? This changes whether `data_guard.py`'s loader-level key-drop is
  sufficient or whether a stronger boundary (separate process, no filesystem access to that CSV at
  all from agent code) is needed.
- **Q3 — Rough iteration/time budget for this hackathon** (order of magnitude only — e.g. "~20
  iterations over a few hours" vs. "~5 iterations total"): needed to size the `budget_tier` default
  threshold (currently a placeholder `0.6`) and the debug-run sample size sensibly rather than
  arbitrarily. Not asking for a precise schedule — you already said keep it to one threshold.

## Decisions made unilaterally (veto any of these)

1. Hybrid architecture (deterministic Python driver + one wrapped LLM client for judgment steps) —
   see Reconciliation above.
2. Action space constrained to a config-driven set (field toggles from a known list, hyperparams, a
   handful of pre-built model variants), not free-form code generation, at least for v0.
3. Debug-first sampling uses a **fixed small N (20,000 rows)**, not "10% of train" — see §1 tradeoff.
4. Hidden-test isolation implemented as "loader never returns the `'test'` key," not a
   filesystem/process-level sandbox — pending Q2.
5. `target_stage` reuses/renames v0.1's `change_type` field rather than adding a second taxonomy.
6. Time-aware planning implemented via prompt-variant selection, not a numeric tier parameter
   threaded through every module (a hook for the latter is left in `action_space.py` for later).
7. Aggregated re-eval (§4b) defaults **on**, given `fm_official`'s std is close in magnitude to the
   convergence epsilon — fully toggleable per your requirement.
8. v0.1's three-file-per-run schema is consolidated into one structured log entry per iteration.
9. Propose extracting `ablation_features.py`'s CWM-13-field-joining logic into a shared `data.py`
   helper rather than letting `action_space.py` re-derive a third copy — flagged, not yet done,
   since it touches a file CLAUDE.md documents as deliberately standalone.
10. `models/fm_v0.py` (and future FM variants) wrap `baseline.FM`/`run_fm` rather than
    reimplementing the training loop, to avoid a fourth copy of the Adam/early-stopping logic.

## Open questions carried over from v0.1 (still open)
- Iteration cap independent of convergence?
- Multi-seed count per iteration — 3 (`ablation_features.py`'s convention) or 5
  (`baseline_scores.json`'s)? Also relevant now to `reeval.py`'s recheck count.
- Precise "suspicious" criteria for Guardrail/Analysis to flag a too-good-to-trust jump.
- Where do user-fed studies/papers get ingested — a `studies/` folder the Hypothesis Agent reads,
  with one short note per study on what it suggests trying here?

## Changelog
- v0.1 — initial outline.
- v0.2 — integrated 4 of R&D-Agent's 6 components (Yang et al. 2025, arXiv:2505.14738): debug-first
  workflow, structured hypothesis pipeline, time-aware planning, simplified aggregated eval +
  error/recovery. Added hard-requirement mapping (wrapped LLM client, unified log schema,
  manual-intervention counter, hidden-test isolation, crash-safe resume). Added phased file-path
  plan. Consolidated v0.1's 3-file run schema into 1 structured log entry. Resolved the "mapping to
  Claude Code mechanics" open question (hybrid). Flagged reconciliation points with
  `ablation_features.py` and `baseline.py`. No code written yet.
- v0.3 — Q1 resolved and implemented: `agent/__init__.py`, `agent/llm_client.py` wired to a local
  Ollama server running `qwen2.5-coder:7b`, smoke-tested (plain call + `json_mode`) against the
  actual running model, token logging to `runs/token_ledger.jsonl` confirmed working off Ollama's
  native `prompt_eval_count`/`eval_count`. `.gitignore` += `runs/*` (+ `runs/.gitkeep`). Rest of
  Phase 0 (`action_space.py`, `models/`, `data_guard.py`, `logging_schema.py`, `resume.py`,
  `manual_intervention.py`) still not started. Q2/Q3 still open.
