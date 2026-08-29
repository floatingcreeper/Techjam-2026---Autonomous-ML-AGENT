# Agentic Research Loop — Strategy Outline

Status: **v0.7** — Phase 0 (partial) + Phases 1, 2, 4 built and verified against real data (real
Ollama, real qwen2.5-coder:7b, real KuaiRand-Pure); Phase 3 folded into Phase 4 rather than kept
separate. Resource consumption now tracked and reported every session (`agent/cost_report.py`) —
see Changelog. Living document; refined as studies/prompts are fed in. Each refinement appends to
the Changelog rather than rewriting history.

Design reference for this revision: Yang et al., "R&D-Agent: An LLM-Agent Framework Towards
Autonomous Data Science," Microsoft Research/GenAI, 2025, arXiv:2505.14738 — **verified against the
paper's own HTML, not taken on secondhand summary**. Its real 6 components: Planning, **Exploration
Path Structuring**, Reasoning Pipeline, Memory Context (Research phase); Coding Workflow, Evaluation
Strategy (Development phase). We adopt 4: Coding Workflow → strategy 1 below, Reasoning Pipeline →
strategy 2, Planning → strategy 3, Evaluation Strategy → strategy 4.

**Correction to the original premise**, worth stating plainly: the ablation does NOT show that
Exploration Path Structuring and Memory Context are the two lowest-priority components. Memory
Context genuinely is the smallest effect (9% relative decline when removed). But **Exploration Path
Structuring is their single highest-impact component** — downgrading it from adaptive (their
DAG-based parallel-branch-exploration-plus-multi-trace-merge, architecturally distinct from MCTS,
which is ML-Master's design, a different paper) to plain chain-based search costs **28% relative
(35.1%→25.3% medal rate)**, more than Planning or Reasoning Pipeline (~24% each). Not adopting it is
a real, acknowledged cost, not a free simplification — see the dedicated section below for the full
reasoning on why we're accepting that cost anyway for this project.

Not adopting: Memory Context (their probabilistic cross-branch kernel) — replaced with a flat,
append-only history fed into the propose prompt's context (matches AIRA's and R&D-Agent's own
finding that this is the lowest-leverage of the six). Exploration Path Structuring (parallel-branch
DAG + merge, or ML-Master-style MCTS) — replaced with a simple greedy/chain loop that always extends
current-best (no branching structure exists in the repo today, so nothing to preserve) — see the
search-strategy section for why, given it's the one substitution with a real, measured cost.

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
**Concrete prompt template lives in `06-Master-Prompts_1.md`** (fed in this revision) — that file,
not a fresh design here, is now the source of truth for the actual propose/repair prompts. Its
schema is richer than v0.2's flat 4-field draft and supersedes it:
```
{
  "problem_identified": "string",           # the bottleneck-identification step, explicit
  "hypothesis": {
    "statement": "string",                  # plain-language, human-readable — see note below
    "target_stage": "features | model | training | sampling | eval_postprocessing",
    "reasoning": "string",
    "expected_effect": "string"
  },
  "implementation_sketch": "string"          # concrete enough for the Coding step, no further Q&A
}
```
`target_stage`'s controlled vocabulary is `06-Master-Prompts_1.md`'s (supersedes v0.2's
`change_type`-derived list — same concept, this is now the authoritative naming).
`eval_postprocessing` means post-processing the model's score array before `evaluate.evaluate()` is
called (e.g. calibration, blending two runs) — **never** a change to `evaluate.py` itself; the
Guardrail step enforces that regardless of what `target_stage` claims.

**Human-readability requirement** (explicit ask this revision): `hypothesis.statement` must read as
one plain sentence a human could understand cold from the dashboard (Phase 6) months later — concrete
nouns from the task (field/model/hyperparam names), not code syntax, not generic ML-speak. The exact
same string (just window-truncated) is what feeds `{{ history_block }}` back into the *next*
iteration's propose prompt — so this isn't just a display nicety, sloppy hypothesis text directly
degrades the next iteration's reasoning quality too. Logging must preserve it verbatim, never
compress it into a diff-shaped shorthand.

LLM output is parsed and validated (all fields present, non-empty); on failure, retry **once**
(matches `06-Master-Prompts_1.md`'s explicit guidance — corrects v0.2's "cap 2 retries" to 1) with the
validation error appended, then log a `hypothesis_generation_failed` event and skip to the next
iteration rather than retry indefinitely.

`{{ history_block }}`: last 8–10 iterations + the single best-ever entry (per `06-Master-Prompts_1.md`'s
own token-budget note), not the full history — this is also the first lever to pull if token spend
runs high.

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
**Recovery**: any failure (guardrail rejection, debug-run not-ok, full-run crash/NaN) → the repair
prompt from `06-Master-Prompts_1.md` (cheap/fast-tier call, no deep reasoning needed — same
`qwen2.5-coder:7b`, just lower stakes per call) returns
`{diagnosis, fixable, fix_description, corrected_code_diff}`. If `fixable: false`, stop repairing
immediately rather than burning remaining attempts. Cap at 3 repair attempts total either way, then
roll back to current-best (a no-op, since current-best is never overwritten until a clean accept)
and move to the *next* iteration with a fresh hypothesis rather than retrying the same one
indefinitely. Each attempt appends one entry to `error_events` on the iteration record:
`{attempt, error_text, diagnosis, fixable, fix_description, repaired}` — `corrected_code_diff` itself
is NOT stored inline in the JSONL (keeps the log scannable, matches the "keep the log lean" principle
from `06-Master-Prompts_1.md`'s history-window note); it lives in the run's
`runs/<NNNN>/code_snapshot/` artifact, referenced by attempt number.

**Aggregated eval, simplified**: no multi-branch merge — but before accepting a new current-best,
optionally re-run with 1–2 additional seeds and require the seed-mean (not a single seed) to beat
current-best by `epsilon`. Toggleable (`RECHECK_TOP_CANDIDATE`), **default on**: `fm_official`'s own
std across seeds (~0.0008) is close enough to the convergence `epsilon` (0.002) that a single-seed
accept is genuinely likely to be noise.

## Search strategy: greedy vs. chain vs. tree — recommendation

**Recommendation: greedy/chain — one active best solution, extended one hypothesis at a time.**
Architecturally this is MLE-STAR's design (Nam et al. 2025, arXiv:2506.15692 — verified via the
paper's HTML): a nested-loop, single linear working solution, no branching in the core refinement
loop (branching appears only in MLE-STAR's *final* ensembling step, after refinement is done, which
is a reasonable pattern to borrow later but isn't core-loop architecture). This is the closest
published precedent to what we're building, not an ad-hoc simplification.

**What we're giving up, stated honestly**: R&D-Agent's own ablation (see above) says their
DAG-based parallel-exploration-plus-merge component is worth up to 28% relative on MLE-Bench —
larger than any other single component, including the two we *are* adopting (Reasoning Pipeline,
Planning). ML-Master (Liu et al. 2025, arXiv:2506.16499 — verified) goes further with genuine
MCTS: UCT-based selection, Draft/Debug/Improve expansion, reward-based backpropagation, asynchronous
branch-parallel workers. Its reported 29.3% medal rate is directionally credible but not cleanly
comparable to R&D-Agent's 35.1% — the two papers' own tables disagree on each other's numbers
(ML-Master's table puts R&D-Agent at 22.4%, not 35.1%), most likely different task subsets/snapshots;
we're not treating either exact percentage as ground truth, just as evidence branching *can* be
worth a lot when done well.

**Why we're accepting that cost for this project anyway:**
1. **Compute**: this repo is CPU-only, numpy-only, one small FM model per run. A branching factor of
   even 2–3 at one "layer" multiplies wall-clock and Ollama-call volume by that factor for every
   iteration it touches — real cost here, not the abstracted cost of a cloud MLE-Bench harness.
2. **Scoring criteria say so directly**: "total LLM tokens + GPU-hours spent" is one of our five
   scored dimensions (your Context section, this revision). Branching directly inflates both.
3. **Reasoning quality mismatch**: R&D-Agent's merge step and ML-Master's backprop/reward step are
   *harder* reasoning tasks than a single linear hypothesize step — synthesizing or arbitrating
   between multiple divergent traces. `qwen2.5-coder:7b` (our wired model, see Q1) is not well-suited
   to that even with `json_mode`; a 7B model doing single-path structured reasoning is a much safer
   bet than the same model doing multi-trace synthesis.
4. **AIRA (Toledo et al. 2025, arXiv:2507.02554 — verified) directly supports sequencing this way**:
   its finding is that *operators* (their term for Draft/Debug/Improve quality) bottleneck agent
   performance, not search policy — "with [baseline] operators, more advanced search policies gain
   no advantage; with improved operators, MCTS outperforms greedy." Strategies 1 and 2 below
   (debug-first, structured reasoning) are exactly operator-quality improvements. The evidence-backed
   sequencing is: get those right on a chain backbone first; branching is a lever to pull *later*,
   only if operator quality alone plateaus — not a starting assumption.
5. AIRA's separate finding — a **9–13% generalization gap** between selecting a "best" solution by
   validation score vs. true held-out performance — is independent support for strategy 4's
   multi-seed recheck-before-accept (§4 below), which is our (much cheaper) substitute for what
   R&D-Agent's multi-trace merge and AIRA's "robust final-node-selection" both are reaching for:
   don't trust one single eval pass to pick a winner.

**Not doing, but flagged as a cheap stretch goal if time allows**: R&D-Agent's own strategy note
("maximize diversity in the first layer") suggests a lightweight version — branch 2–3 ways only at
iteration 0, keep the best, then run the rest of the loop as pure chain from there. Recovers some of
Exploration Path Structuring's value at a bounded, one-time cost. Not in the phased plan below
(needs parallel-execution plumbing the current serial Ollama-call design doesn't have) — worth
reconsidering only after Phases 0–5 are working end to end on the chain design, never before.

**AIRA cross-check on completeness**: AIRA's two design axes are Search Policy (greedy/MCTS/
evolutionary — covered above) and Operators (Draft/Debug/Improve/Memory/Crossover, plus
prompt-adaptive complexity, scoped memory, "think tokens"). Nothing in AIRA suggests our 4 adopted
strategies are missing a load-bearing piece; it doesn't address debug-on-sample or time-aware
budget planning at all (outside its scope, not a contradiction of KompeteAI's or R&D-Agent's
Planning-derived support for those two).

## Explicitly out of scope
No cross-branch/multi-trace memory (flat append-only log instead — matches both R&D-Agent's and
AIRA's own finding that this is comparatively low-leverage). No tree search/MCTS/DAG-parallel-branch
exploration (plain greedy/chain loop, see recommendation above — nothing in the repo today has
branching to preserve, and this is now an evidence-weighed choice, not an assumption). No
RAG/external knowledge base (confirmed nothing RAG-like exists in the repo currently).

## Hard requirements (ours, not from the paper) and where they land
- **One wrapped LLM client, logs input/output tokens** → `agent/llm_client.py`; every Hypothesis /
  Analysis / Coding / Repair call goes through it; running ledger at `runs/token_ledger.jsonl`.
- **One structured JSONL log entry per iteration** (hypothesis, code diff, metrics, error/recovery
  events, token cost, GPU time) → `runs/experiment_log.jsonl`, written by `agent/archivist.py`.
  Hypothesis sub-object is now `06-Master-Prompts_1.md`'s schema verbatim
  (`problem_identified`/`hypothesis.{statement,target_stage,reasoning,expected_effect}`/
  `implementation_sketch`) — see strategy 2 above; this **supersedes v0.1's three-file-per-run
  schema** (`hypothesis.json`/`metrics.json`/`analysis.json`), consolidated into one record; raw
  stdout/code snapshots remain as supplementary, non-"log" artifacts under `runs/<NNNN>/`. GPU time:
  repo is CPU-only today, so this field is logged as `0.0` / wall-clock CPU time with a note, not a
  real GPU metric, unless that changes. A human-facing viewer for this file is now in scope — Phase 6.
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
- `agent/__init__.py` — DONE (v0.3).
- `agent/llm_client.py` — DONE (v0.3), see Q1.
- `models/base.py` — DONE (v0.5): the convention (`train(splits, config) -> results`) plus
  `non_train_splits()`, no base class — kept in the repo's beginner-friendly, minimal-abstraction
  style rather than reaching for `typing.Protocol`/ABCs.
- `models/fm_v0.py` — DONE (v0.5): thin adapter wrapping `baseline.FM` (Adam/forward/backward
  reused unmodified; the epoch/batch/early-stop *driver loop* is necessarily re-derived from
  `run_fm`'s shape, generalized over "whichever non-train splits are present" instead of
  `run_fm`'s hardcoded `enc['test']`, since the agent loop's splits never have a 'test' key —
  see `agent/data_guard.py`). **Verified**: at 3 epochs on the real full train/valid split, valid
  `primary`=0.5993 — already in `fm_official`'s reference range (0.6016 at full convergence,
  `baseline_scores.json`), confirming the adapter faithfully reproduces `baseline.FM`.
- `agent/data_guard.py` — DONE (v0.5): `load_train_valid(data_dir)` drops `'test'` before
  returning. **Verified** against the real data dir: returns exactly `{'train', 'valid'}`
  (1,141,112 / 124,909 rows), `'test'` key physically absent.
- `agent/action_space.py` — NOT STARTED. Needed before Phase 2's Coding step, not before Phase 1.
- `agent/logging_schema.py` — NOT STARTED. Needed before Phase 2/5, not before Phase 1.
- `agent/resume.py` — NOT STARTED. Needed before Phase 5.
- `agent/manual_intervention.py` — NOT STARTED. Needed before Phase 5.
- `.gitignore` — DONE (v0.3): `runs/*` + `runs/.gitkeep`.

**Phase 1 — Debug-first workflow** (strategy 1) — DONE (v0.5)
- `agent/debug_run.py` — fixed-N sample (20,000 train / 10,000 per other split), reduced
  epochs/patience, timed, output-sanity-checked (dict shape, NaN/Inf, `[0,1]` range on
  GAUC/nDCG@5/primary), linear full-run-duration extrapolation (row-ratio × epoch-ratio, documented
  as pessimistic since it ignores early stopping); returns
  `DebugResult{ok, reason, estimated_full_runtime_s, sample_metrics, elapsed_s}`. Never lets a
  candidate's exception propagate — caught and turned into `ok=False`.
- `agent/orchestrator.py` (skeleton) — wires `debug_run()` as the gate in front of a full
  `model.train()` call; not the full loop yet (no Hypothesis/Guardrail/Archivist/resume — Phases
  2-5), just enough to prove the gate works end to end.
- **Verified, all against real data** (`python -m agent.orchestrator --epochs 3`,
  `python -c "..."` for the failure cases): (1) happy path — debug sample OK in ~1s, estimated
  ~1160s for a full 40-epoch run (or ~72s estimated / 35s actual for a capped 3-epoch run — the
  estimate's pessimism confirmed directly, since it doesn't account for early stopping); (2) a
  candidate that raises `KeyError` → caught, `ok=False`, reason names the exception, no crash;
  (3) a candidate returning `NaN` → caught, `ok=False`, reason names the exact NaN field; (4) a
  candidate returning an out-of-`[0,1]`-range score → caught, `ok=False`. All three failure modes
  return in microseconds (fail fast, no wasted debug-sample compute on an obviously broken
  `train_fn`).

**Phase 2 — Structured hypothesis pipeline** (strategy 2) — DONE (v0.6)
- `agent/prompts/hypothesize.md` — DONE: sourced directly from `06-Master-Prompts_1.md`'s Propose
  Prompt (§1 of that file, corrected metric names + the Prior-art block folded in), not written
  fresh.
- `agent/config.py` — DONE (new, small — needed sooner than expected): single source of truth for
  `CONVERGENCE_N`/`CONVERGENCE_EPSILON` (3 / 0.002), per `06-Master-Prompts_1.md`'s own explicit
  demand that these never be duplicated.
- `agent/budget.py` — DONE, **partially** (pulled forward from Phase 3 out of necessity: the
  propose prompt needs `budget_tier_instruction` filled on every call, so this couldn't wait):
  `budget_tier_instruction(budget_fraction)` (3 tiers, thresholds/wording locked to
  `06-Master-Prompts_1.md`'s own comment) and `iteration_budget_fraction()` (the iteration-count
  fallback pending Q3). Confirms the Phase 3 design note from v0.4: this repo's real design is one
  template with an injected instruction block, not three separate `hypothesize_{explore,mid,exploit}.md`
  files as v0.2 first sketched.
- `agent/hypothesis_agent.py` — DONE: `build_state()` assembles all `{{ }}` values (n_users/
  n_items/n_interactions computed from the real loaded splits, not trusted from any secondhand
  figure; `feature_list`/`feedback_signals` built from the actual CSV columns, verified via
  `head -1` on the real files, not guessed); `propose()` calls `llm_client`, parses/validates the
  nested schema, retries **once** on failure (continuing the same conversation with the validation
  error appended — matches `06-Master-Prompts_1.md`'s guidance, corrects v0.2's "cap 2 retries").
  `format_history()` implements the "last 8-10 + best-ever" windowing.
- `agent/logging_schema.py` still NOT built (Phase 5) — `format_history()`'s expected history-entry
  shape is documented as a minimal projection for now; Phase 5 needs to produce/project into it.
- **Bug found and fixed on the first real end-to-end run**: filling `current_best_*` with
  `"0.0000"` when there's no accepted candidate yet (iteration 1) read to the model as "the model
  is broken" rather than "no baseline exists yet" — its `problem_identified` was built around
  exactly that misreading. Fixed: an explicit `"N/A (no accepted candidate yet)"` string instead of
  a formatted zero.
- **Verified against the real stack** (real data, real `qwen2.5-coder:7b` over Ollama): iteration-1
  call returned a fully schema-valid response on the first attempt (no retry needed), real computed
  context (26,210 users, 7,538 items, 1,266,021 interactions), 1798 input / 168 output tokens.
  Retry/validation logic separately verified with deterministic mocks (not the real model, so it's
  reproducible): a flaky call that fails once then succeeds → `ok=True, attempts=2`, correct retry
  message threaded into the conversation; a call that always fails → `ok=False` after exactly 2
  attempts (1 + 1 retry, never more); an invalid `target_stage` value → rejected with the valid
  vocabulary listed in the error.

**Phase 3 — Time-aware planning** (strategy 3) — FOLDED INTO PHASE 4, not a separate phase
Most of Phase 3 already landed in v0.6 as a Phase 2 dependency (`agent/budget.py`'s
`budget_tier_instruction()` + `iteration_budget_fraction()` — the propose prompt needed it
immediately, couldn't wait for its own phase). What was left was two loose ends, both genuinely
small enough not to justify a standalone phase: a `wall_clock_tier()` function (real math, just not
callable meaningfully until Q3 gives it a real budget number — DONE, folded into `agent/budget.py`
in v0.7, dormant until Q3), and threading a tier constraint into `agent/action_space.py` (can't be
done — that module doesn't exist yet, noted as a TODO for whenever it's built, not lost).

**Phase 4 — Error/recovery + simplified aggregated eval** (strategy 4) — DONE (v0.7)
- `agent/prompts/repair.md` — DONE: sourced directly from `06-Master-Prompts_1.md`'s Repair Prompt
  (§2), not written fresh.
- `agent/logging_schema.py` — DONE, built now (originally slated for Phase 5, pulled forward since
  Phase 4's `error_events` needed a defined shape and Phase 2's hypothesis output needed a home to
  slot into unchanged): `new_record()` (the full per-iteration record) and `to_history_entry()`
  (projects a record down to what `agent.hypothesis_agent.format_history()` expects — the bridge
  Phase 5's archivist will use to feed history back into the next propose call).
- `agent/error_recovery.py` — DONE, scoped deliberately narrower than the original one-line plan
  description: `repair()` is ONE LLM call (diagnose + propose a fix), no internal loop. Actually
  re-applying a fix, re-running it through Guardrail → debug_run, and calling `repair()` again with
  attempt_number+1 if it still fails — that's the orchestrator's job (Phase 5), since only the
  orchestrator has execution access; a fake loop here that can't actually re-execute would have
  been misleading rather than useful. `MAX_REPAIR_ATTEMPTS = 3` lives here as the cap the future
  orchestrator reads. `error_event()` builds one `error_events` entry from a `RepairResult`.
- `agent/reeval.py` — DONE: `recheck()`, optional multi-seed recheck before accept,
  `RECHECK_TOP_CANDIDATE` flag (default on) — this project's substitute for AIRA's "robust
  final-node-selection" finding (9–13% generalization gap from trusting a single validation pass),
  see search-strategy section.
- **New this revision, not in the original plan**: `agent/cost_report.py` — reads
  `runs/token_ledger.jsonl` (every LLM call, logged unconditionally by `agent/llm_client.py`) and
  summarizes tokens/latency/failures, total and per-caller. `python -m agent.cost_report` any time.
  GPU-hours always reported as `0.0` with an explicit note (CPU-only repo) rather than omitted.
  This is what makes the "total LLM tokens + GPU-hours spent" scored dimension checkable
  continuously, not just totted up once at the end — added because it's now being tracked and
  reported on every turn, not just phase-by-phase.
- **Verified against the real stack**: `repair()` called for real against a genuine failure
  scenario (a `KeyError` from reading a CSV column that isn't actually loaded by `data.load()` —
  the same class of mistake a hypothesis like "add play_time_ms" from Phase 2's own real output
  could trigger) — got back a schema-valid, sensibly-structured `fixable: false` response. **Real
  limitation surfaced, not just a pass**: the diagnosis said the column "does not exist in the CSV
  data", which is factually wrong (`play_time_ms` genuinely is a column in the raw logs, confirmed
  via `head -1` in this same session — `data.load()` just doesn't read it into row tuples yet). Root
  cause: the synthetic test fed `repair()` only a one-line diff fragment, not real surrounding code,
  so a 7B model reasonably reached for the more common explanation. **Design note for Phase 5**:
  the orchestrator must feed `repair()` enough real code context (not just the failing line) or
  it'll produce confident-sounding but wrong diagnoses — logged here so it isn't lost.
  `recheck()` run against real `fm_v0.train`, real data, 3 real training passes: seed-0/1/2
  primaries 0.599311/0.599816/0.599214 (tight, consistent with `fm_official`'s known ~0.0008 std),
  mean 0.599447; accept path (low bar) correctly accepted, reject path (deliberately high bar,
  `RECHECK_TOP_CANDIDATE` effectively off via `extra_seeds=()`) correctly rejected.
- **Resource consumption this session so far** (`python -m agent.cost_report`, also fixed a
  Windows-console mojibake bug in its own em-dash output while verifying it): 3 LLM calls, 0
  failed, 4,107 input / 438 output tokens (4,545 total), 220.0s cumulative LLM wall-clock, 0.00
  GPU-hours (CPU-only repo, reported explicitly rather than omitted). Per-caller breakdown now
  available any time via that command — this is the ongoing, checkable answer to the "total LLM
  tokens + GPU-hours spent" scored dimension, not a one-time total.

**Phase 5 — Orchestrator wiring**
- `agent/orchestrator.py` (complete) — full loop per the diagram below, reads `resume.py` state.
- `agent/archivist.py` — writes `runs/<NNNN>_<slug>/{stdout/, code_snapshot/}` (supplementary) +
  appends the merged record to `runs/experiment_log.jsonl` + updates `runs/state.json` (last step).
- `agent/cli.py` — `python -m agent.cli run [--iterations N | --budget-seconds N]`,
  `python -m agent.cli note-intervention "..."`, `python -m agent.cli status`.

**Phase 6 — Human-facing run/log viewer** (new this revision — explicit requirement, needed for the
"quality/reasoning behind what the agent tried" scoring criterion, which a human has to actually be
able to read)
- `agent/viewer.py` — reads `runs/experiment_log.jsonl`, `runs/token_ledger.jsonl`, `runs/state.json`,
  `runs/manual_interventions.jsonl`; renders one self-contained static `runs/dashboard.html` (plain
  HTML/CSS/vanilla JS, data embedded inline at generation time — no server, no new dependency,
  works offline, doubles as a demo artifact for judges). Shows: iteration timeline
  (`hypothesis.statement`, `target_stage`, accept/reject, valid `primary` delta, current-best-vs-
  `fm_official`-vs-`oracle_ceiling` reference lines — test score shown but visually de-emphasized,
  never framed as the deciding number), running token cost + cost-per-iteration (feeds the cost
  report), error/recovery events per iteration, manual-intervention count, convergence countdown
  (`stale_count` vs `N`).
- Only depends on `agent/logging_schema.py`'s format (end of Phase 0) — doesn't need the orchestrator
  running to start development; can build/test against a small hand-written synthetic log first.
- Regenerated as the last step of `agent/archivist.py` each iteration (cheap — it's reading a few
  small JSONL files — so the human can just leave `runs/dashboard.html` open and refresh).

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
             → Archivist (runs/<NNNN>/, experiment_log.jsonl, state.json, regenerates
                          runs/dashboard.html [NEW, §6])
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
- **`06-Master-Prompts_1.md`'s original TASK CONTEXT block described NDCG@10/Recall@50 and a
  generic hidden-test-set framing** — doesn't match this repo's actual `evaluate.py` contract
  (GAUC + nDCG@5, `primary` = mean). Fixed directly in that file this revision (metric names,
  `{{ baseline_gauc/ndcg5/primary }}` placeholders, an explicit pointer to Q2 for what "hidden"
  concretely means here) rather than left as a latent bug that would have fed the LLM wrong metric
  names the first time this template was actually instantiated.
- **`06-Master-Prompts_1.md`'s hypothesis/repair JSON schemas supersede v0.2's flatter drafts**
  (§2/§4 above) — not a real conflict since v0.2's schemas were explicitly drafts, but noting the
  supersession since v0.2 was, until this revision, the only concretely-specified version anywhere.

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
11. **Greedy/chain search strategy** (not R&D-Agent's DAG-parallel-merge, not ML-Master's MCTS) —
    see the dedicated section above for the full, citation-backed reasoning. This is the one
    decision in this list with an acknowledged, real performance cost per the source papers, not a
    free simplification — flagging it as the single most important one to veto if you disagree.
12. The lightweight "branch 2–3 ways at iteration 0 only" stretch goal is explicitly NOT in the
    phased plan — noted as a possible later addition, never a Phase 0–5 dependency.
13. `06-Master-Prompts_1.md`'s hypothesis/repair schemas adopted as authoritative over v0.2's drafts
    (richer, already concretely specified, includes fields — `problem_identified`,
    `implementation_sketch` — v0.2 didn't have).
14. Retry cap on hypothesis JSON validation corrected from v0.2's "2 retries" to **1 retry** (2
    attempts total), matching `06-Master-Prompts_1.md`'s explicit guidance.
15. Human-facing viewer (Phase 6) built as a static, dependency-free HTML file regenerated each
    iteration, not a running server (Flask/Streamlit) — matches the repo's minimal-dependency
    posture and needs nothing extra running during a live demo.
16. Fixed `06-Master-Prompts_1.md`'s TASK CONTEXT block's metric mismatch (NDCG@10/Recall@50 →
    GAUC/nDCG@5/primary) directly in that file rather than leaving it as a latent bug — see
    Reconciliation above.

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
- v0.4 — merged `06-Master-Prompts_1.md`'s concrete propose/repair prompt templates in as the
  authoritative hypothesis/repair schema (supersedes v0.2's flatter drafts); fixed a real metric-name
  bug in that file (NDCG@10/Recall@50 → this repo's actual GAUC/nDCG@5/primary) and pinned its
  epsilon/N template vars to a single source of truth (0.002/3, matching `baseline_scores.json`).
  Added a human-readability requirement for `hypothesis.statement` (read by both a human and the
  next iteration's history block). Added Phase 6 (`agent/viewer.py`, static `runs/dashboard.html`).
  Verified R&D-Agent's, ML-Master's, MLE-STAR's, and AIRA's actual architectural/ablation claims
  against the papers' own HTML (not secondhand summary) — **corrected the original premise**: R&D-
  Agent's Exploration Path Structuring is their highest-impact component (28% relative decline when
  simplified), not a safe-to-drop one like Memory Context (9%); flagged an unresolved numeric
  discrepancy between R&D-Agent's and ML-Master's self-reported medal rates. Added a dedicated,
  citation-backed greedy-vs-chain-vs-tree recommendation (greedy/chain, MLE-STAR precedent, cost
  acknowledged not hidden, reasoning grounded partly in AIRA's operators-bottleneck-not-search-policy
  finding). Retry cap on hypothesis validation corrected 2→1. No new implementation code this
  revision (per explicit instruction) — only `06-Master-Prompts_1.md` and this file were edited.
- v0.5 — **Phase 1 built and verified against real data**: `agent/data_guard.py`, `models/base.py`,
  `models/fm_v0.py`, `agent/debug_run.py`, `agent/orchestrator.py` (skeleton). Confirmed: the
  hidden-test guard actually drops `'test'` (real data dir, exact row counts logged above); the
  `fm_v0` adapter reproduces `baseline.FM` (valid primary 0.5993 @ 3 epochs vs. `fm_official`'s
  0.6016 reference); the debug gate's happy path and all three failure modes (crash / NaN /
  out-of-range) behave exactly as designed, each caught in microseconds without touching full-run
  compute. Phase 0's remaining pieces (`action_space.py`, `logging_schema.py`, `resume.py`,
  `manual_intervention.py`) intentionally deferred — not needed until Phase 2/5.
- v0.6 — **Phase 2 built and verified**: `agent/prompts/hypothesize.md`, `agent/config.py`,
  `agent/budget.py` (partial, pulled forward from Phase 3 — the propose prompt needs it),
  `agent/hypothesis_agent.py`. Real end-to-end run against `qwen2.5-coder:7b` produced a fully
  schema-valid hypothesis on the first attempt; retry/validation logic separately verified with
  deterministic mocks (flaky-then-succeeds, always-fails, invalid-vocab-value). Found and fixed a
  real bug from the first live run: `current_best_*="0.0000"` at iteration 1 (no accepted candidate
  yet) was misread by the model as "the model is broken" — replaced with an explicit N/A string.
- v0.7 — **Phase 3 folded into Phase 4** (its remainder was two small loose ends, not enough to
  justify a standalone phase — see Phase 3/4 section). **Phase 4 built and verified**:
  `agent/prompts/repair.md`, `agent/logging_schema.py` (pulled forward from Phase 5, needed sooner),
  `agent/error_recovery.py` (scoped to one LLM call, not a fake execution loop — see Phase 4
  section for why), `agent/reeval.py`, `agent/budget.py`'s `wall_clock_tier()`. Real verification
  surfaced a genuine repair-quality limitation (wrong diagnosis from too little code context — see
  Phase 4 section), not just a pass, and confirmed `recheck()`'s multi-seed accept/reject logic
  against real training runs. **New, not in the original plan**: `agent/cost_report.py` — resource
  consumption is now tracked continuously (`runs/token_ledger.jsonl`) and reportable on demand via
  `python -m agent.cost_report`, per an explicit ask to keep this an ongoing tracked thing rather
  than a one-time total. Fixed a Windows-console mojibake bug in that report's own output.
