# Agentic Research Loop — Strategy Outline

Status: **v0.10** — Phases 0-6 all built and verified against the real stack (real Ollama, real
qwen2.5-coder:7b, real KuaiRand-Pure). `python -m agent.cli run --iterations N` runs the loop;
`runs/dashboard.html` (auto-regenerated every iteration, or `python -m agent.viewer` on demand) is
the visual view. Every phase's file list, review findings, and live-discovered bugs are recorded
in the Changelog below — nothing here is claimed without a verification note next to it. A real
usage report (repeated hypotheses, runs stalling ~5 iterations) led to 3 more real fixes — see the
v0.10 section: a code-level duplicate-config guard + higher propose temperature (repetition), and
a convergence-math fix that was excluding real evidence incorrectly (false-early-stopping).
Remaining, not yet built: `agent/action_space.py`'s `toggle_field`/`swap_model_variant` action
types are named but not executable (need a `models/fm_v1_*.py` variant + `data.encode()` extension
— see `models/base.py`'s docstring; this is also the most direct lever for longer, richer
exploration, since the model's actual first instinct is almost always a features hypothesis); Q2/Q3
from earlier revisions are still open. Living document; refined as studies/prompts are fed in. Each
refinement appends to the Changelog rather than rewriting history.

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

**Phase 5 — Orchestrator wiring** — DONE (v0.8)
Two steps existed nowhere before this phase and had to be built fresh (not sourced from
`06-Master-Prompts_1.md`, which only ever specified propose/repair): **Coding** and **Guardrail**.
- `agent/prompts/code.md`, `agent/coding_agent.py` — turns a hypothesis into ONE action from
  `agent/action_space.py`'s vocabulary, or honestly reports `implementable: false` rather than
  forcing a fake fit. Same JSON-validated, retry-once pattern as the rest of the codebase.
- `agent/action_space.py` — v0's actual constrained action space: only `set_hyperparam` is
  executable (`k`/`lr`/`l2`/`epochs`/`patience`/`batch_size`, bounded ranges); `toggle_field` and
  `swap_model_variant` are named (so `target_stage`'s vocabulary has somewhere to point) but
  raise `ActionNotExecutable` — no model variant beyond `fm_v0` exists yet, and `data.encode()`
  isn't parameterized for extra fields. A real, structural safety property falls out of this for
  free: no action type here can ever write a file or touch `evaluate.py`, because none of them
  touch a file at all — Guardrail doesn't need static code analysis, just bounds-checking.
- `agent/guardrail.py` — deliberately thin (see its own docstring for why that's a property of
  the constrained action space, not laziness): re-validates the action, rejects anything not
  executable, with a clear reason. A Guardrail/Coding rejection never calls `error_recovery` —
  that's for runtime failures of code that ran, not a static rejection of a proposal.
- `agent/resume.py`, `agent/manual_intervention.py` — the two remaining Phase 0 items, needed now.
  `state.json` deliberately holds only `{last_completed_iteration, current_best}` — NOT history,
  which is re-derived from `experiment_log.jsonl` on resume so there's one source of truth, not two
  that can drift.
- `agent/archivist.py`, `agent/orchestrator.py` (full `run_iteration()`/`run_loop()`), `agent/cli.py`
  (`python -m agent.cli {run,status,note-intervention}`) — the actual sequence: Hypothesis → Coding
  → Guardrail → debug_run → (repair-and-reject on failure) → full_run → reeval → Archivist →
  convergence check. `run_loop()` always caps iterations (resolves the open "iteration cap"
  question: yes, always, defaults to `expected_total_iterations`) and resumes from `state.json` if
  present.
- **v0's known, stated limitation, found live, not just designed around**: since only hyperparameter
  actions are executable, and the propose prompt's own reasoning naturally gravitates toward
  feature ideas early on, the FIRST real end-to-end run (iteration 1, real Ollama call) proposed
  "add `is_like` as a feature" and Coding correctly rejected it as not-implementable — proving the
  honesty mechanism works, but also revealing two real gaps, both fixed this revision: (1) the
  rejection *reason* wasn't in `history_block`, only kept/discarded, so the model had no way to
  learn from its own history — fixed in `logging_schema.to_history_entry()` +
  `hypothesis_agent._format_entry()`. (2) nothing warned the model upfront which `target_stage` is
  actually executable, wasting iteration 1's tokens/compute on every fresh run — fixed with an
  explicit "action space today" line added to `agent/prompts/hypothesize.md` (and
  `06-Master-Prompts_1.md`, kept in sync).
- **Then a genuine, previously-undiscovered bug surfaced on re-verification, live**: the very
  first real accepted candidate (hypothesis correctly pivoted to a hyperparameter change, full
  pipeline ran and decided accept=True) crashed at the archiving step —
  `TypeError: Object of type float32 is not JSON serializable`. `evaluate.evaluate()` returns
  `numpy.float32` (fixed, do-not-modify per CLAUDE.md, so this can't be fixed at the source).
  Fixed with a new `agent/json_utils.py` (a `default=` handler duck-typing on numpy scalars'
  `.item()` method), applied at both JSON-writing boundaries (`archivist.py`, `resume.py`).

**Code review findings (`code-review high`, scoped to `agent/` + `models/`) — all 7 fixed**:
1. `orchestrator.py`: the full run + reeval's extra-seed passes weren't wrapped in try/except
   (only `debug_run` was) — a real-scale-only crash (debug_run's own docstring admits this is
   possible) would have propagated out and killed the whole loop. Fixed: same
   repair-and-reject handling as a `debug_run` failure.
2. `error_recovery.py`: `error_event()`'s `repaired` field was set from `bool(fixable)` — the
   model's "this is fixable in principle" claim, not "this was actually fixed" — misleading given
   v0 never re-applies a fix. Fixed: `repaired` now defaults to `False`, a caller must prove it.
3. `reeval.py`: `recheck()` unconditionally ran both extra full-training seeds even for a
   candidate already far below current-best with no realistic chance of the mean crossing the
   bar — wasted compute against the project's own scored "compute spent" dimension. Fixed: a
   cheap short-circuit skips the extra seeds when `original_primary <= current_best_primary`.
4. `debug_run.py`: `_is_plausible()` was called OUTSIDE the try/except guarding `train_fn`, so a
   malformed (non-exception-raising) metrics dict could still crash `debug_run()`, contradicting
   its own documented "never propagates an exception" guarantee. Fixed: both calls now share one
   try/except.
5. `hypothesis_agent.py`/`coding_agent.py`/`error_recovery.py`: ~30 lines of templating/JSON-parse
   logic duplicated near-verbatim three times. Fixed: extracted to `agent/prompt_utils.py`, all
   three now import from there.
6. `reeval.py`: `enabled=RECHECK_TOP_CANDIDATE` as a literal default bound at function-definition
   time, so flipping the module constant later wouldn't affect existing callers. Fixed:
   `enabled=None` default, resolved against the current module value at call time.
7. `action_space.py`: `validate_action()` didn't check that int-typed hyperparams (`k`, `epochs`,
   `patience`, `batch_size`) actually got integer values — `apply_action()` silently truncated a
   fractional value, so the logged pseudo-diff misrepresented what was actually proposed. Fixed:
   rejected explicitly, routes back through the existing retry-once mechanism instead.

**8th finding, from a real resume test, not the static review**: a genuine Ollama timeout (this
machine's CPU shared between Ollama and this repo's numpy training — queuing an LLM call right
after a heavy training pass is exactly when this shows up) raised `agent.llm_client.LLMError` from
inside `propose()`, and it propagated all the way out of `run_loop()`, killing the whole process.
Crash-safe resume then worked *exactly* as designed — a fresh process correctly picked up from
iteration 3, no lost progress — but a transient network timeout shouldn't need a full crash +
manual restart to recover from in the first place; that bar is too blunt for something this
common, and doesn't meet "recovers from failures instead of crashing" for what's really just a
network hiccup. Fixed: `agent/orchestrator.py` now wraps each iteration in
`_run_iteration_with_retry()` — backs off (5s/15s/30s) and retries specifically on `LLMError`
before giving up, at which point crash-safe resume remains the final fallback, not the first
line of defense. Also bumped `agent/llm_client.py`'s default timeout 120s → 180s.

**Phase 6 — Human-facing run/log viewer** — DONE (v0.9)
- `agent/viewer.py` — reads `runs/experiment_log.jsonl`, `runs/state.json`,
  `runs/manual_interventions.jsonl` (via `agent/manual_intervention.py`) and
  `runs/token_ledger.jsonl` (via `agent/cost_report.py`, reused rather than re-parsed); renders one
  self-contained `runs/dashboard.html` — a real standalone HTML document (its own `<!DOCTYPE>`/
  `<html>`/`<head>`/`<body>`, since this is a local file opened directly in a browser, not
  published through Claude's Artifact tool, which wraps that automatically). No server, no
  external library — even the best-primary-over-iterations chart is hand-rolled inline SVG, not a
  charting dependency, matching decision #15 (static file, not a running server). Light/dark via
  `prefers-color-scheme`.
- Shows: header stats (current-best primary, delta vs `fm_official`, delta vs `oracle_ceiling`,
  convergence countdown `stale/N`, manual-intervention count), the SVG chart (dashed reference
  lines for `fm_official`/`oracle_ceiling`; the line only moves on ACCEPTED iterations, so a
  rejected iteration doesn't visually look like progress), a per-iteration table
  (`hypothesis.statement`, `target_stage`, accept/reject badge, primary, token cost, wall-clock,
  and — for rejected iterations — the actual rejection reason inline, the same
  `to_history_entry()`-sourced reason the model itself learns from), the resource-consumption
  breakdown from `agent/cost_report.py`, and the manual-intervention log. Test-split scores never
  appear anywhere — the dashboard only has access to what `agent/data_guard.py` ever returns.
- Wired into `agent/archivist.py`'s existing best-effort `_regenerate_dashboard()` (written in
  Phase 5 to silently no-op via `ImportError` until this module existed) — **verified this
  actually activates now**: a direct `archivist.archive()` call regenerated `runs/dashboard.html`
  with no code change to `archivist.py` needed, and a follow-up check confirmed it points at the
  real default log files (not whatever alternate path a caller happened to pass for testing).
- **Verified against the real data already in `runs/`** (3 real iterations from Phase 5
  end-to-end testing): valid, complete HTML (`<!doctype html>` → `</html>`), current-best primary
  0.5995 rendered correctly, both real hypothesis statements rendered correctly (including
  iteration 1's exact rejection reason — "batch size of 32 is outside the allowed range
  [256, 65536]" — the same text the model itself read from history before proposing the
  boundary-correct 256 in iteration 2), reference lines fm_official=0.6016 / oracle=0.8484
  matching `baseline_scores.json` exactly.
- Not done in this pass: no JS interactivity (static tables only — sufficient for v1, no
  functional gap, just a possible future nicety), no automated screenshot/visual-render check (only
  content-level verification — opening it in an actual browser is still worth a manual look).

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

## v0.10 — real-usage bug report and fixes (repeated hypothesis + capped-around-5 runs)

**Reported**: real runs kept stalling around iteration 5, and the agent kept proposing "increase
the learning rate to 0.01" repeatedly. Investigated directly against the actual `runs/` logs
rather than guessing:

- **The repetition was real and worse than it looked**: iterations 3, 4, and 5 all proposed the
  *identical* hypothesis and got a *bit-identical* `primary` (`0.5741060972213745`, matching to
  the last float32 digit) — because this pipeline is fully deterministic given a fixed seed and
  unchanged current-best, so the same config always reproduces the same result. Not just weak
  model diversity — three full debug_run+full_run+reeval cycles rediscovering an answer already
  known, pure wasted compute.
- **Root cause, two parts**: (1) `agent/hypothesis_agent.py`'s `propose()` never overrode
  `llm_client.call`'s low default temperature (0.2) — near-deterministic sampling made
  `qwen2.5-coder:7b` converge on the same "obvious" hyperparameter change regardless of context.
  (2) nothing enforced the prompt's existing "don't repeat a discarded hypothesis" instruction —
  it was a soft suggestion a small model didn't reliably follow.
- **The timeout in the ledger traces to a real, already-identified bug class**: a genuine Ollama
  `"timed out"` entry appeared in `runs/token_ledger.jsonl` from an actual run. Two likely
  contributors: the LLMError-retry fix from v0.8 not yet having reached whatever process was
  running (Python doesn't hot-reload edited `.py` files — a long-lived `cli run` process started
  before a fix lands keeps executing the old in-memory code until restarted), and Ollama's default
  5-minute `keep_alive` unloading the model during a real full-scale training run's multi-minute
  gap, forcing a slow reload on the next call that can itself exceed even a generous timeout.

**Fixed**:
1. `agent/hypothesis_agent.py`: `propose()` now defaults to `temperature=0.8` (was inheriting
   `llm_client.call`'s 0.2) — json_mode's grammar constraint keeps output schema-valid regardless
   of temperature, so this trades away nothing on validity while directly increasing variety.
2. `agent/prompts/hypothesize.md` (and `06-Master-Prompts_1.md`, kept in sync): the anti-repetition
   instruction rewritten from a soft suggestion to an explicit, high-stakes rule — states plainly
   that a repeated hyperparameter change will be detected and skipped without even training.
3. **New: a code-level duplicate-config guard**, not just a prompt fix — `agent/logging_schema.py`
   now stores `resulting_config` (the FULL config after the action is applied, not just the raw
   action — comparing the full config, not the delta, avoids false positives if current-best has
   changed since a prior attempt) on every record; `agent/orchestrator.py`'s
   `_find_duplicate_config()` checks the last `DUPLICATE_CHECK_WINDOW=5` rejected iterations for an
   exact match BEFORE `debug_run`, and skips straight to a cheap, clearly-labeled rejection if
   found ("identical resulting config already tried and rejected in iteration N") — this doesn't
   rely on the model behaving; it structurally prevents the waste even if the prompt fix alone
   isn't enough for a 7B model.
4. `agent/llm_client.py`: added `"keep_alive": "30m"` to the Ollama request payload, so the model
   stays resident across a normal training-run-sized gap instead of Ollama's 5-minute default.
5. Documented plainly (for the user, not a code fix): if a `cli run` process was started before
   any of v0.8's fixes landed, it needs to be **restarted** to pick them up — editing `.py` files
   does not affect an already-running Python process.

**A 6th, likely bigger fix, found while verifying the first 5**: continuing the real run from
iteration 6, the loop reported "converged" after a single non-executed iteration (rejected at
Coding, never reached training). Traced it to `_converged()`/`_stale_count()` comparing the "last
N iterations" by raw position, without distinguishing an iteration that actually ran training from
one rejected before ever reaching it — so a run of not-implementable/duplicate-skipped rejections
was counted as "N real attempts without improvement" and triggered false-early convergence. Fixed:
`agent/logging_schema.to_history_entry()` now exposes `executed: bool(metrics present)`; both
`agent/orchestrator.py`'s `_converged()` and `agent/hypothesis_agent.py`'s `_stale_count()` filter
to executed iterations only before applying the N/epsilon window.

**Verified, clean state, all fixes together** (real Ollama, real full-scale-ish config — 12 epochs/
patience 4, not the earlier heavily-shortened test config): 4 iterations, **4 genuinely distinct
hypotheses** (`epochs=50`, `lr=0.005`, `lr=0.01`, `epochs=30`) — no exact repeats. Real progress
found: current-best moved from the old 0.5995 to **0.6015** (iteration 1's `epochs=50`, beating the
old best via a better early-stopping point). Convergence then triggered after iteration 4 — but
this time for a **legitimate** reason: 3 genuinely different, fully-executed hypotheses in a row
(iterations 2-4) each failed to beat iteration 1's new best by more than epsilon. Not a bug this
time — confirmed by checking `executed=True` on all 4 entries before the convergence check fired.

**Honest caveat, not another bug**: converging in ~4-6 iterations may often be the *correct*
outcome right now, not something left to fix — v0's action space is deliberately hyperparameter-
only (decision #2), a narrow space that a reasonably-tuned baseline doesn't leave much room to beat
via hyperparameters alone. Nearly every real run so far has the model's actual first instinct
being a **features** hypothesis ("add `is_like`", "add `play_time_ms`") — rejected as
not-implementable every time. If more, longer exploration is wanted, the principled lever is
expanding the action space (the already-flagged, not-yet-built `toggle_field` path — needs a
`models/fm_v1_*.py` variant + a `data.encode()` extension), not loosening
`CONVERGENCE_N`/`CONVERGENCE_EPSILON`, which are the officially-specified values from
`baseline_scores.json`'s own `convergence_rule`, not ours to relax unilaterally.

**Also worth noting, not fixed (minor, not the headline)**: iteration 4's `resulting_config` had
`epochs=30`, genuinely different from iteration 1's `epochs=50`, so the duplicate-config guard
correctly did NOT skip it — but early stopping (patience=4) kicked in around epoch 11 in both
cases, well below either cap, so the actual trained result came out bit-identical anyway
(`seed_primaries` matched iteration 1's exactly). The guard is doing its job correctly (it only
promises to catch identical *configs*, not configs that are merely *functionally* equivalent due
to early stopping) — flagged as a possible future refinement, not a defect to fix now.

**Also confirmed important for the user**: if a `cli run` process was started before these (or
v0.8's) fixes landed in the source files, it must be restarted — Python does not hot-reload edited
`.py` files, so an already-running process keeps executing whatever code was loaded at its start.

## v0.11 — real feature-selection capability (resolves decision #9)

**Asked directly**: what can the agent do besides tweaking hyperparameters, and is it actually
rewriting any of the codebase? Honest answer at the time: nothing — `set_hyperparam` was the only
executable action, and the Coding step never touched a `.py` file. Every real run so far had the
model's actual first instinct (a features hypothesis) rejected as not-implementable.

**Implemented, not just designed**: the agent can now genuinely add/remove real CWM signals, not
just retune existing hyperparameters. This is **config-driven feature selection** (the agent picks
from a fixed, pre-built list of real columns, executed by code written once, in advance) — NOT
free-form code generation (the agent does not write novel Python at runtime). That distinction is
deliberate: true arbitrary code-generation would need a real Guardrail overhaul (static analysis
of generated code, sandboxing) before it's safe to hand to a 7B model that's already needed a lot
of structural scaffolding to behave reliably — flagged as a possible future ask, not attempted here.

- `data.py` — additive only (FIELDS/`encode()`/`load()` completely unchanged, still exactly what
  `baseline.py`/`submit.py`/`ablation_features.py` use): `EXTRA_FIELDS` (music_id, video_type,
  upload_type, follow_user_num_range, register_days_range, fans_user_num_range,
  friend_user_num_range, user_active_degree) and `encode_with_extra_fields()` — resolves decision
  #9 (the one shared join-logic implementation now, so nothing re-derives a third copy of what
  `ablation_features.py` already prototypes standalone).
- `models/fm_v1.py` — new variant, adds `extra_fields`/`data_dir` config keys on top of
  `fm_v0.DEFAULT_CONFIG`; `extra_fields=[]` is byte-for-byte identical to `fm_v0` (**verified**:
  exact `primary` match). Now the default model in `agent/orchestrator.py`.
- `agent/action_space.py` — `toggle_field` moved from named-but-not-executable to genuinely
  executable (`{"type": "toggle_field", "field": <one of EXTRA_FIELDS>, "op": "add"|"remove"}`).
  `swap_model_variant` stays not-executable — `fm_v1` already subsumes `fm_v0`, nothing to swap.
- `agent/prompts/code.md` + `agent/coding_agent.py` — the Coding step now knows `toggle_field` is
  real, sees which extra fields are currently active, and is explicitly told to prefer it over a
  hyperparameter tweak when the hypothesis is genuinely about a feature.
- `agent/hypothesis_agent.py` — `feature_list` context now correctly says these fields are
  genuinely toggleable (previously said they'd need code changes — no longer true).
- **Real bug caught and fixed while wiring this in**: `fm_v1.DEFAULT_CONFIG` hardcodes its own
  `data_dir` (for the side-info CSVs) separately from the `data_dir` the orchestrator actually
  loads interaction logs from — a latent mismatch if anyone ever ran with a non-default
  `--data_dir`. Fixed in `agent/orchestrator.py`: the actual invocation's `data_dir` always
  overrides whatever a variant's `DEFAULT_CONFIG` says.
- **Verified**: `fm_v1` with `extra_fields=[]` exactly reproduces `fm_v0` (regression check, bit-
  identical `primary`). `fm_v1` with real extra fields (`music_id`, `user_active_degree`) trains
  and evaluates correctly, produces a sane, different metrics dict. `toggle_field` unit-tested
  (add/remove, bad field, bad op all behave correctly). Full end-to-end real run in progress as of
  this entry — see the next Changelog entry for whether the agent actually chose `toggle_field`
  for a real features hypothesis.

## v0.12 — real code generation + AIDE-style solution tree + commit/revert decision tree

**Asked for**: let the coding agent rewrite the codebase and decide whether to commit or revert,
with decision trees for the revert call and a good sense of which branch to pursue — citing AIDE's
tree architecture for efficient computation.

**This reverses the v0.4 search-strategy decision, deliberately and on request.** v0.4 recommended
greedy/chain (MLE-STAR's shape) and deferred tree search. The tree now earns its keep for a reason
that only became visible once real code generation existed: a 7B model writing a whole model module
fails *often*, and a chain has nowhere to put a failed-but-promising attempt except the bin. A tree
gives it a DEBUG child instead. Still NOT adopting ML-Master's MCTS (no rollouts, no UCT, no
backpropagation, no parallel branch fan-out) — this is AIDE's greedy tree, one node expanded per
iteration.

### Scope, stated plainly
The agent now **writes and executes real Python modules** — genuinely new model architectures, not
config mutations. What it does NOT do is edit existing repo files: it writes *new* modules into
`models/generated/`, and `evaluate.py`/`data.py`/`agent/*`/`submit.py` remain untouchable. That
boundary is deliberate, not a shortcut: letting generated code rewrite the scoring contract or the
train-only-fitting logic is how a competition entry gets silently invalidated, and it is exactly
the risk v0.11 flagged as needing "a real Guardrail overhaul" before code generation was safe.

### What was built
- `agent/code_guardrail.py` — the Guardrail overhaul v0.11 promised. Real AST static analysis of
  every generated module *before* it touches disk. Central rule: **generated code performs no file
  or network I/O at all** — no `open()`, no `os`/`subprocess`/`sockets`/`urllib`/`pickle`, no
  `exec`/`eval`/`compile`/`__import__`, no dunder introspection escapes, import allowlist only,
  must define `train()`, and the literal string `'test'` is refused outright. That one no-I/O rule
  does most of the safety work: generated code cannot read KuaiRand's CSVs itself, so the only data
  it can ever see is the `splits` dict handed to it — which `agent/data_guard.py` already stripped
  the test split from. It cannot leak what it cannot reach, and it cannot damage a repo it cannot
  write to. **Verified against 9 adversarial cases** (file read, `os` import, `subprocess`,
  `exec`, `().__class__.__bases__[0].__subclasses__()` escape, test-split reference, missing
  `train()`, syntax error, plus a legitimate module that correctly passes).
- `agent/prompts/write_model.md` + `agent/codegen_agent.py` — generation in three modes
  (draft/debug/improve), fence-stripping, static analysis, and **retry with the specific rejection
  reasons fed back** (up to 3 attempts). Nothing unsafe is ever written to disk.
- `agent/solution_tree.py` — the AIDE tree. Nodes are complete solutions; operations are DRAFT /
  DEBUG / IMPROVE. Where the efficiency comes from: DEBUG and IMPROVE start from an existing
  module's source so the model edits rather than re-derives; selection is greedy (exactly one node
  expanded per iteration, no rollouts); `MAX_DEBUG_DEPTH=3` caps how much compute one broken idea
  can absorb. Selection order — **debug first** (cheapest possible win: the idea is already
  written, it just doesn't run), **draft while the tree is narrow** (`MIN_DRAFTS=2`, so it doesn't
  lock onto the first working idea), **improve the best working node** otherwise.
- `agent/decision.py` — the explicit commit/revert decision tree, replacing scattered `if accept:`
  checks. Every verdict records the exact branch path it took (`decision.path`), logged per
  iteration, so "why was this kept or thrown away" is auditable rather than reconstructed.
  Outcomes: `COMMIT` / `KEEP_NODE` / `REVERT` / `REJECT_UNSAFE`. **`KEEP_NODE` is the outcome that
  only makes sense once there's a tree**: in the old chain, "didn't beat current-best" and "throw
  it away" were the same thing; in a tree, a solution that runs correctly but scores slightly below
  the incumbent is still a legitimate base to IMPROVE from later — it's marked WORKING and kept,
  it just doesn't move current-best. Only genuinely broken candidates become BUGGY, and only BUGGY
  nodes are DEBUG-eligible. **Verified across all 8 decision paths.**
- **"Commit" is deliberately two separate things.** `COMMIT` at the tree/state level means "accept
  as the new current-best". Writing an actual **git** commit is a separate, opt-in side effect
  (`agent/cli.py --git-snapshot`, `decision.GIT_SNAPSHOT`, **off by default**) — an autonomous loop
  writing to real git history should be a choice the user makes deliberately, and the `runs/`
  artifacts already record everything needed to reproduce a result without it.
- Orchestrator routing, integration into `logging_schema` (`decision`, `node_id` fields),
  `agent/cli.py` (`--git-snapshot`, tree rendering in `status`).

### Bugs found and fixed while building this
- **Selection policy re-debugged already-fixed branches**: a BUGGY node whose DEBUG child had
  already reached WORKING stayed DEBUG-eligible forever, burning an LLM call + training run per
  iteration re-fixing something already fixed. Caught in unit-testing `select()`; fixed with
  `_has_working_descendant()`.
- **`_format_action` crashed on the first non-hyperparameter action** (`KeyError: 'param'` — it
  assumed `set_hyperparam`'s shape unconditionally). Found live the first time the agent actually
  chose a `toggle_field` action. Fixed and made generic over action type so it can't recur for a
  future action type either.

### The first real generated module — a representative result worth recording
Asked for "a per-user popularity prior blended with the FM score", `qwen2.5-coder:7b` produced a
module that **passed static analysis on the first attempt** and followed the contract correctly
(right imports, `non_train_splits()`, train-only fitting, proper `evaluate()` usage) — but was
genuinely buggy: it indexed a numpy array with user_id *strings* (`pop_prior[user]`). `debug_run`
caught it on the 20k sample in seconds, the decision tree returned `REVERT`/`is_buggy=True`, and it
became a DEBUG-eligible node. That is the whole design working end to end, and it is also the
honest expectation to set: **a 7B model writing whole model modules will fail often**, which is
precisely why the debug-first gate and the tree's DEBUG operation matter more here than they did
when the action space was config-only.

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
- v0.8 — **Phase 5 built** (`action_space.py`, `coding_agent.py` + `prompts/code.md`,
  `guardrail.py`, `resume.py`, `manual_intervention.py`, `archivist.py`, full `orchestrator.py`,
  `cli.py`). **Full `code-review high` pass** on `agent/`+`models/`: 7 findings, all fixed (see
  Phase 5 section for the list — crash exposure in the full-run/reeval path, a misleading
  `repaired` log field, wasted reeval compute, an uncaught-exception gap in `debug_run`, 3x
  duplicated templating code extracted to `agent/prompt_utils.py`, a mutable-default-argument bug,
  silent int-truncation in the action space). **2 more bugs found live, through actual runs, not
  static review**: (1) `evaluate.py` returns `numpy.float32`, which crashed the very first real
  archiving step (`TypeError: not JSON serializable`) — fixed with `agent/json_utils.py`. (2) a
  real Ollama timeout under CPU contention crashed the whole loop, correctly triggering (and
  proving out) crash-safe resume, but that's too blunt a recovery path for a network hiccup —
  fixed with `_run_iteration_with_retry()`'s backoff-retry in `orchestrator.py` + a bumped
  `llm_client.py` timeout. **Also found and fixed, mid-Phase-5**: the propose prompt had no way to
  tell the model which `target_stage` is actually executable, so iteration 1 reliably wasted a
  full LLM round-trip on an unimplementable "features" hypothesis every fresh run; and rejection
  reasons never made it into `history_block`, so the model couldn't learn from its own history —
  both fixed, and the fix was then observed working live (iteration 2 read iteration 1's exact
  rejection reason and proposed the boundary-correct value). Verified end-to-end multiple times:
  full loop with real accept/reject decisions, a genuine cross-process resume (kill+restart
  simulated via two separate interpreter invocations reading the same `runs/state.json`), `cli
  status`/`note-intervention`. `python -m agent.cli run --iterations N` is a real, working
  entrypoint. Only Phase 6 (dashboard) remains unstarted.
- v0.9 — **Phase 6 built**: `agent/viewer.py` → `runs/dashboard.html`, a standalone (own
  doctype/html/head/body — not Artifact-published, a local auto-regenerating file) dependency-free
  page with a hand-rolled inline-SVG progression chart, per-iteration table (including inline
  rejection reasons), resource-consumption breakdown, and manual-intervention log. Wired into
  `agent/archivist.py`'s existing best-effort hook (no code change needed there — it was already
  waiting for this module to exist). Verified against the real 3-iteration log already in `runs/`:
  valid HTML, correct current-best/reference-line numbers, correct hypothesis text and rejection
  reasons. Verified the auto-regeneration wiring activates for real (a direct `archivist.archive()`
  call produced `runs/dashboard.html` with no `archivist.py` change), and that it always points at
  the real default log files regardless of what path a given `archive()` call used. All 6 phases
  from the original plan are now built and verified end-to-end.
