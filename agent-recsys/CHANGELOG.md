# agent-recsys: change log and handoff brief

**Purpose of this document:** this project (`agent-recsys`) is an autonomous ML research agent built for a hackathon (KuaiRand-Pure recommender-systems benchmark, organizer-provided `kuairand-starter-kit`). It is a deterministic Python controller loop that calls Claude Sonnet 5 once per iteration to propose a hypothesis + one full-file code rewrite, runs it in a sandbox, and logs results against the organizer's fixed convergence rule. Everything below was built and debugged incrementally, in the order listed (oldest first), across real runs against the actual data and a real Anthropic API key. Each entry says what changed, why, which files it touched, and how it was verified. A final section lists changes that were **discussed and recommended but not yet implemented** — these are open opportunities for the next pass.

Use this as a briefing before reviewing the code: it explains *why* the code looks the way it does in several non-obvious places (see especially items 9, 10, 12, 13, 20, 21 — each fixes a real bug found via an actual failed run, not a hypothetical).

---

## Repository layout

- `kuairand-starter-kit/` — organizer-provided files (`data.py`, `evaluate.py`, `baseline.py`, `submit.py`, `ablation_features.py`), annotated with explanatory comments (see item 1). Its `KuaiRand-Pure/data/` subfolder is the canonical dataset location.
- `agent-recsys/` — the autonomous agent, sibling folder to `kuairand-starter-kit/`:
  - `pipeline/` — seeded copies of the annotated starter-kit files; `data.py` and `baseline.py` are the only two files the agent is allowed to rewrite.
  - `agent/context.py` — builds the LLM prompt (static domain knowledge + iteration history + current best code).
  - `agent/llm_client.py` — the one place that calls the Anthropic API; also a `DryRunClient` for testing the harness without spending tokens.
  - `agent/sandbox.py` — applies a proposed change to a throwaway scratch copy, runs it under a timeout, parses the result.
  - `agent/convergence.py` — implements the organizer's exact stopping rule (ε=0.002, N=3) plus iteration/wall-clock caps.
  - `agent/controller.py` — the outer loop tying all of the above together; owns promotion, logging, retries.
  - `agent/finalize.py` — builds the final submission CSV from the best snapshot.
  - `run_agent.py` — CLI entrypoint.
  - `hypothesis-ledger.html` — a standalone dashboard that visualizes `logs/iteration_log.jsonl` (charts + metrics table).

---

## Change log (oldest to newest)

### 1. Annotated the organizer's starter kit
**What:** added "what this file does" / "how it connects to the other files" comment blocks and inline comments to all five starter-kit files, plus a plain-English explanation of GAUC / nDCG@5 / primary in `evaluate.py`.
**Why:** requested, to make the provided code and metrics understandable before building on top of it.
**Files:** `kuairand-starter-kit/{data,evaluate,baseline,submit,ablation_features}.py`.
**Verified:** re-ran `baseline.py --model fm` after editing and confirmed identical scores to the original (valid 0.6015 / test 0.5953) before delivering.

### 2. Built the autonomous agent from scratch
**What:** designed and implemented the full harness described in "Repository layout" above: a deterministic controller that calls an LLM for one hypothesis + one full-file rewrite per iteration, sandboxes execution, and enforces the organizer's exact convergence rule and budget caps (50 iterations / 6h wall-clock).
**Why:** the core deliverable — an agent that iterates on the FM baseline the way a researcher would, with the LLM only ever producing a `Proposal` (hypothesis + rewrite), never given direct control of the real pipeline directory.
**Files:** all of `agent-recsys/` (new).
**Verified:** exercised end-to-end via `--dry-run` (a `DryRunClient` with three hand-written, legitimate hypotheses) before any real API key was used.

### 3. Fixed: relative `data_dir` resolved against the wrong directory
**What:** `controller.run()` now resolves `data_dir` to an absolute path up front.
**Why:** `sandbox.run_scratch()` executes the pipeline with `cwd` set to a throwaway temp directory, so a relative `data_dir` silently resolved against that temp directory instead of the real data location.
**Files:** `agent/controller.py`.
**Verified:** found via the author's own dry-run testing (not user-reported); confirmed fixed by rerunning.

### 4. Fixed: `DryRunClient` burned through its whole plan within one iteration
**What:** `DryRunClient.propose()` now selects its next hardcoded hypothesis by the `iteration` parameter, not an internal call counter.
**Why:** the call counter advanced once per `propose()` call including repair-attempt retries within the same iteration, so a single iteration with 2-3 retries could exhaust the entire 3-step hardcoded plan before iteration 0 even finished.
**Files:** `agent/llm_client.py` (`DryRunClient`).
**Verified:** found via dry-run testing; confirmed each iteration now proposes a consistent hypothesis across retries.

### 5. Fixed: `numpy.float32` not JSON-serializable
**What:** added a recursive `_jsonable()` sanitizer.
**Why:** `evaluate.py`'s GAUC/nDCG@5/primary values are `numpy.float32`, which `json.dumps()` cannot serialize directly, crashing every iteration's result-reporting step.
**Files:** `pipeline/_run_iteration.py`, `agent/finalize.py`.
**Verified:** found via dry-run testing; confirmed `run_agent.py --dry-run` completes without the `TypeError`.

### 6. Fixed: Windows `UnicodeDecodeError` on non-ASCII source comments
**What:** added explicit `encoding="utf-8"` to every file read/write across the codebase.
**Why:** `pathlib`'s `read_text()`/`write_text()`/`open()` default to the OS locale codec (cp1252 on Windows), which cannot decode the organizer's original Chinese-language comments still present in the kit files, or any non-ASCII text an LLM proposal might contain.
**Files:** `agent/context.py`, `agent/llm_client.py`, `agent/sandbox.py`, `agent/controller.py`, `run_agent.py`.
**Verified:** user-reported traceback (`charmap codec can't decode byte 0x8d`); fixed and confirmed via a fresh dry-run rerun on the user's machine.

### 7. Updated defaults after the user reorganized folders
**What:** updated `--data_dir` and related paths after the user moved `kuairand-starter-kit` into a shared `TECHJAM/` folder alongside `agent-recsys/`.
**Why:** keep the CLI defaults matching the real on-disk layout.
**Files:** `run_agent.py`.
**Note:** this update initially pointed at a *non-existent* path (`pipeline/KuaiRand-Pure/data`); the correct fix is item 11 below.

### 8. Switched the LLM model to Claude Sonnet 5
**What:** changed `AnthropicClient`'s and `run_agent.py`'s default model from `claude-sonnet-4-5` to `claude-sonnet-5`.
**Why:** explicit user request, alongside a cost estimate (Sonnet 5: $2/$10 per MTok in/out) before committing a real API budget.
**Files:** `agent/llm_client.py`, `run_agent.py`.

### 9. Fixed: LLM response truncated at `max_tokens=8192`
**What:** raised `max_tokens` from 8192 to 32000 in `AnthropicClient.propose()`.
**Why:** a full rewrite of `baseline.py` (200+ commented lines) has to fit *inside a JSON string*, with every newline/quote escaped — that alone can exceed 8192 tokens before the model finishes the string, producing an `Unterminated string` JSON parse error. Claude Sonnet 5's standard (non-beta) ceiling is 128K output tokens, so 32000 gives real margin. Raising the cap doesn't cost more by itself (billing is by tokens actually generated).
**Files:** `agent/llm_client.py`.
**Verified:** root-caused directly from the user's first real-run traceback (partial JSON cut off mid-`new_content`, well-formed hypothesis about a BPR loss change).

### 10. Fixed: SDK refused the large `max_tokens` request without streaming
**What:** switched `AnthropicClient.propose()` from `messages.create()` to `messages.stream()` (draining the stream, then `stream.get_final_message()`).
**Why:** immediately after item 9, a new error appeared: `ValueError: Streaming is required for operations that may take longer than 10 minutes`. The Anthropic Python SDK refuses a large-`max_tokens` non-streaming call because a long-idle single HTTP request risks being killed by a proxy before completion; streaming keeps the connection alive incrementally.
**Files:** `agent/llm_client.py`.
**Verified:** user re-ran after the fix; got past this error into a real LLM response.

### 11. Fixed: `--data_dir` default pointed at a nonexistent folder
**What:** corrected the default to `../kuairand-starter-kit/KuaiRand-Pure/data` (sibling of `agent-recsys/`).
**Why:** the default set in item 7 (`pipeline/KuaiRand-Pure/data`) never actually existed on disk — confirmed by directly listing the user's folders. The real run crashed with `FileNotFoundError`, and the LLM burned 3 retries (~$0.13 in real tokens) trying to "fix" it by rewriting `data.py`'s file-search logic, which could never work since the file wasn't anywhere under that tree.
**Files:** `run_agent.py`.
**Verified:** confirmed the corrected path resolves to the user's real data folder via `device_list_dir` before shipping.

### 12. Fixed: strict JSON parsing rejected raw control characters
**What:** `_parse_json_response()` now calls `json.loads(text, strict=False)`.
**Why:** when embedding a full source file inside a JSON string, the model sometimes pastes a literal raw newline/tab byte instead of the escaped `\n`/`\t` sequence — readable code, but invalid *strict* JSON. `strict=False` is a documented `json` module option that allows this without weakening any other validation.
**Files:** `agent/llm_client.py`.
**Verified:** reproduced the exact failure (`Invalid control character at char 13902`) in isolation with a synthetic string containing a raw newline, confirmed `strict=False` parses it correctly, confirmed `strict=True` still fails the same way (so nothing else was silently loosened).

### 13. Fixed: an LLM-side failure crashed the entire process instead of retrying
**What:** wrapped `llm_client.propose()` in `controller.py`'s retry loop in its own `try/except`, so a bad response is treated as a recoverable iteration error like a sandbox failure, instead of propagating uncaught.
**Why:** this was the actual root cause behind *every* "the whole script crashed" failure so far (items 9, 10, 12's errors). The code's own comments claimed failures would "retry with the traceback fed back to the model," but `proposal = llm_client.propose(...)` sat outside the `try/except` block that made that true — so any LLM-side exception killed the run immediately, losing the rest of the iteration budget.
**Files:** `agent/controller.py`.
**Verified:** wrote and ran two real functional tests against the actual controller logic (not just inspection): (a) a stub LLM that always raises — confirmed the process now retries 4x per iteration, logs a clean failed record, and proceeds to the next iteration instead of crashing; (b) a stub that fails once then succeeds — confirmed normal recovery and promotion still work.

### 14. Built the Hypothesis Ledger dashboard
**What:** a self-contained HTML page (published as a Cowork Artifact, plus a local copy in `agent-recsys/`) that loads `logs/iteration_log.jsonl` client-side and renders KPI cards, charts (validation-primary trend, GAUC/nDCG@5, spend per iteration, wall-clock), and an expandable per-iteration hypothesis/error log.
**Why:** requested, so iteration results are readable at a glance instead of raw JSONL.
**Files:** `agent-recsys/hypothesis-ledger.html` (new).

### 15. Fixed: charts silently rendered nothing
**What:** corrected the Chart.js CDN URL from `chart.js` (lowercase) / version `4.4.4` to `Chart.js` (capital C, cdnjs's actual directory name) / version `4.5.1`, and added a visible on-page error message for this failure mode going forward.
**Why:** the original URL 404'd (verified directly by fetching it), so `Chart` stayed `undefined` and the chart-rendering code silently no-opped — no error was ever surfaced to the user, just blank space under each chart header.
**Files:** `hypothesis-ledger.html`.
**Verified:** fetched the corrected URL directly to confirm it returns real JavaScript; ran a full headless-browser render test (Playwright) with the corrected library loaded locally, confirmed all four canvases have real drawn pixels and zero console/page errors.

### 16. Fixed: duplicate iteration numbers across separate runs
**What:** sort records by `timestamp` (not the `iteration` field), and detect run boundaries (iteration counter resetting) to label points as `r1·0`, `r2·0`, etc. instead of showing "0, 1, 2, 0" on one axis.
**Why:** `iteration_log.jsonl` is appended to across separate `python run_agent.py` invocations, and each run restarts its own counter at 0 — sorting naively by iteration value would have scrambled an already-chronological file (interleaving two different runs' entries).
**Files:** `hypothesis-ledger.html`.
**Verified:** ran the parsing logic in Node against the user's real log file (which does contain two genuine "iteration 0" entries), confirmed correct chronological ordering and disambiguated labels.

### 17. Added a full metrics table to the dashboard
**What:** a dedicated table (between the charts and the hypothesis cards) showing every iteration's valid/test GAUC, nDCG@5, and primary in one scannable grid, best score highlighted.
**Why:** requested — the user wanted the raw numeric values more prominent, not only reachable by expanding each card.
**Files:** `hypothesis-ledger.html`.

### 18. Added "Connect & auto-reload" via the File System Access API
**What:** a "Connect & auto-reload" button that opens a native file picker once, after which a "Reload" button re-reads that same file's current disk contents with no further dialog; the connection is remembered via the page's own IndexedDB across page reopens (Chromium may ask to reconfirm access once per browser restart). Drag-and-drop / plain file input remain as a universal fallback.
**Why:** requested "automatically loaded, hardcoded path." A literal zero-click hardcoded-path auto-load isn't something a browser permits (it would let any page read arbitrary local files) — this is the closest real equivalent.
**Files:** `hypothesis-ledger.html`.

### 19. Made Chart.js load failures visible
**What:** an on-page error box that explains when the charting library failed to load (e.g. no internet on first open) and reassures that the metrics table and log still work without it.
**Why:** closes the exact silent-failure gap found in item 15, so this class of bug can't recur invisibly.
**Files:** `hypothesis-ledger.html`.

### 20. Fixed: successful-but-worse iterations compounded regressions
**What:** an iteration is now only *adopted* (promoted into the live `pipeline/` directory that the next iteration's prompt reads from) if its validation primary matches or beats the best score seen so far. A regression is still logged honestly (`"adopted": false`) but discarded — the next iteration's prompt is built from the best-known code, never from a worse state.
**Why:** the real run showed iteration 0 (valid primary 0.6007) followed by three consecutive iterations that were each individually "successful" (the harness ran fine) but scored progressively worse (0.5983 → 0.5981 → 0.5975) — because each iteration's code was built on top of the *previous* iteration's regression, compounding the drift instead of exploring fresh ideas from the best point found.
**Files:** `agent/controller.py`, `agent/context.py` (prompt now states the current best explicitly).
**Verified:** a scripted functional test with a controlled score sequence (0.60 → 0.55 → 0.58) confirmed iteration 1's regression is correctly rejected, and — critically — iteration 2 is judged against the true best (0.60), not against iteration 1's rejected 0.55; confirmed the actual file on disk retained iteration 0's content afterward.

### 21. Added `LLMFatalError`: distinguish unrecoverable API failures from recoverable content failures
**What:** a new exception class for failures where the API call itself never completed (network down, bad/expired key, rate limit, quota exhausted, a 5xx, an unknown model name). These now abort the *entire run* immediately, rather than burning up to `MAX_REPAIR_ATTEMPTS` more identical, instantly-retried failures. Plain `LLMError` (truncation, malformed JSON — the call succeeded but the output was unusable) remains retryable as before.
**Why:** requested ("kill switch once LLM, code has error and disconnected"). Retrying an infrastructure failure with no backoff cannot fix it and just wastes time (and, for anything that reached the server, real tokens).
**Files:** `agent/llm_client.py`, `agent/controller.py`.
**Verified:** a functional test confirmed a persistent fatal error stops the run after exactly 1 call (not 4); a second test called `AnthropicClient.propose()` directly (not a stub) against a fake client that raises inside `messages.stream()`, confirming the real code path wraps it as `LLMFatalError`.

### 22. Fixed: token accounting under-reported spend on multi-attempt iterations
**What:** `controller.py` now sums LLM token usage across *every* attempt within an iteration (including failed repair attempts), not just the final attempt.
**Why:** a repair-attempt retry is a real, separately billed API call; only recording the last attempt's tokens meant the log (and the dashboard cost figures built on it) understated true spend whenever an iteration needed more than one try.
**Files:** `agent/controller.py`.
**Verified:** a functional test with a scripted LLM (3 failures + 1 success, fixed token counts per call) confirmed the logged total sums all 4 attempts, not just the last.

### 23. Added cost and consecutive-failure kill switches
**What:** two new CLI flags: `--max_cost_usd` (default 4.5, checked before each iteration starts) and `--max_consecutive_failures` (default 3, counting iterations that fully failed after exhausting repairs or hit a fatal error — not individual retries). Either stops the run cleanly with a clear printed reason.
**Why:** requested error-recovery/kill-switch behavior; the default cost cap is set with headroom under the user's stated $5 API budget.
**Files:** `agent/controller.py`, `run_agent.py`.
**Verified:** two functional tests — an always-failing stub confirmed the run stops after exactly 3 failed iterations (12 LLM calls = 3 × 4 attempts, no 4th iteration attempted); a stub reporting inflated token counts confirmed the run stops once cumulative estimated cost reaches the cap.

### 24. Added expected-vs-actual improvement tracking
**What:** the prompt (`context.py`) now asks the model for its own `expected_valid_primary_delta` prediction and states the current best score explicitly ("best so far: 0.6007"). `controller.py` prints the hypothesis and expected delta *before* running the sandbox, then prints the actual delta next to it once the result is known, and logs both fields (`expected_delta`, `actual_delta`, `reference_primary`, `adopted`) into `iteration_log.jsonl`.
**Why:** requested, to compare the model's stated confidence against reality and see whether its self-predictions are calibrated over a longer run.
**Files:** `agent/context.py`, `agent/llm_client.py` (`Proposal.expected_delta` field), `agent/controller.py`.
**Verified:** covered by the same functional tests as items 20-23 (the expected/actual fields are asserted directly in the regression test).

### 25. Implemented: multiple candidate proposals per iteration, keep the best (proposed item 1)
**What:** `controller.run()` gained a `candidates_per_iteration` parameter (default 2, CLI: `--candidates_per_iteration`). Each iteration now asks the LLM for that many independent proposals in the same slot — each applied to its own scratch copy and run under the usual timeout — and only the highest-scoring successful candidate is kept: promoted (if it beats the reference score), snapshotted to `best/` (if it's a new global best), and logged. The losing candidates' scratch copies are deleted; their hypotheses and scores are NOT logged as separate iterations, only summarized in the winning record's new `candidates_tried` / `candidate_scores` fields. A fatal LLM/API error on any candidate still aborts the whole run immediately (after keeping whatever earlier candidate in that same iteration already succeeded, rather than discarding a good result).
**Why:** the real 4-iteration run showed iteration 0 as the best result three times in a row across separate runs — a single proposal per iteration has no way to recover from an LLM guess that happens to be poorly tuned, and the organizer's ε/N convergence rule allows very few iterations before a mandatory stop. Trying 2 independent guesses per slot raises the odds at least one is good, for roughly proportional token cost (still bounded by `--max_cost_usd`).
**Files:** `agent/controller.py` (new `_propose_and_run_one_candidate()` helper), `agent/context.py` (prompt now takes `candidate_idx`/`candidates_per_iteration`), `run_agent.py` (new flag).
**Verified:** a functional test with 2 scripted candidates (scores 0.55 and 0.60) confirmed the higher scorer wins, gets promoted/snapshotted, and its sibling's scratch dir is deleted; token counts summed correctly across both candidates; a second test confirmed a fatal error on candidate 2 still keeps and logs candidate 1's already-successful result before halting the run.

### 26. Implemented: refine-before-pivot — later candidates see earlier candidates' results (proposed item 2)
**What:** a new `format_sibling_candidates()` in `context.py`, included in the prompt only when `candidates_per_iteration > 1`. Each candidate after the first is shown the hypothesis and score (or error) of every candidate already tried *this same iteration*, and is explicitly told it may either refine a near-miss with a concrete different parameterization or pivot to a genuinely different hypothesis — with a preference for refining when a sibling scored close to (or above) the current best but looks under-tuned.
**Why:** without this, a second candidate in the same iteration is just a second independent guess; it can't build on a first candidate that was directionally right but needed different hyperparameters. This is distinct from `format_history_summary()`, which only covers *past* iterations, not siblings in the current one.
**Files:** `agent/context.py`.
**Verified:** a functional test confirmed candidate 2's prompt contains candidate 1's hypothesis text and its exact score, and that candidate 1's own prompt correctly states "candidate 1 of N" with no sibling section when it's the first.

### 27. Implemented: isolated-change constraint (proposed item 3)
**What:** added an explicit paragraph to the prompt's constraints section: touch only what the hypothesis targets, and copy every other hyperparameter/constant/code path unchanged from the current best file — no incidental "while I'm in there" tweaks alongside the real change.
**Why:** a full-file rewrite makes it easy to change five things at once, which makes `actual_delta` an unreliable read on the one thing the hypothesis claimed to test, and makes it impossible for a later iteration to cleanly keep just the part that worked.
**Files:** `agent/context.py`.
**Verified:** confirmed present in the built prompt (functional test asserts the constraint text renders for both single- and multi-candidate prompts).

### 28. Fixed: a second run had no memory of the first run's best result (cross-run best persistence)
**What:** `best/` now also carries a small `_best_meta.json` (score, iteration, hypothesis, timestamp) written every time a new best is snapshotted. At the start of `run()`, if this file exists, both the score to beat *and* the actual `data.py`/`baseline.py` content are restored from `best/` into `pipeline/` before iteration 0 — not just the score.
**Why:** found by reading the user's own real `logs/iteration_log.jsonl`: a second real run (started after item 20's regression-compounding fix was already deployed) began with `best_valid_primary = -1.0` / `reference_primary: null`, meaning it had zero memory of the *first* run's 0.6007 result sitting correctly in `best/` — and had even inherited a worse `pipeline/` state left over from the first run predating item 20's fix. Every prior fix in this log only protected a single `run()` call; nothing previously persisted a validated high-water mark *across* separate `python run_agent.py` invocations.
**Files:** `agent/controller.py` (`_load_best_meta()`, `_write_best_meta()`, resume block at the top of `run()`).
**Verified:** a functional test seeded `best/` with a prior 0.6007 result and a stale, worse `pipeline/`, then ran two more iterations (one deliberately worse, one better): confirmed the resume message and restored code, confirmed the worse candidate was correctly rejected against the *resumed* 0.6007 (not silently accepted for lack of a reference), confirmed the second, better candidate was adopted and became the new best, and confirmed `pipeline/` ends up holding exactly the winning code with no trace of the discarded stale/worse states. Two bugs were caught by this test pass before shipping: (a) a failed iteration immediately after a resume was feeding the convergence monitor's stall-detection window a `-1.0` sentinel instead of the real resumed best score (harmless before this feature existed, since `best_valid_primary` could only be `-1.0` when nothing had ever succeeded — actively wrong once resume can seed a real score before iteration 0), caught and fixed with a dedicated test that spies on `ConvergenceMonitor.record()`; (b) `sandbox.promote()`'s generic "copy every file" behavior was copying the bookkeeping `_best_meta.json` itself into `pipeline/` (and from there into every scratch copy) on resume — harmless (nothing reads unknown files) but sloppy, now explicitly stripped right after the resume-copy.

### 29. Fixed: the hidden test set was scored on every iteration, not once (confirmed rule violation)
**What:** `pipeline/_run_iteration.py` now pops `'test'` out of `splits` before calling `run_fm()`, so the per-iteration harness cannot compute a test score at all — not "isn't asked to," structurally absent. `run_fm()`'s required contract shrank to `{'valid': {...}}` (was `{'valid':..., 'test':...}`); `sandbox.run_scratch()`'s output validation and `agent/context.py`'s prompt were both updated to match, and the prompt now explicitly warns the agent not to reach for `splits['test']` inside `run_fm()`. `agent/finalize.py`'s one-time `train_and_predict()` call remains the sole, unchanged path allowed to see the test split — it calls `data.load()` directly, not through the stripped-down per-iteration path.
**Why:** re-auditing against the full challenge brief (pasted by the user after the previous walkthrough) confirmed this codebase's `test` split IS the challenge's hidden test set — the starter kit's own published FM baseline numbers (GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946) are a byte-for-byte match to the brief's "Published hidden-test scores." The brief states, four separate times, that the hidden test set must never be touched during development, only scored once. The prior interface contract *required* `run_fm()` to return real test-split metrics every iteration (`sandbox.run_scratch()` failed any iteration missing them), so every iteration of every real run to date scored against the hidden test set. The mitigating fact is that `agent/context.py`'s prompt never surfaced test numbers to the LLM (decision-making was always valid-only, per item 20's adoption logic), but the log itself — a required deliverable — carried the evidence trail regardless of whether it influenced anything.
**Files:** `pipeline/_run_iteration.py`, `agent/sandbox.py`, `agent/context.py`, `agent/controller.py` (`_write_best_meta()`'s `test_primary` param is now optional/`None` during iteration — see item 31).
**Verified:** a real subprocess test (not a stub) ran the actual patched `_run_iteration.py` against a fake `baseline.py` that records exactly which split keys it received — confirmed `'test'` is absent from what `run_fm()` sees, and confirmed the JSON output has no `'test'` key. A second test used a deliberately stale `baseline.py` that still reads `splits['test']` — confirmed it fails loudly with `KeyError: 'test'` (a real, retryable iteration failure the agent would see and self-correct from) rather than silently succeeding. A third test confirmed `sandbox.run_scratch()`'s real JSON-parsing path now accepts a valid-only metrics dict as success.

### 30. Added: code diff logged per iteration (required run-log field, was missing entirely)
**What:** each iteration record now includes `code_diff` — a unified diff (`difflib.unified_diff`, capped at 20,000 chars) of the winning candidate's `new_content` against `target_file`'s pre-change content in `pipeline/`.
**Why:** the challenge brief's run-log deliverable is explicit: "each iteration should record its hypothesis, **the code diff applied**, the resulting metrics, and any error/recovery events." The log never recorded a diff, or even the full proposed content — only the hypothesis text and metrics. This is a required deliverable field that was simply absent.
**Files:** `agent/controller.py` (`_compute_diff()`, wired into `_propose_and_run_one_candidate()`'s returned outcome and the winning record).
**Verified:** a functional test proposed a one-line change to a 3-line fixture file and confirmed the logged `code_diff` contains the actual `-`/`+` lines of the real change, present both in the in-memory record and the persisted `iteration_log.jsonl` line.

### 31. Added: explicit baseline-reproduction check before iteration 0 of a fresh run
**What:** `_check_baseline_reproduction()` runs the untouched, seeded pipeline once — no LLM call, no code change — before iteration 0 of any run that isn't resuming from a prior `best/` (see item 28), compares the result to `baseline_scores.json`'s official validation number, prints a one-line confirmation, and writes `logs/baseline_reproduction.json`. Best-effort: a failure here is reported but never blocks iteration 0 from starting, since a genuinely broken seeded pipeline will surface the identical error through the normal iteration/repair path anyway.
**Why:** the brief's step 1 is explicit: "Reproduce the official baseline... confirm it reaches the official baseline's reported validation score," before iterating. Nothing in the codebase ever did this for a real run — iteration 0 went straight into proposing a change (confirmed by the user's own real run: iteration 0's hypothesis was already a curriculum-schedule loss change, not a reproduction check). The only place a reproduction step existed was `DryRunClient`'s hardcoded first plan step, which only runs under `--dry-run`, never in a real, scored run.
**Files:** `agent/controller.py` (`_check_baseline_reproduction()`, called from `run()`).
**Verified:** a functional test confirmed `logs/baseline_reproduction.json` is written with the correct measured/official comparison on a fresh run, and confirmed it is *not* written (and consumes no extra sandbox run) on a resumed run, since that evidence already exists from whichever earlier run first produced the `best/` snapshot being resumed from.

### 32. Added: automatic finalize + manual-intervention summary at the end of a run
**What:** `run_agent.py` now calls `agent/finalize.py` automatically once the run stops (new `--submission_out` / `--skip_finalize` flags), rather than requiring a second, manual command — this is the one sanctioned, one-time scoring of the hidden test set, and the final printed summary includes the hidden-test primary score and its delta over the official baseline. `finalize()` also now patches the real test score into `best/_best_meta.json`'s `test_primary` field (left `None` by every iteration since item 29) once it's computed. Separately, `controller.run()`'s return value now includes `manual_interventions`, an aggregate count of every record's `manual_intervention` flag across the whole run, printed in the final summary.
**Why:** the brief frames the submission as something "the agent designates as final," but producing it was a separate manual step a human had to remember to run. Deliverable 3 also explicitly asks for "a short summary reporting the number of manual interventions during the run" — the per-record flag existed, but nothing ever aggregated or reported it.
**Files:** `run_agent.py`, `agent/finalize.py` (patches `_best_meta.json`'s `test_primary`), `agent/controller.py` (`manual_interventions` in the return dict).
**Verified:** a functional test built a fake `best/` snapshot with `train_and_predict()`/`submit.py` stand-ins, called `finalize()` directly, and confirmed the submission CSV is written, the returned metrics carry the test score, and `_best_meta.json` is patched with `test_primary` while `valid_primary`/`iteration`/`hypothesis` are preserved unchanged. A second test confirmed the `manual_interventions` aggregation is 0 for a fully autonomous run and correctly counts a flag flipped by hand.

---

## Proposed, discussed, but **not yet implemented**

Items 1-3 from two rounds ago are implemented — see items 25-27. What's left:

1. **Meta-strategy, not code:** the ε/N rule governs the organizer-scored run, but nothing prevents an unscored exploratory run with a much higher iteration cap to find a promising direction offline first, then a single clean official run seeded from that finding.
2. **Local LLM inference (discussed, not started):** the user asked about running the reflect+revise step on a local RTX 5060 (8GB GDDR7) instead of the Anthropic API. `controller.py` already treats the LLM as a swappable `.propose(prompt, iteration) -> Proposal` dependency (see `AnthropicClient`/`DryRunClient` in `agent/llm_client.py`), so this is additive, not a redesign — add a sibling class (e.g. an Ollama-backed `LocalOllamaClient`) and a `--provider` flag. Flagged concern going in: this workload's real prompts run ~18K input / ~17K output tokens per call (see item 24's fields in the real log), well past the ~8K-context comfort zone generally recommended for smooth performance on an 8GB card, and a quantized 7B-class model is meaningfully weaker than Sonnet at holding the strict interface contract — expect more repair-attempt loops, which could offset any latency/cost win. Worth a short manual single-iteration trial before committing a full run to it.
3. **Numpy-only constraint vs. the brief's explicit allowance of any framework (discussed, deliberately left as-is this round).** `agent/context.py`'s prompt currently forbids torch/pandas/sklearn. The challenge brief explicitly allows "any open-source library or framework (PyTorch, RecBole, TorchRec, LightGBM, ...)" and 20% of scoring (Innovation & Problem Insight) specifically rewards going "beyond naive baseline tweaks" — the numpy-only constraint is self-imposed, not organizer-required, and forecloses the more sophisticated directions already listed in `KNOWN_HEADROOM_DIRECTIONS` (DIN/SIM sequence modeling, multi-task learning) from ever being properly implemented. Compute isn't the blocker (GPU-hours aren't capped, only reported) — this is a scope decision the user deferred rather than a technical one.
4. **Section 2.3's metric definition conflicts with the rest of the brief (organizer clarification needed, not a code fix).** The brief's "Limits" row states `NDCG@10 / Recall@50, click = positive`, which contradicts the GAUC/nDCG@5/`long_view` definition stated consistently everywhere else (2.4, 2.6, and `evaluate.py` itself — the sole scoring authority per the starter kit). Very likely a copy-paste artifact from a different track's template, but worth a one-line clarification request to organizers rather than assuming either way.

---

## Known constraints worth preserving in any refactor

- `data.py` and `baseline.py` are the *only* two files the agent may rewrite (enforced in `sandbox.apply_change()`); the interface contract (`load()`, `run_fm()`, `train_and_predict()`) must stay stable regardless of what the agent changes internally — see `pipeline/README.md`.
- The convergence rule (ε=0.002, N=3, 50-iteration cap, 6h wall-clock) is organizer-mandated and must be implemented exactly, not approximated. This is evaluated per `run()` call (i.e. per `python run_agent.py` invocation) — it does not span across separate runs, only the *best score* does (see item 28).
- All file I/O must use `encoding="utf-8"` explicitly (see item 6) — this codebase runs on the user's Windows machine, where the default locale codec is not UTF-8.
- Claude Sonnet 5 pricing ($2/$10 per MTok in/out) is hardcoded in `agent/llm_client.py` (`PRICE_PER_MTOK_IN`/`PRICE_PER_MTOK_OUT`) and duplicated in `hypothesis-ledger.html`'s JS — keep both in sync if pricing changes, or consider centralizing.
- `best/_best_meta.json` is bookkeeping owned by `controller.py` (see item 28), not part of the pipeline's interface contract — `sandbox.promote()`/`snapshot_to()` stay generic "copy everything" functions and know nothing about it; if you add another file to `best/` that shouldn't leak into `pipeline/` on resume, strip it explicitly the same way, rather than teaching `sandbox.py` about specific filenames.
- **The hidden test set (item 29) is the single most important constraint in this codebase to preserve.** `run_fm()` must never receive or be able to compute against the `'test'` split during iteration — only `agent/finalize.py`'s one-time `train_and_predict()` call, after the run has already stopped, is allowed to. If you ever restructure `_run_iteration.py` or the interface contract, re-verify this hasn't quietly regressed (e.g. by re-running the subprocess test in this round's changes) before running against real data again.
