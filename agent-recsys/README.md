# agent-recsys

An autonomous ML research agent for the KuaiRand-Pure challenge, built as a
deterministic Python controller wrapped around the verified, organizer-
matching starter kit (`pipeline/`) -- not a rewrite of it. See
`pipeline/README.md` for exactly what the agent is and isn't allowed to
edit, and the interface contract that keeps the controller decoupled from
whatever model/features the agent ends up with.

## Layout

```
pipeline/            seeded copy of the starter kit + one harness file
  data.py             <- the agent edits this (feature engineering)
  baseline.py         <- the agent edits this (model / training strategy)
  evaluate.py          fixed official scoring code -- never edited
  submit.py             used once at the end, by agent/finalize.py
  _run_iteration.py    harness-owned; the stable data.load()/baseline.run_fm()
                        contract the controller depends on every iteration
  baseline_scores.json  official baseline numbers + convergence rule (from
                        the problem statement: epsilon=0.002, N=3)

agent/
  context.py           builds the LLM prompt: data profile, known dead ends
                        + ranked headroom directions (from the starter kit
                        README), bounded iteration history, current code
  llm_client.py         AnthropicClient (real calls) + DryRunClient (no
                        API key, fixed hypothesis list -- for testing)
  sandbox.py             snapshot -> apply one-file rewrite -> run under a
                        timeout -> promote only if it ran clean
  convergence.py        epsilon/N stopping rule + iteration cap + wall-clock
                        cap + token/time accounting
  controller.py           the outer loop tying the above together, with
                        retry-then-rollback on a failed iteration
  finalize.py             runs once at the end: best/ snapshot -> real
                        predictions -> submission CSV -> validated exactly
                        like `submit.py --check`

run_agent.py          CLI entrypoint
logs/                 iteration_log.jsonl + resource_summary.json, written
                       at runtime (this is your Run & Iteration Logs
                       deliverable, produced automatically)
best/                 validation-best pipeline snapshot, written at runtime,
                       plus _best_meta.json (score/iteration/timestamp) so
                       the NEXT run can resume from it -- see "Resuming
                       across runs" below
```

## Hidden test set: never touched during iteration

`data.load()`'s `'test'` split IS the challenge's hidden test set -- its
official baseline numbers match this repo's `baseline_scores.json` exactly.
Per the challenge rules ("Teams develop on train + validation only; the
hidden test set is scored once"), it must never be scored during
development. `pipeline/_run_iteration.py` enforces this structurally, not
just by instruction: it pops `'test'` out of `splits` before calling
`run_fm()`, so a rewritten `baseline.py` cannot compute a test score during
iteration even if it tries to -- the rows simply aren't there (see
`pipeline/README.md` for the exact contract). The only place the hidden
test set is ever scored is the one-time finalize step described below.

There is a **second, independent check** on top of that: `agent/sandbox.py`
raises `HiddenTestViolation` and stops the run outright if any iteration's
output carries a `'test'` key at all. It is deliberately not a recoverable
iteration error -- no alternative hypothesis could fix it, so retrying
would just burn repair attempts while the violation continued.

That redundancy was earned. The stripping line in `_run_iteration.py` was
silently lost once when a `git checkout` restored a commit predating the
fix, and nothing detected it -- the run kept going, the logs kept recording
test metrics, and this README kept claiming enforcement that was no longer
happening. The structural fix lives in a file that tooling outside this
repo can revert; the behavioural check catches the consequence on the very
next run regardless of how the revert happened. If you ever see
`RUN STOPPED -- HIDDEN TEST SET COMPLIANCE VIOLATION`, restore the
stripping line in `pipeline/_run_iteration.py` before doing anything else.

## Before you run it: point it at your data

`pipeline/` does not include the dataset. `--data_dir` defaults to
`../kuairand-starter-kit/KuaiRand-Pure/data`, i.e. it assumes `agent-recsys/`
and `kuairand-starter-kit/` sit side by side directly under `TECHJAM/` (which
is where both live as of this writing) -- so if that layout holds, you don't
need to pass `--data_dir` at all. Pass it explicitly to override, e.g. if you
move the dataset again:

```powershell
python run_agent.py --dry-run --data_dir "C:\path\to\KuaiRand-Pure\data" --max_iterations 3
```

## Try it with --dry-run first (no API key, no cost)

`--dry-run` uses a fixed 3-step hypothesis list instead of calling an LLM,
so you can confirm the whole loop -- sandboxing, promotion, logging,
convergence, finalize -- works on your machine before spending any real LLM
tokens. This is exactly what I ran before handing this over: 3 dry-run
iterations against your actual KuaiRand-Pure data, all three completed
successfully (valid primary 0.6015 / 0.6008 / 0.6007), the best snapshot
was correctly saved to `best/`, and `agent/finalize.py` then regenerated a
170,588-row test submission from it and validated it cleanly. Full command
sequence (run from inside `agent-recsys/`):

```powershell
python run_agent.py --dry-run --max_iterations 3
```

(a `--dry-run` also auto-finalizes by default, same as a real run -- see
"Running it for real" below for what that produces.)

## Running it for real

1. Get an Anthropic API key and `pip install anthropic` (or
   `pip install anthropic --break-system-packages` if your Python is
   externally managed).
2. `set ANTHROPIC_API_KEY=sk-...` (PowerShell: `$env:ANTHROPIC_API_KEY="sk-..."`)
3. `python run_agent.py --max_iterations 50`

Before iteration 0 of a genuinely fresh run (nothing to resume from -- see
"Resuming across runs" below), it first runs the untouched, seeded pipeline
once -- no LLM call, no code change -- and compares the result against the
official validation number, printing a line like:

```
baseline reproduction check: measured valid primary 0.6012 vs. official
0.6016 (delta -0.0004)
```

and writing the same comparison to `logs/baseline_reproduction.json`. This
is the explicit "confirm it reaches the official baseline's reported
validation score" step the brief asks for as step 1, as its own auditable
artifact rather than an assumption that iteration 0 happens to start from
unmodified code.

That measured score also becomes the floor iteration 0 has to clear: a
first hypothesis that scores below it is rejected (`"adopted": false`)
exactly like any later iteration would be, and `best/` falls back to the
unmodified baseline itself rather than sitting empty, so a run where
nothing ever beats the baseline still leaves a valid, submission-ready
state behind. Earlier this was not the case -- iteration 0 had no
reference to be judged against and was adopted unconditionally, so a
first hypothesis that made things worse could still become "current."

It will then stop on its own per the organizer's convergence rule (validation
primary not improving by more than 0.002 over 3 iterations), the 50-
iteration cap, or the 6-hour wall-clock ceiling -- whichever comes first.
The convergence rule is evaluated per `run()` call, i.e. per invocation of
this command -- see "Resuming across runs" below for what *does* persist
across separate invocations.

Two kill switches stop the run early if something is going wrong, so a
disconnect or a run of bad LLM output can't silently burn your whole
budget unattended:

```powershell
python run_agent.py --max_iterations 50 --max_cost_usd 4.5 --max_consecutive_failures 3
```

- `--max_cost_usd` (default 4.5): stops before starting an iteration once
  estimated spend (Claude Sonnet 5's published per-token rate) reaches this
  many dollars. Pass `0` to disable.
- `--max_consecutive_failures` (default 3): stops after this many
  iterations in a row fail outright (exhausted repair attempts, or a fatal
  API error) -- not each individual retry within an iteration.

A fatal API failure (network down, bad/expired key, rate limit, quota
exhausted, a 5xx, an unknown model name) stops the run immediately on its
own, regardless of either flag above, since retrying instantly with no
backoff can't fix any of those.

By default each iteration tries `--candidates_per_iteration` (default 2)
independent proposals in the same slot and keeps whichever one scores
best; the others are discarded, not logged as separate iterations. This
roughly proportionally raises token spend per iteration (still bounded by
`--max_cost_usd`) in exchange for a better hit rate per iteration slot,
since the organizer's convergence rule only allows a handful of real
iterations before a mandatory stop. Pass `--candidates_per_iteration 1` to
reproduce the older single-candidate-per-iteration behavior. A candidate
past the first is shown what earlier candidates in the same iteration
scored, so it can refine a near-miss instead of only ever pivoting to
something unrelated.

Every iteration appends one record to `logs/iteration_log.jsonl`:
hypothesis, **a unified diff of the code change applied** (`code_diff`),
the resulting metrics (or the error if it failed), token usage, and
wall-clock -- this is your Run & Iteration Logs deliverable, in the schema
the challenge brief's deliverables ask for ("the code diff applied" is a
required field, not optional). Each record also carries the model's own
predicted `expected_delta` for that hypothesis (asked for explicitly in
the prompt) alongside the `actual_delta` once the result is known,
`reference_primary` (the best score this iteration was judged against),
and `adopted` (whether it beat that reference and was promoted -- see
"Adoption vs. logging" below).
`manual_intervention` is hardcoded `false` throughout an unattended run; if
you ever step in by hand (edit a file yourself, restart after a hang),
that's the field to flip to `true` for that iteration. The run's final
printed summary (and `result["manual_interventions"]` if you're calling
`controller.run()` directly) reports the total count across the whole run
-- the number the brief asks for in its deliverables and Autonomy scoring.

Once the run stops, it automatically finalizes: `agent/finalize.py` runs
once against the validation-best snapshot in `best/`, scoring the hidden
test set for the first and only time, and writes `submission.csv` (override
with `--submission_out`). This is the agent autonomously designating and
producing its final submission, not a manual step you have to remember --
pass `--skip_finalize` if you'd rather run `agent/finalize.py` yourself
later instead. The final printed summary includes the hidden-test primary
score and its delta over the official baseline, alongside the resource
numbers (tokens, wall-clock, iterations used) the Feasibility & Practicality
criterion is scored on.

## Adoption vs. logging: regressions are recorded, never built on

Every iteration's result is logged honestly, but only *adopted* (promoted
into the live `pipeline/` directory the next iteration's prompt is built
from) if its validation primary matches or beats the best score seen so
far. A worse result is kept in the log with `"adopted": false` and simply
discarded -- the next iteration still reasons from the best-known code,
never from a regression. Without this, a chain of individually-"successful"
iterations (the harness ran fine each time) can silently drift the actual
pipeline code worse and worse, each iteration compounding the last one's
regression instead of exploring fresh ideas from the best point found.

## Resuming across runs

`best/` carries a small `_best_meta.json` alongside the snapshot,
recording the score, iteration number, hypothesis, and timestamp of
whatever's saved there. The *next* time you run `python run_agent.py` --
even in a brand new process, hours or days later -- it reads that file (if
present) and resumes from it: both the score to beat and the actual
`data.py`/`baseline.py` content are restored from `best/` into `pipeline/`
before iteration 0, printing a line like:

```
resuming from a prior best: valid primary 0.6007 (iteration 2 of an earlier
run, saved 2026-...) -- pipeline/ restored to that state
```

Without this, a second invocation would only know about whatever was last
left in `pipeline/`, which is not necessarily the best code ever found (for
example if an earlier run predates the adoption-vs-logging fix above, or
was interrupted mid-iteration). `best/` is the one place a validated
high-water mark is guaranteed to live, so it's always the source of truth
on resume, never `pipeline/`.

## What the agent can and can't change

Only `data.py` and `baseline.py`, one full-file rewrite per iteration --
see `pipeline/README.md` for the exact interface contract that keeps this
safe (why `evaluate.py` is off-limits, what `_run_iteration.py` depends on).
The prompt built in `agent/context.py` already seeds the LLM with the
starter kit's own findings, so it won't waste iterations re-testing "add
more raw feature columns" or "bump embedding size" (both confirmed
non-improvements) -- and steers it toward the organizer-ranked open
directions instead: loss function (pointwise -> pairwise/listwise),
sequence modeling, multi-task learning, censored watch-time regression,
before architecture changes.

It also learns from this run's own history, not just the organizers' --
past iterations are shown to the LLM split into PROMISING (adopted;
"refine or extend this") and DEAD ENDS (ran but scored below what it was
judged against; "don't repeat or near-vary this, consider the opposite
direction instead"), rather than a flat chronological list. The
adopted/rejected verdict and the actual score delta were already recorded
per iteration -- this just makes the pattern explicit in the prompt instead
of leaving the LLM to infer it from raw numbers.

## Running against a local Ollama model

`agent/llm_client.py` also ships `OllamaClient`, which talks to a local
`ollama serve` instance instead of the Anthropic API -- no API key, no
per-token cost, `--max_cost_usd` becomes a no-op:

```powershell
ollama pull qwen2.5-coder:7b
python run_agent.py --local_model qwen2.5-coder:7b --candidates_per_iteration 1
```

`--ollama_host` (default `http://localhost:11434`) points at a remote
Ollama server if you're not running it on the same machine; `--ollama_num_ctx`
(default 32768) sets the context window requested from the model.

**Don't lower `--ollama_num_ctx` to fit VRAM.** Ollama silently *truncates*
a prompt longer than `num_ctx` rather than erroring, and the part most
likely to be cut is the tail -- which is exactly where this prompt keeps
its response-format spec and constraints, so the symptom is unexplainable
garbage output, not an error message. `OllamaClient` therefore estimates
the prompt size before sending and refuses outright if it plus a 10K-token
response reserve wouldn't fit, and separately warns after the fact if the
real token counts Ollama reports filled the whole window.

Measured against the actual pipeline files, the prompt is **~14.5K tokens**
(50,874 characters), so ~24.5K is needed once the response reserve is
counted. That fits `num_ctx=32768` and does *not* fit 16384 -- which is the
real bind on an 8GB card: 32768 tokens of KV cache is roughly 1.8 GB on top
of ~4.7 GB of weights for a 7B model, and Windows is already holding some
VRAM for the desktop. Expect Ollama to offload layers to CPU, which shows
up as slow generation *and* low GPU utilization. Shrinking the prompt (or
using a card with more VRAM) is the only real fix; lowering `num_ctx` just
trades a visible problem for a silent one. Requests are sent with
Ollama's `format: "json"` option, which constrains decoding to valid JSON
at the token-sampling level rather than relying on the prompt instruction
alone -- this matters more for a small model than a large one.

**What to actually expect, on an 8GB-class GPU (e.g. RTX 5060):** this
prompt runs roughly 15-20K input tokens (data profile, dead ends/headroom
directions, bounded history, both full pipeline files) and asks for a
complete file rewrite back -- easily 10K+ output tokens for `baseline.py`.
Published RTX 5060 Ollama benchmarks for 7-8B models cluster around
30-70 tok/s ([databasemart](https://www.databasemart.com/blog/ollama-gpu-benchmark-rtx5060),
[runaihome](https://runaihome.com/blog/best-llm-every-rtx-50-series-gpu-2026/)),
but every one of those numbers is measured at short-to-moderate context
that fits entirely in 8GB VRAM (roughly a 4K-8K token window at Q4_K_M);
both sources explicitly note that pushing past that on an 8GB card forces
KV-cache spillover to CPU, and don't publish a number for what throughput
becomes at that point. At this prompt's actual size, expect generation
closer to the low tens of tokens/sec than the high end of that range --
call it tens of minutes per candidate, not seconds. Over a run that
converges in ~6 iterations (like the real API run in this README's
history), that's plausibly 3+ hours wall-clock for one `--candidates_per_iteration 1`
run, versus the ~26 minutes the same iteration count took against the
Anthropic API.

Speed aside, a 7-8B model is also simply weaker than Claude Sonnet at two
things this task leans on hard: reliably emitting fence-free, schema-exact
JSON (mitigated somewhat by `format: "json"` above, but not eliminated),
and coherently rewriting an entire 500-800 line Python file without
corrupting logic unrelated to its stated hypothesis -- the "isolated
change" constraint in the prompt is much easier for a large model to
honor at this file size than a small one. Expect a higher failed/repaired
iteration rate, and likely a smaller, noisier improvement over baseline
than the Claude-driven runs already in this project's history -- possibly
none at all, if repair attempts eat most of the small iteration budget the
convergence rule allows (epsilon=0.002, N=3). Treat a local run as a
zero-marginal-cost way to explore or to keep iterating after a cost
budget is spent, not as a way to get a *better* result than the hosted
model produced.

Recommendations if you do run this: pick a coding-tuned model over a
general chat one at the same size (`qwen2.5-coder:7b` over a generic
`llama3.1:8b`) -- this task is 100% "rewrite a Python file correctly,"
which is exactly what a coding-tuned model is trained for. Use
`--candidates_per_iteration 1` (the default of 2 doubles wall-clock for a
cost tradeoff that no longer applies locally). And budget real time --
this is not a "kick it off and check back in twenty minutes" run the way
the API version is.

**What actually happened on the first real local run (`qwen2.5-coder:7b`,
default settings):** every iteration failed, and every failure was the
same shape -- `AttributeError: module 'baseline' has no attribute
'run_fm'`, `NameError: name '_train_fm_hybrid' is not defined`,
`NameError: name 'encode' is not defined`. The model wasn't writing wrong
logic; it was writing an *incomplete* file -- silently dropping helper
functions that live elsewhere in the ~800-line original while claiming
its output was "the COMPLETE new content." Repair attempts (the traceback
fed back, up to `MAX_REPAIR_ATTEMPTS` times) didn't reliably fix this,
since it isn't a small, nameable bug each time -- it's a coherence limit
at this file size. `--ollama_temperature` (default `0.2`, down from
Ollama's chat-tuned `0.8`) biases decoding toward reproducing the
existing file rather than paraphrasing it, which should help somewhat,
but don't expect it to close the gap entirely -- if a coding-tuned 7B
model at low temperature still can't hold the whole file together, that's
a real capability ceiling for this specific "rewrite an entire large file
losslessly except one function" task shape, not a settings problem.

## Swapping the LLM provider

`controller.py` only depends on an object with `.propose(prompt, iteration)
-> Proposal` and `.estimate_cost(input_tokens, output_tokens) -> float`
(see `agent/llm_client.py`). `AnthropicClient` and `OllamaClient` are the
two real implementations shipped here, but nothing about the controller is
tied to either -- add a sibling class calling whichever API/model you
choose (a different local runtime, another hosted provider, ...) and pass
an instance of it into `controller.run()` in `run_agent.py` instead.
