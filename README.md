# Autonomous ML Research Agent — KuaiRand-Pure

An LLM-driven agent that runs the **entire ML research loop** on a frozen recommender benchmark:
it reproduces the official baseline, states the bottleneck it sees, proposes an experiment, edits real
model code or configuration, has that change validated before any training, trains, evaluates, compares
against a control with a paired bootstrap, decides what to investigate next, assembles a complementary
model portfolio, and writes a validated submission — then stops when the **organizer's own convergence
rule** says to.

Built for **TikTok TechJam 2026, Track 2** (Autonomous ML Research Agent for Recommender Systems).

```
OBSERVE → HYPOTHESIZE → PLAN → CODE → GUARD → TRAIN → EVALUATE
   → COMPARE → REFLECT → DECIDE → next experiment → ENSEMBLE → FINALIZE
```

---

## The problem

For each user, rank *that user's own* logged impressions by `long_view` (a binary watch-completion
signal). Nothing is retrieved from a global catalogue — this is **within-user ranking**, which is why
purely user-side features carry no signal at all.

Score: `primary = ½(GAUC + nDCG@5)`, computed by a frozen evaluator.

Two numbers govern everything: the **oracle ceiling is ≈0.85, not 1.0** (38% of users are all-positive
or all-negative, so no model changes their nDCG), and the **paired validation noise floor is σ ≈ 0.0009**.
Real gains are measured in thousandths, and an agent that cannot tell signal from noise will chase its
own variance.

## Current result

From `runs/run_20260831_090457` (live Gemini run — 4 executed experiments, 450 s, 12,593 tokens,
**0 manual interventions**):

| | primary |
|---|---|
| FM baseline (reproduced exactly by the agent's root node) | 0.60147 |
| Best single model discovered | 0.60366 |
| Portfolio, tuned on all of valid — *optimistic, in-sample* | 0.60463 |
| **Portfolio, honest 5-fold user-level CV** | **0.60409 ± 0.00141** (**+0.00259** over FM) |

Stopped on `official_convergence` (ε=0.002, N=3). Submission passed the frozen `submit.py --check`.

**We quote the cross-validated number, not the tuned one.** The system labels the larger value as
in-sample and separately measured the size of that optimism (≈+0.0007).

## What makes it autonomous

* **It states a problem before proposing one** — `problem_identified` is a required, first field.
* **It writes and adopts real code**, not just hyper-parameters: one of six block bodies, or a whole
  model family.
* **It refuses to fabricate.** Asked for a position-bias tower, it replied that position features do
  not exist in this dataset — and it was factually right.
* **It interprets results statistically.** Every node is compared to its control with a paired
  user-level bootstrap; `P(Δ>0)` drives the verdict, not a raw delta.
* **It reasons about diversity, not just accuracy** — keeping a model with a *worse* standalone score
  because its errors are decorrelated.
* **It disbelieves implausible results.** Its behavior-aware-history experiment showed **+0.0165**
  (18× the noise floor); the agent's own machinery surfaced the magnitude, the mechanism was diagnosed
  as an evaluation artifact, and the effect became **−0.00167** under an honest policy. The feature
  ships **disabled**. See [SUBMISSION.md §10](docs/SUBMISSION.md#10-self-correction-the-rejected-breakthrough).

## Architecture at a glance

Two layers behind a fixed boundary — *what* is searched is separate from *how* it is searched.

```
FROZEN HARNESS   data.py · evaluate.py · submit.py · pipeline/run_node.py · pipeline/contracts.py
─────────────────────────────── trust boundary ───────────────────────────────
AGENT (agent/)                          SOLUTION SPACE (pipeline/)
  orchestrator   control loop             baseline_blocks/  FM control (root node)
  roles/         Proposer·Coder·Reflector lib/din_blocks/   DeepFM + DIN
  tree           best-first + ε-explore   lib/lgbm_blocks/  LightGBM LambdaRank
  blockspec      honoured-knob contract   lib/…             losses · trainer · caches
  stats          paired user bootstrap
  portfolio      valuation + K-fold CV  OBSERVABILITY
  memory/ledger  evidence, cross-run      dashboard/research-console.html
```

**The frozen trust boundary** is the central design decision: five files are SHA-256-pinned and
verified at the start of every run, so the agent can never — accidentally or via a hallucinated
"fix" — modify the code the score depends on. Every capability routes *around* that boundary, never
through it. Details in [SYSTEM.md §3](docs/SYSTEM.md#3-frozen-trust-boundary).

## Quick start

```bash
# 1. Install
pip install numpy torch lightgbm google-genai pydantic pyyaml scipy

# 2. Data (~195 MB, no registration) into KuaiRand-Pure/data/  — from https://kuairand.com

# 3. Run
python -m agent.run --smoke     # reproduce the FM baseline exactly + verify frozen.lock  (~35 s)
python -m agent.run --mock      # the full research loop offline — no API key, no credits
python -m agent.run --faults    # inject a crash, verify recovery, still finalize a submission
python -m agent.run             # LIVE (needs GEMINI_API_KEY or .env.local)
```

The first run builds `runs/_cache/` (~60 s); every run after reuses it.

**Tests** (all deterministic, no API key needed). Each is a standalone module:

```bash
python -m tests.test_stats          # bootstrap matches the frozen evaluator to <1e-5
python -m tests.test_blockspec      # honoured-knob contract, incl. runtime bit-identity checks
python -m tests.test_leakage        # holdout isolation; hostile blocks rejected
python -m tests.test_orchestration  # convergence semantics, dedup, evidence ≠ status
python -m tests.test_sequence       # chronology + feedback-state leakage safety
```

Two interpreters are supported; prefer `cudaenv/Scripts/python.exe` (Python 3.12 + CUDA torch) — FM and
LightGBM are deterministic and interpreter-independent, only DIN differs and runs much faster on GPU.

## Research Console

A live/replay view of the agent's actual research loop, built for the demo. It renders artifacts the
agent emitted — it never simulates execution.

```bash
python -m agent.console_server        # http://127.0.0.1:8712/
```

**▶ Replay** replays a completed run at 1×–12× (no API key, network or GPU — this is what a demo video
records). **◉ Live** streams events from a running agent.

## Where to read next

| Document | For | Contents |
|---|---|---|
| **[docs/SUBMISSION.md](docs/SUBMISSION.md)** | judges & reviewers | The competition narrative: why this is a researcher rather than an AutoML script, results with honest uncertainty, the rejected-breakthrough episode, resource usage, rubric mapping, the 4-minute demo script |
| **[docs/SYSTEM.md](docs/SYSTEM.md)** | engineers | How it works now: trust boundary, six-block solution space, cache and leakage architecture, agent internals, config validation, statistical machinery, portfolio assembly, convergence semantics, configuration reference, run artifacts, extension rules, known limitations |
| **[docs/RESEARCH.md](docs/RESEARCH.md)** | reviewers & researchers | The scientific record: metric derivations, the noise model, statistical methodology, every measured finding **including the negative ones**, literature grounding, residual uncertainties, future directions |
| [docs/PROBLEM_STATEMENT.pdf](docs/PROBLEM_STATEMENT.pdf) | reference | The organizer's brief (immutable) |

Evidence labels **[VERIFIED] / [MEASURED] / [LITERATURE] / [PROPOSED]** are used throughout
RESEARCH.md so implemented facts, measurements and future work are never confused.

## Honest status

The margin over baseline is modest (+0.0026 honest) in a benchmark whose ceiling is 0.8484 and whose
noise floor is 0.0009. Every number is validation-side; test performance is unmeasured by
construction. Several open questions — notably all auxiliary-task arms — are **inconclusive for lack of
statistical power**, and are reported as such rather than rounded into wins. Full limitations:
[SUBMISSION.md §11](docs/SUBMISSION.md#11-current-limitations) and
[SYSTEM.md §24](docs/SYSTEM.md#24-known-engineering-limitations).
