# SUBMISSION.md — TechJam 2026, Track 2

**Autonomous Machine Learning Research Agent for Recommender Systems** — competition-facing summary.

Readable without the rest of the repository. For how it works see [SYSTEM.md](SYSTEM.md); for the
evidence behind every number see [RESEARCH.md](RESEARCH.md); to run it see the [README](../../README.md).

---

## 1. Executive summary

We built an LLM-driven agent that runs the entire ML research loop on the KuaiRand-Pure within-user
video-ranking benchmark: it reproduces the official baseline, states the bottleneck it sees, proposes an
experiment, edits real model code or configuration, has that change **validated before any training**,
trains, evaluates, compares against a control with a **paired user-level bootstrap**, decides what to
investigate next, assembles a complementary model portfolio, and writes a validated submission — then
stops when **the organizer's own convergence rule** says to.

**Verified result** (`runs/run_20260831_090457`, live Gemini run, 4 executed experiments, 450 s,
12,593 tokens):

| Quantity | Value |
|---|---|
| FM baseline | 0.60147 |
| Best single model discovered | 0.60366 |
| Portfolio, tuned on all of valid — *optimistic, in-sample* | 0.60463 |
| **Portfolio, honest 5-fold user-level CV** | **0.60409 ± 0.00141** (**+0.00259** over FM) |
| Stop reason | `official_convergence` (ε=0.002, N=3) |
| Submission | PASSED the frozen `submit.py --check` |

**We quote 0.60409 ± 0.00141, not 0.60463.** The system itself labels the larger number as tuned and
in-sample, and measured the size of that optimism (≈+0.0007) rather than assuming it away.

**The result we are most proud of is a rejected one.** The agent's behavior-aware-history experiment
produced an apparent **+0.0165** — eighteen times the measured noise floor. Rather than banking it, the
magnitude was treated as implausible, the mechanism was investigated, a train/serve evaluation mismatch
was found, the experiment was rebuilt under an honest policy, and the effect became **−0.00167**. The
supposed breakthrough was rejected and the feature ships **disabled by default** (§10).

---

## 2. The challenge

Track 2 asks for an agent that autonomously runs:

```
understand problem → form hypothesis → modify pipeline/code → train → evaluate
  → interpret result → decide next experiment → repeat → select final solution
```

**Inner task.** For each user, rank *that user's own* logged impressions by `long_view`.
`primary = ½(GAUC + nDCG@5)`. Official FM baseline: 0.6016 valid / 0.5946 test.

**Two facts govern everything.** The oracle ceiling is **≈0.86, not 1.0** (27.1% of test users are
all-negative, 9.2% all-positive — no model changes their nDCG), and the paired validation noise floor
is **σ ≈ 0.0009**. So real gains are measured in thousandths, and an agent that cannot tell signal from
noise will confidently chase its own variance.

---

## 3. Why this is an autonomous ML researcher, not an AutoML script

An AutoML script searches a hyper-parameter grid. This agent does six things a grid search cannot:

| Behaviour | Concretely |
|---|---|
| **States a problem before proposing** | `Hypothesis.problem_identified` is a required field, emitted *first* |
| **Writes and adopts real code** | rewrites one of six block bodies, or adopts a whole model family; every edit is snapshotted and hashed |
| **Refuses to fabricate** | when asked for a position-bias tower it replied that position features do not exist in this dataset — and it was **factually right**, there is no slot column |
| **Interprets results statistically** | every node is compared to its control with a paired user bootstrap; `P(Δ>0)` drives the verdict, not a raw delta |
| **Learns from negative results** | a `rejected` verdict changes what it proposes next; negative evidence is first-class |
| **Reasons about diversity, not just accuracy** | it keeps a model with a *worse* standalone score because its errors are decorrelated |

And one thing most agents do not do: **it disbelieves its own best result when the magnitude is
implausible** (§10).

---

## 4. The end-to-end autonomous research loop

```
OBSERVE → HYPOTHESIZE → PLAN → CODE → GUARD → [DEBUG] → TRAIN → EVALUATE
   → COMPARE → REFLECT → DECIDE → next experiment → ENSEMBLE → FINALIZE
```

Three LLM roles as operators over a deterministic policy: **Proposer** (what problem to attack),
**Coder** (how to implement it in one block), **Reflector** (why it failed and how to recover).
Search is best-first with an ε exploration valve — deliberately not MCTS, because each node is a real
training run and the benchmark legitimately ends after ~4–6 experiments, far too few for rollouts to
produce reliable value estimates.

Every step emits a structured, auditable event. Nothing in the demo is narration.

---

## 5. Trust and integrity architecture

This is the part we would most like a technical judge to scrutinise.

**A frozen trust boundary.** Five files — `data.py`, `evaluate.py`, `submit.py`,
`pipeline/run_node.py`, `pipeline/contracts.py` — are SHA-256-pinned and verified at the start of every
run. The agent can never, accidentally or via a hallucinated "fix", modify the code the score depends
on. Every capability added during development routes *around* that boundary, never through it.

**Layered holdout protection — and an honest claim about it.** We found that our own earlier
documentation claimed label access was *"physically impossible."* It was not. Hidden-test labels and
the auxiliary `is_click` column were sitting inside the directory handed to every block, and
**ranking the validation set by `is_click` alone scores 0.7466 — 58.8% of the entire headroom above the
baseline, with no training at all.** That hole is now closed by five layers (separate holdout
directory, loader refusal, build-time assertion, static read guard, plausibility tripwire), and the
claim in our documentation was rewritten to what is actually true. This matters: METR observed explicit
reward hacking in 39 of 128 agent runs on RE-Bench with no prompting to cheat.

**Config-effectiveness validation.** A knob that the mounted model cannot read is rejected *before*
training rather than silently producing a baseline result. This closed a defect that had been
converting the strongest lever in the project into a fabricated negative result (§9).

**Provenance.** Each node records what was *intended* versus what was *executed*, with per-block source
hashes, cache version and a code-state hash — so no claim rests on the assumption that the agent's
proposal is what actually ran.

---

## 6. Scientific methodology

| Principle | Implementation |
|---|---|
| Separate the three variances | training stochasticity (0.00000 for FM/LightGBM, σ≈0.00025 for DIN) vs **validation-sample noise** (σ≈0.0009) vs cross-seed generalisation — never conflated |
| Reduce the variance that binds | a **paired user bootstrap** on saved predictions (≈2 s, no re-training) attacks validation noise; multi-seed *re-training* is reserved for stochastic models where it measures something real |
| Evidence ≠ adoption status | a node can be the champion and still `inconclusive`; standalone-`rejected` and still a portfolio asset |
| Honest reporting by construction | tuned/in-sample and cross-validated estimates are separate fields in every report |
| Respect multiple comparisons | one arm at P=0.91 out of 12 tests is explicitly **not** reported as a finding |
| Refuse to pool incompatible evidence | cross-run pooling requires the same cache version *and* code-state hash |

---

## 7. Example research trajectory (the live run, verbatim)

```
root  reproduce the official FM baseline                      0.60147
it 1  A  "Change the loss function to BPR to optimize
         pairwise ranking"                                    0.60361  confirmed  P(Δ>0)=1.00
it 2  D  "Adopt LightGBM as the model family"                 0.60205  rejected  (standalone)
         → but rank corr 0.860, the most decorrelated model found → kept as a portfolio candidate
it 3  B  "Adopt the DIN model family with BPR loss"           0.60366  inconclusive  P(Δ>0)=0.52
it 4  —  proposal rejected as a STRUCTURAL NO-OP before training → re-proposed
it 4  C  "Add auxiliary tasks (click and like)"               0.60247  rejected
         → rank corr 0.974, EMC +0.00000 → redundant, not carried into the portfolio
stop     official_convergence (ε=0.002, N=3) after 4 executed experiments of a permitted 50
final    portfolio n1+n2+n3+root  →  honest 5-fold CV 0.60409 ± 0.00141   submission PASSED
```

Four experiments, five proposals, one corrected before it wasted a training run. Note iteration 2:
the agent recorded LightGBM as standalone-*rejected* and simultaneously retained it as a portfolio
asset — those are different questions and the system answers them separately.

---

## 8. Results

### Honest vs tuned

| Estimate | Value | What it means |
|---|---|---|
| `final_valid_tuned` | 0.60463 | blend weights tuned on the *same* users it is measured on — **optimistic** |
| **`final_valid_honest`** | **0.60409 ± 0.00141** | 5-fold user-level CV **of the whole assembly procedure** — the defensible number |
| CV gain over best single | +0.00064 ± 0.00051 | roughly one SE: real, consistent, modest |

We measured the size of the optimism directly: over eight random user splits, weight-selection
optimism averaged **+0.00072**, i.e. roughly half of a naively reported ensemble gain.

### Baseline comparison

| Model | valid primary | source |
|---|---:|---|
| random | 0.4834 | `baseline_scores.json` |
| item popularity | 0.5807 | `baseline_scores.json` |
| **FM (official baseline)** | **0.60147** | reproduced exactly by our root node |
| FM + BPR (the discovered lever) | 0.60361 | +0.00214, P(Δ>0)=1.00 |
| best single (DIN) | 0.60366 | |
| **agent portfolio, honest** | **0.60409 ± 0.00141** | **+0.00259 over FM** |
| oracle ceiling | 0.8484 | |

### Portfolio contribution — the interesting part

| Member | Standalone | Rank corr | Marginal contribution |
|---|---:|---:|---:|
| FM + BPR | 0.60361 | 0.906 | +0.00025 |
| **LightGBM** | **0.60205** *(weaker)* | **0.860** *(most decorrelated of those found)* | **+0.00028** |
| DIN | 0.60366 *(best single)* | 1.000 | +0.00028 |
| FM + BCE (root) | 0.60147 *(weakest)* | **0.852** *(most decorrelated overall)* | +0.00017 |
| DIN + aux | 0.60247 | 0.974 | **+0.00000** *(redundant)* |

**A weaker model can be worth more than a stronger one.** LightGBM contributes **more than the
FM+BPR node** (+0.00028 vs +0.00025) while scoring 0.0016 *lower* standalone, because its errors are
decorrelated; the auxiliary-head node, with a standalone score *above* LightGBM's, contributes
exactly nothing. Selecting on standalone score alone would have kept the wrong
model.

### Robustness and recovery

`python -m agent.run --faults` injects a crash into a block. The Reflector patches it, the node
re-runs, and — notably — the recovered node is classified **`no_effect`**, because the patched block is
functionally identical to its parent. The recovery works *and* the system refuses to record it as a
scientific result. The run finalizes a valid submission. No run in this project has ever crashed.

### Resource usage (feasibility)

| | Live run |
|---|---|
| Executed experiments | 4 of a permitted 50 |
| Wall-clock | 450 s of a permitted 6 h |
| LLM tokens | 12,593 |
| Manual interventions | 0 |

**We do not try to spend the quota.** Under the organizer's rule a healthy run converges after ~4–6
experiments; the cap and the wall-clock ceiling are limits, not targets. The engineering goal was to
make each of those few experiments *real*, not to manufacture more of them.

---

## 9. Innovation

1. **Config-effectiveness validation.** We found that a proposed knob could be silently ignored by the
   mounted model — a hypothesis of "use BPR" trained plain BCE and returned a byte-identical baseline
   result, in 4 of 21 recorded runs. The agent recorded that as *"BPR → 0.6015, rejected"* and its own
   memory then told it not to retry its strongest lever. Fixing it recovered **+0.00214**. The general
   lesson — *intended intervention ≠ executed intervention* — is now a validated, recorded property of
   every experiment.
2. **Evidence separated from bookkeeping.** Scientific verdicts come from a paired bootstrap, not from
   whether a node beat its parent by an arbitrary margin. Before this, the agent's memory labelled its
   own best model "REJECTED — don't repeat."
3. **Portfolio-aware valuation.** Rank correlation, pairwise blend gain and leave-one-out marginal
   contribution are computed for every node from saved predictions — no re-training — and surfaced to
   the Proposer, so a diverse-but-weak model is treated as a success rather than a failure.
4. **Honest ensemble estimation.** K-fold cross-validation *of the whole assembly procedure*, with the
   four data roles (choose members / tune weights / report / refit) never conflated.
5. **Model identity preserved.** Test predictions come from the same trained instance whose validation
   predictions drove selection. For a stochastic model this matters: two trainings of one DIN config
   differ at rank correlation 0.926.
6. **Cheap analysis over expensive re-training.** Bootstraps, correlations and blend evaluation cost
   seconds on saved predictions; multi-seed re-training is spent only where training noise is the
   actual question.

---

## 10. Self-correction: the rejected breakthrough

**The single strongest demonstration that this is a research agent and not a leaderboard fitter.**

**The hypothesis.** In an autoplay short-video feed a skip is meaningful negative evidence and an
impression is not a positive — a mechanism with direct literature support (RecSys'23). Our DIN saw
history as video IDs only: a skipped video and a loved video looked identical. So each history event
was given a feedback state.

**The apparent result.** Behavior-aware DIN scored **0.61925** against **0.60275** without, across three
paired training repetitions: **+0.0165, P(Δ>0) = 1.00.**

**The skepticism.** Every other lever in this benchmark moves ~0.002. An effect eighteen times the
measured noise floor, from one extra embedding lookup, is not a plausible modelling gain — it is a
symptom. The result was investigated rather than banked.

**The diagnosis.** Feedback coverage was asymmetric:

| split | history events whose outcome the model could see |
|---|---:|
| train | 100.0% |
| **valid** | **100.0%** |
| **test** | **75.8%** |

A validation row could see the outcomes of *earlier validation rows*. A test row could not — a
submission scores all test rows at once, with no feedback in between. **The gain was structurally
available on the set used to choose models and structurally unavailable on the set that would be
scored.** It would have corrupted every selection decision in the run and delivered nothing on test.

**The rebuild.** The policy was changed so that only *training-window* outcomes may ever become
features, making validation and test symmetric. Under that honest policy:

| Arm (3 paired repetitions) | Δ vs no-feedback | P(Δ>0) |
|---|---:|---:|
| DIN + feedback states | **−0.00167** | 0.00 |
| DIN + feedback states, with feedback dropout | **−0.00070** | 0.14 |

The diagnosed remedy (masking states during training to match inference) recovered 58% of the loss —
**confirming the mechanism** — but the lever remained unhelpful. It was not rescued by portfolio
diversity either (rank correlation 0.942, blend gain +0.00002).

**The verdict.** A literature-supported hypothesis, an apparent 18σ win, and a **negative** final
answer. `use_fb` ships **disabled by default**. The prediction that this would be the project's biggest
lever was recorded in our own roadmap; it was disproven and the document was corrected rather than
re-argued. Total cost of finding out: about four minutes of GPU time.

---

## 11. Current limitations

Stated plainly, because they are the first things a reviewer should probe.

* **Every number is validation-side.** The honest CV estimate corrects for weight-tuning optimism, not
  for valid→test distribution shift. Test performance is unmeasured by construction.
* **The margin over baseline is small in absolute terms** (+0.0026 honest) — but the oracle ceiling is
  0.8484 and the noise floor is 0.0009, so this is a narrow band by nature.
* **The validation set cannot resolve effects below ~0.002.** Several open questions (all auxiliary
  tasks) are inconclusive for lack of statistical power, not for lack of design.
* **Cross-run evidence is currently empty.** The compatibility guard correctly refuses to pool the 14
  historical entries across cache/code changes — correct, but it means the ledger has not yet paid off.
* **The random-exposure surface runs for the FM family only**; DIN and LightGBM need sibling caches.
* **`max_llm_usd` is declared but not enforced** — there is no hard LLM spend cap.
* **Small repetition counts** (n=2–3) underpin the DIN and auxiliary measurements.
* **A live run is short by design** (4 experiments). That is the organizer's convergence rule working,
  not the agent giving up — but it does limit how much science one run can contain.

Full engineering limitations: [SYSTEM.md §24](SYSTEM.md#24-known-engineering-limitations).

---

## 12. Reproducibility

Deterministic paths need no API key, no network and no GPU.

```bash
python -m agent.run --smoke     # reproduce the FM baseline exactly (0.60147) + verify frozen.lock
python -m agent.run --mock      # the full research loop offline, scripted operators
python -m agent.run --faults    # inject a crash, verify recovery, still finalize a valid submission
python -m tests.test_stats      # bootstrap matches the frozen evaluator to <1e-5
python -m tests.test_blockspec  # honoured-knob contract, incl. runtime bit-identity checks
python -m tests.test_leakage    # holdout isolation; hostile blocks rejected
python -m tests.test_orchestration  # convergence semantics, dedup, evidence ≠ status
python -m tests.test_sequence   # chronology + feedback-state leakage safety
```

A live run needs `GEMINI_API_KEY` (or `.env.local`):

```bash
python -m agent.run
```

**An honesty note on `--mock`.** The mock path replays *scripted* operator moves from
`tests/mock_moves.py`. Its "0 interventions" is true of the machinery, but the research decisions are
scripted, not autonomous. **Every headline number in this document comes from a live Gemini run**
(`runs/run_20260831_090457`), where the hypotheses, the code edits and the decisions were the model's.
`--mock` exists so that the whole loop — including the integrity guards and the recovery path — can be
exercised deterministically with no API key.

Every run writes `run_log.jsonl` (per node), `events.jsonl` (per research event),
`resource_report.json` (accounting + tuned vs honest results) and a `submit.py --check`-validated
submission.

---

## 13. Research Console and the demo

**Presentation infrastructure, not a model improvement.** The console renders artifacts the agent
emitted; it never simulates execution and never computes a metric of its own.

```bash
python -m agent.console_server        # http://127.0.0.1:8712/
```

* **▶ Replay** — deterministic replay of a completed run at 1×–12×. No API key, network or GPU, so a
  recording never depends on live timing.
* **◉ Live** — polls a running agent and streams new events.

The status bar keeps two things visually distinct that are easy to conflate: the **official**
convergence state versus internal research bookkeeping, and **executed experiments** versus proposal
attempts.

### Suggested 4-minute demo

Replay `run_20260831_090457` at 2×:

| Time | Beat |
|---|---|
| 0:00–0:20 | Run start; frozen-boundary verification and holdout isolation appear as real guard events |
| 0:20–1:20 | Observation → BPR hypothesis → effective intervention → training → `P(Δ>0)=1.00 confirmed` |
| 1:20–2:10 | LightGBM: standalone **rejected**, yet rank corr 0.860 keeps it as a portfolio candidate |
| 2:10–2:40 | A proposal caught as a structural no-op and **corrected before training** |
| 2:40–3:20 | Portfolio panel: the weakest standalone member has the largest marginal contribution |
| 3:20–4:00 | Official convergence, honest CV estimate, submission PASSED |

If time allows, the strongest single beat is §10 — the +0.0165 result the agent rejected.

---

## 14. Mapping to the judging criteria

| Criterion | Weight | What we offer |
|---|--:|---|
| **Technical Execution** | 35% | +0.00259 honest over baseline with a stated confidence interval; a frozen trust boundary; five layers of holdout protection; config-effectiveness validation; content-based dedup; provenance; recovery verified by fault injection; five passing test suites |
| **Innovation & Problem Insight** | 20% | Intended-vs-executed intervention validation; evidence separated from bookkeeping; portfolio-aware node valuation; K-fold-honest ensemble estimation; a diagnosed and rejected evaluation artifact |
| **Impact & Relevance (autonomy)** | 20% | **0 manual interventions**; the agent proposes, codes, guards, trains, interprets statistically, self-corrects and stops under an explicit rule |
| **Feasibility & Practicality** | 15% | 450 s and 12,593 tokens for a complete run that beats the baseline; 4 experiments of a permitted 50; cheap analysis replaces expensive re-training wherever it can |
| **Presentation & Communication** | 10% | A console that renders the real research loop; four authoritative documents; every claim traceable to a run artifact or a test |

**The honest framing we would defend in person:** the score margin is modest, and we could have quoted
a larger one. We report the cross-validated estimate instead, we document the negative results
alongside the positive ones, and the single most interesting thing the agent did all project was throw
away its own best number.

---

*[README](../../README.md) · [SYSTEM.md](SYSTEM.md) · [RESEARCH.md](RESEARCH.md)*
