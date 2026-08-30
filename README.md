# Autonomous ML Agent — TikTok TechJam 2026 (PS2)

> **New to this repo?** Start at [`deliverables/00-START-HERE.md`](deliverables/00-START-HERE.md) — results, submission, run-log evidence and the demo script are all collected there, along with the open items that still need a human.

An LLM agent that runs the ML-engineering loop on its own — read the problem, form a hypothesis,
write code, validate it, train, evaluate, reflect, repeat — against a recommender-systems
benchmark, and keeps a full audit trail of everything it tried.

The agent runs on a **local Ollama model**, uses **numpy only** (no ML framework, no autograd, no
GPU), and never sees the hidden test split.

---

## Results

| Model | valid primary | test primary |
|---|---|---|
| `random` (sanity check) | 0.4834 | 0.4753 |
| `item_popularity` (official non-trained baseline) | 0.5807 | 0.5715 |
| **`fm_official` (official baseline to beat)** | **0.6016** | **0.5946** |
| `fm_bpr` (ours, single model) | 0.6031 | 0.5970 |
| **`fm_bpr` × 5-seed ensemble (ours, submitted)** | **0.6039** | **0.5974** |
| *oracle ceiling (scores from true labels)* | *0.8484* | *0.8645* |

**+0.0028 test primary over the official baseline.** Submission:
`deliverables/submission/submission_ens_test.csv`,
validated by `submit.py` (170,588 rows, row-alignment verified).

Context for the size of that number: `fm_official`'s own seed-to-seed standard deviation is
0.0008, and the entire hyperparameter space of the baseline model spans 0.0040 end to end. The
gain is real and reproducible across seeds, but this is a benchmark where movement is small.

---

## The task

**KuaiRand-Pure** short-video interaction logs — 27K users, 7.6K items, 1.4M interactions.

**Within-user ranking.** For each user, rank the impressions logged in the eval window so the
videos they actually watched long (`long_view = 1`) come first.

**Metric** (`evaluate.py`, fixed — do not modify):

```
primary = mean(GAUC, nDCG@5)
```

Both halves are *within-user* rankings — they never compare one user's rows to another's. That
single fact drives the entire result below.

**Splits are by date, not shuffled** — train on the past (04-08→04-21), validate on the near
future (04-22→04-28), test on the far future (04-29→05-08). Vocabularies and bucket edges are fit
on **train only**; unseen values map to a per-field UNK slot. This is deliberate — it simulates
deployment and avoids leakage.

---

## Agent architecture

```
                    ┌──────────────────────────────────────────┐
                    │  solution_tree.select()                  │
                    │  debug / draft / improve  + which node   │
                    └────────────────┬─────────────────────────┘
                                     ▼
   ┌──────────────┐        ┌──────────────────┐
   │ data_guard   │        │ hypothesis_agent │  problem → hypothesis → reasoning
   │ strips       │        │ prompts/         │  → implementable action
   │ 'test' key   │        │ hypothesize.md   │
   └──────┬───────┘        └────────┬─────────┘
          │                         ▼
          │      ┌──────────────────┴──────────────────┐
          │      │                                     │
          │      ▼ (improve: cheap path first)         ▼ (draft/debug, or if config can't express it)
          │  ┌──────────────┐                   ┌──────────────┐
          │  │ coding_agent │                   │codegen_agent │  writes a COMPLETE module
          │  │ action_space │                   │prompts/      │
          │  │ set_hyperparam                   │write_model.md│
          │  │ toggle_field │                   └──────┬───────┘
          │  └──────┬───────┘                          ▼
          │         ▼                          ┌────────────────┐
          │  ┌─────────────┐                   │ code_guardrail │ AST: no I/O, no eval,
          │  │ guardrail   │                   │ static analysis│ no 'test', API-contract
          │  └──────┬──────┘                   └───────┬────────┘ + use-without-import
          │         └──────────────┬───────────────────┘
          ▼                        ▼
    ┌─────────────────────────────────────┐
    │ debug_run   20k rows, 2 epochs      │  ── fails ──▶ error_recovery (prompts/repair.md)
    │ crash + plausibility + coverage gate│                └─▶ node marked BUGGY, diagnosis
    └──────────────┬──────────────────────┘                    attached for the next DEBUG
                   ▼
    ┌─────────────────────────────────────┐
    │ runtime budget gate                 │  5× the reference model's own estimate
    └──────────────┬──────────────────────┘
                   ▼
    ┌─────────────────────────────────────┐
    │ full_run (1.1M rows) → reeval        │  multi-seed recheck, VALID only
    └──────────────┬──────────────────────┘
                   ▼
    ┌─────────────────────────────────────┐
    │ decision.decide()                   │  COMMIT / KEEP_NODE / REVERT / REJECT_UNSAFE
    └──────────────┬──────────────────────┘
                   ▼
    ┌─────────────────────────────────────┐
    │ archivist → experiment_log.jsonl     │  + resume.save_state (crash-safe)
    │           → solution_tree.json       │  + viewer → dashboard.html
    └──────────────┬──────────────────────┘
                   └────────────▶ back to select()
```

### Search strategy

**Greedy / chain**, following **MLE-STAR** (Nam et al. 2025) rather than R&D-Agent's parallel
branches or ML-Master's MCTS — chosen for a hackathon compute and token budget. It is implemented
as an AIDE-style solution tree (`agent/solution_tree.py`) whose `select()` policy is:

1. **DEBUG** a broken-but-promising node first — the idea is already written, it just doesn't run,
   so one targeted fix is the cheapest possible win. Capped by `MAX_DEBUG_DEPTH` per *branch*.
2. **DRAFT** a new solution while the tree has fewer than `MIN_DRAFTS` working nodes, so the
   search doesn't commit to one early local optimum.
3. **IMPROVE** the best working node otherwise — pure greedy exploitation.

What we give up by not doing tree+MCTS: no cross-branch credit assignment and no principled
exploration bonus, so a promising-but-currently-worse branch gets abandoned. That was an accepted
trade for the timeline.

### The four R&D-Agent strategies

Design reference: Yang et al., *R&D-Agent: An LLM-Agent Framework Towards Autonomous Data
Science*, 2025 (arXiv:2505.14738). We implement the four components their ablation shows matter
most in a resource-constrained setting, not the full six.

| # | Strategy | Where it lives |
|---|---|---|
| 1 | **Debug-first coding workflow** — validate on a small sample and estimate full-run cost before committing compute | `agent/debug_run.py` |
| 2 | **Structured hypothesis pipeline** — force problem → hypothesis → reasoning → action, never one unstructured "improve the model" prompt | `agent/hypothesis_agent.py`, `agent/prompts/hypothesize.md` |
| 3 | **Time-aware planning** — early/mid/late stage instructions change what kind of idea is appropriate | `agent/budget.py` |
| 4 | **Error recovery + aggregated eval** — diagnose failures instead of crashing; confirm wins across seeds | `agent/error_recovery.py`, `agent/reeval.py` |

Strategy 1 is independently corroborated by **KompeteAI** (arXiv:2508.10177). **AIRA**
(arXiv:2507.02554) and **ML-Master** (arXiv:2506.16499) were used to cross-check the search-strategy
decision.

### Safety and isolation

- **Hidden-test isolation is structural, not a convention.** `agent/data_guard.load_train_valid()`
  physically removes the `'test'` key before any agent-facing code runs. No agent code path can
  read it even by accident. Only two human-run scripts (`make_submission.py`,
  `ensemble_submission.py`) and `submit.py` ever touch test.
- **Generated code is statically analysed before it runs** (`agent/code_guardrail.py`): an import
  allowlist, no `open`/`exec`/`eval`/`__import__`, no dunder introspection, no reference to the
  string `'test'`, plus API-contract checks for the specific mistakes a small model repeats
  (wrong `encode()` signature, clamping scores, full-batch stepping, use-without-import).
- **The action space is constrained** (`agent/action_space.py`): validated dict mutations only, so
  a config action can never write a file or reach the data itself.

---

## Repository layout

```
baseline.py          the kit's models: random / pop / FM (hand-rolled Adam + backprop)
data.py              load → date splits → categorical encode (train-only vocabs)
evaluate.py          GAUC + nDCG@5. FIXED — the scoring contract, do not modify
submit.py            submission format, validation, scoring

models/
  fm_v0.py           adapter over baseline.FM
  fm_v1.py           + optional CWM side-info fields
  fm_bpr.py          OURS — within-user pairwise (BPR) objective. Best single model.
  generated/         56 modules the agent wrote itself

agent/
  cli.py             entrypoint: run / status / note-intervention
  orchestrator.py    the loop (699 lines) — wires every stage below
  solution_tree.py   AIDE-style tree + select() policy
  hypothesis_agent.py / coding_agent.py / codegen_agent.py
  action_space.py    guardrail.py    code_guardrail.py
  debug_run.py       decision.py     reeval.py       error_recovery.py
  data_guard.py      budget.py       llm_client.py   (Ollama)
  archivist.py       logging_schema.py  resume.py    manual_intervention.py
  cost_report.py     viewer.py       prompts/*.md
  
make_submission.py     human-run: write a submission from a models/ variant
ensemble_submission.py human-run: N-seed ensemble submission
refresh_deliverables.py  snapshot runs/ into deliverables/evidence/

deliverables/          ── EVERYTHING A TEAMMATE OR JUDGE NEEDS ──
  00-START-HERE.md     index + open items. Read this first.
  RESULTS.md           scores, verification commands, honest caveats
  DEMO_SCRIPT.md       demo video script (pre-flight checklist inside)
  submission/          the submission CSV + two comparison baselines
  evidence/            snapshot of runs/: experiment_log.jsonl, tree, cost report
run_for_hours.ps1      supervisor: restart the loop if the process dies
runs/                  experiment_log.jsonl, solution_tree.json, state.json,
                       token_ledger.jsonl, dashboard.html
```

---

## Running it

```bash
# Reproduce the kit's baselines
python baseline.py --model random     # sanity check — must land near primary 0.475
python baseline.py --model pop        # official non-trained baseline
python baseline.py --model fm         # official baseline to beat

# Run the agent loop
python -m agent.cli run --iterations 20
python -m agent.cli run --iterations 200 --max-hours 2 --ignore-convergence
python -m agent.cli run --model fm_v1          # start from a different variant
python -m agent.cli status                     # current-best, tree, cost
python -m agent.cli note-intervention "why you touched it"

# Supervised long run (restarts if the process dies)
.\run_for_hours.ps1 -Hours 6

# Submissions (human-run — the only code allowed to touch test)
python make_submission.py --model fm_bpr --split test --out submission_test.csv
python ensemble_submission.py --seeds 5 --split test --out submission_ens_test.csv

# Verify the submission that ships in deliverables/
PYTHONIOENCODING=utf-8 python submit.py \
    --score deliverables/submission/submission_ens_test.csv --split test

# Refresh the evidence snapshot before packaging or recording the demo
python refresh_deliverables.py
```

> On Windows, prefix `submit.py` with `PYTHONIOENCODING=utf-8` — it prints a `✓` that crashes
> cp1252 consoles *after* validation has already succeeded.

---

## Deliverables → where to find them

| Judging criterion | Evidence |
|---|---|
| **Delta over baseline at convergence** | +0.0028 test primary. `deliverables/RESULTS.md` |
| **Manual interventions** | `runs/manual_interventions.jsonl`, `python -m agent.cli status` |
| **LLM tokens + GPU-hours** | `python -m agent.cost_report` — 579 calls, 1.33M tokens, **0 GPU-hours** (CPU-only) |
| **Quality of reasoning** | `runs/experiment_log.jsonl` — every record carries `problem_identified`, `hypothesis`, `implementation_sketch` |
| **Recovery from failures** | `error_events` in the log; `agent/error_recovery.py`; `run_for_hours.ps1`; crash-safe `agent/resume.py` |
| **Required per-iteration log** | `deliverables/evidence/experiment_log.jsonl` (live copy: `runs/`) — `{hypothesis, code_diff, metrics, error_events, decision, token_cost, wall_clock_s, node_id, resulting_config}` |

The loop recovers at three distinct levels: a bad candidate is logged and skipped (normal
operation), a transient LLM failure is retried with backoff (`LLM_RETRY_DELAYS_S`), and a dead
process is restarted from `state.json` by the supervisor.

---

## What we actually learned

The most useful output of this project is a set of **measured negative results**. They are all
recorded in `models/fm_bpr.py`'s docstring and fed back into the agent's own prompts, so neither
a human nor the loop re-derives them.

**The winning change was the objective, not the features.**

`baseline.FM` trains pointwise binary cross-entropy against a globally imbalanced label — it
spends most of its capacity getting each row's absolute probability right. But the metric is a
*within-user* ranking, which throws that calibration away. `models/fm_bpr.py` samples
(positive, negative) pairs **from the same user** and maximises `σ(z⁺ − z⁻)`. It adds **zero
features** and reuses baseline.FM's forward pass, embeddings and Adam untouched — only the loss
gradient changes. Sampling positives uniformly also weights each user by their positive count,
which is exactly how GAUC averages.

**Item-side features are saturated — three separate attempts, all negative.**

| Attempt | Result |
|---|---|
| Target-encoded per-video/author long_view rates | 0.5906 vs 0.6015 baseline — **worse**, degrading from epoch 1 |
| Blending FM with the popularity prior | best α=0.05 → **+0.00003**, then monotonic decline |
| `video_features_statistic_pure.csv` aggregates | dropped — same signal |

The FM's `video_id` linear weight `W[video_id]` **already is** a learned per-video propensity,
fitted on the same train rows a popularity lookup counts. Re-feeding it is redundant capacity and
extra parameters to overfit. One cheap offline sweep — remixing scores an already-trained model
had produced, no retraining — settled all three.

**Hard negative mining is actively harmful here — monotonically.**

```
neg_candidates   1 → 0.60306 | 2 → 0.60041 | 4 → 0.58238 | 8 → 0.57176 | 16 → 0.56862
```

`long_view` is noisy implicit feedback, so a user's highest-scoring negative is very often a
mislabeled positive — someone who did watch but wasn't logged. Mining aims every gradient step at
exactly the rows most likely to be wrong.

**Hyperparameter search was exhausted before iteration 1.** Every working configuration of the
baseline model across 75 iterations scored between 0.5976 and 0.6016 — a 0.0040 spread against a
0.0008 noise floor. The same held for the new model (k=32 worse, more pairs worse, every raised
lr worse). This is now stated with its evidence in `agent/prompts/hypothesize.md`, so the agent
stops proposing it.

**What did help on top:** seed ensembling (`ensemble_submission.py`) — averaging
per-user-standardised scores over 5 seeds. This works where the popularity blend did not because
it averages out *initialisation and minibatch variance* rather than adding information.

---

## Known limitations

Stated plainly, because they are more useful to a reader than a claim of completeness.

- **The loop has not beaten a hand-written model.** Across ~45 logged iterations on the current
  run and 75 before it, the best node the agent produced never exceeded the incumbent it was
  seeded with. Every gain in the results table was hand-built; the agent's contribution is the
  search infrastructure, the audit trail, and the negative results that redirected the search.
- **Generated modules drift back to the pointwise FM.** Even on `improve` operations that are
  correctly handed `fm_bpr.py` as parent source, the local model rewrites a pointwise FM — because
  `prompts/write_model.md` also ships two complete pointwise reference modules, and concrete code
  outweighs a prose rule for a small model. The fix (make the reference block conditional on
  operation) is identified but not implemented.
- **The hypothesis agent repeats itself when stuck** — seven near-identical "introduce a feature
  that captures the frequency of…" proposals in one run.
- **`CONVERGENCE_EPSILON` is 0.002**, wider than any gain that exists on this dataset, so the loop
  logs real improvements as `KEEP_NODE` rather than `COMMIT`. Recalibrating to ~0.001 (≈3× seed
  noise) is a scoring-policy decision left to the team.
- **The runtime estimator runs ~12× hot** — it extrapolates epochs linearly while early stopping
  fires around epoch 11, and scales the one-off `encode()` cost as if it repeated. Made harmless by
  expressing the budget as a *multiple of the reference model's own estimate* so both sides carry
  the same bias, but the estimate itself is still too noisy to be a useful signal.
- **`runs/manual_interventions.jsonl` records 1 intervention**, but the true human involvement in
  this repo is far higher — debugging the loop, rewriting its prompts, and hand-building the
  winning model. `note-intervention` was not called consistently. Report the honest number.

---

## References

- Yang et al. (2025) *R&D-Agent* — arXiv:2505.14738 — the four strategies above
- Nam et al. (2025) *MLE-STAR* — arXiv:2506.15692 — chain-based search, our structure
- Liu et al. (2025) *ML-Master* — arXiv:2506.16499 — tree+MCTS, the road not taken
- Toledo et al. (2025) *AIRA* — arXiv:2507.02554 — design-space cross-check
- Kulibaba et al. (2025) *KompeteAI* — arXiv:2508.10177 — corroborates debug-first
- KuaiRand dataset — see `KuaiRand-Pure/LICENSE`
