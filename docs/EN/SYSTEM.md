# SYSTEM.md — Engineering Reference

**The authoritative description of the system as it exists now.** For the scientific record
(evidence, derivations, negative results) see [RESEARCH.md](RESEARCH.md); for the competition
narrative see [SUBMISSION.md](SUBMISSION.md); for orientation see the [README](../../README.md).

This document tracks the code. Where the two disagree, the code is authoritative and this document
is the defect.

---

## Contents

1. [System overview](#1-system-overview) · 2. [Design principles](#2-design-principles) ·
3. [Frozen trust boundary](#3-frozen-trust-boundary) · 4. [Repository architecture](#4-repository-architecture) ·
5. [Six-block solution space](#5-six-block-solution-space) · 6. [Node representation](#6-node--experiment-representation) ·
7. [Data & cache architecture](#7-data-loading-and-cache-architecture) · 8. [Integrity protections](#8-leakage--integrity-protections) ·
9. [Model families & levers](#9-model-families-and-implemented-levers) · 10. [Agent architecture](#10-agent-architecture) ·
11. [Config-effectiveness validation](#11-config-effectiveness-validation) · 12. [Dedup & no-op handling](#12-deduplication-and-no-op-handling) ·
13. [Statistical decision machinery](#13-statistical-decision-machinery) · 14. [Research memory & ledger](#14-research-memory-and-cross-run-ledger) ·
15. [Portfolio machinery](#15-portfolioensemble-machinery) · 16. [Convergence & budget](#16-convergence-and-benchmark-budget-semantics) ·
17. [Failure recovery](#17-failure-recovery) · 18. [Finalization](#18-submission-and-finalization) ·
19. [Research Console](#19-research-console-architecture) · 20. [Configuration reference](#20-configuration-reference) ·
21. [Running & testing](#21-running-and-testing) · 22. [Run artifacts & schemas](#22-run-artifacts-and-schemas) ·
23. [Extension rules](#23-extension-rules) · 24. [Known limitations](#24-known-engineering-limitations)

---

## 1. System overview

An LLM-driven agent that runs the whole ML research loop on the KuaiRand-Pure within-user video
ranking benchmark: it reproduces the official Factorization-Machine baseline, states the bottleneck it
sees, proposes an experiment, edits real model code or configuration, has that change validated before
any training, trains, evaluates, compares against a control with a paired bootstrap, decides what to
investigate next, assembles a portfolio, and writes a validated submission — then stops when the
organizer's own convergence rule says to.

**Task.** For each user, rank *that user's own* logged impressions by `long_view`. Nothing is retrieved
from a global catalogue. Score is `primary = ½(GAUC + nDCG@5)`, computed by the frozen `evaluate.py`.

**Current verified result** (`runs/run_20260831_090457`, live Gemini run):

| quantity | value |
|---|---|
| FM baseline (root node) | 0.60147 |
| best single model | 0.60366 (DIN, node `n3`) |
| portfolio, tuned on all of valid — **optimistic** | 0.60463 |
| portfolio, **honest** 5-fold user-level CV | **0.60409 ± 0.00141** |
| stop reason | `official_convergence` |
| submission | PASSED `submit.py --check` |

The honest CV estimate is the number to quote. See [RESEARCH.md](RESEARCH.md#12-ensemble--portfolio-findings)
for why tuned ≠ honest.

---

## 2. Design principles

1. **Two layers behind a fixed boundary.** *What* the agent searches (`pipeline/`, the solution space)
   is separated from *how* it searches (`agent/`, the policy). The orchestrator drives an FM, a
   LightGBM or a DIN without knowing anything about them.
2. **The agent is the deliverable, not just the model.** Track 2 asks for an autonomous researcher,
   so the loop — autonomy, robustness, recovery, cost — is engineered as carefully as the levers.
3. **Trust the score.** Everything the score depends on is frozen and hash-pinned (§3).
4. **Best-checkpoint invariant.** The submission is always the validated best object, tracked
   independently of the (possibly non-monotonic) search path.
5. **Cheap analysis over expensive re-training.** Bootstraps, rank correlations, blend evaluation and
   memory synthesis all run on saved predictions and never launch a training run (§13).
6. **Evidence, not adoption status, is the scientific record.** A node's tree status and the evidence
   about its effect are separate fields (§14).
7. **Compute is not the objective.** The benchmark's iteration cap and wall-clock ceiling are limits,
   not targets (§16).

---

## 3. Frozen trust boundary

Five files are SHA-256-pinned in `agent/frozen.lock` and verified by
`agent/guardrails.py::ensure_frozen()` at the top of every run. A mismatch aborts:

```
data.py   evaluate.py   submit.py   pipeline/run_node.py   pipeline/contracts.py
```

Everything the agent may change lives downstream: `agent/*`, `pipeline/lib/*`,
`pipeline/*_blocks/*`, new `pipeline/*.py`, `tests/*`, `dashboard/*`.

**If a change appears to require editing a frozen file, it is the wrong change — route around it.**
Three established patterns:

| Need | Pattern |
|---|---|
| A knob the frozen `Cfg` cannot hold | `cfg_ext.json` sidecar, read via `pipeline/lib/ext.py` (§6) |
| A second inference split from one training pass | the infer block writes it itself, `pipeline/lib/extra_infer.py` (§18) |
| A fast-fail gate the runner has no flag for | subsample the *cache* and call the same runner, `pipeline/debug_cache.py` (§17) |
| Withhold labels from blocks | act at load time in `datced.load_bundle` (§8) |

A second static guard, `executor.check_imports`, parses every agent-written block *before* it runs and
rejects disallowed imports, forbidden builtins and holdout path literals (§8).

---

## 4. Repository architecture

```
FROZEN HARNESS  data.py · evaluate.py · submit.py · pipeline/run_node.py · pipeline/contracts.py
──────────────────────────── trust boundary ────────────────────────────
AGENT (agent/)                        SOLUTION SPACE (pipeline/)
  orchestrator  control loop            baseline_blocks/   FM control (the root node)
  tree          best-first search       lib/din_blocks/    DeepFM+DIN family
  roles/        Proposer·Coder·Reflector lib/lgbm_blocks/  LightGBM LambdaRank family
  llm/          Gemini | Mock driver    lib/fm.py          numpy FM backbone
  blockspec     honoured-knob contract  lib/losses.py      BCE·BPR·softmax-CE·IPS-BCE
  mutate        node materialisation    lib/train_np.py    point/pair/group trainer
  provenance    intended vs executed    lib/din.py         DIN + aux head + feedback embedding
  executor      sandbox·gates·guards    lib/gbm.py         LightGBM features + ranker
  datced        cache builder/loader    lib/seq_build.py   chronological history + feedback states
  stats         paired user bootstrap   lib/aux_build.py   auxiliary labels (train slice only)
  portfolio     valuation + K-fold CV   lib/rand_build.py  random-exposure split
  memory        run_log + research state lib/ext.py        extension-config sidecar
  ledger        cross-run evidence      lib/extra_infer.py extra split from the same training pass
  events        structured event stream lib/debug_cache.py subsampled cache for the debug gate
  reeval        multi-seed re-training
  champion      cross-run champion
  console_server Research Console server
OBSERVABILITY  dashboard/research-console.html · dashboard/hypothesis-ledger.html
```

---

## 5. Six-block solution space

A node's model is six Python files. The frozen runner executes the first five in order; `combine` is
the assembly-phase hook, called by the portfolio machinery rather than by the runner:

```python
build_features(bundle, cfg) -> FeatureSet
build_model(meta, cfg)      -> model exposing .logits/.apply_grad/.predict (or a wrapper)
build_loss(cfg)             -> lossfn(z, batch) -> (loss, g)      # g = dL/dz per row
train(model, lossfn, feats, bundle, cfg) -> model                 # validation-best, early-stopped
infer(model, feats, split)  -> np.ndarray aligned to bundle row order
combine(base, cfg)          -> np.ndarray                         # assembly-phase hook
```

`pipeline/baseline_blocks/` is the FM+BCE baseline **and** the ablation control (the root node).
`build_loss` returns `make_loss(cfg)`, so `loss_type` is a genuine configuration knob. A block that
hardcodes its loss instead silently disables Lever A and returns a baseline result under a BPR label
(see [RESEARCH.md](RESEARCH.md#7-bpr-theory-and-measured-result)) — which is what §11 exists to catch.

`FeatureSet` is frozen and cannot gain a field. Anything extra is either computed in the trainer, or
carried in an existing optional field — the behavior-aware history rides as an optional third element
of the `seq` tuple.

---

## 6. Node / experiment representation

**One node = one experiment** = a snapshot of the six block sources + a `Cfg` + an optional
`cfg_ext.json` sidecar. Full snapshots (not diffs) make every node independently runnable and let two
nodes run concurrently without collision.

On disk: `runs/<run_id>/nodes/<id>/{blocks/*.py, cfg.json, cfg_ext.json, provenance.json,
metrics.json, val_scores.npy, test_scores.npy, rand_scores.npy}`.

Three mutation kinds:

| Kind | What changes | Cost |
|---|---|---|
| **config** | `Cfg` / `cfg_ext` values only | near-zero tokens |
| **block edit** | the Coder rewrites exactly one block body | gated by `check_imports` |
| **block-set adoption** | `Hypothesis.adopt_blockset: "din" \| "lgbm"` swaps in a whole pre-built family from `pipeline/lib/<name>_blocks/` | `cfg.model_type` follows the blocks |

`model_type` is **managed by the harness** — a hypothesis may not set it directly. It must follow the
mounted blocks, otherwise every node collapses into one ensemble family and no portfolio forms.

### The extension sidecar

`pipeline/contracts.py` is frozen, so `Cfg` cannot gain fields and `Cfg.from_dict` silently drops
unknown keys. Knobs added later (`use_fb`, `fb_dropout`, the `gbm_*` hyper-parameters) live in
`cfg_ext.json` next to `cfg.json`. A block reads it via `pipeline/lib/ext.py::load(__file__)` — the
runner loads blocks with `importlib.util.spec_from_file_location`, so `__file__` is
`<node_dir>/blocks/<name>.py` and the sidecar is two levels up. The sidecar participates in the node's
content signature and provenance hash, so it cannot smuggle an undetected change past deduplication.

---

## 7. Data loading and cache architecture

Re-reading ~106 MB of CSV per experiment would dominate the budget, so everything is encoded **once**
into memory-mapped `.npy` files under `runs/_cache/`.

**`CACHE_VERSION = 10`.** Bump it whenever the cached array layout changes; it forces one rebuild.
History: 6 = base+gbm+seq+aux+rand · 7 = holdout isolation · 8 = chronological seq + feedback states ·
9 = split-filtered history · 10 = honest `fb_policy=train_only`.

| Directory | Per-row arrays | Notes |
|---|---|---|
| `runs/_cache/` | `{split}_X.npy` (N×5), `_u.npy`, `_vid.npy`; `_y.npy` for **train/valid only** | base encoded fields |
| `runs/_cache/gbm/` | `{split}_X.npy` (N×22), `_y.npy`, `_u.npy` | LightGBM features |
| `runs/_cache/seq/` | `{split}_{seq,fb,slen,tgt}.npy` | chronological history + feedback states |
| `runs/_cache/aux/` | `train_aux.npy`, `train_vid.npy` | **train slice only** |
| `runs/_cache/rand/` | `rand_{X,y,u}.npy` | random-exposure split (public labels) |
| `runs/_holdout/` | `test_y.npy`, `aux/{valid,test}_aux.npy` | **never passed to a block** |

All arrays are per-row aligned, which is what makes the debug subsample and every cross-cache join
correct. `datced._assert_aux_aligned` compares `aux/train_vid.npy` against the base `train_vid.npy`
and refuses to run on drift.

### Sequence construction (chronological)

`pipeline/lib/seq_build.py` re-reads `time_ms` from the raw logs (mirroring `data.load()`'s file order
and date filter, then asserting row-alignment against `{split}_vid.npy`), walks every event in true
`(user, time_ms, split, row)` order, and snapshots each row's history **before** appending that row.

Three guarantees, asserted at build time and independently re-verified in `tests/test_sequence.py`:

1. **No future event appears in any history.** The build raises if any within-user time inversion
   survives the sort. (The previous row-order-based construction violated this for 30.83% of train,
   20.89% of valid and 31.54% of test rows — see [RESEARCH.md](RESEARCH.md#10-sequence-chronology-and-the-temporal-guarantee).)
2. **No train/valid row sees a test-window event.** Filtering on split index as well as time makes
   this structural. It matters because `date` and `time_ms` disagree at the split boundary: 28
   test-dated rows carry timestamps earlier than the last valid row.
3. **Arrays are written back in `data.load()` row order**, so every sibling cache stays aligned.

`{split}_fb.npy` carries the feedback state of each history event
(`PAD/SKIP/SHORT/NORMAL/LONG/EXPLICIT/UNKNOWN`). Under the default `fb_policy="train_only"`, **only
train-window outcomes may become features**; valid- and test-window events are `UNKNOWN`. This is the
only policy under which validation is an unbiased proxy for test, and it was chosen after a measured
failure of the alternative ([RESEARCH.md](RESEARCH.md#11-behavior-aware-history-and-the-00165-artifact)).

---

## 8. Leakage / integrity protections

The competition-relevant risk is a model that scores well by seeing labels it should not. Protections
are layered; none is described as absolute.

| Layer | Mechanism | Where |
|---|---|---|
| Data interface | `load_bundle` never populates `y["test"]`; `load_aux` raises `KeyError` for valid/test | `agent/datced.py`, `pipeline/lib/aux_build.py` |
| On-disk separation | hidden-test labels and the `is_click` proxy live in `runs/_holdout/`, a **sibling** of the cache that is never passed to a block | `agent/datced.py` |
| Build-time assertion | the run refuses to start if a holdout array reappears in the block-visible cache | `datced._assert_aux_aligned` |
| Static guard | `check_imports` rejects disallowed imports, `open`/`eval`/`exec`/`compile`/`__import__`, and path literals matching `_holdout`, `test_y`, `test_aux`, `valid_aux`, `KuaiRand-Pure`, `log_standard` | `agent/executor.py` |
| Runtime tripwire | any node scoring above `leak_tripwire_primary` (0.70) is **quarantined**, excluded from the tree and the portfolio | `agent/orchestrator.py` |
| Sequence policy | `fb_policy="train_only"` (§7) | `pipeline/lib/seq_build.py` |

**The claim, stated exactly:** *agent-written blocks are not given hidden-test labels or current-row
outcome proxies, and label-derived holdout data is kept outside the block-visible data interface,
backed by a static read guard and a plausibility tripwire.* It is deliberately **not** stated as
"label access is physically impossible" — that would be stronger than the mechanisms support. See
[RESEARCH.md](RESEARCH.md#9-label-leakage-the-risk-and-the-containment).

Why the tripwire is calibrated where it is: ranking valid by `is_click` alone scores **0.7466**, which
is 58.8% of the entire oracle headroom above FM. Real progress here is measured in thousandths, so a
jump of that size is a bug or a leak, never a model.

`tests/test_leakage.py` asserts all of the above (26 checks), including that six hostile block sources
are rejected and five legitimate ones still pass.

---

## 9. Model families and implemented levers

| Lever | Idea | Status | Where |
|---|---|---|---|
| **A** | loss alignment: BCE · BPR · softmax-CE · IPS-BCE | implemented, `loss_type` is a real knob | `lib/losses.py`, `lib/train_np.py` |
| **B** | sequences: DeepFM + Deep Interest Network | implemented | `lib/din.py`, `lib/din_blocks/` |
| **C** | multi-task auxiliary heads (click/like/follow/comment/forward) | implemented, `mtl_arch="shared"` only | `lib/din.py`, `lib/aux_build.py` |
| **D** | model family: LightGBM LambdaRank | implemented **and tunable** via `gbm_*` sidecar knobs | `lib/gbm.py`, `lib/lgbm_blocks/` |
| **E** | exposure: random-exposure surface + inverse-popularity weighting | implemented; rand surface is FM-family only | `lib/rand_build.py`, `train_np._ips_weights` |
| **F** | portfolio: rank-space blend | implemented, K-fold cross-validated | `agent/portfolio.py` |
| — | behavior-aware history (`use_fb`) | implemented, **default OFF** — measured negative | `lib/seq_build.py`, `lib/din.py` |

Two naming points that matter scientifically: `train_np._ips_weights` computes
`w ∝ 1/√freq(item)`, which is **inverse-popularity weighting, not inverse propensity**; and the
random-exposure split is reported as a **second robustness surface**, never as the competition target.

Not implemented: `mtl_arch ∈ {mmoe, ple}` (rejected before a training launch by `blockspec`), a native
numpy LambdaRank loss (`Cfg.lambdarank` is reserved; LightGBM supplies LambdaRank), and the rand
surface for DIN/LightGBM (needs sibling rand caches).

---

## 10. Agent architecture

The orchestrator is deterministic **policy**; the LLM roles are the **operators**.

| Role | Question | Prompt content |
|---|---|---|
| **Proposer** | What scientific problem should be attacked next? | phase/budget, evidence-graded research state, honoured-knob map, per-lever table, plateau signal, re-proposal feedback |
| **Coder** | How is this implemented in one block? | the target block's current source, the import allowlist, an instruction to decline honestly rather than fabricate |
| **Reflector** | Why did execution fail and how should it recover? | failure class + traceback tail |

`Hypothesis` requires `problem_identified` **first** — the agent must state the bottleneck before
proposing a change. All LLM output is Pydantic-schema-constrained (`agent/llm/schemas.py`), so parsing
never fails; `MockDriver` replays scripted moves for offline testing.

### Search policy

Best-first with an ε exploration valve (`agent/tree.py::select`). Deliberately **not** MCTS: each node
costs a real training run, and under the official convergence rule a run legitimately ends after ~4–6
experiments — far too few for rollouts and backups to produce reliable value estimates.

Under a pre-convergence research plateau, `prefer_diverse=True` makes exploration expand the most
*decorrelated* viable node rather than a random one. Plateau escalation changes **what is proposed**;
it can never postpone convergence (§16).

### One iteration

```
select parent → Proposer → validate config (§11) → [Coder → check_imports] → materialise node
  → content-signature dedup (§12) → [debug gate for torch] → train (+test inference)
  → evaluate → paired bootstrap vs control (§13) → portfolio valuation (§15)
  → adoption status → research-information verdict → append to memory, events, best_series
```

A proposal that fails validation, duplicates an earlier node, or is honestly declined is **corrected
and re-proposed** (up to `max_reproposals`) without training and without advancing the experiment
counter (§12, §16).

---

## 11. Config-effectiveness validation

`agent/blockspec.py` declares, per block set, the `Cfg` fields and `cfg_ext` keys that **provably reach
an execution path**, plus allowed values and ranges.

```
fm   : batch epochs grad_clip group_filter ips k l2 loss_type lr neg_ratio patience seed tau
din  : aux_tasks aux_weights batch epochs fb_dropout k l2 loss_type lr mtl_arch neg_ratio
       patience seed use_fb
lgbm : seed gbm_learning_rate gbm_num_boost_round gbm_num_leaves gbm_min_data_in_leaf
       gbm_feature_fraction gbm_lambda_l2 gbm_bagging_fraction
```

`validate_delta` classifies every key of a proposed mutation:

| Class | Meaning | Consequence |
|---|---|---|
| `effective` | honoured, valid, and different from the current value | reaches `cfg.json` |
| `ineffective` | honoured but already at that value | dropped, reported |
| `not_honoured` | a real field this block set never reads | dropped, fed back to the Proposer |
| `invalid` | unknown key, out-of-domain value, or a managed field | dropped, fed back |

A mutation whose effective set is empty and which carries no block edit or adoption is a **structural
no-op**: provably incapable of changing execution, so it is never trained and never enters the
scientific record.

`agent/provenance.py` records, per node, the intended delta, the effective delta, every rejected key
with its reason, the effective config hash, per-block source hashes, cache version, code-state hash and
seed — so *intended intervention* and *executed intervention* can always be compared.
`intervention_matched` is `False` when the executed experiment is narrower than the proposal.

`tests/test_blockspec.py` verifies the declarations at runtime on a subsampled cache: a honoured knob
must change predictions, a not-honoured knob must leave them **bit-identical**.

---

## 12. Deduplication and no-op handling

**Identity is content-based.** `signature = sha256(cfg + cfg_ext + all six block sources)`. Hashing a
unified diff *against the parent* instead gives two nodes with identical configs and identical block
sources different signatures whenever they are reached from different parents — which is how a re-run
of an existing node ends up recorded as a new scientific finding.

Three post-hoc classes, kept distinct:

| Class | Test | Is it evidence? |
|---|---|---|
| `STRUCTURAL_NOOP` | decided before execution: no effective change | **No** |
| `EXACT_NOOP` | predictions bit-identical to the parent's | **No** |
| `NEAR_NOOP` | rank correlation > 0.9999 but not identical | **Yes** — legitimate evidence of a negligible effect |

Near-identity is deliberately *not* treated as proof of a missing intervention: two trainings of an
identical DIN config correlate 0.926, so a stochastic family can produce genuinely different models
that look similar.

---

## 13. Statistical decision machinery

Three distinct variances exist and are never conflated:

| Source | Magnitude | Reduced by re-training? |
|---|---|---|
| training stochasticity — fm, lgbm | **0.00000** at a fixed seed | n/a |
| training stochasticity — din (torch) | σ ≈ 0.00025 | yes |
| **validation-sample noise** (paired user bootstrap) | **σ ≈ 0.0009** | **no** |

`agent/stats.py` provides an `Evaluator` that reproduces the frozen evaluator's semantics
(Mann-Whitney U with average-rank tie correction, positive-weighted GAUC over discriminative users,
nDCG@5 with a stable descending sort) and agrees with it to **< 1e-5**, verified in
`tests/test_stats.py` on both synthetic and real node predictions.

`paired_bootstrap` resamples **users** with replacement and rescores both models on the same resample,
so the strong per-user correlation between two models cancels. It reports Δprimary, ΔGAUC, ΔnDCG,
bootstrap SE, 95% CI and `P(Δ>0)` — for ~2 s per comparison and **no re-training**.

Evidence classes come from `P(Δ>0)`: `confirmed` ≥ 0.90 · `promising` 0.60–0.90 · `inconclusive` ·
`rejected` ≤ 0.10.

Multi-seed **re-training** (`agent/reeval.py`) is reserved for stochastic families, where it measures
something the bootstrap cannot. It is never run for FM or LightGBM, which are exactly deterministic at
a fixed seed. At finalize it reports training variance for a stochastic finalist **without**
substituting the model (§18).

---

## 14. Research memory and cross-run ledger

`agent/memory.py` writes the append-only `run_log.jsonl` and synthesises the state the Proposer sees.

**Scientific evidence and tree status are separate.** A node carries `status` (tree/adoption
bookkeeping), `evidence` (what the bootstrap says), and `noop_class` (whether an intervention happened
at all). Collapsing them into a single label is what gets an agent told "REJECTED — don't repeat"
about its own best model. Buckets are: `confirmed`, `promising`, `inconclusive`, `rejected`, `no_effect`,
`unsupported_capability`, plus an explicit `diverse_portfolio_candidates` list and a separately
surfaced champion.

`inconclusive` is worded so the Proposer knows a repeat will not settle it — the effect is below what
the validation set can resolve, so the mechanism or the design must change.

`agent/ledger.py` persists every executed experiment to `runs/_ledger.jsonl` keyed by an **arm**
(`family|loss|aux`, deliberately excluding seed, since seeds are repetitions not arms). Pooling is
gated by `compatible()`, which requires the same `cache_version` **and** the same `code_state` hash.
This guard is load-bearing: applying it retires a cross-run conclusion that was drawn across a cache
change ([RESEARCH.md](RESEARCH.md#8-auxiliary-task-investigation)). `ResearchInsight` records are
generated rule-based from the ledger, with no extra LLM call.

---

## 15. Portfolio/ensemble machinery

Base learners live on incomparable score scales and the metric cares only about within-user order, so
blending happens in **rank space**: per user, `r_i = rank_u(s_i)/(|I_u|−1) ∈ [0,1]` — monotone (cannot
hurt a single model) and scale-free.

**Valuation** (cheap, from saved `val_scores.npy`, never trains): `rank_corr_to_best`,
`pair_blend_gain`, and `emc` (leave-one-out contribution to the full-pool blend). These are surfaced
to the run log, research memory, the Proposer context and the console — because standalone score alone
would discard the models that make the portfolio work.

**Assembly** is a user-level **K-fold cross-validation of the whole procedure**:

```
A(S) = de-duplicate (rank corr > 0.999) → greedy forward member selection → weight grid search,
       computed ONLY from users S

for each fold k:  members_k, weights_k = A(valid \ fold_k)      # fold_k never consulted
                  score_k              = primary on fold_k
honest estimate = mean(score_k) ± sd/√K
final artefact  = A(all of valid)      # produces the test blend; NOT reported as an estimate
```

Four data roles are never conflated: choose members / tune weights / report honestly / refit for the
submission. Only the third may be quoted as unbiased. Regularisation: coarse weight lattice
(step 0.25), at most 4 members.

---

## 16. Convergence and benchmark budget semantics

**Three stopping concepts, kept strictly separate.**

| Concept | Rule | May it end a healthy run? |
|---|---|---|
| **OFFICIAL** | `eps = 0.002, N = 3` over the best-so-far series of executed experiments; plus `max_iter = 50` hard cap and a 6 h wall-clock backstop | **Yes — this is the only one that should** |
| **Research bookkeeping** | `research_stall`, plateau escalation | No. It changes *what* is proposed |
| **Liveness guard** | `proposal_guard_limit` — the Proposer cannot emit anything executable | Only in the pathological case; reported as `proposal_guard`, never as convergence |

`OFFICIAL_EPS` and `OFFICIAL_N` are read at import from `baseline_scores.json → convergence_rule`, so
the code cannot drift from the organizer's artifact. `docs/PROBLEM_STATEMENT.pdf` is an image-only scan
and cannot arbitrate, so the JSON is authoritative: **ε = 0.002, N = 3.** Any larger `N` loosens the
rule rather than tightening it — `N = 6` needs 7 scored nodes without improvement before it fires,
`N = 3` needs 4.

**Compute is deliberately not the binding constraint.** Under the official rule a run converges after
roughly 4–6 experiments. The objective is *not* to spend the quota — it is to make every
pre-convergence experiment a real one. There is deliberately **no** minimum-experiment rule and no
research-driven extension of a run.

### Accounting

| Counter | Counts | Compared against |
|---|---|---|
| `experiments_executed` | nodes that trained and produced metrics | `max_iter`; feeds `best_series` → convergence |
| `proposal_attempts` | every Proposer call, including rejected ones | nothing — observability only |
| `wall_clock_used` | seconds since run start | the 6 h backstop |

A proposal that never trained a model is not a benchmark iteration. Both counters are separately
observable so the claim is auditable rather than asserted. `adopt_eps` governs tree shape only and is
**not** a stopping rule.

---

## 17. Failure recovery

`executor.run_node` launches each node as an isolated subprocess with a timeout and forced UTF-8 I/O,
and classifies outcomes into a metrics dict or a typed `Failure(kind ∈ {code, timeout, numerical})`.

| Failure | Recovery |
|---|---|
| `code` | Reflector supplies a corrected block; re-gated by `check_imports`; re-run → else abandon |
| `timeout` | `degrade` (smaller epochs/history via a config delta) → else abandon |
| `numerical` | `degrade` (grad clip, lower lr) → else abandon |

**Debug-first gate** (torch families only): `pipeline/debug_cache.py` builds a row-consistent subsample
of *every* per-row cache array, and the node runs on it with capped epochs through the same frozen
runner. A crash costs seconds instead of a full training run. Cheap FM nodes skip it.

Every iteration is wrapped so one bad step can never kill a run, and `finalize` still emits the best
validated submission even if every branch failed. Verified by `python -m agent.run --faults`, which
injects a crash, patches it, and finalizes a valid submission.

---

## 18. Submission and finalization

**Model identity is preserved.** `extra_split="test"` is passed on **every** node run, so
`test_scores.npy` is written by the same training pass that produced `val_scores.npy`. There is no
finalize-time re-training. This matters because re-training a DIN produces a genuinely different model
(rank correlation 0.926 between two runs of one config), which would break the best-checkpoint
invariant for torch families.

`run_node` is frozen and can infer only one extra split per invocation, so the second one (`rand`, the
random-exposure surface) is written by the infer block itself via `pipeline/lib/extra_infer.py`.

Finalize then: builds portfolio members → computes valuation → runs K-fold CV assembly (§15) → blends
the saved test arrays → writes `best/submission_test.csv` → validates it with the frozen
`submit.py --check` → writes `resource_report.json` and `results.md` → appends to the cross-run ledger
→ optionally saves the champion.

---

## 19. Research Console architecture

Presentation infrastructure. **It renders artifacts the agent emitted; it never simulates execution,
never computes a metric of its own, and is not a model-performance improvement.**

| Component | Role |
|---|---|
| `agent/events.py` | append-only `runs/<id>/events.jsonl`; 17 event types (`RUN_START OBSERVE HYPOTHESIZE PLAN CODE GUARD DEBUG TRAIN EVALUATE COMPARE REFLECT RECOVER ENSEMBLE DECIDE CONVERGENCE FINALIZE RUN_END`), flushed per line so a live UI can tail it |
| `agent/console_server.py` | stdlib `http.server`; `/api/runs`, `/api/events?run&since`, `/api/log`, `/api/report`. Read-only |
| `dashboard/research-console.html` | the console: activity stream, evidence-graded research state, portfolio table with EMC, experiment + integrity tables |

`run_log.jsonl` keeps its per-node schema; `events.jsonl` is **additive**, so the older
`dashboard/hypothesis-ledger.html` still works.

Two modes share the same components: **Replay** (deterministic, from a completed run — no API key, GPU
or network) and **Live** (polls `since=<seq>` for new events). Replay may compress timing; it never
alters ordering, metrics, hypotheses, decisions or outcomes.

The status bar separates the **official** convergence state from internal research bookkeeping, and
executed experiments from proposal attempts.

---

## 20. Configuration reference

**`Cfg`** (`pipeline/contracts.py`, frozen, per-node): `seed, use_seq, L, use_vstat, use_aux,
model_type, k, loss_type, alpha, tau, neg_ratio, lambdarank, group_filter, lr, l2, epochs, batch,
patience, grad_clip, aux_tasks, aux_weights, mtl_arch, ips, ensemble_members`.
Which of these actually take effect depends on the mounted block set — see §11.
`use_seq, use_vstat, use_aux, lambdarank, ensemble_members, L, alpha` are declared but read by no block.

**`cfg_ext.json`** (sidecar): `use_fb, fb_dropout` (din) · `gbm_learning_rate, gbm_num_boost_round,
gbm_num_leaves, gbm_min_data_in_leaf, gbm_feature_fraction, gbm_lambda_l2, gbm_bagging_fraction` (lgbm).

**`agent/config.py`** (agent-side, not frozen):

| Group | Fields (defaults) |
|---|---|
| `Config` | `data_dir`, `cache_dir=runs/_cache`, `runs_dir=runs`, `seed=0`, `gpu=auto`, `debug_gate=True`, `debug_train_n=20000`, `debug_other_n=10000`, `debug_epochs=2`, `recheck=True`, `recheck_seeds=(1,2)`, `recheck_top_k=3`, `resume=False`, `champion_dir=runs/_champion`, `ledger_path=runs/_ledger.jsonl`, `use_ledger=True`, `unbiased_eval=False`, `events=True` |
| `Budget` (official) | `max_iter=50` (hard cap), `wall_clock_hours=6.0` (backstop), `per_iter_timeout_s=900`, `eps=0.002`, `N=3`, `adopt_eps=0.001` (tree shape only) |
| `Research` | `bootstrap_B=1000`, `adopt_p=0.90`, `promising_p=0.60`, `ens_eps=0.0002`, `diversity_corr=0.90`, `max_reproposals=3`, `proposal_guard_limit=12`, `plateau_after=2`, `explore_p_escalated=0.40`, `cv_folds=5`, `max_members=4`, `weight_step=0.25`, `leak_tripwire_primary=0.70` |
| `Phases` | `breadth_until=12`, `depth_until=40`, `ablation_every=6` (unused), `explore_p=0.15` |
| `LLM` | `provider=gemini`, per-role model ids, `temperature=0.4`, `max_retries=5`, `max_llm_usd=0.0` (declared, **not enforced**) |

`Config.load(path)` merges an optional `agent/config.yaml` over the defaults.

---

## 21. Running and testing

Two interpreters: system `python` 3.14 (CPU torch) and `cudaenv/` 3.12 with torch 2.6.0+cu124 (**GPU**).
FM and LightGBM are numpy/booster-deterministic and interpreter-independent; only DIN differs and runs
much faster on `cudaenv`. Prefer `cudaenv/Scripts/python.exe` for all runs.

```bash
python -m agent.run --smoke        # M0 gate: build cache, reproduce FM (~0.6015), verify frozen.lock
python -m agent.run --mock         # full loop via MockDriver — no API key, no credits
python -m agent.run --faults       # robustness: inject a crash, verify recovery, still finalize
python -m agent.run                # LIVE (needs GEMINI_API_KEY or .env.local); --max-iter to cap
python -m agent.console_server     # Research Console at http://127.0.0.1:8712/
```

Targeted suites (each is standalone and prints PASS/FAIL per check):

```bash
python -m tests.test_stats          # bootstrap matches the frozen evaluator (<1e-5)
python -m tests.test_blockspec      # honoured-knob contract, incl. runtime bit-identity checks
python -m tests.test_leakage        # holdout isolation, hostile blocks rejected
python -m tests.test_orchestration  # convergence semantics, dedup, no-op classes, evidence≠status
python -m tests.test_sequence       # chronology + feedback-state leakage safety
```

**Run `--smoke` after any change touching the cache, the blocks, or the frozen boundary.**

---

## 22. Run artifacts and schemas

`runs/<run_id>/`:

| File | Contents |
|---|---|
| `run_log.jsonl` | one line per node — the research ledger and the agent's memory |
| `events.jsonl` | fine-grained structured research events (§19) |
| `resource_report.json` | benchmark accounting, tuned vs honest results, portfolio valuation, training variance, stop reason |
| `results.md` | short human-readable results table |
| `best/submission_test.csv` | the submission (passes `submit.py --check`) |
| `nodes/<id>/` | blocks snapshot, `cfg.json`, `cfg_ext.json`, `provenance.json`, `metrics.json`, `val_scores.npy`, `test_scores.npy`, `rand_scores.npy` |
| `reeval/` | multi-seed re-training scores for a stochastic finalist (§13), when `recheck` fires |

Persistent: `runs/_cache/` (DataBundle), `runs/_holdout/` (label-derived, never block-visible),
`runs/_champion/` (cross-run champion), `runs/_ledger.jsonl` (cross-run evidence).

**What is committed.** The evidence trail ships with the repository: `run_log.jsonl`,
`events.jsonl`, `resource_report.json` and `results.md` for **every** run in the project's history,
plus `runs/_ledger.jsonl` and `agent/frozen.lock`. That is what every number in these documents is
checked against, and it is all the Research Console needs to replay a run — so a fresh clone can
verify the claims and watch the run without training anything.

Deliberately **not** committed: `runs/_cache/` (a 475 MB derived cache, rebuilt in ~60 s),
`runs/_holdout/` (**it holds the hidden test labels — publishing it would leak exactly what the
benchmark withholds**), `runs/_champion/`, and the per-node score arrays, block snapshots and
submission CSVs (tens of MB of regenerable binary).

**`run_log.jsonl` record**: `iter, phase, node_id, parent_id, lever, hypothesis, problem_identified,
config, cfg_ext, code_diff, metrics{GAUC, nDCG@5, primary_valid, primary_unbiased, …},
status, evidence{class, delta_primary, delta_GAUC, delta_nDCG, boot_se, p_gt0, ci_lo, ci_hi,
control_id}, portfolio{rank_corr_to_best, pair_blend_gain, emc, standalone_primary},
provenance{…}, noop_class, informative, events, cost, signature`.
`status ∈ {root, improved, no_gain, abandoned, duplicate, rejected_proposal, quarantined}`.

**`events.jsonl` record**: `{seq, ts, type, node_id, phase, summary, data}`.

---

## 23. Extension rules

1. Never edit a frozen file. 2. `--smoke` after every change. 3. All cache arrays stay per-row aligned.
4. Bump `CACHE_VERSION` on any cache-layout change.

**Add a config knob:** if the frozen `Cfg` already has a suitable field, use it and declare it in
`blockspec` for the relevant block set. Otherwise put it in the `cfg_ext.json` sidecar and add it to
that block set's `ext_honoured`. **A knob that is not declared in `blockspec` will be rejected before
training** — that is the point.

**Add a lever / model family:**
1. Implement the model in `pipeline/lib/<name>.py`.
2. Create `pipeline/lib/<name>_blocks/` honouring the six signatures.
3. If it needs new cached data, add `<name>_build.py`, wire it into `datced.build_or_load`, **assert
   row-alignment against `{split}_vid.npy`**, bump `CACHE_VERSION`, and add its arrays to
   `debug_cache.PER_ROW`.
4. Add a `BlockSetSpec` to `agent/blockspec.py` listing the knobs it genuinely reads, and declare
   whether it is stochastic.
5. Add a `tests/mock_moves.py` move that adopts it and run `--mock`.

**Common pitfalls:** grouping and family logic key on `cfg.model_type`, which must follow the mounted
blocks; `bundle` arrays are `mmap_mode='r'` (wrap in `np.asarray` before heavy indexing); adding a
required field to a Pydantic schema breaks every direct constructor including the mock moves.

---

## 24. Known engineering limitations

* **`max_llm_usd` is declared but never enforced.** There is no LLM spend cap.
* **Dead `Cfg` fields.** `use_seq, use_vstat, use_aux, lambdarank, ensemble_members, L, alpha` are read
  by no block. `Cfg.loss_type`'s in-file comment advertises a `blend` value that `make_loss` does not
  implement — `blockspec` rejects it before a wasted launch.
* **`Phases.ablation_every` is unused.**
* **Random-exposure surface is FM-family only.** DIN and LightGBM need sibling rand caches for their
  `seq`/`gbm` features. `unbiased_eval` defaults `False` because it adds an inference pass per node.
* **`AblationRead` schema is defined and mocked but used by no role.**
* **The `Cfg` sidecar split is a workaround**, not a clean contract. It exists because
  `pipeline/contracts.py` is frozen. Two places must now be consulted to know a node's configuration.
* **Cross-run pooling is usually empty in practice.** The compatibility guard is strict by design, so
  a cache or code change invalidates the whole ledger. This is correct but means the ledger only pays
  off across a stable code state.
* **Behavior-aware history (`use_fb`) is implemented but disabled** — it measured negative under the
  honest policy ([RESEARCH.md](RESEARCH.md#11-behavior-aware-history-and-the-00165-artifact)).
* **The debug gate is torch-only** by design, so a broken FM/LightGBM block still costs a full run.
* **`resume: false` (the default) does not lower the achievable score.** The cross-run champion changes
  only the *comparison baseline* — which node gets labelled "improved" — not any node's actual score.
  Cold-starting simply relabels the first breakthrough as an improvement.
* **`--mock` replays scripted operator moves.** Its "0 interventions" is true of the *machinery*, but
  the research decisions come from `tests/mock_moves.py`, not from a model. Only a **live** run
  demonstrates autonomous research decisions; the headline result in
  [SUBMISSION.md](SUBMISSION.md#8-results) is from a live run for exactly this reason.

---

*[README](../../README.md) · [RESEARCH.md](RESEARCH.md) · [SUBMISSION.md](SUBMISSION.md)*
