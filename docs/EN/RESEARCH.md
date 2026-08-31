# RESEARCH.md — Scientific Record

**The authoritative scientific record**: what was asked, what was measured, what was rejected, and what
remains uncertain. For the engineering reference see [SYSTEM.md](SYSTEM.md); for the competition
narrative see [SUBMISSION.md](SUBMISSION.md); for orientation see the [README](../../README.md).

## Evidence labels

Every substantive claim carries one:

| Label | Meaning |
|---|---|
| **[VERIFIED]** | Confirmed by reading the current source. |
| **[MEASURED]** | Actually measured on this repository's data or run artifacts. Reproduction in §20. |
| **[LITERATURE]** | Supported by an external paper, cited in §18. |
| **[PROPOSED]** | Not implemented. Future work. |

Where an older document disagrees with a current measurement, the measurement wins — including where
the older document is an earlier version of *this* one (§8 records one such self-correction).

---

## Contents

1. [Research question](#1-research-question) · 2. [Data properties that shape the science](#2-benchmark-and-data-properties-that-shape-the-science) ·
3. [Metrics](#3-metrics-and-mathematical-definitions) · 4. [Noise & uncertainty](#4-noise-and-uncertainty-model) ·
5. [Statistical methodology](#5-statistical-methodology) · 6. [Why BCE is mismatched](#6-why-bce-is-mismatched) ·
7. [BPR](#7-bpr-theory-and-measured-result) · 8. [Auxiliary tasks](#8-auxiliary-task-investigation) ·
9. [The holdout leak](#9-the-holdout-leak-that-was-open) · 10. [Sequence chronology](#10-sequence-chronology-a-corrected-guarantee) ·
11. [Behavior-aware history & the +0.0165 artifact](#11-behavior-aware-history-and-the-00165-artifact) ·
12. [Ensemble & portfolio](#12-ensemble--portfolio-findings) · 13. [DIN](#13-din-findings) · 14. [LightGBM](#14-lightgbm-findings) ·
15. [Exposure & randomized data](#15-exposure-and-randomized-data-findings) · 16. [Cross-run evidence](#16-cross-run-evidence) ·
17. [Tried and rejected](#17-what-was-tried-and-rejected) · 18. [Literature](#18-literature-grounding) ·
19. [Conclusions, uncertainties, future work](#19-current-conclusions-residual-uncertainties-and-future-directions) ·
20. [Reproduction](#20-reproducing-the-measurements)

---

## 1. Research question

**Outer question.** Can an LLM-driven agent autonomously discover a stronger recommender than the
official baseline — proposing hypotheses, modifying real code, interpreting evidence statistically, and
correcting itself — under a fixed benchmark contract?

**Inner question.** Maximise `primary = ½(GAUC + nDCG@5)` on KuaiRand-Pure within-user ranking.

The two are not the same, and the more interesting result of this project is the outer one: the agent
produced an apparent +0.0165 breakthrough and then **rejected its own result** on evidence (§11).

---

## 2. Benchmark and data properties that shape the science

**Within-user ranking.** For each user, the candidate set is exactly that user's logged impressions.
A purely user-side feature is constant inside a user's ranking group and therefore carries **zero**
within-user ordering signal. **[VERIFIED]** — this is why the LightGBM feature set is item-centric and
why "add static user features" is encoded as a hard dead-end in the Proposer prompt.

**Splits** (fixed by the organizers, by date): train 2022-04-08…04-21 (1,141,112 rows) · valid
04-22…04-28 (124,909) · test 04-29…05-08 (170,588, hidden) · plus a public randomly-exposed log
(1,186,059 rows).

**Evaluation-surface statistics** **[MEASURED]**:

| split | rows | users | median impressions/user | positive rate |
|---|---:|---:|---:|---:|
| valid | 124,909 | 22,377 | **4** | 0.313 |
| rand | 1,186,059 | 27,285 | **22** | 0.085 |

Valid user composition: **30.3% all-negative, 11.9% all-positive, 57.8% discriminative.** A median of
4 impressions per user is the root cause of the noise floor in §4 — nDCG@5 over a 4-item list is an
extremely coarse per-user statistic.

**Auxiliary label statistics (train)** **[MEASURED]**:

| task | positive rate | P(long_view=1 \| task=1) | corr with long_view |
|---|---:|---:|---:|
| click | 0.4634 | 0.7233 | **0.7605** |
| like | 0.0187 | 0.6763 | 0.0992 |
| comment | 0.0026 | 0.8857 | 0.0590 |
| follow | 0.0010 | 0.7084 | 0.0250 |
| forward | 0.0010 | 0.6743 | 0.0226 |

Three orders of magnitude separate `click` from `follow`, so equal auxiliary weights do not mean equal
gradient contribution. And `click` is so correlated with `long_view` that using it as an auxiliary task
is closer to label smoothing than to multi-task learning — while `P(long_view=1 | click=0) = 0.0026`
makes it a near-oracle proxy, which is what made §9 serious.

**Raw log columns** **[VERIFIED]**: `user_id, video_id, date, hourmin, time_ms, is_click, is_like,
is_follow, is_comment, is_forward, is_hate, long_view, play_time_ms, duration_ms, profile_stay_time,
comment_stay_time, is_profile_enter, is_rand, tab`. Notably **there is no slot/position column** — so
the agent's honest refusal to build a position-bias tower was factually correct, not merely
well-formed.

---

## 3. Metrics and mathematical definitions

Notation: for user $u$, $\mathcal{I}_u$ are their impressions; $y_i\in\{0,1\}$ is `long_view`; $z_i$ is
the model logit; $\sigma(x)=1/(1+e^{-x})$.

### 3.1 AUC as the Mann–Whitney U statistic

For one user with $n^+$ positives and $n^-$ negatives, AUC is the probability a random positive
outscores a random negative. Instead of the $O(n^2)$ double sum, `auc()` sorts by score, assigns
**average ranks** to ties, and uses the rank-sum identity: with $R^+$ the sum of positive ranks,

$$\text{AUC}=\frac{R^+-\tfrac{n^+(n^++1)}{2}}{n^+n^-}.$$

The numerator is exactly the count of correctly-ordered pairs (the U statistic); average ranks give the
½ tie credit for free. $O(n\log n)$ and tie-correct.

### 3.2 GAUC

$$\text{GAUC}=\frac{\sum_{u:\,0<n^+_u<|\mathcal{I}_u|} n^+_u\,\text{AUC}_u}{\sum_{u:\,0<n^+_u<|\mathcal{I}_u|} n^+_u}.$$

All-positive and all-negative users are **excluded** (AUC undefined). Weighting by $n^+_u$ gives users
with more positives more influence. **Consequence:** GAUC is a sum of per-user pairwise-ordering
probabilities, so its natural surrogate is a **pairwise** loss (§7).

### 3.3 nDCG@k

With items sorted by predicted score, gain $g_i=2^{y_i}-1$ (=$y_i$ for binary) and discount
$1/\log_2(i+2)$:

$$\text{DCG@}k=\sum_{i=0}^{k-1}\frac{2^{y_i}-1}{\log_2(i+2)},\qquad \text{nDCG@}k=\frac{\text{DCG@}k}{\text{IDCG@}k}.$$

Top-heavy, and averaged over **all** users — an all-negative user has IDCG=0 and contributes 0.
**Consequence:** its natural surrogate is a **listwise, top-weighted** loss.

### 3.4 Why the ceiling is ≈0.86, not 1.0

On test, 27.1% of users are all-negative (nDCG always 0) and 9.2% all-positive (nDCG always 1),
regardless of the model. A perfect ranker therefore reaches GAUC 1.0, nDCG@5 ≈ 0.729,
**primary ≈ 0.8645 (test), 0.8484 (valid)**. Judge progress against ≈0.86. The FM baseline already
captures ≈31% of the usable range, which is why per-lever gains are small in absolute terms.

### 3.5 The FM backbone

With active embeddings $E=V[X]\in\mathbb{R}^{F\times k}$, linear weights $W[X]$, and $S_f=\sum_i E_{i,f}$:

$$z=b+\sum_i W_i+\tfrac12\sum_{f=1}^{k}\Big[S_f^2-\sum_i E_{i,f}^2\Big],$$

using $\sum_{i<j}\langle v_i,v_j\rangle=\tfrac12(\|\sum_i v_i\|^2-\sum_i\|v_i\|^2)$ to turn the
$O(F^2k)$ pairwise sum into $O(Fk)$.

**One gradient rule for every loss.** Since $\partial z/\partial W_i=1$ and
$\partial z/\partial E_{i,f}=S_f-E_{i,f}$, given per-row $g_j=\partial L/\partial z_j$ the backward pass
scatters $\partial L/\partial W_i=g$ and $\partial L/\partial E_{i,f}=g(S_f-E_{i,f})$, then Adam-updates.
**This factoring is the key design move**: the FM knows nothing about the objective, so any loss
producing $g$ drives the same backbone. **[VERIFIED]**

With BCE's $g=\sigma(z)-y$ this reproduces `baseline.py` line-for-line — which is why the M0 gate lands
on `primary_valid = 0.60147` to the digit. **[MEASURED]**

---

## 4. Noise and uncertainty model

**The single most consequential set of numbers in the project.** Three distinct variances exist, and
conflating them produces wrong decisions. **[MEASURED]**

| Source | What it is | Magnitude | Reduced by re-training? |
|---|---|---|---|
| Training stochasticity — FM | same cfg, same seed | **0.00000** (n=24 nodes, all exactly 0.60147) | n/a |
| Training stochasticity — LightGBM | same cfg, same seed | **0.00000** (n=14 nodes, all exactly 0.60205) | n/a |
| Training stochasticity — DIN (torch) | same cfg, **same seed** | **σ ≈ 0.00025**, range 0.00110 (n=13) | yes |
| **Validation-sample noise** | paired user-bootstrap SE of a primary *delta* | **σ ≈ 0.0008–0.0010** | **NO** |
| Cross-seed generalisation | published FM test std | 0.0008 | partly |

Two consequences that reshaped the whole design:

1. **The published `σ = 0.0008` is a *test* cross-seed number.** The quantity that actually limits the
   agent's decisions is the *paired validation delta* SE, ≈0.0009 — which happens to be similar but is
   a different thing, and is **not** reduced by re-running with more seeds.
2. **Multi-seed re-training attacks the wrong variance for two of three families.** FM and LightGBM are
   exactly deterministic at a fixed seed, so re-seeding them measures nothing that a free paired
   bootstrap does not. Multi-seed re-training is therefore reserved for stochastic (DIN) finalists.

**The practical resolution limit:** at SE ≈ 0.0009, an effect of +0.001 reaches only P(Δ>0) ≈ 0.87.
Effects below ~0.002 cannot be settled on this validation set by repetition alone — a fact that recurs
throughout §8 and §11.

---

## 5. Statistical methodology

**Paired user-level bootstrap.** For a candidate against its control, resample the 22,377 valid
**users** with replacement B=1000 times and rescore *both* models on the same resample, so the strong
per-user correlation between two models cancels. Report Δprimary, ΔGAUC, ΔnDCG, bootstrap SE, 95% CI
and `P(Δ>0)`. Cost ≈2 s per comparison, **no re-training**. **[VERIFIED]**

`agent/stats.py` reproduces the frozen evaluator's semantics exactly enough to decide on — agreement
**< 1e-5** on both synthetic data (including heavy score ties and degenerate users) and real node
predictions, two orders of magnitude below the smallest effect the project can resolve. **[MEASURED]**

**Evidence classes** from `P(Δ>0)`: `confirmed` ≥ 0.90 · `promising` 0.60–0.90 · `inconclusive` ·
`rejected` ≤ 0.10. These describe *evidence about an effect*, deliberately not tree adoption status —
a node can be `inconclusive` and still be the champion, or standalone-`rejected` and still earn a
portfolio slot.

**Pairing matters more than repetition.** Running an arm and its control inside the same repetition
removes confounding by code state, hardware and run conditions. This is what made a cross-run result
interpretable at n=8 that was uninterpretable when pooled unpaired (§8, §16).

**Multiple comparisons are taken seriously.** §8 reports one arm at P=0.91 out of 12 simultaneous
tests and explicitly declines to call it a finding.

---

## 6. Why BCE is mismatched

The organizers' strongest hint: the baseline optimises **pointwise logloss** but is scored on
**ranking** metrics. Since `primary = ½(GAUC + nDCG@5)`, and GAUC's natural surrogate is pairwise
(§3.2) while nDCG's is listwise (§3.3), the aligned objective should be a ranking loss. This is the
cheapest possible lever — only the loss changes, on a proven backbone. **[LITERATURE + VERIFIED]**

**The thesis turned out to be half right**, which is a genuine finding rather than a story:
the pairwise surrogate wins (§7); the listwise one **loses** (§17).

### 6.1 The listwise surrogate (softmax cross-entropy)

For a user group with logits $z_i$, define the target as uniform over the positives,
$p_i = y_i/\sum_j y_j$, and the model distribution $s_i = \text{softmax}(z/\tau)_i$. The listwise
cross-entropy and its gradient are

$$L = -\sum_i p_i \log s_i, \qquad \frac{\partial L}{\partial z_i} = \frac{1}{\tau}\big(s_i - p_i\big).$$

This is the ListNet / Plackett–Luce cross-entropy; with binary relevance it is a **bound on nDCG**
(Bruch et al. 2019), so it targets the nDCG@5 half of the metric. `_fit_group` packs whole users per
batch so each group's softmax is complete. **[LITERATURE + VERIFIED]**

**Why it loses here [MEASURED].** The model is a bare-ID FM. The listwise loss pushes hard on each
user's within-group ordering, which the high-cardinality `user_id × video_id` embeddings can simply
memorise — so it overfits fast (best epoch ≈2, then valid declines to **0.59971**, below baseline).
The pairwise BPR is better regularised on this data. See §17.

---

## 7. BPR theory and measured result

For a within-user positive $i^+$ and negative $i^-$ with $d=z_{i^+}-z_{i^-}$:

$$L_{\text{BPR}}=-\log\sigma(d),\qquad \frac{\partial L}{\partial z_{i^+}}=-\sigma(-d),\quad \frac{\partial L}{\partial z_{i^-}}=+\sigma(-d).$$

AUC is $\frac{1}{n^+n^-}\sum\mathbf{1}[d>0]$; replacing the step with the smooth $\log\sigma(d)$ makes
$\sum\log\sigma(z_i-z_j)$ a differentiable surrogate, and summing within-user pushes up **GAUC**
directly. **[LITERATURE]**

Sampling is fully vectorised: `_build_pair_index` precomputes per-user positive rows plus a flat
negative pool with `(start, len)` offsets, so each epoch draws `neg_ratio` negatives per positive with
one numpy call. An earlier per-positive Python loop was ~50× slower.

### The defect that hid this lever, and the measured recovery

`pipeline/baseline_blocks/loss.py` **hardcoded BCE and never read `cfg.loss_type`.** A hypothesis of
`{"loss_type": "bpr"}` as a config mutation therefore trained plain BCE and returned metrics
byte-identical to the root node. **[VERIFIED]**

**[MEASURED]** across the run history:

| condition | primary | n | note |
|---|---:|---:|---|
| config-only `loss_type=bpr`, before the fix | **0.60147** | 4 runs | identical to root; rank correlation to root **1.000** |
| BPR via an explicit block edit | **0.60358 ± 0.00010** | 25 nodes | 3 distinct values, from 3 Coder variants |
| config-only `loss_type=bpr`, **after** the fix | **0.60361** | direct test | |
| live Gemini run, node `n1` | **0.60361** | | `confirmed`, Δ+0.00214 vs root, P(Δ>0)=1.00 |

**BPR is a real, reliable lever worth ≈ +0.0021 over FM** — the largest single-model effect in the
project and the only one comfortably above the ~0.0009 resolution (≈2.3 SE). The defect had converted
it into a *fabricated negative result*: the agent recorded "BPR → 0.6015, rejected" and its memory then
told the Proposer not to repeat it.

**Lesson generalised.** The system now validates that a proposed knob provably reaches an execution
path before training ([SYSTEM.md §11](SYSTEM.md#11-config-effectiveness-validation)). *Intended
intervention ≠ executed intervention* is now a recorded, checkable property.

---

## 8. Auxiliary-task investigation

**Status: INCONCLUSIVE.** Not an established win, and not an established loss.

### Round 1 screen **[MEASURED]**

7 arms × 2 objectives × 2 training repetitions = 28 runs on the chronological v10 cache, each arm
paired against its own no-aux control at the same seed:

| arm | BPR Δ vs no-aux | P(Δ>0) | BCE Δ vs no-aux | P(Δ>0) |
|---|---:|---:|---:|---:|
| +click | +0.00015 | 0.78 | +0.00011 | 0.45 |
| +like | −0.00000 | 0.42 | −0.00013 | 0.67 |
| +follow | +0.00009 | 0.83 | +0.00026 | 0.86 |
| +comment | −0.00020 | 0.49 | +0.00013 | 0.49 |
| +forward | +0.00032 | **0.91** | +0.00011 | 0.63 |
| +click+like | +0.00043 | 0.73 | −0.00036 | 0.33 |

**Every effect is smaller than the ~0.0009 resolution of the validation set (§4).** The single P=0.91
is **not** reported as a finding: it is one marginal hit among 12 simultaneous comparisons, exactly the
rate chance produces. Round 2 of the sequential ladder therefore has nothing to eliminate, and the
ladder correctly stops rather than spending 30 more trainings chasing noise.

### A self-correction, recorded rather than quietly dropped

An earlier version of this analysis claimed, from 8 historical runs that happened to contain both arms,
that `aux[click,like]` hurts DIN+BPR — **worse in 8/8 runs, paired mean −0.00037, SE 0.00009,
t = −4.16**. That measurement was real, but it was taken on the **pre-chronological cache and an older
code state**. The compatibility rule built into `agent/ledger.py` says entries must not be pooled
across a cache or code change — and applying that rule to our own earlier claim, **it does not carry
forward.** It is recorded here as history, not as current evidence.

Two independent observations still lean negative and are worth keeping:

* the live Gemini run's own iteration 4 measured DIN+aux(click,like) at **0.60247** against a
  **0.60366** DIN control (Δ−0.0012, classified `rejected`), with `rank_corr 0.974` and
  **`EMC +0.00000`** — redundant rather than complementary;
* §11's leakage-free behavior-aware result is negative too.

**Why MMoE/PLE were *not* built.** The ladder is: shared trunk → measure task compatibility → only if
negative transfer is *demonstrated*, task-specific embeddings (STEM-lite) → task-specific bottoms →
expert routing. Evidence points to auxiliary heads being unhelpful *at all* here, which is a reason not
to add expert routing. `blockspec` rejects `mtl_arch ∈ {mmoe, ple}` before a training launch is wasted.
**[LITERATURE + MEASURED]**

**Next discriminating test [PROPOSED].** The binding constraint is **power**, not design. At σ ≈ 0.0009
an effect of 0.0003 needs a far larger evaluation surface — the argument for §15, not for more
repetitions on valid.

---

## 9. The holdout leak that was open

**[MEASURED]** `is_click` correlates **0.7605** with `long_view` on train (0.7515 on valid), and
`P(long_view=1 | is_click=0) = 0.0026`. Ranking the validation set **by `is_click` alone** scores:

| | primary | GAUC | nDCG@5 |
|---|---:|---:|---:|
| `is_click` as the score | **0.7466** | 0.8664 | 0.6268 |
| FM baseline | 0.60147 | | |
| oracle ceiling | 0.8484 | | |

That is **58.8% of the entire remaining headroom above FM**, with no training at all.

**[VERIFIED]** Until this was closed, `runs/_cache/test_y.npy` and
`runs/_cache/aux/{valid,test}_aux.npy` sat inside the directory handed to every block as
`bundle.cache_dir`, and `numpy` is on the executor's import allowlist — one `np.load` away.

**[LITERATURE]** This is not hypothetical. METR observed explicit reward hacking in **39 of 128** o3
runs on RE-Bench (30.4%) with no prompting to cheat; MLE-bench and the reward-hacking agent benchmarks
report the same failure mode.

**Resolution.** Label-derived arrays moved to a sibling `runs/_holdout/`; `load_aux` refuses non-train
splits; a build-time assertion fails the run if they reappear; a static guard rejects the relevant path
literals and `open`/`eval`/`exec`; a tripwire quarantines any node above 0.70.
See [SYSTEM.md §8](SYSTEM.md#8-leakage--integrity-protections).

**A retired claim.** Earlier documentation described the guard as making label access *"physically
impossible."* That was **overstated** and is withdrawn. The accurate claim is layered and is stated as
such in SYSTEM.md.

---

## 10. Sequence chronology: a corrected guarantee

**The old claim.** Earlier documentation stated that temporal safety was "structural: we process rows
in global time order."

**[MEASURED] It was not true.** `data.load()` does not sort — it appends rows in CSV order and filters
by date — and the KuaiRand logs are not time-ordered within a user. On
`log_standard_4_22_to_5_08_pure.csv`: **47,742 contiguous user runs for 25,877 users**, and **18,763
per-user `time_ms` inversions**. Replaying the old construction, the share of rows whose "prior"
history contained a **later-dated** item was:

| split | rows | rows with a future item in history |
|---|---:|---:|
| train | 1,141,112 | **30.83%** |
| valid | 124,909 | **20.89%** |
| test | 170,588 | **31.54%** |

With ids-only histories the damage was bounded (video ids carry no label, and no row's history crossed
a split boundary), so it did not invalidate published scores — but it **did** invalidate the stated
guarantee, and it was a hard blocker for §11.

**[MEASURED] After the fix**: histories are built in true `(user, time_ms)` order and independently
re-verified by recomputing expected histories from the raw logs — **0 violations on all three splits**,
400 sampled rows each, plus exact agreement of sampled histories with an independent rebuild.

**A second, smaller finding.** `date` and `time_ms` disagree at the split boundary: **28 test-dated
rows carry timestamps earlier than the last valid row** (the logs' `date` is a local calendar day; the
timestamps are epoch ms). Time order alone would place a handful of test-window events into valid
histories. Filtering on split index as well as time makes "a train/valid row never sees a test-window
event" structural rather than dependent on a timestamp quirk.

---

## 11. Behavior-aware history and the +0.0165 artifact

**This is the project's most important scientific episode.** A literature-supported mechanism produced
an apparent breakthrough, the breakthrough was disbelieved on the basis of its own magnitude,
diagnosed, and rejected.

### The hypothesis **[LITERATURE]**

In an autoplay short-video feed, videos play automatically and the user *passively receives* content,
so a skip is meaningful negative evidence and an impression is not a positive. RecSys'23's
*Understanding and Modeling Passive-Negative Feedback for Short-video Sequential Recommendation*
proposes encoding both positive and passive-negative feedback in the sequence. Our DIN saw history as
video ids only: every history event looked identical regardless of whether the user skipped it, watched
it, or liked it.

**Implementation.** `seq_build` emits `{split}_fb.npy` with 7 states
(`PAD/SKIP/SHORT/NORMAL/LONG/EXPLICIT/UNKNOWN`); `DIN` gains an `fb` embedding added to each history
event: `eh = emb(seq) + fb(fbstate)`. One extra embedding lookup. Gated by `use_fb`.

### Step 1 — the naive version looked spectacular **[MEASURED]**

With feedback states available for every event preceding the scored row, behavior-aware DIN scored
**0.61925** against **0.60275** without (3 training repetitions each, paired by seed):
**+0.0165, ~18× the noise floor**, P(Δ>0) = 1.00.

**That is not a plausible effect size in a benchmark where every other lever moves ~0.002**, so it was
investigated instead of banked.

### Step 2 — the diagnosis **[MEASURED]**

| split | history events whose outcome the model could see |
|---|---:|
| train | 100.0% |
| valid | **100.0%** |
| test | **75.8%** |

A valid row could see the outcomes of *earlier valid rows*; a test row could see no test-window
outcomes, because a submission scores all test rows at once with no feedback in between. The gain was
**structurally available on the selection set and structurally unavailable on the scored set** —
transductive leakage of the validation labels. It would have inflated every model-selection decision in
the run and delivered nothing on test.

### Step 3 — the honest policy, and the real answer **[MEASURED]**

`fb_policy="train_only"` (now the default) lets **only train-window outcomes** become features, so
valid and test are treated identically. Known-outcome coverage becomes train 100% / valid 81.7% /
test 52.3%, and:

| arm (3 training reps, paired by seed) | mean primary | Δ vs no-feedback | per-seed Δ | P(Δ>0) |
|---|---:|---:|---|---:|
| DIN, no feedback states | 0.60265 | — | — | — |
| DIN + feedback states | 0.60099 | **−0.00167** | −0.00167, −0.00178, −0.00155 | 0.00 |
| DIN + feedback states, `fb_dropout=0.5` | 0.60196 | **−0.00070** | −0.00062, −0.00080, −0.00068 | 0.14 |

### Step 4 — the diagnosed remedy confirms the mechanism but does not rescue the lever

The model trains on 100% known states and scores against 52–82% unknown ones. Masking states to
`UNKNOWN` during training (`fb_dropout=0.5`) recovers **58% of the loss** (−0.00167 → −0.00070), which
confirms train/serve mismatch is the mechanism — but the lever is still not beneficial.

### Step 5 — nor is it rescued by portfolio diversity **[MEASURED]**

Against the plain DIN: rank correlation **0.942**, best 2-way blend gain **+0.00002**. Negative on both
axes (§12), so it is not retained as a "diverse-but-weak" member either.

### Conclusion

The mechanism is real and literature-supported, but **this benchmark's offline split does not deliver
the feedback it needs at scoring time**: only 52% of a test row's history has a usable outcome. The
capability remains implemented, tested and leak-safe; **`use_fb` defaults to OFF**.

**A roadmap prediction disproven.** This was previously called "the top new lever" and the best
candidate for gains beyond +0.005. The measurement contradicts it. It is corrected here rather than
re-argued. Cost of finding out: ~4 minutes of GPU.

**Next discriminating test [PROPOSED]:** condition on feedback *recency*, and evaluate on the
random-exposure split where per-user impression counts are 5× higher (§15).

---

## 12. Ensemble / portfolio findings

**Standalone score is a poor predictor of portfolio value.** **[MEASURED]** In the live run:

| member | family | standalone | rank corr to champion | EMC (leave-one-out) |
|---|---|---:|---:|---:|
| `n1` | fm + BPR | 0.60361 | 0.906 | +0.00025 |
| `n2` | lgbm | **0.60205** (2nd weakest) | **0.860** (most decorrelated of the models the agent *discovered*) | **+0.00028** |
| `n3` | din | 0.60366 (best single) | 1.000 | +0.00028 |
| `root` | fm + BCE | 0.60147 (weakest overall) | **0.852** (most decorrelated overall) | +0.00017 |
| `n4` | din + aux | 0.60247 | 0.974 | **+0.00000** |

Two readings that matter. **LightGBM contributes more than the FM+BPR node (+0.00028 vs +0.00025)
despite scoring 0.0016 lower standalone** — it is the most decorrelated model the agent *discovered*
(the FM baseline itself is more decorrelated still, at 0.852, but it was given, not found). And the
auxiliary-head node is **redundant** — corr 0.974, EMC exactly **0.00000** — despite a standalone
score *above* LightGBM's. Ranking experiments by standalone primary alone would have kept the wrong
one. Note the absolute contributions are small (2–3 in the fourth decimal); the ordering is the
finding, not the magnitude.

**In-sample ensemble numbers are optimistic, and the size of the bias was measured.** **[MEASURED]**
Eight random 50/50 user splits, weights tuned on half A and evaluated on half B:

| quantity | mean over 8 splits |
|---|---|
| held-out ensemble gain over the best single member | **+0.00074** |
| weight-selection optimism (in-sample optimum − held-out) | **+0.00072** |
| in-sample gain reported by a naive tuned-on-everything procedure | +0.00159 |

**Roughly half of a naively reported ensemble gain is weight-tuning overfit.**

**The reporting procedure was corrected as a result.** An earlier proposal used greedy selection scored
on a "holdout" half and then reported that half's number — but a subset consulted once per greedy step
to choose members is a *selection* set, and reporting on it is biased by exactly the mechanism above.
The implemented procedure is a user-level K-fold cross-validation of the whole assembly procedure, with
four data roles never conflated ([SYSTEM.md §15](SYSTEM.md#15-portfolioensemble-machinery)).

**Live run result [MEASURED]:** tuned-on-all-valid **0.60463** (optimistic) vs honest 5-fold CV
**0.60409 ± 0.00141**; CV gain over the best single member **+0.00064 ± 0.00051**. Note the gain is
roughly one SE — real, consistently positive, and modest.

---

## 13. DIN findings

**Pure attention is not enough.** A pure-sequence DIN *underperforms* FM (0.5895 historical), because
attention over video history carries no user identity and cannot memorise user-specific preference.
Adding a **DeepFM part** (FM linear + 2nd-order cross over the base 5 fields) lifts it above baseline.
Sequences help *on top of* the FM signal, not instead of it. **[MEASURED, historical]**

**Interest unit** (no softmax — DIN's signature, so raw activation magnitude carries interest
intensity):

$$a_h=\text{MLP}\big([\,e_h\,\|\,e_t\,\|\,e_h\odot e_t\,\|\,e_h-e_t\,]\big),\qquad u=\sum_h m_h a_h e_h,$$

$$z = z_{\text{fm}} + \text{MLP}\big([\,u\,\|\,e_t\,\|\,\textstyle\sum_i e^{\text{base}}_i\,]\big).$$

The four-way concatenation expresses both similarity ($e_h\odot e_t$) and contrast ($e_h-e_t$).

**Current measured position [MEASURED]:** on the chronological v10 cache, DIN/BPR scores
**0.60261 ± 0.00044** (n=3) and was the best single model in the live run at **0.60366** — but its
delta against the BPR-FM champion was **inconclusive** (P(Δ>0) = 0.52). DIN is *competitive*, not
*established as better*, and its main value in the live run was portfolio membership.

**DIN is the only stochastic family.** Two trainings of an identical config at the same seed produce
models with rank correlation **0.926** and primaries differing by 0.00042 (§4). This is why the
submitted predictions must come from the *same trained instance* that was validated
([SYSTEM.md §18](SYSTEM.md#18-submission-and-finalization)).

---

## 14. LightGBM findings

`objective='lambdarank'`, query groups = users. For each within-query pair the gradient is scaled by
$|\Delta\text{nDCG}_{ij}|$, focusing gradient on swaps that move the top-5. Its axis-aligned tabular
inductive bias is **complementary** to the embedding models. **[LITERATURE + VERIFIED]**

**Leakage-safe, item-centric feature design.** Because ranking is within-user, only features that vary
across a user's impressions carry signal: item `long_view` rate
$\hat r_v=\frac{\text{pos}_v+\alpha\bar r}{\text{imp}_v+\alpha}$ (smoothed, $\alpha=20$, **train only**),
author rate, log item/author impression counts, log duration, `tab`, and 16 global engagement columns
z-scored on train. No current-row `play_time` — that is the label's source.

**It was un-tunable, and that was costly.** **[MEASURED]** All 14 LightGBM nodes ever run scored
**exactly 0.60205, std 0.00000**, because `gbm.train_ranker` read only `cfg.seed` and hardcoded every
hyper-parameter. The agent could not tune the member with the *largest* ensemble marginal contribution.
Hyper-parameters now come from the `cfg_ext.json` sidecar; defaults reproduce 0.60205 exactly. A first
tuned configuration scored **0.60152** (Δ−0.00053, P(Δ>0)=0.10) — i.e. tuning is now *possible*, and
that particular setting is worse. Recorded as-is.

---

## 15. Exposure and randomized data findings

KuaiRand ships a **randomly-exposed** log (items shown independent of the policy) — an unbiased sample,
and the reason the dataset exists. **[LITERATURE]**

**Two levers, correctly separated:**

* **E1 — random-exposure evaluation.** Implemented; reported as `primary_rand` alongside
  `primary_valid` plus the gap. It is a **second robustness surface, never the competition target**.
  Interpretation: valid↑/rand↑ is stronger generalisation evidence; valid↑/rand↓ suggests exploitation
  of the logging policy; valid↓/rand↑ is scientifically interesting but not automatically the champion.
* **E2 — popularity down-weighting.** `train_np._ips_weights` computes $w \propto 1/\sqrt{\text{freq}(item)}$,
  mean-normalised. **This is inverse-popularity weighting, not inverse propensity.** A true IPS
  estimator needs $1/P(\text{exposure}\mid u,\text{context},\text{policy})$, and the square root makes it
  not even an unbiased popularity correction. Earlier documentation over-described it as IPS; that is
  corrected here. **[VERIFIED]**
* **E3 — true propensity / SNIPS correction [PROPOSED].** Estimate $P(\text{exposure})$ from the random
  log, then apply self-normalised weights. Genuinely novel for this benchmark; not built.

**A historical measurement worth keeping [MEASURED]:** FM scores 0.6015 on biased valid but ≈0.364 on
the random-exposure log — a +0.24 gap, which is the exposure-bias signal the surface is designed to
expose. The random log's `long_view` positive rate is 0.085 versus 0.313 on biased valid.

**Scope limit [VERIFIED]:** the rand surface currently runs for the FM family only. DIN and LightGBM
need sibling rand caches for their `seq`/`gbm` features. `unbiased_eval` defaults `False` because it
costs an inference pass per node.

**Statistical caution [PROPOSED]:** the rand split is 9.5× larger with 5× more impressions per user, so
it *should* offer better resolution — but the rand-side uncertainty must be **measured**, not assumed
from row count, because the per-user structure and positive rate differ substantially.

---

## 16. Cross-run evidence

`agent/ledger.py` persists every executed experiment keyed by an **arm** (`family|loss|aux`,
deliberately excluding seed — seeds are repetitions, not arms).

**Compatibility is enforced, and it is load-bearing.** Pooling requires the same `cache_version` *and*
the same `code_state` hash. Pooling incompatible results is worse than not pooling: it is exactly what
produced the retracted t = −4.16 claim in §8. At the time of writing the ledger holds 14 entries and
**0 are compatible** with the current cache and code state — the guard correctly refusing to pool
across the v6→v10 cache rebuilds and the implementation changes.

**Repeated trainings on the same validation users are not independent datasets.** They estimate
training stochasticity, not validation-sample uncertainty and not hidden-test generalisation. The three
are kept in separate fields wherever they surface.

**Honest consequence:** the ledger only pays off across a *stable* code state. It has not yet
accumulated usable cross-run evidence in this project's lifetime.

---

## 17. What was tried and rejected

| Direction | Verdict | Evidence |
|---|---|---|
| **Softmax-CE (listwise, nDCG surrogate)** | **Rejected — statistically supported negative** | 0.59971, *below* baseline (n=13, std 0.00000). Live run: Δ−0.00391 vs the BPR champion, P(Δ>0)=0.00. The bare-ID FM memorises each user's within-group order and overfits fast (best epoch ~2). Half the loss↔metric thesis fails, and that is a finding. **[MEASURED]** |
| **Behavior-aware history** | **Rejected — negative under honest evaluation** | §11 |
| **MMoE / PLE expert routing** | **Not built — evidence points away** | §8 |
| **MCTS over pipeline configs** | **Rejected — wrong tool for the budget** | Each node is a real training run; under the official rule a run ends after ~4–6 experiments, far too few for rollouts and backups to produce reliable value estimates. Best-first + ε-exploration extracts more signal per evaluation. **[LITERATURE + VERIFIED]** |
| **Whole-pipeline generation each run** | **Rejected** | More failure modes, more tokens, and it lets the agent alter the code the score depends on. |
| **Adding static user-side features** | **Rejected — organizer-proven dead end** | Constant within a user ⇒ zero within-user ranking signal (§2). |
| **Raising embedding size `k` for its own sake** | **Rejected — organizer-proven dead end** | |
| **A single blended loss (α·BPR + (1−α)·softmax-CE)** | **Deprioritised** | Once softmax-CE proved to overfit, diversity is captured better by *ensembling* an FM, a DIN and a LightGBM than by blending two FM losses. |
| **Live web search per iteration (MLE-STAR style)** | **Deferred** | A curated lever playbook in the Proposer prompt is cheaper and encodes the organizer dead-ends so no iteration is wasted. |
| **1B-parameter recommenders, industrial RL, large generative RecSys** | **Rejected** | Incompatible with the compute contract; Feasibility is scored on tokens and wall-clock. |
| **A learned second-stage reranker over model scores** | **Deferred** | It stacks a second layer of valid-set selection on top of the +0.00072 optimism already measured in §12. |

---

## 18. Literature grounding

Each citation is attached to the specific claim it supports.

| Claim it supports | Source |
|---|---|
| Passive-negative (skip) feedback is a distinct, valuable signal in autoplay short-video feeds (§11) | *Understanding and Modeling Passive-Negative Feedback for Short-video Sequential Recommendation*, RecSys 2023 — arXiv:2308.04086 |
| Implicit/negative feedback learning in industrial short-video recommenders (§11) | arXiv:2308.13249 |
| KuaiRand is a randomised-exposure dataset recording 12 feedback signals (§15) | KuaiRand — arXiv:2208.08696 |
| Shared embeddings themselves cause negative transfer; shared + task-specific embeddings restore it (§8) | STEM, AAAI-24 — arXiv:2308.13537 |
| Task-specific bottom representations as an intermediate step before expert routing (§8) | DTRN — arXiv:2308.05996 |
| Ablation-guided refinement of individual code blocks + learned ensembling is the current SOTA ML-engineering agent design (§12, block-level search) | MLE-STAR, NeurIPS 2025 — arXiv:2506.15692 |
| Code-space tree search as an ML-engineering agent (node = runnable solution, best-first selection) | AIDE — arXiv:2502.13138 |
| Best-first tree search managed by an experiment-manager agent (orchestrator-as-policy split) | AI-Scientist-v2 — arXiv:2504.08066 |
| The hypothesis → implement → validate → feedback loop | RD-Agent (Microsoft) |
| MCTS over pipeline configs — considered and deliberately not followed (§17) | SELA — arXiv:2410.17238 |
| Autonomous ML agents reward-hack at a high base rate (§9) | MLE-bench — arXiv:2410.07095; METR RE-Bench (39/128 o3 runs) |
| BPR as an AUC surrogate (§7) | Rendle et al. 2009 |
| Softmax-CE as an nDCG bound for LTR (§6, §17) | Bruch et al. 2019 |
| DIN target attention over behavior (§13) | Zhou et al. 2018 |
| DeepFM wide+deep memorisation (§13) | Guo et al. 2017 |
| LightGBM LambdaRank (§14) | Ke et al. 2017 |

**Two corrections to how this literature is used.** MLE-STAR's ensembling result supports making the
search ensemble-aware, but MLE-STAR ensembles *independently generated full solutions*, not tree
siblings — our greedy forward selection over all viable nodes is the closer analogue. And the framing
that "search topology should follow opportunity density" is right in general, but the operative
constraint here is **statistical resolution** (§4): with SE ≈ 0.0009 no search policy can steer on
+0.0004 effects. Widening the evaluation surface dominates changing the search algorithm.

---

## 19. Current conclusions, residual uncertainties, and future directions

### Established findings **[MEASURED]**

1. **BPR is a real lever worth ≈+0.0021** over the FM baseline (0.60147 → 0.60361), reliable
   (std 0.00010, n=25), and the only single-model effect comfortably above the resolution limit.
2. **Softmax-CE is a statistically supported negative result** (0.59971, below baseline).
3. **Standalone score does not predict portfolio value**: LightGBM (corr 0.860, standalone 0.60205)
   contributes more than the FM+BPR node (standalone 0.60361), while an auxiliary-head node with a
   *better* standalone score than LightGBM's is redundant (EMC exactly 0.00000).
4. **In-sample ensemble numbers are optimistic by ≈+0.00072**; the honest gain over the best single
   member is ≈+0.0006–0.0007.
5. **Behavior-aware history is negative under leakage-free evaluation** (−0.00167; −0.00070 with
   dropout).
6. **FM and LightGBM are exactly deterministic at a fixed seed; DIN is not** (σ ≈ 0.00025).
7. **The validation set cannot resolve effects below ~0.002** by repetition alone.

### Promising but not established

* DIN as a standalone model (best single in the live run at 0.60366, but P(Δ>0)=0.52 vs the BPR-FM
  champion — competitive, not proven better).
* The random-exposure split as a higher-resolution decision surface (§15) — plausible from its size,
  but its uncertainty has not been measured.

### Inconclusive

* **All auxiliary-task arms** (§8). Effects are below the resolution limit in both directions.
* Whether the BCE-vs-BPR interaction on auxiliary tasks is real (the sign differs, but neither
  direction is significant).

### Residual uncertainties

* **Test-set generalisation is unmeasured by construction.** Every number here is validation-side. The
  honest CV estimate accounts for weight-tuning optimism, not for valid→test distribution shift.
* **Cross-run evidence is currently empty** (§16): no ledger entries are compatible with the current
  cache and code state.
* **The exposure gap is large and unexplained.** FM: 0.6015 biased-valid vs ≈0.364 random-exposure.
  Whether closing it would help the competition metric is untested.
* **`n=2–3` repetitions** underpin the DIN and auxiliary measurements. They are enough to establish
  sign consistency for large effects (§11) but not to resolve small ones (§8).

### Future directions **[PROPOSED]**

1. **Widen the evaluation surface** (§15) — the highest-leverage change available, because power is
   the binding constraint on nearly every open question.
2. **Recency-conditioned feedback states** (§11) — the named next test for the one mechanism whose
   failure mode is understood.
3. **True propensity / SNIPS correction** (§15 E3).
4. **Behavior-aware / hard negative sampling** — deferred: it depends on the feedback-state signal,
   which did not survive honest evaluation.
5. **A native numpy LambdaRank loss** (`Cfg.lambdarank` is reserved; LightGBM currently supplies it).
6. **Alternative sequence backbones** (SASRec, FMLP-Rec) — only after DIN is exhausted, and any
   replacement must keep the FM part (§13).

---

## 20. Reproducing the measurements

All numbers above come from this repository. Representative procedures; run from the repo root with
`cudaenv/Scripts/python.exe`.

| Measurement | How |
|---|---|
| Bootstrap matches the frozen evaluator | `python -m tests.test_stats` |
| Honoured-knob contract, incl. bit-identity | `python -m tests.test_blockspec` |
| Holdout isolation + `is_click` severity (0.7466) | `python -m tests.test_leakage` |
| Convergence semantics, dedup, no-op classes | `python -m tests.test_orchestration` |
| Chronology (0 violations) + feedback-state policy | `python -m tests.test_sequence` |
| Live end-to-end result | `runs/run_20260831_090457/resource_report.json` |
| Per-node evidence, portfolio, provenance | `runs/<run>/run_log.jsonl` |
| Pooled historical scores by arm | iterate `runs/run_*/run_log.jsonl`, key by `(family, loss_type, aux_tasks)`; infer family from `code_diff` for pre-fix runs where `model_type` was mislabelled |

The behavior-aware and auxiliary-task screens were run as offline measurement scripts (28 and 12
training runs respectively) rather than as agent iterations — under the official `eps=0.002, N=3` rule
a compliant benchmark run converges after ~4 experiments, so a multi-round ladder cannot fit inside
one, and manufacturing iterations to make it fit would violate the contract
([SYSTEM.md §16](SYSTEM.md#16-convergence-and-benchmark-budget-semantics)).

---

*[README](../../README.md) · [SYSTEM.md](SYSTEM.md) · [SUBMISSION.md](SUBMISSION.md)*
