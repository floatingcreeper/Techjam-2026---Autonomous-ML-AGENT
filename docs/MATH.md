# The Math — metrics, models, and losses

Full derivations behind the implementation. This is the companion to the
[README](../README.md); every formula here corresponds to code you can read, and the section
anchors are the ones the README links to.

Notation: for a user $u$, $\mathcal{I}_u$ is the set of that user's logged impressions;
$y_i\in\{0,1\}$ is the `long_view` label of impression $i$; $s_i=z_i$ is the model's score
(logit) for it; $\sigma(x)=1/(1+e^{-x})$.

---

## 1. The metrics

The score is `primary = ½(GAUC + nDCG@5)`, computed in `evaluate.py`. Both halves are
**within-user** — each user is ranked only against their own impressions.

### 1.1 AUC as the Mann–Whitney U statistic

For one user with $n^+$ positives and $n^-$ negatives, AUC is the probability that a random
positive outscores a random negative:

$$\text{AUC}=\frac{1}{n^+ n^-}\sum_{i\in\text{pos}}\sum_{j\in\text{neg}}
\Big(\mathbf{1}[s_i>s_j]+\tfrac12\mathbf{1}[s_i=s_j]\Big).$$

Rather than the $O(n^2)$ double sum, `auc()` sorts by score, assigns **average ranks** to ties,
and uses the rank-sum identity. If $R^+$ is the sum of ranks of the positives (ranks starting at 1),

$$\text{AUC}=\frac{R^+-\tfrac{n^+(n^++1)}{2}}{n^+ n^-}.$$

The numerator $R^+-\frac{n^+(n^++1)}{2}$ is exactly the count of correctly-ordered pairs (the U
statistic); dividing by $n^+n^-$ turns the count into a probability. Average ranks give the
$+\tfrac12$ tie credit for free. This is why the code is $O(n\log n)$ and tie-correct.

### 1.2 GAUC — group AUC

$$\text{GAUC}=\frac{\sum_{u:\,0<n^+_u<|\mathcal{I}_u|} n^+_u\cdot \text{AUC}_u}
{\sum_{u:\,0<n^+_u<|\mathcal{I}_u|} n^+_u}.$$

Users that are all-positive or all-negative are **excluded** — AUC is undefined for them (no
pos–neg pair exists). The weighting by $n^+_u$ gives users with more positives more influence.
**Consequence for modeling:** GAUC is a sum of per-user pairwise-ordering probabilities, so the
natural surrogate is a **pairwise** loss (BPR, §3.1).

### 1.3 nDCG@k

With items sorted by predicted score, gain $g_i=2^{y_i}-1$ (which is just $y_i$ for binary
labels) and discount $1/\log_2(i+2)$ (0-indexed rank $i$):

$$\text{DCG@}k=\sum_{i=0}^{k-1}\frac{2^{y_i}-1}{\log_2(i+2)},\qquad
\text{nDCG@}k=\frac{\text{DCG@}k}{\text{IDCG@}k},$$

where IDCG is the DCG of the ideal ordering (labels sorted descending). nDCG is **top-heavy** (the
discount shrinks fast) and is averaged over **all** users — an all-negative user has IDCG $=0$ and
contributes nDCG $=0$. **Consequence:** the natural surrogate is a **listwise, top-weighted** loss
(softmax-CE / LambdaRank, §3.2).

### 1.4 Why the ceiling is 0.86, not 1.0

On the test set, 27.1 % of users are all-negative (nDCG always 0) and 9.2 % all-positive (nDCG
always 1) regardless of the model. A perfect ranker (using the true labels as scores) therefore
reaches only

$$\text{GAUC}=1.0,\quad \text{nDCG@5}\approx0.729,\quad \text{primary}\approx0.8645\ (\text{test}),\;0.8484\ (\text{valid}).$$

So progress should be judged against $\approx0.86$, not $1.0$; the FM baseline (0.5946 test) already
captures $\approx31\%$ of the usable range. This is why the per-lever gains are small in absolute
terms — the attainable band is narrow.

---

## 2. The Factorization Machine

`pipeline/lib/fm.py`. Each of the 5 fields contributes one active index into a flat space of size
`dim`; let the active indices of a row be the multiset in $X$, with embeddings $E=V[X]\in\mathbb{R}^{F\times k}$
and linear weights $W[X]$.

### 2.1 Forward (the $O(nk)$ trick)

$$z=b+\sum_{i}W_i+\underbrace{\tfrac12\sum_{f=1}^{k}\Big[\big(\textstyle\sum_i E_{i,f}\big)^2-\sum_i E_{i,f}^2\Big]}_{\text{2nd-order interaction}}.$$

Let $S_f=\sum_i E_{i,f}$ (sum over the $F$ active fields). The interaction is
$\tfrac12\sum_f(S_f^2-\sum_i E_{i,f}^2)$ — computed as `0.5*((S**2).sum(1) - (E**2).sum((1,2)))`.
This is the standard identity $\sum_{i<j}\langle v_i,v_j\rangle = \tfrac12(\|\sum_i v_i\|^2-\sum_i\|v_i\|^2)$,
which turns the $O(F^2k)$ pairwise sum into $O(Fk)$.

### 2.2 Backward — one gradient rule for every loss

The model exposes `apply_grad(X, g, cache)` where $g_j=\partial L/\partial z_j$ is the **per-row loss
gradient** (whatever the loss). Because $z$ is a smooth function of $V,W,b$:

$$\frac{\partial z}{\partial W_i}=1,\qquad
\frac{\partial z}{\partial E_{i,f}}=S_f-E_{i,f}\quad(\text{since }\partial S_f/\partial E_{i,f}=1).$$

So per row, $\partial L/\partial W_i=g$ and $\partial L/\partial E_{i,f}=g\,(S_f-E_{i,f})$ — exactly
`np.add.at(gW, X, g)` and `np.add.at(gV, X, g[:,None,None]*(S[:,None,:]-E))`. Parameters are updated
with Adam (first/second moment with bias correction). **This factoring is the key design move**: the
FM knows nothing about the objective; any loss that can produce $g=\partial L/\partial z$ (BCE, BPR,
softmax-CE) drives the same backbone.

### 2.3 Why BCE reproduces the baseline exactly

For pointwise logloss $L=-\big[y\log\sigma(z)+(1-y)\log(1-\sigma(z))\big]$, we get $\partial L/\partial z=\sigma(z)-y$.
Feeding $g=(\sigma(z)-y)/B$ into `apply_grad` reproduces `baseline.py`'s hand-derived FM update
line-for-line — hence the M0 gate lands on `primary_valid = 0.6015` to the digit.

---

## 3. Ranking losses (Lever A)

`pipeline/lib/losses.py`. All produce $g=\partial L/\partial z$ per row and declare a `.mode`
(`point`/`pair`/`group`) so `train_np.py` batches correctly.

### 3.1 BPR — a smooth AUC surrogate → GAUC

For a within-user positive $i^+$ and negative $i^-$, BPR maximizes the probability the positive
outscores the negative:

$$L_{\text{BPR}}=-\log\sigma(z_{i^+}-z_{i^-}),\qquad d=z_{i^+}-z_{i^-}.$$

$$\frac{\partial L}{\partial z_{i^+}}=-\big(1-\sigma(d)\big)=-\sigma(-d),\qquad
\frac{\partial L}{\partial z_{i^-}}=+\sigma(-d).$$

`bpr_pair(zp, zn)` returns exactly these. **Why it targets GAUC:** AUC $=\frac{1}{n^+n^-}\sum_{i,j}\mathbf{1}[z_i>z_j]$;
replacing the step $\mathbf{1}[d>0]$ with its smooth lower bound $\log\sigma(d)$ makes
$\sum_{i,j}\log\sigma(z_i-z_j)$ a differentiable surrogate for (a monotone function of) AUC. Summed
over within-user pairs and averaged over users, maximizing it directly pushes up per-user AUC, i.e.
**GAUC**. Empirically this is the winning lever: `0.6015 → 0.6036`.

**Vectorized sampling** (`train_np._fit_pair` + `_build_pair_index`): we precompute, per user, the
list of positive rows and a flat pool of negative rows with per-user `(start, len)` offsets. Each
step draws `neg_ratio` negatives per positive with a single vectorized `rng` call (`start + ⌊u·len⌋`),
forwards positives and negatives together, and scatters $g$. No Python-level pair loop — the earlier
per-positive loop was ~50× slower and blew the wall-clock budget.

### 3.2 Softmax cross-entropy — an nDCG surrogate → nDCG@5

For a user group with logits $z_i$ ($i\in\mathcal{I}_u$), define the target distribution as uniform
over the positives, $p_i=y_i/\sum_j y_j$, and the model distribution $s_i=\text{softmax}(z/\tau)_i$.
The listwise cross-entropy is

$$L=-\sum_i p_i\log s_i,\qquad \frac{\partial L}{\partial z_i}=\frac{1}{\tau}\big(s_i-p_i\big).$$

`_softmax_ce` computes precisely $g_i=(s_i-p_i)/\tau$ per group (skipping degenerate groups). This is
the Plackett–Luce / ListNet cross-entropy; Bruch et al. (2019) show softmax-CE with binary relevance
is a **bound on nDCG**, so it targets the **nDCG@5** half of the metric.

**Why it *loses* here (a real finding):** the model is a bare-ID FM. The listwise loss pushes hard on
each user's within-group ordering, which the high-cardinality `user_id × video_id` embeddings can
memorize — so it overfits fast (best epoch ~2, then valid declines to 0.5997, below baseline). The
pairwise BPR is better-regularized on this data. Grouping (`_fit_group`) packs whole users per batch so
the per-group softmax is complete.

### 3.3 LambdaRank weighting (cfg `lambdarank`, planned)

LambdaRank multiplies each pair's gradient by $|\Delta\text{nDCG}_{ij}|$ — the change in nDCG from
swapping items $i,j$ at their current ranks — focusing gradient on swaps that move the top-5. It needs
the full within-group ranking each step (a `group`-mode computation). The `Cfg.lambdarank` flag is
reserved; the current losses implement BPR and softmax-CE, and LightGBM (§4) supplies a
production-grade LambdaRank via its own objective.

---

## 4. LightGBM LambdaRank (Lever D)

`pipeline/lib/gbm.py`. A gradient-boosted tree ensemble trained with `objective='lambdarank'`,
`query` groups = users. LightGBM's LambdaRank computes, for each pair in a query, a gradient scaled by
$|\Delta\text{nDCG}|$ and fits trees to those pseudo-gradients — a non-parametric ranker whose
inductive bias (axis-aligned splits on tabular features) is **complementary** to the embedding models.

**Feature design (leakage-safe, item-centric).** Because ranking is within-user, features that are
constant across a user's impressions (pure user-side) carry **no** ordering signal — the same insight
the organizers proved for the FM. So the features are the ones that *vary within a user*:

- item `long_view` rate $\hat r_v=\dfrac{\text{pos}_v+\alpha\,\bar r}{\text{imp}_v+\alpha}$ (smoothed
  with a global-mean prior $\bar r$, $\alpha=20$), computed from **train only**;
- author `long_view` rate (same smoothing); $\log$ item and author impression counts; $\log$ duration; `tab`;
- 16 global engagement columns from `video_features_statistic_pure.csv`, z-scored on train.

No current-row `play_time` is used (it *is* the label's source — that would be leakage). Standalone it
scores ~0.6021; its value is diversity for the ensemble (§6).

---

## 5. The DIN model (Lever B)

`pipeline/lib/din.py`. `DIN` is a **DeepFM + Deep Interest Network** hybrid. Let $e_t$ be the target
video embedding, $\{e_{h}\}_{h=1}^{L}$ the history embeddings (padded, with mask $m_h$), and
$\{e^{\text{base}}_1,\dots,e^{\text{base}}_5\}$ the base 5-field embeddings.

### 5.1 The interest unit (DIN)

A local attention MLP scores each history item against the target and pools **without softmax** (DIN's
signature — the raw activation magnitude carries interest intensity):

$$a_h=\text{MLP}_{\text{att}}\big([\,e_h\,\|\,e_t\,\|\,e_h\odot e_t\,\|\,e_h-e_t\,]\big),\qquad
u=\sum_{h=1}^{L} m_h\,a_h\,e_h.$$

The four-way concatenation (raw, target, elementwise product, difference) lets the unit express both
similarity ($e_h\odot e_t$) and contrast ($e_h-e_t$).

### 5.2 The DeepFM part (why it is essential)

Pure attention has **no user identity**, so it cannot memorize user-specific preferences and
underperforms FM (0.5895). We add an FM over the base fields — linear plus 2nd-order cross:

$$z_{\text{fm}}=b+\sum_{i} w^{\text{base}}_i+\tfrac12\Big[\big(\textstyle\sum_i e^{\text{base}}_i\big)^2-\sum_i (e^{\text{base}}_i)^2\Big]\!\cdot\!\mathbf{1},$$

(the cross term summed over the $k$ dimensions). The final logit combines the FM part with a deep MLP
over the interest, target, and base-sum:

$$z=z_{\text{fm}}+\text{MLP}_{\text{deep}}\big([\,u\,\|\,e_t\,\|\,\textstyle\sum_i e^{\text{base}}_i\,]\big).$$

Adding the DeepFM part lifts DIN from 0.5895 to **0.6031** — above the FM baseline. Training (`fit_din`)
uses the same BPR objective as §3.1 (torch autograd), early-stops on valid primary, and runs on GPU
when `torch.cuda.is_available()`.

### 5.3 Temporal safety of the history

The history for a row is built in `seq_build.build` by processing rows in global time order and
appending the current item to the user's `deque(maxlen=L)` **after** snapshotting — so no row sees its
own outcome or any future interaction. This is a structural guarantee, not a post-hoc filter (see the
README §6).

---

## 6. Ensembling (Lever F)

`orchestrator.assemble` + `finalize`. The base learners (FM/BPR, LightGBM, DIN) live on
**incomparable score scales** (FM logits vs. tree outputs vs. DIN logits), and the metric only cares
about **within-user order**. So we blend in **rank space**.

### 6.1 Per-user percentile rank

For each user group, transform scores to $r_i=\text{rank}_u(s_i)/(|\mathcal{I}_u|-1)\in[0,1]$
(`_per_user_rank`). This is monotone (order-preserving, so it cannot hurt a single model's own metric)
and scale-free (so heterogeneous models combine sensibly).

### 6.2 Weighted blend + grid search

For members $m=1..M$ with rank vectors $r^{(m)}$ and weights $w_m\ge0$, $\sum_m w_m=1$:

$$\hat r_i=\sum_{m=1}^{M} w_m\, r^{(m)}_i.$$

`_weight_grids` enumerates the weight simplex (step 0.1 for $M=3$, a 1-D sweep for $M=2$);
`assemble` evaluates each on validation and keeps the best, but only accepts the blend if it beats the
best single learner. In the reference run, $M=3$ with $w=(0.3,0.4,0.3)$ over (FM, DIN, LightGBM)
yields **0.6050 > 0.6036** — DIN's largest weight reflects that the sequence model adds the most
independent signal.

### 6.3 Applying it to the hidden test

`finalize` re-runs the chosen members on the **test** split (the members were trained on train and
selected on valid), rank-transforms and blends with the same weights, writes `best/submission_test.csv`,
and validates it with `submit.py --check`. The submission is always the validation-best object — the
best-checkpoint invariant.

---

## 7. Debiasing with the random-exposure log (Lever E — planned)

KuaiRand ships a **randomly-exposed** log (items shown independent of the policy), which is an unbiased
sample. The planned Lever E uses it two ways: (1) as an **unbiased validation guard** — if a model's
biased-valid score rises while its random-log score falls, it is overfitting exposure bias; and (2) for
**IPS-weighted** training, reweighting each logged example by $1/\hat p(\text{exposure})$ (with
self-normalization to tame variance). Not yet built — `primary_unbiased` is currently `null`. See the
README §14.

---

*Back to the [README](../README.md) · design rationale in [DESIGN.md](DESIGN.md).*
