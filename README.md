# KuaiRand-Pure Starter Kit

## Dependencies

Python 3.9+ and numpy. **Nothing else.** No torch, pandas, or sklearn required.

## Data

Download from https://kuairand.com (direct Zenodo link, no registration required):

```bash
# Run inside the Starter Kit directory; extracting gives you ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Running

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`; specify it explicitly if the data is elsewhere.

`--model` can be `fm` (official baseline) / `pop` (trivial baseline) / `random` (lower bound, for sanity-checking the evaluation code).
FM takes about 40 seconds end to end (CPU, single core).

## Task definition (the conventions are hard-coded, do not change them)

| | |
|---|---|
| Task | **within-user ranking** -- each user only ranks their own impressions in the evaluation set, no full-corpus retrieval |
| Relevance label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **primary = the mean of the two** |
| Data split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Zero-positive users | nDCG counts as 0.0 and is included in the mean; GAUC only counts users with `0 < #positives < #impressions`, weighted by #positives |
| nDCG gain | `2^rel − 1` (equivalent to identity under binary labels) |

See `evaluate.py` for the implementation; all conventions are in the file's header comment.

## Baseline ladder

Scores on the test set. **The line to beat is FM.**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (lower bound, for sanity checks) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ The real range of the metric: the ceiling for nDCG@5 is 0.729, not 1.0

Among the 23,875 users in the test set:

| | Share | Effect on the metric |
|---|---|---|
| All-negative users (none of the user's impressions are long_view) | **27.1%** | nDCG is always **0**, no model can save it; not counted in GAUC |
| All-positive users | **9.2%** | nDCG is always **1**; not counted in GAUC |
| Discriminative users | **63.7%** | the actual sample for GAUC |

So even using the true labels as prediction scores (oracle, perfect ranking) can only achieve:

| | random | FM baseline | **oracle ceiling** | fraction FM has captured |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Please measure evaluation progress against the oracle as the denominator.** Seeing 0.5946 and thinking "still far from a perfect 1.0" is a misjudgment --
the baseline has already captured 30% of the usable range, and the remaining headroom is 0.27, not 0.41.

FM's std across 5 random seeds is **0.0008** in every case. Based on this, the convergence criterion is **ε = 0.002 (≈2.5σ), N = 3**:
if the validation primary score improves by no more than 0.002 for 3 consecutive iterations, it is judged to have converged.

> Sanity check: if your evaluation code doesn't get primary ≈ 0.475 (±0.001) when running `--model random`, the harness is broken -- fix it first.

## Submission format

CSV, with header, one row per row of the evaluation set:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| Field | Description |
|---|---|
| `row_id` | increases contiguously from 0, corresponding to the row order of `data.load()[split]` (deterministic: read `log_standard_4_08_to_4_21_pure.csv` first, then `log_standard_4_22_to_5_08_pure.csv`, keeping the original file order after filtering by date) |
| `user_id` / `video_id` | redundant fields, only used to verify alignment |
| `score` | the score your model assigns to this row, any real number, only relative magnitude matters; no NaN / Inf allowed |

> **Why `row_id` is required:** `(user_id, video_id)` is **not unique** in the evaluation set --
> the test set has 3.06% duplicate pairs, repeated up to 12 times. So it cannot serve as a primary key.

Generate and validate:

```bash
python3 submit.py --make  --split test  submission.csv    # generate a sample submission using the official FM baseline
python3 submit.py --check --split test  submission.csv    # validate format and alignment
python3 submit.py --score --split valid submission.csv    # validate and score (valid is available locally)
```

`--check` will reject: wrong header, mismatched row count, `row_id` gaps, `user_id`/`video_id` misaligned with the evaluation set,
non-numeric `score`, or NaN/Inf. **Please run `--check` yourself before submitting.**

## Where to start modifying

The ordering below is **empirically tested**, not guessed. Dead ends the organizers have already tried are marked directly, so you don't repeat them.

### Already tested: these two yield no gains, don't waste iterations

| Tried | Result |
|---|---|
| **Adding static features** -- wiring in all 13 of CWM's feature fields (+`music_id`/`video_type`/`upload_type` + 6 coarse user-side buckets) | primary **0.5940** vs the 5-field **0.5950**, no difference within noise, if anything slightly lower |
| **Adding model capacity** -- embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887, barely moves |

Reason: the `user_id × video_id` cross already captures most of the learnable signal. Coarse buckets like `follow_user_num_range`
are redundant in the presence of `user_id`; and 1.14M rows can't support larger capacity. **The bottleneck is not features or capacity.**

⚠️ Also note: **the first-order terms of pure user-side features contribute exactly 0 to the score.** Because ranking is done within each user, any term that is constant within a user
does not change the intra-group order (verified: `item_pop × user bias` and pure `item_pop` produce scores identical to the last digit). User-side features can only take effect through
**cross terms with the item side**.

### Unexplored: the headroom should be here

Ordered by the likelihood we judge (**these have not been tested by the organizers, they are left for you**):

1. **Change the loss function.** It is currently pointwise logloss, but the metrics (GAUC / nDCG) are **ranking metrics**.
   Switch to pairwise (BPR) or listwise (softmax over the user's impressions) -- aligning the objective with the evaluation convention,
   this is the one we think is most likely to work.
2. **User history sequences.** The current features **make no use of behavior sequences at all**. In KuaiRand each user has hundreds to thousands of interactions in train;
   interest modeling of the DIN / SIM family is a completely blank direction.
3. **Multi-objective.** The logs also contain `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `play_time_ms`,
   which can be used for multi-task auxiliary support of the `long_view` primary task.
4. **Modeling watch time.** [CWM](https://github.com/hyz20/CWM)'s contribution is exactly this: it models watch time as **censored regression**
   (when the video plays to the end, the true watch time is truncated, so it uses a one-sided loss instead of squared error). This is a research-depth direction.
5. **Change the model.** DeepFM / DCN / xDeepFM. Given that capacity is empirically not the bottleneck, **rank this after 1-4**.
6. **Temporal features and distribution drift.** `hourmin`, `date`, and the drift between train and test.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` is a random-exposure log (1.18M rows),
   which can serve as an additional unbiased validation set to check whether the model only overfits to biased traffic.

## Use your own model (including CWM)

`evaluate.py` is completely decoupled from the model; it only needs three equal-length arrays:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores can come from any model
```

- `user_ids`: the user_id of each row in the evaluation set
- `labels`: that row's `long_view` (0/1)
- `scores`: the score your model assigns to that row (any real number, only relative magnitude matters)

So you can skip `baseline.py` entirely and use PyTorch, LightGBM, or [CWM](https://github.com/hyz20/CWM)'s xDeepFM,
as long as you hand the `scores` to `evaluate()` at the end. **The scoring convention is solely determined by `evaluate.py`.**

> Note when using CWM: it depends on `torch==1.6.0` (a 2020 version, probably won't install on new GPUs),
> and its loss optimizes counterfactual watch time, while its evaluation label is its own reconstructed `long_view2`.
> It is the research code for a watch-time debiasing paper and can serve as an **advanced reference**; it is not recommended as a starting point.

## Files

| | |
|---|---|
| `evaluate.py` | metric implementation + all conventions. **Do not change.** |
| `data.py` | data loading, official split, feature encoding. Modify here to add features. |
| `baseline.py` | the three baselines. FM is the one to beat. |
| `baseline_scores.json` | officially released scores + seed variance + convergence parameters. |
| `submit.py` | generate / validate submission files. |
| `ablation_features.py` | feature-ablation experiment, reproduces the "adding features yields no gain" set of numbers. |
