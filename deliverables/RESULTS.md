# Results

## Scores

Metric is `primary = mean(GAUC, nDCG@5)`, within-user ranking. Higher is better.

| Model | valid | test | notes |
|---|---|---|---|
| `random` | 0.4834 | 0.4753 | sanity check — confirms the scoring code works |
| `item_popularity` | 0.5807 | 0.5715 | official non-trained baseline |
| **`fm_official`** | **0.6016** | **0.5946** | **the official baseline to beat** |
| `fm_v1` (baseline retrained here) | 0.6015 | 0.5953 | same-machine control |
| `fm_bpr` single model | 0.6031 | 0.5970 | ours |
| **`fm_bpr` × 5-seed ensemble** | **0.6039** | **0.5974** | **submitted** |
| *oracle ceiling* | *0.8484* | *0.8645* | *scores from true labels — not reachable* |

**Delta over the official baseline: +0.0028 test primary.**
**Delta over the baseline retrained on this machine: +0.0021 test primary.**

Both are quoted because the second is the fairer comparison — it rules out the gain being an
artifact of comparing against a number recorded on different hardware.

## Is that delta real, or noise?

Yes, real. Three independent checks:

**1. Multi-seed.** Five seeds each, on valid:

| | mean | std |
|---|---|---|
| `fm_v1` | 0.60157 | 0.00032 |
| `fm_bpr` | 0.60274 | 0.00049 |

Delta +0.00117, roughly 2.4× the combined seed noise. `fm_bpr`'s *worst* seed (0.60187) sits within
a whisker of `fm_v1`'s *best* (0.60204) — it wins on effectively every pairing.

**2. It holds on test**, and is in fact slightly larger there (+0.0017 same-machine) than on valid
(+0.0012). The gain is not a valid-set artifact.

**3. Format-validated.** `submit.py --score` on the submission: 170,588 rows, row-alignment against
`data.load()`'s deterministic order verified field by field.

### Context for the size

0.0028 sounds small. Calibrate it against:

- `fm_official`'s own seed-to-seed std is **0.0008**
- the *entire* hyperparameter space of the baseline model spans **0.0040** end to end (measured
  across 75 agent iterations — every working config scored between 0.5976 and 0.6016)
- 27.1% of test users are all-negative, so their nDCG is 0 for *any* model — the oracle ceiling is
  0.8645, not 1.0

This is a benchmark where movement is small. Say this out loud when presenting the number.

## Reproducing

```bash
# The submission (~5 min, CPU only, no GPU)
python ensemble_submission.py --seeds 5 --split test --out submission_ens_test.csv

# The single model
python make_submission.py --model fm_bpr --split test --out submission_test.csv

# The baseline control
python make_submission.py --model fm_v1  --split test --out submission_test_fm_v1.csv

# Verify any of them
PYTHONIOENCODING=utf-8 python submit.py --score <file> --split test
```

## What produced the gain

**Changing the objective, not the features.**

`baseline.FM` trains pointwise binary cross-entropy — it optimises each row's absolute
probability. But `primary` is a **within-user** ranking; it never compares one user's rows to
another's, so all the cross-user calibration that pointwise training buys is discarded by the
metric.

`models/fm_bpr.py` samples (positive, negative) pairs **from the same user** and maximises
`σ(z⁺ − z⁻)`. It adds **zero features**, and reuses `baseline.FM`'s forward pass, embedding table,
L2 term and Adam optimiser untouched — only the loss gradient differs. Sampling positives
uniformly also makes each user's training weight proportional to their positive count, which is
exactly how GAUC averages users.

On top of that, **seed ensembling** (`ensemble_submission.py`) averages per-user-standardised
scores across 5 seeds: +0.0008 valid, +0.0004 test.

## What did NOT work — measured, not assumed

These are the most useful output of the project. All are recorded in `models/fm_bpr.py`'s
docstring and fed back into the agent's prompts so nothing re-derives them.

| Attempt | Result |
|---|---|
| Target-encoded per-video/author `long_view` rates as features | **0.5906** vs 0.6015 — worse, degrading from epoch 1 |
| Blending FM with the popularity prior | best α=0.05 → **+0.00003**, then monotonic decline |
| `video_features_statistic_pure.csv` item aggregates | dropped — same signal as above |
| Hard negative mining (`neg_candidates` 2→16) | **monotonically worse**: 0.60041 → 0.56862 |
| Hyperparameter search (k, lr, l2, epochs, patience, batch) | flat — 0.0040 total spread vs 0.0008 noise |

**Why the feature work failed.** The FM's `video_id` linear weight `W[video_id]` *already is* a
learned per-video propensity, fit on the same train rows a popularity lookup counts. Every one of
those three attempts was a different way of feeding the model information it had already
extracted — redundant capacity, and extra parameters to overfit. One cheap offline sweep settled
all three: train the FM once, then remix its own valid scores against the popularity score at a
range of weights, no retraining required.

**Why hard negatives failed.** `long_view` is noisy implicit feedback, so a user's
highest-scoring negative is very often a mislabelled positive — someone who *did* watch but wasn't
logged as a long view. Mining aims every gradient step at exactly the rows most likely to be wrong.

## Honest limitations

- **The agent did not produce the winning model.** Across ~120 iterations its best node never
  exceeded the incumbent it was seeded with. Every gain above was hand-built. What the agent
  contributed was the search infrastructure, the audit trail, and the negative results that
  redirected the work from features to the loss function — which is genuinely how we found it, but
  is not the same as autonomous discovery. Present it that way.
- **The agent's generated modules drift back to the pointwise FM**, even on `improve` operations
  handed `fm_bpr.py` as parent source, because the codegen prompt also ships two complete pointwise
  reference modules and concrete code outweighs a prose rule for a small local model. Diagnosed,
  not fixed.
- **`CONVERGENCE_EPSILON` is 0.002** — wider than any gain that exists on this dataset, so the loop
  logs real improvements as `KEEP_NODE` rather than `COMMIT`. A judge reading the log will see zero
  accepted candidates. Recalibrating to ~0.001 (≈3× seed noise) is a team decision.
- **Possible metric mismatch** — see item 2 in [`00-START-HERE.md`](00-START-HERE.md). Unresolved,
  and it would invalidate this entire document if the kit is wrong.
