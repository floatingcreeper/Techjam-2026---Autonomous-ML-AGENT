# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A recommendation-ranking kit built on the KuaiRand-Pure dataset (short-video interaction logs). The
task is **within-user ranking**: given the impressions logged for a user in the eval window, rank
them so that videos the user actually watched long (`long_view=1`) come first. Comments in the code
are written for ML beginners (bilingual EN/中文) — this is a teaching/competition kit (`baseline.py`
line 1 calls the FM model "起步模型，学生从这里往上改" — the model students are meant to improve on).

Only dependency beyond the stdlib is `numpy` — no ML framework, no autograd. `baseline.py`'s FM
implements forward pass, gradients, and an Adam optimizer by hand.

## Commands

```bash
# Run a baseline (from repo root; data_dir defaults to ./KuaiRand-Pure/data)
python baseline.py --model random   # sanity check only — verifies evaluate.py/data.py aren't broken
python baseline.py --model pop      # official baseline: non-trained popularity lookup
python baseline.py --model fm       # trainable Factorization Machine (the model to improve on)
python baseline.py --model fm --k 32 --lr 0.001 --epochs 40 --seed 0

# Compare feature-set ablations (adds CWM's 13 feature fields vs. the kit's default 5)
python ablation_features.py [data_dir]

# Generate / validate / score a submission (see submit.py header for full spec)
python submit.py --make  submission.csv --split test   # writes a sample submission using the FM baseline
python submit.py --check submission.csv --split test   # validate format + row alignment only
python submit.py --score submission.csv --split valid  # validate and score (only valid has local labels)
```

There is no test suite, linter, or build step in this repo. `run_random` in `baseline.py` is the de
facto smoke test — if `--model random` doesn't land near `primary≈0.475` (see `baseline_scores.json`),
the bug is in `evaluate.py` or `data.py`, not in a model change.

## Architecture

Data flow: `data.load()` → `data.encode()` → a model (`baseline.FM` or `run_pop`) → `evaluate.evaluate()`.

- **`data.py`** — loads the two interaction log CSVs plus `video_features_basic_pure.csv` (for a
  video→author join), splits rows into train/valid/test **by date range** (`SPLITS`, not random
  shuffling — this is a "train on past, eval on future" split), and encodes 5 categorical fields
  (`FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']`) into integer ids.
  - Rows are plain tuples accessed **by position** everywhere downstream:
    `x[0]=date, x[1]=user_id, x[2]=video_id, x[3]=author_id, x[4]=tab, x[5]=duration_ms, x[6]=label`.
    Adding a column means updating every positional `x[N]` read after it, in this file and in
    `ablation_features.py`.
  - `dur_bucket` is `duration_ms` quantile-bucketed into 10 bins (edges computed from **train only**,
    same as the per-field vocabularies) so the model — which only handles categorical inputs — can
    treat duration as a categorical field.
  - Vocabularies are built from train only; unseen values in valid/test map to a per-field UNK slot.
    This is deliberate (simulates deployment / avoids leakage), not a bug.
  - `encode()` offsets each field's local ids into a disjoint slice of one shared id space
    (`offsets`, `field_dims`) so a single embedding table (`FM.V`) can serve all fields without
    collisions. `FIELDS` order must match the order values are packed in `raw()`.

- **`baseline.py`** — three models behind `--model {random,pop,fm}`:
  - `random`: sanity check only, no learning.
  - `pop` (`run_pop`): official non-trained baseline — Bayesian-smoothed per-video long-view rate.
  - `fm` (`FM` class + `run_fm`): a Factorization Machine — bias + per-field linear term + pairwise
    feature interactions via embeddings, trained with hand-rolled backprop + Adam, batches of 8192,
    early-stopped on validation `primary` score (patience-based, not on training loss). This is the
    file students extend with new features/architectures — `FIELDS` in `data.py` is "student's most
    likely edit point" per the file's own comments.

- **`evaluate.py`** — the fixed, do-not-modify scoring contract (see its own docstring). Computes
  per-user GAUC (weighted by positive count, only over users with `0 < positives < impressions`) and
  nDCG@5 (zero for all-negative users, still counted in the mean). `primary = mean(GAUC, nDCG@5)` is
  the ranking metric for everything in this repo.

- **`ablation_features.py`** — standalone experiment script (duplicates a chunk of `data.py`/
  `baseline.py`'s logic inline rather than importing it) that trains the FM model 3x per feature-set
  variant (`base`=5 fields, `item`=+4 item-side CWM fields, `cwm13`=all 13 CWM fields) to test whether
  additional user/item side features help. Not part of the main pipeline; run standalone to answer
  "are richer features worth it before I build something fancier."

- **`submit.py`** — writes/validates/scores submission CSVs (`row_id,user_id,video_id,score`).
  `row_id` exists because `(user_id, video_id)` isn't unique in the eval set (dup pairs occur, up to
  12x in test) so it can't be the join key; row order must exactly match `data.load()`'s deterministic
  read order (`log_standard_4_08_to_4_21_pure.csv` then `log_standard_4_22_to_5_08_pure.csv`, filtered
  by split date range, original file order preserved). `--check`/`--score` re-derive the expected rows
  from `data.load()` and diff every field, so any reordering or filtering change in `data.py` will
  break existing submissions' alignment.

- **`baseline_scores.json`** — recorded reference scores (mean over multiple seeds) for `random`,
  `item_popularity`, `fm_official`, and an `oracle_ceiling` (scores from true labels — the practical
  upper bound, since `nDCG` can't reach 1.0 when 27.1% of test users are all-negative). Use this to
  judge whether a change is a real improvement or noise — `fm_official`'s `std_over_5_seeds` is ~0.0008
  on `primary`, so treat any delta below that as noise.

- **`KuaiRand-Pure/`** — the raw dataset (CSVs + upstream `load_data_pure.py`, not used by this repo's
  own `data.py`) and its own LICENSE. Treat as read-only input data, not code to edit.

## Working in this repo

- `evaluate.py`'s metric definitions are fixed by the competition spec ("口径全部写死在这里，不要改" —
  don't change the scoring contract). Extend the model/features, not the scoring.
- Any change to `data.py`'s split logic, field list, or vocab-building must stay train-only for
  vocab/bucket-edge fitting — sourcing those from valid/test is leakage that invalidates the score.
- When adding a field to `FIELDS`, add a matching value in `raw()`'s return list (same order) and
  check for other positional `x[N]` reads of the row tuple in `data.py` and `ablation_features.py`.
- After any change to `data.py` or `evaluate.py`, rerun `python baseline.py --model random` first —
  it isolates whether a change broke the pipeline vs. changed model quality.
