"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""

# ============================================================================
# WHAT THIS FILE DOES (plain English)
# ----------------------------------------------------------------------------
# This is the data-loading and feature-encoding step of the pipeline.
# It reads the raw KuaiRand-Pure CSV files, splits them into train / valid /
# test by date (the official split, fixed by the organizers), and turns the
# handful of categorical columns (user_id, video_id, ...) into integer IDs
# that a model can consume.
#
# HOW IT CONNECTS TO THE OTHER FILES:
#   - baseline.py    imports load() and encode() from here to get its
#                     training/validation/test matrices before training FM.
#   - submit.py       imports load() and encode() the same way, to rebuild
#                     the exact same rows/features when generating or
#                     checking a submission file.
#   - ablation_features.py reimplements a variant of load()/encode() itself
#                     (it does not import this file's functions) to test
#                     extra feature columns not used here.
#   - evaluate.py     does NOT depend on this file at all — it only needs
#                     plain (user_ids, labels, scores) arrays, so any model
#                     built on top of this file's output can be scored by it.
#
# This is also the file the README points to as "the place to add features":
# the FIELDS list below defines exactly what the model sees.
# ============================================================================

import csv, os, collections
import numpy as np

LABEL = 'long_view'                              # the target column we are predicting (0/1)

SPLITS = {'train': (20220408, 20220421),         # fixed official date ranges — do not change,
          'valid': (20220422, 20220428),         # these come straight from the problem statement
          'test':  (20220429, 20220508)}

# 5 个特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
# The 5 categorical feature columns the model uses. This is the main list to
# extend if you want to add new features to the pipeline.
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']


def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。
    Reads the interaction logs plus the video-side feature file, and returns
    a dict {'train': [...], 'valid': [...], 'test': [...]} of raw rows,
    already filtered into the correct date range for each split."""

    # Step 1: build a lookup video_id -> author_id from the video features file.
    # This is needed because the raw log rows don't carry author_id directly.
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    # Step 2: read both interaction log files (older + newer date range) and
    # turn every row into one flat tuple we can work with:
    # (date, user_id, video_id, author_id, tab, duration_ms, label)
    #     index:  x[0]    x[1]     x[2]      x[3]     x[4]     x[5]      x[6]
    # Every other function in this file (and in baseline.py's inline row
    # access) reaches into this same tuple by position — e.g. x[1] always
    # means user_id, x[6] always means the label. Keep this order in mind
    # when reading raw()/encode() below, since they don't repeat the names.
    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0))

    # Step 3: split all rows into train/valid/test purely by date, using the
    # official ranges above. Order within each split is preserved (this
    # matters later — submit.py relies on this exact row order for row_id).
    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out


def _bucket_edges(durations, n=10):
    # Helper: computes n-1 quantile cut points from the training durations,
    # used to turn a continuous duration into a discrete "duration bucket"
    # feature (dur_bucket). Only training data is used to compute the edges,
    # so validation/test never leak into this calculation.
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])


def encode(splits):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    返回 (X, y, users) per split，X 为 int32 (N, len(FIELDS))，以及 field_dims。
    Turns each categorical feature into a small integer ID. Any value never
    seen during training falls into that field's own UNK ("unknown") slot,
    so validation/test rows with unseen IDs don't crash. Returns, per split,
    (X, y, users): X is the encoded feature matrix, y is the label array,
    users is the list of user_ids (needed later by evaluate.py to group rows
    per user)."""

    tr = splits['train']
    # Duration buckets are learned from train only (see _bucket_edges above).
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        # Turns one row into its 5 raw (not-yet-integer) feature values, in
        # the same order as FIELDS: user_id, video_id, author_id, tab,
        # dur_bucket (duration turned into a bucket index string).
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    # Build one vocabulary (value -> integer id) per field, using only the
    # training split — this is what makes validation/test "unseen" values
    # possible, and is why the UNK slot exists.
    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽
                                                    # one reserved "unknown" slot per field
    field_dims = [len(v) + 1 for v in vocabs]      # size of each field's vocabulary (+1 for UNK)
    # offsets let us pack all 5 fields into one shared embedding table: field
    # i's ids are shifted by offsets[i] so they don't collide with field i-1.
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                # look up this value's id in the training vocabulary; unseen
                # values fall back to that field's UNK id (see unk above)
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    # field_dims summed = total size of the shared embedding table any model
    # (like the FM in baseline.py) needs to allocate.
    return enc, int(sum(field_dims))
