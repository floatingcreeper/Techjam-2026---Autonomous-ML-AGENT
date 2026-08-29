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
# THIS ITERATION'S CHANGE:
#   Added a 9th field, 'user_author_click_affinity': a leave-one-out,
#   additive-smoothed estimate of how often THIS user clicks (is_click,
#   NOT the long_view label) on videos by THIS author, using ONLY the
#   training split to build the (user,author) click/impression counts.
#   This mirrors the existing 'user_author_affinity' feature's mechanics
#   exactly (same leave-one-out discipline, same additive-smoothing prior,
#   same quantile-bucket discretization), but targets a DIFFERENT, much
#   denser auxiliary signal: is_click fires far more often per user than
#   long_view, so for the many users whose OWN training history has few or
#   zero long_view positives with a given author, the click-based estimate
#   is still statistically meaningful, whereas the long_view-based
#   affinity for those (user,author) pairs collapses to (or near) the
#   global mean. This is organizer direction #3 (multi-task / auxiliary
#   feedback signals) applied as a FEATURE-ENGINEERING technique (an
#   auxiliary-label-derived input feature) rather than as an auxiliary
#   training loss, so it needs no model-architecture change at all -- it
#   slots into the FM's existing shared embedding table exactly like any
#   other categorical field.
#   No leakage risk: is_click is a directly-observed attribute of each row
#   itself (not derived from the long_view label being predicted), and the
#   leave-one-out subtraction on train rows prevents a row from reading off
#   information that includes its own click outcome.
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
#   - evaluate.py     does NOT depend on this file at all -- it only needs
#                     plain (user_ids, labels, scores) arrays, so any model
#                     built on top of this file's output can be scored by it.
#
# This is also the file the README points to as "the place to add features":
# the FIELDS list below defines exactly what the model sees.
# ============================================================================

import csv, os, collections
import numpy as np

LABEL = 'long_view'                              # the target column we are predicting (0/1)

SPLITS = {'train': (20220408, 20220421),         # fixed official date ranges -- do not change,
          'valid': (20220422, 20220428),         # these come straight from the problem statement
          'test':  (20220429, 20220508)}

# 9 个特征域。想加特征就往这里加 -- 这是学生最该动的地方之一。
# The feature columns the model uses. This is the main list to extend if you
# want to add new features to the pipeline. 'user_author_affinity' and
# 'hour' were added in previous iterations; 'user_author_click_affinity' is
# new this iteration -- see module docstring above.
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'tab_x_dur',
          'user_author_affinity', 'hour', 'user_author_click_affinity']

# Additive-smoothing strength for the leave-one-out user x author affinity
# feature -- same idea as `prior` in baseline.py's run_pop. Larger = more
# shrinkage toward the global mean rate for (user, author) pairs with few
# observations.
_AFFINITY_PRIOR = 15.0
_AFFINITY_BUCKETS = 10

# Same-style smoothing prior for the NEW click-based affinity feature. Kept
# at the same value as _AFFINITY_PRIOR (not separately tuned) so this
# iteration's change is isolated to "add the feature", not "add the feature
# AND tune its smoothing strength".
_CLICK_AFFINITY_PRIOR = 15.0
_CLICK_AFFINITY_BUCKETS = 10


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
    # (date, user_id, video_id, author_id, tab, duration_ms, label, hour, is_click)
    # NOTE: 'hour' and 'is_click' are appended at the END of the tuple
    # (indices 7 and 8) rather than inserted earlier, so every existing
    # index reference elsewhere in this file (e.g. x[6] for the long_view
    # label) stays correct and unchanged.
    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                hm = r.get('hourmin', '0')
                try:
                    hour = str(int(hm) // 100)          # HHMM -> hour-of-day 0-23
                except ValueError:
                    hour = 'UNK'
                is_click = 1 if r.get('is_click', '0') != '0' else 0
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0,
                             hour, is_click))

    # Step 3: split all rows into train/valid/test purely by date, using the
    # official ranges above. Order within each split is preserved (this
    # matters later -- submit.py relies on this exact row order for row_id).
    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out


def _bucket_edges(values, n=10):
    # Helper: computes n-1 quantile cut points from a list of training-only
    # values, used to turn a continuous quantity into a discrete bucket
    # index feature. Only training data is used to compute the edges, so
    # validation/test never leak into this calculation.
    if len(values) == 0:
        return np.array([], dtype=np.float64)
    return np.quantile(np.asarray(values), np.linspace(0, 1, n + 1)[1:-1])


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

    # ---- user x author historical long_view affinity feature (earlier iteration) ----
    # Aggregate per (user_id, author_id) positive/impression counts using
    # ONLY the training split.
    ua_pos = collections.Counter()
    ua_imp = collections.Counter()
    tot_pos = 0
    for x in tr:
        key = (x[1], x[3])          # (user_id, author_id)
        ua_pos[key] += x[6]
        ua_imp[key] += 1
        tot_pos += x[6]
    gmean = (tot_pos / len(tr)) if tr else 0.0

    def affinity_rate(x, is_train):
        # Smoothed rate of this user long_view'ing videos by this author.
        # is_train=True subtracts the row's own label out of its own
        # (user,author) bucket first (leave-one-out), so train rows can't
        # just read off their own label through this feature.
        key = (x[1], x[3])
        p = ua_pos.get(key, 0)
        n = ua_imp.get(key, 0)
        if is_train:
            p -= x[6]
            n -= 1
        return (p + _AFFINITY_PRIOR * gmean) / (n + _AFFINITY_PRIOR)

    # Quantile bucket edges for the affinity rate, computed from the
    # TRAIN-split leave-one-out values only (same no-leakage discipline as
    # dur_bucket's edges above).
    aff_edges = _bucket_edges([affinity_rate(x, True) for x in tr], n=_AFFINITY_BUCKETS)

    # ---- NEW: user x author historical CLICK affinity feature (this iteration) ----
    # Same mechanics as the long_view affinity above, but the smoothed
    # target is is_click (x[8]) instead of long_view (x[6]). is_click fires
    # much more often per user than long_view, so this gives a denser,
    # lower-variance per-(user,author) estimate -- useful especially for
    # users whose own training history has few/no long_view positives with
    # a given author (where the long_view-based affinity above collapses
    # toward the global mean).
    ua_click_pos = collections.Counter()
    ua_click_imp = collections.Counter()
    tot_click = 0
    for x in tr:
        key = (x[1], x[3])
        ua_click_pos[key] += x[8]
        ua_click_imp[key] += 1
        tot_click += x[8]
    gmean_click = (tot_click / len(tr)) if tr else 0.0

    def click_affinity_rate(x, is_train):
        # Smoothed rate of this user clicking videos by this author.
        # Same leave-one-out discipline as affinity_rate above, but on the
        # is_click column instead of the long_view label.
        key = (x[1], x[3])
        p = ua_click_pos.get(key, 0)
        n = ua_click_imp.get(key, 0)
        if is_train:
            p -= x[8]
            n -= 1
        return (p + _CLICK_AFFINITY_PRIOR * gmean_click) / (n + _CLICK_AFFINITY_PRIOR)

    click_aff_edges = _bucket_edges([click_affinity_rate(x, True) for x in tr], n=_CLICK_AFFINITY_BUCKETS)

    def raw(x, is_train):
        # Turns one row into its raw (not-yet-integer) feature values, in
        # the same order as FIELDS: user_id, video_id, author_id, tab,
        # dur_bucket, tab_x_dur, user_author_affinity, hour,
        # user_author_click_affinity.
        _db = str(int(np.searchsorted(edges, x[5])))
        _aff = str(int(np.searchsorted(aff_edges, affinity_rate(x, is_train))))
        _caff = str(int(np.searchsorted(click_aff_edges, click_affinity_rate(x, is_train))))
        return [x[1], x[2], x[3], x[4], _db, f"{x[4]}_{_db}", _aff, x[7], _caff]

    # Build one vocabulary (value -> integer id) per field, using only the
    # training split (with leave-one-out affinity values) -- this is what
    # makes validation/test "unseen" values possible, and is why the UNK
    # slot exists.
    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x, True)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽
                                                    # one reserved "unknown" slot per field
    field_dims = [len(v) + 1 for v in vocabs]      # size of each field's vocabulary (+1 for UNK)
    # offsets let us pack all fields into one shared embedding table: field
    # i's ids are shifted by offsets[i] so they don't collide with field i-1.
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        is_train = (name == 'train')
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x, is_train)):
                # look up this value's id in the training vocabulary; unseen
                # values fall back to that field's UNK id (see unk above)
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    # field_dims summed = total size of the shared embedding table any model
    # (like the FM in baseline.py) needs to allocate.
    return enc, int(sum(field_dims))
