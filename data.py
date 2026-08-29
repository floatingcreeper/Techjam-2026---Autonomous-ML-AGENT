"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections
import numpy as np

# NOTE: LABEL = which CSV column counts as "the answer" the model is trying to
# predict (0 or 1: did the user watch this video long enough to count as a
# "long view"). Everything downstream treats this as ground truth.
LABEL = 'long_view'
# NOTE: SPLITS defines train/valid/test purely by DATE RANGE (inclusive on both
# ends), not by randomly shuffling rows. This matters: it's a "future data" split —
# the model is trained on earlier days and evaluated on later days, which is more
# realistic than random shuffling (in the real world you can't train on the future).
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 5 个特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
# NOTE ("feature/field" for non-ML folks): a "field" here just means one column of
# categorical input the model gets to look at — e.g. "which user", "which video".
# This list's ORDER matters: it must match the order values are packed into `raw()`
# inside encode() below. Add a new field here + a matching entry in raw() to give
# the model a new signal to learn from.
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。"""
    # NOTE: video_features_basic_pure.csv is a separate table that maps each
    # video_id -> its author_id. Building this dict first is a manual "join":
    # instead of an SQL JOIN, we're pre-loading a lookup table in memory so that
    # while reading the (much bigger) interaction logs below we can attach each
    # row's author_id with a plain dict lookup (`vid2author.get(...)`).
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    # NOTE: `rows` ends up as a big list of plain tuples, one per logged
    # impression (one row = "this user was shown this video on this date, and
    # here's what happened"). Tuples are accessed by POSITION elsewhere in this
    # file and in baseline.py (x[0]=date, x[1]=user_id, x[2]=video_id,
    # x[3]=author_id, x[4]=tab, x[5]=duration_ms, x[6]=label) — there's no field
    # names attached at this point, just position, so if you add a column here
    # you must update every x[N] index that reads after it everywhere else.
    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0))

    # NOTE: this is where the date-range split from SPLITS actually gets applied —
    # each row's date (x[0]) decides which bucket (train/valid/test) it lands in.
    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out

# NOTE: why bucket duration_ms at all? The model (FM, in baseline.py) only knows
# how to handle CATEGORICAL features — it looks up an embedding vector for a
# discrete id, it can't take a raw continuous number like "3421 milliseconds" as
# input directly. So this chops the training set's duration values into n=10
# buckets of roughly EQUAL POPULATION (quantiles, not equal-width ranges — e.g.
# bucket boundaries might be 500ms, 1200ms, 3000ms... not 0-1000-2000...), turning
# "duration" into a categorical field ("dur_bucket", the 5th FIELDS entry) with 10
# possible values the model can learn an embedding for.
def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])

def encode(splits):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    返回 (X, y, users) per split，X 为 int32 (N, len(FIELDS))，以及 field_dims。"""
    # NOTE: this whole function's job is "turn human-readable category values
    # (a user_id string, a video_id string, ...) into small integers a numpy
    # array can hold" — this is standard categorical encoding, similar in spirit
    # to sklearn's LabelEncoder, just hand-rolled with plain dicts.
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])   # NOTE: bucket edges computed from
    # TRAIN ONLY — never from valid/test. Same reasoning as vocabs below: peeking
    # at future/held-out data while building preprocessing rules is "leakage" and
    # gives you a fake, over-optimistic score.

    # NOTE: raw(x) reads one row tuple (see load()'s x[0..6] comment above) and
    # returns its 5 field VALUES in FIELDS order — this is the actual place the
    # 5 fields get pulled together for one training example.
    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    # NOTE: build one vocabulary (value -> integer id) PER FIELD, using ONLY the
    # training set. Why train-only: this simulates real deployment, where you
    # build your dictionary once from known data, and later you WILL see values
    # you've never seen before (a brand-new user_id in valid/test that didn't
    # exist during training). Using valid/test to build the vocab would be
    # "cheating" (data leakage) — the model would look artificially good because
    # it "knew about" the future.
    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    # NOTE: `unk` = one reserved "unknown" slot id per field, placed right after
    # all the known ids for that field. Any value not seen in training (e.g. a
    # new user_id first appearing in valid/test) maps to this UNK id instead of
    # crashing — see `vocabs[i].get(v, unk[i])` below.
    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽
    field_dims = [len(v) + 1 for v in vocabs]      # NOTE: +1 for the UNK slot.
    # NOTE: this is the "flatten multiple categorical fields into one shared
    # embedding table" trick used by baseline.py's FM. Each field has its own
    # small id range (0..field_dims[i]-1), but FM.V is ONE big table indexed by a
    # single combined id. `offsets` shifts each field's local ids into their own
    # non-overlapping slice of that big table — e.g. if field 0 (user_id) has
    # 1000 possible values, field 1 (video_id) ids all get shifted by +1000, so
    # "user #5" and "video #5" never collide on the same row of V. `sum(field_dims)`
    # (returned below as `dim`) is exactly the size FM.V needs to be.
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        # NOTE: X is (N rows, 5 fields) of already-offset integer ids — this is
        # exactly the array baseline.py's FM.logits(X) expects: `self.V[X]` does
        # one big lookup into the shared embedding table for every field of every
        # row at once.
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []   # NOTE: kept as a separate plain list (not put in X) because
        # evaluate.py needs raw user_id to group rows by user for GAUC/nDCG — it's
        # not a model input, just bookkeeping for scoring.
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))
