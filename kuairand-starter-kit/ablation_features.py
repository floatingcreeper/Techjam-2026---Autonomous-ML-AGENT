"""把 CWM 的 13 个特征域接进来，验证「用户侧特征在 FM 里是否有用」。"""

# ============================================================================
# WHAT THIS FILE DOES (plain English)
# ----------------------------------------------------------------------------
# This is a standalone EXPERIMENT script, not part of the main pipeline. It
# reproduces the "adding more features doesn't help" finding documented in
# the README: it trains 3 versions of the same FM model with progressively
# more feature columns (5 fields -> 9 fields -> 13 fields, the last matching
# the CWM reference paper's feature set) and reports mean +/- std over 3
# seeds for each. The result: more features ~= no real gain, within noise.
#
# HOW IT CONNECTS TO THE OTHER FILES:
#   - It does NOT import data.py. It reimplements its own version of
#     load/encode inline (mode-aware, so it can build 5, 9, or 13-field
#     versions of the same rows) rather than reusing data.encode().
#   - It imports the FM class from baseline.py (`import baseline as B`) to
#     reuse the exact same model/training code as the official baseline —
#     only the FEATURES differ between this script's runs, not the model.
#   - It imports evaluate() from evaluate.py, same as everywhere else, so
#     its numbers are directly comparable to baseline.py's output.
#
# You don't need to run this yourself — its conclusion is already in the
# README. It's included so the "more features didn't help" claim is
# reproducible, not just asserted.
# ============================================================================

import csv, os, collections, statistics
import numpy as np
from evaluate import evaluate
import baseline as B

import sys
D = sys.argv[1] if len(sys.argv) > 1 else './KuaiRand-Pure/data'    # optional data dir override, e.g. `python3 ablation_features.py /path/to/data`
SPLITS={'train':(20220408,20220421),'valid':(20220422,20220428),'test':(20220429,20220508)}  # same official date split as data.py

# CWM 的 13 个域
# The 13 feature columns used in the CWM reference paper: 5 are user-side
# (from user_features_pure.csv), and 4 are item/video-side (from
# video_features_basic_pure.csv, on top of author_id which is already in
# the base 5-field kit).
USER_FE=['follow_user_num_range','register_days_range','fans_user_num_range',
         'friend_user_num_range','user_active_degree']
VID_FE=['author_id','music_id','video_type','upload_type']

# Build lookup tables: user_id -> [its 5 extra feature values], and
# video_id -> [its 4 extra feature values], read once up front.
u_ext={}
with open(f'{D}/user_features_pure.csv') as fh:
    for r in csv.DictReader(fh): u_ext[r['user_id']]=[r[k] for k in USER_FE]
v_ext={}
with open(f'{D}/video_features_basic_pure.csv') as fh:
    for r in csv.DictReader(fh):
        v_ext[r['video_id']]=[r[k] for k in VID_FE[:1]+VID_FE[1:]]

# Load the raw interaction rows, same idea as data.load() but inlined here:
# (date, user_id, video_id, tab, duration_ms, label)
rows=[]
for f in ('log_standard_4_08_to_4_21_pure.csv','log_standard_4_22_to_5_08_pure.csv'):
    with open(f'{D}/{f}') as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r['date']), r['user_id'], r['video_id'], r['tab'],
                         float(r['duration_ms']), 1 if r['long_view']!='0' else 0))
splits={n:[x for x in rows if lo<=x[0]<=hi] for n,(lo,hi) in SPLITS.items()}
print({k:len(v) for k,v in splits.items()})

UNKU=['UNK']*len(USER_FE); UNKV=['UNK']*len(VID_FE)   # fallback values for users/videos missing from the feature files


def build(mode):
    """mode: 'base'=5域(现kit) / 'item'=只加物品侧 / 'cwm13'=CWM全13域
    Builds one encoded feature matrix, with the feature set controlled by
    `mode`: 'base' matches the shipped 5-field kit, 'item' adds the 4 extra
    video-side fields, 'cwm13' additionally adds the 5 user-side fields
    (the full CWM-paper feature set). Mirrors data.py's encode() logic
    (train-only vocab + UNK slot) but built inline so it can vary the field
    count per mode."""
    # duration buckets, learned from train only — same idea as data._bucket_edges
    edges=np.quantile([x[4] for x in splits['train']], np.linspace(0,1,11)[1:-1])

    def raw(x):
        # look up each row's extra user/video features, falling back to UNK
        # placeholders if that user/video id is missing from the feature files
        ue=u_ext.get(x[1],UNKU); ve=v_ext.get(x[2],UNKV)
        f=[x[1], x[2], ve[0], x[3], str(int(np.searchsorted(edges,x[4])))]   # 5 域基线 — same 5 base fields as data.py
        if mode in ('item','cwm13'): f += ve[1:]                              # +music/type/upload — adds the 3 remaining video-side fields
        if mode=='cwm13':            f += ue                                  # +6 用户侧 — adds all user-side fields (the "13 total" config)
        return f

    n=len(raw(splits['train'][0]))
    # build a train-only vocabulary per field, exactly like data.encode()
    vocabs=[dict() for _ in range(n)]
    for x in splits['train']:
        for i,v in enumerate(raw(x)):
            if v not in vocabs[i]: vocabs[i][v]=len(vocabs[i])
    unk=[len(v) for v in vocabs]; dims=[len(v)+1 for v in vocabs]
    off=np.cumsum([0]+dims[:-1]).astype(np.int32)
    enc={}
    for name,rws in splits.items():
        X=np.empty((len(rws),n),dtype=np.int32); y=np.empty(len(rws),dtype=np.float32); us=[]
        for j,x in enumerate(rws):
            for i,v in enumerate(raw(x)): X[j,i]=vocabs[i].get(v,unk[i])+off[i]
            y[j]=x[5]; us.append(x[1])
        enc[name]=(X,y,us)
    return enc, int(sum(dims)), n


# Main experiment loop: train the SAME FM model (imported from baseline.py)
# on each of the 3 feature configurations, 3 random seeds each, and print
# the mean +/- std test score for a fair, noise-aware comparison.
for mode,desc in [('base','5 域（当前 kit）'),('item','+4 物品侧 = 9 域'),('cwm13','CWM 全 13 域')]:
    enc,dim,nf=build(mode)
    Xtr,ytr,_=enc['train']; Xva,yva,uva=enc['valid']; Xte,yte,ute=enc['test']
    scores=[]
    for seed in range(3):
        # same FM class, same hyperparameters (k=16, lr=0.001) and same
        # early-stopping logic as baseline.py's run_fm — only the input
        # features (dim, and what's packed into X) differ here
        m=B.FM(dim,k=16,lr=0.001,seed=seed); rng=np.random.default_rng(seed)
        best=-1; bs=8192; bad=0; state=None
        for ep in range(40):
            idx=rng.permutation(len(ytr))
            for i in range(0,len(idx),bs): m.step(Xtr[idx[i:i+bs]], ytr[idx[i:i+bs]])
            p=evaluate(uva,yva,m.predict(Xva))['primary']
            if p>best+1e-5: best=p; bad=0; state=(m.V.copy(),m.W.copy(),np.float32(m.b))
            else:
                bad+=1
                if bad>=4: break
        m.V,m.W,m.b=state
        scores.append(evaluate(ute,yte,m.predict(Xte)))
    # aggregate the 3 seeds: mean score plus population std, so the report
    # shows whether a difference between modes is real or just seed noise
    g=statistics.mean(s['GAUC'] for s in scores); n5=statistics.mean(s['nDCG@5'] for s in scores)
    pr=statistics.mean(s['primary'] for s in scores); sd=statistics.pstdev([s['primary'] for s in scores])
    print(f"{desc:20s} ({nf:2d}域) | test GAUC {g:.4f} | nDCG@5 {n5:.4f} | primary {pr:.4f} ± {sd:.4f}")
