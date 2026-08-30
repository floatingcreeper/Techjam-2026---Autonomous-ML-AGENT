"""Generate and validate submission files.

Submission format (CSV, with header):
    row_id,user_id,video_id,score

  row_id   : row number starting from 0, corresponding to the row order of data.load()[split]
             (deterministic: read log_standard_4_08_to_4_21_pure.csv first, then
             log_standard_4_22_to_5_08_pure.csv, keeping the original file order after filtering by date)
  user_id  : the row's user_id (redundant field, only used to check alignment)
  video_id : the row's video_id (redundant field, only used to check alignment)
  score    : the score your model assigns to this row, any real number, only relative magnitude matters

Why include row_id: (user_id, video_id) is **not unique** in the evaluation set
(the test set has 3.06% duplicate pairs, repeated up to 12 times), so it cannot serve as a primary key.

Usage:
    python3 submit.py --make   submission.csv     # generate a sample submission using the official FM baseline
    python3 submit.py --check  submission.csv     # validate format and alignment
    python3 submit.py --score  submission.csv     # validate and score (only available locally for valid)
"""
import argparse, csv, sys
from data import load, encode
from evaluate import evaluate

HEADER = ['row_id', 'user_id', 'video_id', 'score']

def write_submission(path, rows, scores):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (x, s) in enumerate(zip(rows, scores)):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])

def read_submission(path, rows):
    """Read and validate alignment row by row, returning scores. Any mismatch raises a readable error."""
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        head = next(r, None)
        if head != HEADER:
            raise ValueError(f"header must be {','.join(HEADER)}, but got {head}")
        scores, n = [], 0
        for ln, rec in enumerate(r, start=2):
            if len(rec) != 4:
                raise ValueError(f"line {ln} has {len(rec)} fields, expected 4")
            rid, uid, vid, sc = rec
            if int(rid) != n:
                raise ValueError(f"line {ln} row_id={rid}, expected {n} (must increase contiguously from 0)")
            if n >= len(rows):
                raise ValueError(f"submission has more rows than the evaluation set (evaluation set has {len(rows)} rows)")
            if uid != rows[n][1] or vid != rows[n][2]:
                raise ValueError(f"line {ln} alignment error: submission ({uid},{vid}), "
                                 f"row {n} of the evaluation set is ({rows[n][1]},{rows[n][2]})")
            try:
                v = float(sc)
            except ValueError:
                raise ValueError(f"line {ln} score cannot be parsed as a number: {sc!r}")
            if v != v or v in (float('inf'), float('-inf')):
                raise ValueError(f"line {ln} score is NaN/Inf, not allowed")
            scores.append(v); n += 1
    if n != len(rows):
        raise ValueError(f"submission has {n} rows, evaluation set has {len(rows)} rows, count mismatch")
    return scores

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--split', default='test', choices=['valid', 'test'])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--make',  action='store_true', help='generate a sample submission using the official FM baseline')
    g.add_argument('--check', action='store_true', help='only validate format and alignment')
    g.add_argument('--score', action='store_true', help='validate and score')
    a = ap.parse_args()

    splits = load(a.data_dir)
    rows = splits[a.split]

    if a.make:
        from baseline import run_fm
        import baseline as B, numpy as np
        enc, dim = encode(splits)
        Xtr, ytr, _ = enc['train']
        Xva, yva, uva = enc['valid']
        X, y, u = enc[a.split]
        m = B.FM(dim, k=16, lr=0.001, seed=0)
        rng = np.random.default_rng(0)
        best, state, bad = -1, None, 0
        for ep in range(40):
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), 8192):
                m.step(Xtr[idx[i:i+8192]], ytr[idx[i:i+8192]])
            p = evaluate(uva, yva, m.predict(Xva))['primary']
            if p > best + 1e-5: best, bad, state = p, 0, (m.V.copy(), m.W.copy(), m.b)
            else:
                bad += 1
                if bad >= 4: break
        m.V, m.W, m.b = state
        write_submission(a.path, rows, m.predict(X))
        print(f"wrote {a.path}: {len(rows):,d} rows (split={a.split}, official FM baseline)")
    else:
        scores = read_submission(a.path, rows)
        print(f"✓ format and alignment validation passed: {len(scores):,d} rows, split={a.split}")
        if a.score:
            r = evaluate([x[1] for x in rows], [x[6] for x in rows], scores)
            print(f"  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
