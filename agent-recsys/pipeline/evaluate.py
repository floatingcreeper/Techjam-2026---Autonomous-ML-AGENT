"""
KuaiRand-Pure 官方评测脚本 —— 口径全部写死在这里，不要改。

任务         : 用户内排序 (within-user ranking over logged impressions)
相关性标签   : long_view (原生列, 0/1)
指标         : GAUC, nDCG@5  (主分 = 两者的平均)
排序范围     : 每个用户只对其在评测集中的曝光排序, 不做全库检索
零正例用户   : nDCG 记为 0.0 并计入平均 (与 CWM 一致)
              GAUC 只统计 0 < 正例数 < 曝光数 的用户, 按正例数加权
nDCG gain    : (2^rel - 1), 二元标签下等价于 identity
数据划分     : train 20220408-20220421 / valid 20220422-20220428 / test 20220429-20220508
"""

# ============================================================================
# WHAT THIS FILE DOES (plain English)
# ----------------------------------------------------------------------------
# This is the official scoring code. It is the single source of truth for
# how GAUC and nDCG@5 are computed, and the problem statement / README both
# say explicitly: DO NOT MODIFY THIS FILE. Any change here would make your
# local scores incomparable with the organizer's leaderboard.
#
# WHAT GAUC, nDCG@5, AND primary MEAN (in simple words)
# ----------------------------------------------------------------------------
# Both metrics ask the same basic question: "for this one user, did the
# model rank the videos they actually liked (long_view=1) above the ones
# they didn't?" They just answer it in two different ways:
#
#   GAUC ("Grouped AUC") — for one user, pick a liked video and a
#   not-liked video at random from their impressions: GAUC is the chance
#   the model gave the liked one a higher score. 0.5 = random guessing,
#   1.0 = always got the order right. It's computed per user, then
#   averaged across users (weighted by how many likes each user has —
#   that's the "Grouped" part). Only users with SOME likes and SOME
#   not-likes count here; a user who liked everything or nothing has no
#   order to get right or wrong, so they're skipped for this metric only.
#
#   nDCG@5 — look at just the TOP 5 videos the model ranked highest for
#   a user: how many were actually liked, and did the likes land near
#   position 1 (good) or position 5 (still fine, but less good)? It's
#   scaled so 1.0 = a perfect top-5 for that user, 0.0 = none of their
#   top 5 were liked. Unlike GAUC, EVERY user is included, even one who
#   never liked anything — they simply score 0.0 automatically.
#
#   primary — just the average of the two: (GAUC + nDCG@5) / 2. This is
#   the single number used to rank submissions, to decide when the
#   agent's iterations have "converged" (stopped improving), and to
#   check whether you've beaten the official baseline.
#
# HOW IT CONNECTS TO THE OTHER FILES:
#   - baseline.py             calls evaluate() after every training epoch to
#                             check validation score (for early stopping) and
#                             at the end to report valid/test scores.
#   - submit.py               calls evaluate() when you run `--score`, to
#                             locally check how a submission file would score
#                             on the validation split.
#   - ablation_features.py    calls evaluate() to score its own experimental
#                             feature configurations.
#   - data.py                 is NOT imported here. This file is fully
#                             model-agnostic: it only needs plain
#                             (user_ids, labels, scores) arrays, so ANY model
#                             you build (numpy, PyTorch, LightGBM, etc.) can
#                             be scored just by calling evaluate() at the end.
# ============================================================================

import math, collections


def auc(labels, scores):
    """Mann-Whitney U，含并列修正，等价于 sklearn.metrics.roc_auc_score。
    Computes AUC (area under the ROC curve) for one user's impressions,
    using the Mann-Whitney U statistic (with a tie-correction for equal
    scores). This gives the same result as sklearn's roc_auc_score, just
    implemented from scratch with no extra dependency.
    In simple words: this is the per-user building block for GAUC — "how
    often did a liked video get a higher score than a not-liked one?"."""

    # Sort all (score, label) pairs by score, ascending.
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    # Assign ranks, giving tied scores the *average* rank among the tied
    # group (the standard tie-correction for Mann-Whitney U / AUC).
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    npos = sum(l for _, l in pairs)
    nneg = len(pairs) - npos
    if npos == 0 or nneg == 0:
        # No way to rank positives vs negatives if one class is missing —
        # return the "no information" AUC of 0.5. (Note: evaluate() below
        # actually filters these users out before calling auc() at all, via
        # the "0 < npos < len(labs)" check, so this branch is a safety net.)
        return 0.5
    # sum of ranks assigned to the positive-labeled items
    srank = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    # standard Mann-Whitney U -> AUC conversion formula
    return (srank - npos * (npos + 1) / 2.0) / (npos * nneg)


def ndcg_at_k(labels, k):
    """labels 已按预测分降序排列。
    Computes nDCG@k for one user. `labels` must already be sorted by the
    model's predicted score, highest first — this function does not sort
    them itself.
    In simple words: "out of this user's top k ranked videos, how many
    were actually liked, with likes near the top counting more?" — scaled
    so 1.0 = a perfect top-k, 0.0 = no likes in the top k."""
    disc = [math.log2(i + 2) for i in range(k)]              # position discount: log2(rank+1)
    # DCG: sum of (2^relevance - 1) / discount, for the actual (predicted)
    # order, truncated to the top k items.
    dcg = sum(((2 ** t) - 1) / disc[i] for i, t in enumerate(labels[:k]))
    # IDCG: same formula, but for the *ideal* order (labels sorted by true
    # relevance, best case) — this is the normalizer.
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(((2 ** t) - 1) / disc[i] for i, t in enumerate(ideal))
    # nDCG = DCG / IDCG. If IDCG is 0 (user has zero positive labels, so no
    # ranking could ever score above 0), nDCG is defined as 0 rather than
    # dividing by zero — matching the README's "zero-positive users get 0".
    return 0.0 if idcg == 0 else dcg / idcg


def evaluate(user_ids, labels, scores, k=5):
    """返回 {'GAUC':…, 'nDCG@5':…, 'primary':…}。primary = 两者平均，用于排名。
    Main entry point. Takes three parallel arrays — one row per impression —
    and returns the official metric dict. This is the ONLY function other
    files should call; auc() and ndcg_at_k() above are internal helpers."""

    # Group all (score, label) pairs by user, since both metrics are
    # computed *within* each user's own impressions, never across users.
    byu = collections.defaultdict(list)
    for u, y, s in zip(user_ids, labels, scores):
        byu[u].append((s, y))

    gnum = gden = 0.0     # running numerator/denominator for the *weighted* GAUC average
    nd = []                # list of each user's individual nDCG@k score
    for u, lst in byu.items():
        lst.sort(key=lambda x: -x[0])              # sort this user's impressions by score, best first
        labs = [y for _, y in lst]
        npos = sum(labs)
        if 0 < npos < len(labs):
            # GAUC only counts users who have *both* some positive and some
            # negative labels (a user with all-positive or all-negative
            # impressions has no meaningful ranking to evaluate for AUC).
            # Each such user's AUC is weighted by their number of positives.
            gnum += npos * auc(labs, [s for s, _ in lst])
            gden += npos
        # nDCG, unlike GAUC, is computed and averaged for EVERY user,
        # including those with zero positives (who score exactly 0.0).
        nd.append(ndcg_at_k(labs, k))

    gauc = gnum / gden if gden else 0.5
    ndcg = sum(nd) / len(nd) if nd else 0.0
    # primary score = simple average of GAUC and nDCG@5 — this is the number
    # used for ranking, for the convergence check, and for "beat the
    # baseline" comparisons.
    # In simple words: GAUC = "did we rank likes above non-likes, on
    # average?"; nDCG@5 = "were the top 5 shown to each user actually
    # good?"; primary = the one overall score = average of both.
    return {'GAUC': gauc, f'nDCG@{k}': ndcg, 'primary': (gauc + ndcg) / 2.0,
            'users': len(byu), 'rows': len(labels)}
