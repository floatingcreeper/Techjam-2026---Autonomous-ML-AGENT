"""Lever A -- ranking losses that align with the metric halves.

Every loss returns (loss_value, g) with g = dL/dz per row (batch-normalised), so it plugs
into any model exposing logits/apply_grad. Losses declare a `.mode`:
    point  -> rows are i.i.d.        (BCE)                      random minibatching
    group  -> rows come in user groups (BPR / softmax-CE / blend)  group batching

Surrogate mapping (the headline insight):
    BPR         is an AUC   surrogate  -> targets GAUC
    softmax-CE  is an nDCG  surrogate  -> targets nDCG@5
    blend = alpha*BPR + (1-alpha)*softmax-CE  -> targets mean(GAUC, nDCG@5)
"""
import numpy as np
from pipeline.lib.fm import sigmoid


def _segments(seg):
    """Contiguous [start,end) bounds for a group-id array like [0,0,1,1,1,2,...]."""
    cuts = np.flatnonzero(np.diff(seg)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(seg)]))
    return list(zip(starts.tolist(), ends.tolist()))


def _softmax_ce(z, y, seg, tau, group_filter):
    g = np.zeros_like(z, dtype=np.float32)
    loss, used = 0.0, 0
    for a, b in _segments(seg):
        zi = z[a:b] / tau
        yi = y[a:b]
        pos = float(yi.sum())
        if pos <= 0 or (group_filter and pos >= (b - a)):
            continue                                   # no ranking signal in this group
        zi = zi - zi.max()
        e = np.exp(zi); s = e / e.sum()
        p = yi / pos                                   # target: uniform over positives
        g[a:b] = ((s - p) / tau).astype(np.float32)
        loss += float(-(p * np.log(s + 1e-9)).sum())
        used += 1
    if used:
        g /= used; loss /= used
    return loss, g


def bpr_pair(zp, zn):
    """Vectorised BPR over aligned positive/negative logit vectors.

    Returns (loss, gp, gn) where gp/gn are per-instance dL/dz (not yet /B).
    d = zp - zn ; L = -log sigmoid(d) ; dL/dzp = -(1-sigmoid(d)), dL/dzn = +(1-sigmoid(d)).
    """
    d = zp - zn
    w = sigmoid(-d).astype(np.float32)                 # 1 - sigmoid(d)
    loss = float(-np.log(sigmoid(d) + 1e-9).mean())
    return loss, -w, w


class PointLoss:
    """BCE / logloss -- the baseline objective (Lever A ablation control)."""
    mode = "point"

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, z, batch):
        y = batch["y"]; B = len(z)
        p = sigmoid(z)
        g = ((p - y) / B).astype(np.float32)
        loss = float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
        return loss, g


class SoftmaxLoss:
    """Within-user listwise softmax cross-entropy (group mode). nDCG surrogate."""
    mode = "group"

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, z, batch):
        return _softmax_ce(z, batch["y"], batch["seg"], self.cfg.tau, self.cfg.group_filter)


class PairLoss:
    """Within-user pairwise BPR (pair mode). AUC surrogate. The pairing/sampling is done by
    the trainer's pair path; this object carries the config and the pair kernel (bpr_pair)."""
    mode = "pair"

    def __init__(self, cfg):
        self.cfg = cfg


def make_loss(cfg):
    t = cfg.loss_type
    if t == "bce":
        return PointLoss(cfg)
    if t == "softmax_ce":
        return SoftmaxLoss(cfg)
    if t == "bpr":
        return PairLoss(cfg)
    raise ValueError(f"unknown loss_type: {t}")
