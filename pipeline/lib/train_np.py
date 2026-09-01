"""Shared numpy trainer -- dispatches batching on the loss `.mode`.

    point  -> random minibatching        (reproduces baseline.py's FM loop exactly)
    group  -> user-group batching         (for BPR / softmax-CE / blend)

Both paths early-stop on validation primary and restore the best parameters, so the
returned model is always the validation-best checkpoint.
"""
import numpy as np
from evaluate import evaluate


def fit(model, lossfn, feats, cfg):
    mode = getattr(lossfn, "mode", "point")
    if mode == "group":
        return _fit_group(model, lossfn, feats, cfg)
    if mode == "pair":
        return _fit_pair(model, lossfn, feats, cfg)
    return _fit_point(model, lossfn, feats, cfg)


def _eval_valid(model, feats):
    return evaluate(feats.users["valid"], feats.y["valid"], model.predict(feats.X["valid"]))


def _loop(model, feats, cfg, epoch_fn):
    best, best_state, bad, best_ep = -1.0, None, 0, 0
    for ep in range(1, cfg.epochs + 1):
        epoch_fn(ep)
        va = _eval_valid(model, feats)
        if va["primary"] > best + 1e-5:
            best, bad, best_ep = va["primary"], 0, ep
            best_state = model.state()
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state(best_state)
    model._best_valid = float(best)
    model._best_epoch = int(best_ep)
    return model


def _ips_weights(Xtr, field=1):
    """Lever E: w ~ 1/sqrt(train item frequency), normalised to mean 1, so over-exposed (popular)
    items are down-weighted.

    NOTE the name overstates this: it is INVERSE-POPULARITY weighting, not inverse propensity. A true
    IPS estimator needs 1/P(exposure | u, context, policy), and the square root makes this not even an
    unbiased popularity correction. See docs/EN/RESEARCH.md §15 (E2)."""
    items = np.asarray(Xtr[:, field])
    freq = np.bincount(items)[items].astype(np.float32)
    w = 1.0 / np.sqrt(freq + 1.0)
    return (w / w.mean()).astype(np.float32)


def _fit_point(model, lossfn, feats, cfg):
    Xtr, ytr = feats.X["train"], feats.y["train"]
    rng = np.random.default_rng(cfg.seed)
    N = len(ytr)
    # Lever E: IPS row weights when requested (loss_type='ips_bce' or cfg.ips)
    row_w = None
    if getattr(cfg, "loss_type", "") == "ips_bce" or getattr(cfg, "ips", False):
        row_w = _ips_weights(np.asarray(Xtr))

    def epoch(ep):
        idx = rng.permutation(N)
        for i in range(0, N, cfg.batch):
            b = idx[i:i + cfg.batch]
            Xb = np.asarray(Xtr[b]); yb = np.asarray(ytr[b])
            z, cache = model.logits(Xb)
            batch = {"y": yb} if row_w is None else {"y": yb, "w": row_w[b]}
            _, g = lossfn(z, batch)
            model.apply_grad(Xb, g, cache, grad_clip=cfg.grad_clip)

    return _loop(model, feats, cfg, epoch)


def _build_pair_index(ytr, utr):
    """Per-user positive rows + a flat negative pool with per-user (start,len) offsets.
    Only users with >=1 positive AND >=1 negative contribute. Fully vectorisable sampling."""
    order = np.argsort(utr, kind="stable")
    bounds = np.flatnonzero(np.diff(utr[order])) + 1
    groups = np.split(order, bounds)
    pos_rows, neg_pool, negstart, neglen = [], [], [], []
    cur = 0
    for rows in groups:
        yy = np.asarray(ytr[rows])
        p = rows[yy > 0]; n = rows[yy <= 0]
        if len(p) == 0 or len(n) == 0:
            continue
        neg_pool.append(n)
        for _ in p:
            negstart.append(cur); neglen.append(len(n))
        pos_rows.append(p)
        cur += len(n)
    return (np.concatenate(pos_rows).astype(np.int64),
            np.concatenate(neg_pool).astype(np.int64),
            np.asarray(negstart, np.int64), np.asarray(neglen, np.int64))


def _fit_pair(model, lossfn, feats, cfg):
    from pipeline.lib.losses import bpr_pair
    Xtr, ytr, utr = feats.X["train"], feats.y["train"], np.asarray(feats.users["train"])
    P, negpool, ns, nl = _build_pair_index(ytr, utr)
    nr = max(1, int(cfg.neg_ratio))
    rng = np.random.default_rng(cfg.seed)
    M = len(P)

    def step(pb):
        pt = np.repeat(P[pb], nr)                       # each positive repeated nr times
        ns_r = np.repeat(ns[pb], nr); nl_r = np.repeat(nl[pb], nr)
        off = (rng.random(len(ns_r)) * nl_r).astype(np.int64)
        Ng = negpool[ns_r + off]
        rows = np.concatenate([pt, Ng])
        Xb = np.asarray(Xtr[rows])
        z, cache = model.logits(Xb)
        m = len(pt)
        _, gp, gn = bpr_pair(z[:m], z[m:])
        g = np.empty(len(rows), np.float32)
        g[:m] = gp / m; g[m:] = gn / m
        model.apply_grad(Xb, g, cache, grad_clip=cfg.grad_clip)

    def epoch(ep):
        perm = rng.permutation(M)
        for i in range(0, M, cfg.batch):
            step(perm[i:i + cfg.batch])

    return _loop(model, feats, cfg, epoch)


def _fit_group(model, lossfn, feats, cfg):
    Xtr, ytr, utr = feats.X["train"], feats.y["train"], np.asarray(feats.users["train"])
    order = np.argsort(utr, kind="stable")
    bounds = np.flatnonzero(np.diff(utr[order])) + 1
    groups = np.split(order, bounds)                       # list of row-index arrays, one per user
    rng = np.random.default_rng(cfg.seed)

    def step(chunk):
        rows = np.concatenate(chunk)
        seg = np.repeat(np.arange(len(chunk)), [len(c) for c in chunk])
        Xb = np.asarray(Xtr[rows]); yb = np.asarray(ytr[rows])
        z, cache = model.logits(Xb)
        _, g = lossfn(z, {"y": yb, "seg": seg, "n_groups": len(chunk)})
        model.apply_grad(Xb, g, cache, grad_clip=cfg.grad_clip)

    def epoch(ep):
        perm = rng.permutation(len(groups))
        chunk, count = [], 0
        for gi in perm:
            rows = groups[gi]
            chunk.append(rows); count += len(rows)
            if count >= cfg.batch:
                step(chunk); chunk, count = [], 0
        if chunk:
            step(chunk)

    return _loop(model, feats, cfg, epoch)
