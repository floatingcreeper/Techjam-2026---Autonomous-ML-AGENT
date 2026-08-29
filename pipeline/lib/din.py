"""Lever B -- Deep Interest Network with base features (target attention over user history).

The organizers' #1 unexplored direction. A local attention unit weights each history item by
its relevance to the target video and pools an interest vector; that interest is concatenated
with the target-video embedding AND a sum of the base 5-field embeddings (so the model keeps
FM-style user/item memorization) and scored by an MLP. Trains with BPR (pairwise ~ GAUC) or
BCE. Uses the GPU when torch.cuda is available, else CPU.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate import evaluate
from pipeline.lib.train_np import _build_pair_index


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class DIN(nn.Module):
    """DeepFM + DIN: an FM (linear + 2nd-order cross) over the base 5 fields keeps the strong
    user x item memorization, plus a DIN interest vector from target-attention over history."""

    def __init__(self, V, fm_dim, k=16, att_hid=32, mlp_hid=64):
        super().__init__()
        self.emb = nn.Embedding(V + 2, k, padding_idx=0)     # video: seq + target
        self.base = nn.Embedding(fm_dim, k)                  # base 5 fields (FM offset space)
        self.base_w = nn.Embedding(fm_dim, 1)                # FM linear weights
        nn.init.normal_(self.emb.weight, 0, 0.01)
        nn.init.normal_(self.base.weight, 0, 0.01)
        nn.init.zeros_(self.base_w.weight)
        self.bias = nn.Parameter(torch.zeros(1))
        self.att = nn.Sequential(nn.Linear(4 * k, att_hid), nn.ReLU(), nn.Linear(att_hid, 1))
        self.mlp = nn.Sequential(nn.Linear(3 * k, mlp_hid), nn.ReLU(), nn.Linear(mlp_hid, 1))

    def forward(self, tgt, seq, basex):
        et = self.emb(tgt)                                   # (B,k)
        eh = self.emb(seq)                                   # (B,L,k)
        mask = (seq > 0).float().unsqueeze(-1)               # (B,L,1)
        etb = et.unsqueeze(1).expand_as(eh)
        a = torch.cat([eh, etb, eh * etb, eh - etb], -1)     # (B,L,4k)
        w = self.att(a) * mask                               # DIN: no softmax
        u = (w * eh).sum(1)                                  # (B,k) interest
        be = self.base(basex)                                # (B,5,k)
        s = be.sum(1)                                        # (B,k)
        fm_cross = 0.5 * ((s ** 2) - (be ** 2).sum(1)).sum(1)   # (B,) FM interaction
        lin = self.base_w(basex).sum(1).squeeze(-1)          # (B,) FM linear
        deep = self.mlp(torch.cat([u, et, s], -1)).squeeze(-1)  # (B,) DIN deep part
        return self.bias + lin + fm_cross + deep


def predict(model, feats, split, dev, bs=8192):
    tgt, seq = feats.seq[split]
    X = np.asarray(feats.X[split])
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(tgt), bs):
            t = torch.as_tensor(tgt[i:i + bs], dtype=torch.long, device=dev)
            s = torch.as_tensor(seq[i:i + bs], dtype=torch.long, device=dev)
            x = torch.as_tensor(X[i:i + bs], dtype=torch.long, device=dev)
            outs.append(model(t, s, x).float().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def fit_din(model, feats, cfg, dev):
    tgt_tr, seq_tr = feats.seq["train"]
    X_tr = np.asarray(feats.X["train"])
    y_tr = np.asarray(feats.y["train"]); u_tr = np.asarray(feats.users["train"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
    rng = np.random.default_rng(cfg.seed)
    bs = cfg.batch
    best, best_state, bad, best_ep = -1.0, None, 0, 0

    pair = None
    if cfg.loss_type == "bpr":
        P, negpool, ns, nl = _build_pair_index(y_tr, u_tr)
        pair = (P, negpool, ns, nl, max(1, int(cfg.neg_ratio)))

    def fwd(idx):
        t = torch.as_tensor(tgt_tr[idx], dtype=torch.long, device=dev)
        s = torch.as_tensor(seq_tr[idx], dtype=torch.long, device=dev)
        x = torch.as_tensor(X_tr[idx], dtype=torch.long, device=dev)
        return model(t, s, x)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        if pair is not None:
            P, negpool, ns, nl, nr = pair
            perm = rng.permutation(len(P))
            for i in range(0, len(P), bs):
                pb = perm[i:i + bs]
                pt = np.repeat(P[pb], nr)
                off = (rng.random(len(pt)) * np.repeat(nl[pb], nr)).astype(np.int64)
                ng = negpool[np.repeat(ns[pb], nr) + off]
                zp, zn = fwd(pt), fwd(ng)
                loss = -F.logsigmoid(zp - zn).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        else:
            perm = rng.permutation(len(y_tr))
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                yb = torch.as_tensor(y_tr[idx], dtype=torch.float32, device=dev)
                loss = F.binary_cross_entropy_with_logits(fwd(idx), yb)
                opt.zero_grad(); loss.backward(); opt.step()

        va = evaluate(feats.users["valid"], feats.y["valid"], predict(model, feats, "valid", dev))
        if va["primary"] > best + 1e-5:
            best, bad, best_ep = va["primary"], 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model._best_valid = float(best); model._best_epoch = int(best_ep)
    return model
