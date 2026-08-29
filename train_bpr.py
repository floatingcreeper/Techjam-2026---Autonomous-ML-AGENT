"""KuaiRand-Pure BPR (Bayesian Personalized Ranking) Factorization Machine.
Directly optimizes pairwise user-level ranking loss for GAUC and nDCG.
"""
import argparse, collections, time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

class BPR_FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                   # (B, F, k)
        S = E.sum(1)                                    # (B, k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step_bpr(self, X_pos, X_neg):
        B = len(X_pos)
        z_pos, E_pos, S_pos = self.logits(X_pos)
        z_neg, E_neg, S_neg = self.logits(X_neg)

        diff = z_pos - z_neg
        prob = sigmoid(diff)                            # P(pos > neg)
        # BPR loss = -ln(sigmoid(diff))
        # grad w.r.t diff: (prob - 1)
        g_pos = ((prob - 1.0) / B).astype(np.float32)   # (B,)
        g_neg = -g_pos                                  # (B,)

        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X_pos, g_pos[:, None])
        np.add.at(gW, X_neg, g_neg[:, None])
        np.add.at(gV, X_pos, g_pos[:, None, None] * (S_pos[:, None, :] - E_pos))
        np.add.at(gV, X_neg, g_neg[:, None, None] * (S_neg[:, None, :] - E_neg))

        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        
        loss = float(np.mean(-np.log(prob + 1e-9)))
        return loss

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

def build_user_pools(Xtr, ytr, utr):
    """Group training samples per user into positive and negative index arrays."""
    user_pos = collections.defaultdict(list)
    user_neg = collections.defaultdict(list)
    for i, (u, y) in enumerate(zip(utr, ytr)):
        if y > 0.5:
            user_pos[u].append(i)
        else:
            user_neg[u].append(i)
    
    valid_users = [u for u in user_pos if u in user_neg]
    pos_dict = {u: np.array(user_pos[u], dtype=np.int32) for u in valid_users}
    neg_dict = {u: np.array(user_neg[u], dtype=np.int32) for u in valid_users}
    return valid_users, pos_dict, neg_dict

def sample_pairs(valid_users, pos_dict, neg_dict, rng, neg_per_pos=2):
    """Sample neg_per_pos negative items for each positive item of each user."""
    pos_list, neg_list = [], []
    for u in valid_users:
        pos_arr = pos_dict[u]
        neg_arr = neg_dict[u]
        n_pos = len(pos_arr)
        pos_samples = np.repeat(pos_arr, neg_per_pos)
        neg_samples = rng.choice(neg_arr, size=n_pos * neg_per_pos, replace=True)
        pos_list.append(pos_samples)
        neg_list.append(neg_samples)
    
    all_pos = np.concatenate(pos_list)
    all_neg = np.concatenate(neg_list)
    perm = rng.permutation(len(all_pos))
    return all_pos[perm], all_neg[perm]

def run_fm_bpr(splits, k=16, lr=0.002, epochs=40, bs=8192, neg_per_pos=2, patience=4, seed=0, verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']

    print("Building user positive/negative pools...")
    valid_users, pos_dict, neg_dict = build_user_pools(Xtr, ytr, utr)
    total_pos = sum(len(v) for v in pos_dict.values())
    print(f"Valid users: {len(valid_users)}, Pos items: {total_pos}, Pairs/epoch: {total_pos * neg_per_pos}")

    m = BPR_FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        pos_idx, neg_idx = sample_pairs(valid_users, pos_dict, neg_dict, rng, neg_per_pos=neg_per_pos)
        
        losses = []
        for i in range(0, len(pos_idx), bs):
            p_b = pos_idx[i:i + bs]
            n_b = neg_idx[i:i + bs]
            losses.append(m.step_bpr(Xtr[p_b], Xtr[n_b]))
        
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break

    m.V, m.W, m.b = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data', help='KuaiRand-Pure data dir')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.002)
    ap.add_argument('--neg_per_pos', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")

    res = run_fm_bpr(splits, k=a.k, lr=a.lr, epochs=a.epochs, neg_per_pos=a.neg_per_pos, seed=a.seed)
    print(f"\n=== FM with BPR Loss (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
