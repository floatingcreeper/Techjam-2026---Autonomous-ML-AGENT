"""Numpy Factorization Machine -- the baseline backbone, factored so the loss is pluggable.

Unlike the starter kit's FM (which bakes BCE gradients into .step), this version separates
concerns so *any* ranking loss can drive it (Lever A):

    z, cache = model.logits(X)          # forward
    loss, g  = lossfn(z, batch)         # g = dL/dz per row  (loss owns the objective)
    model.apply_grad(X, g, cache)       # backward + Adam     (model owns the parameters)

With the BCE loss (g = (sigmoid(z) - y) / B) this reproduces baseline.py's FM exactly.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FMModel:
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
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        z = self.b + self.W[X].sum(1) + inter
        return z, (E, S)

    def apply_grad(self, X, g, cache, grad_clip=0.0):
        """g: (B,) = dL/dz per row (already batch-normalised by the loss)."""
        E, S = cache
        g = np.asarray(g, dtype=np.float32)
        if grad_clip > 0.0:
            np.clip(g, -grad_clip, grad_clip, out=g)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

    # ---- best-checkpoint support ----
    def state(self):
        return (self.V.copy(), self.W.copy(), np.float32(self.b))

    def load_state(self, s):
        self.V, self.W, self.b = s[0].copy(), s[1].copy(), np.float32(s[2])
