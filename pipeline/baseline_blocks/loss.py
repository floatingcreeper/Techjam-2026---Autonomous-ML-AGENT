"""BASELINE loss block -- pointwise BCE / logloss. Ablation control for Lever A.

Returns lossfn(z, batch) -> (loss_value, g), where g = dL/dz per row, batch-normalised.
For BCE:  g = (sigmoid(z) - y) / B   -- reproduces baseline.py's FM gradient exactly.
"""
import numpy as np
from pipeline.lib.fm import sigmoid


def build_loss(cfg):
    def lossfn(z, batch):
        y = batch["y"]
        B = len(z)
        p = sigmoid(z)
        g = ((p - y) / B).astype(np.float32)
        loss = float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
        return loss, g
    return lossfn
