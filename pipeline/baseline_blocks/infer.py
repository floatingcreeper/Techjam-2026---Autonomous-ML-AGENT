"""BASELINE infer block -- score a split in its native row order (aligned for submission)."""
import numpy as np


def infer(model, feats, split):
    return np.asarray(model.predict(feats.X[split]))
