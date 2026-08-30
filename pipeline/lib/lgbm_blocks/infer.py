"""LightGBM block set -- infer: predict on a split's engineered feature matrix."""
import numpy as np


def infer(model, feats, split):
    bi = getattr(model.booster, "best_iteration", None)
    return model.booster.predict(feats.X[split], num_iteration=bi).astype(np.float32)
