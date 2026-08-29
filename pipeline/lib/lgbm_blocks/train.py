"""LightGBM block set -- train: fit the LambdaRank booster (loss block unused)."""
from pipeline.lib import gbm


def train(model, lossfn, feats, bundle, cfg):
    model.booster = gbm.train_ranker(bundle.cache_dir, cfg)
    model._best_valid = 0.0
    model._best_epoch = int(getattr(model.booster, "best_iteration", 0) or 0)
    return model
