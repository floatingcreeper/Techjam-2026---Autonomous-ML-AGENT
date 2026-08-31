"""LightGBM block set -- train: fit the LambdaRank booster (loss block unused).

Hyper-parameters come from the `cfg_ext.json` sidecar because the frozen `Cfg` cannot gain fields
(docs/RESEARCH.md §14 (LightGBM findings)). With no sidecar the defaults reproduce the previously hardcoded values, so an
untuned lgbm node still scores exactly 0.60205.
"""
from pipeline.lib import ext, gbm


def train(model, lossfn, feats, bundle, cfg):
    model.booster = gbm.train_ranker(bundle.cache_dir, cfg, ext=ext.load(__file__))
    model._best_valid = 0.0
    model._best_epoch = int(getattr(model.booster, "best_iteration", 0) or 0)
    return model
