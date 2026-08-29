"""LightGBM block set (Lever D) -- features: engineered item/author/stat features."""
from pipeline.contracts import FeatureSet, Meta
from pipeline.lib import gbm


def build_features(bundle, cfg):
    X, y, u = gbm.load_features(bundle.cache_dir)
    n = int(X["train"].shape[1])
    return FeatureSet(X=X, y=y, users=u, meta=Meta(dim=n, field_dims=None, n_fields=n))
