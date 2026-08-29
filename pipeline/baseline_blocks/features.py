"""BASELINE features block -- the 5 base fields, exactly as data.encode produces them.

Also serves as the ablation control for Lever B/C/D (sequence / aux / vstat off).
"""
from pipeline.contracts import FeatureSet, Meta


def build_features(bundle, cfg):
    return FeatureSet(
        X=bundle.X, y=bundle.y, users=bundle.users,
        meta=Meta(dim=bundle.dim, field_dims=bundle.field_dims, n_fields=bundle.n_fields),
    )
