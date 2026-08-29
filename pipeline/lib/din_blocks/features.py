"""DIN block set (Lever B) -- features: load per-user behavior sequences."""
from pipeline.contracts import FeatureSet, Meta
from pipeline.lib import seq_build


def build_features(bundle, cfg):
    seq = {}
    for name in ("train", "valid", "test"):
        tgt, s, _ = seq_build.load_split(bundle.cache_dir, name)
        seq[name] = (tgt, s)
    m = seq_build.meta(bundle.cache_dir)
    # meta.dim = video vocab (seq/target embedding); meta.field_dims carries the FM offset dim
    # (base 5-field embedding) so DIN keeps user/item memorization.
    return FeatureSet(X=bundle.X, y=bundle.y, users=bundle.users,
                      meta=Meta(dim=m["V"], field_dims=bundle.dim, n_fields=5), seq=seq)
