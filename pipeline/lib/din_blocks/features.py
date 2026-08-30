"""DIN block set (Lever B) -- features: load per-user behavior sequences (+ Lever C aux labels)."""
from pipeline.contracts import FeatureSet, Meta
from pipeline.lib import seq_build, aux_build


def build_features(bundle, cfg):
    seq = {}
    for name in ("train", "valid", "test"):
        tgt, s, _ = seq_build.load_split(bundle.cache_dir, name)
        seq[name] = (tgt, s)
    m = seq_build.meta(bundle.cache_dir)
    # Lever C: attach auxiliary labels only when the config asks for them.
    aux = None
    if getattr(cfg, "aux_tasks", ()):
        aux = {name: aux_build.load_aux(bundle.cache_dir, name)
               for name in ("train", "valid", "test")}
    # meta.dim = video vocab (seq/target embedding); meta.field_dims carries the FM offset dim
    # (base 5-field embedding) so DIN keeps user/item memorization.
    return FeatureSet(X=bundle.X, y=bundle.y, users=bundle.users,
                      meta=Meta(dim=m["V"], field_dims=bundle.dim, n_fields=5), seq=seq, aux=aux)
