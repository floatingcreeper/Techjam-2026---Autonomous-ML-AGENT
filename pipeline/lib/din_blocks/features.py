"""DIN block set (Lever B) -- features: per-user behavior sequences (+ Lever C aux labels).

`use_fb` (extension knob, see pipeline/lib/ext.py) attaches the FEEDBACK STATE of each history event
alongside its video id -- docs/RESEARCH.md §11. It is carried as an optional third element of the
`seq` tuple because the frozen `FeatureSet` cannot gain a field.

Leakage discipline: the feedback array is built by the chronological cache (docs/RESEARCH.md §10), where a history
event is always strictly earlier than the row being scored, test-window outcomes are recorded as
FB_UNKNOWN, and a train/valid row can never see a test-window event at all. A candidate row's own
outcome is never an input.
"""
from pipeline.contracts import FeatureSet, Meta
from pipeline.lib import aux_build, ext, seq_build


def build_features(bundle, cfg):
    e = ext.load(__file__, {"use_fb": False})
    use_fb = bool(e.get("use_fb"))

    seq = {}
    for name in ("train", "valid", "test"):
        tgt, s, _ = seq_build.load_split(bundle.cache_dir, name)
        seq[name] = (tgt, s, seq_build.load_fb(bundle.cache_dir, name)) if use_fb else (tgt, s)

    m = seq_build.meta(bundle.cache_dir)
    # Lever C: attach auxiliary labels only when the config asks for them.
    # TRAIN ONLY -- fit_din reads aux["train"] and nothing else, and the valid/test aux slices are
    # near-oracle label proxies held outside the block-visible cache (docs/SYSTEM.md §8).
    aux = None
    if getattr(cfg, "aux_tasks", ()):
        aux = {"train": aux_build.load_aux(bundle.cache_dir, "train")}
    # meta.dim = video vocab (seq/target embedding); meta.field_dims carries the FM offset dim
    # (base 5-field embedding) so DIN keeps user/item memorization.
    return FeatureSet(X=bundle.X, y=bundle.y, users=bundle.users,
                      meta=Meta(dim=m["V"], field_dims=bundle.dim, n_fields=5), seq=seq, aux=aux)
