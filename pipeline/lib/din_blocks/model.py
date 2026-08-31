"""DIN block set -- model: build the DIN on the available device (GPU if present)."""
from pipeline.lib import ext, seq_build
from pipeline.lib.din import DIN, device


def build_model(meta, cfg):
    n_aux = len(getattr(cfg, "aux_tasks", ()) or ())
    if n_aux and getattr(cfg, "mtl_arch", "shared") != "shared":
        raise NotImplementedError(f"mtl_arch={cfg.mtl_arch!r} not implemented; only 'shared' (v1)")
    e = ext.load(__file__, {"use_fb": False})
    n_fb = seq_build.N_FB_STATES if e.get("use_fb") else 0
    return DIN(meta.dim, meta.field_dims, k=cfg.k, n_aux=n_aux, n_fb=n_fb).to(device())
