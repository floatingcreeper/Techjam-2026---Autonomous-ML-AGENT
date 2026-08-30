"""DIN block set -- model: build the DIN on the available device (GPU if present)."""
from pipeline.lib.din import DIN, device


def build_model(meta, cfg):
    n_aux = len(getattr(cfg, "aux_tasks", ()) or ())
    if n_aux and getattr(cfg, "mtl_arch", "shared") != "shared":
        raise NotImplementedError(f"mtl_arch={cfg.mtl_arch!r} not implemented; only 'shared' (v1)")
    return DIN(meta.dim, meta.field_dims, k=cfg.k, n_aux=n_aux).to(device())
