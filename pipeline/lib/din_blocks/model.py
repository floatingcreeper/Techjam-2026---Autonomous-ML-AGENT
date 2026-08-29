"""DIN block set -- model: build the DIN on the available device (GPU if present)."""
from pipeline.lib.din import DIN, device


def build_model(meta, cfg):
    return DIN(meta.dim, meta.field_dims, k=cfg.k).to(device())
