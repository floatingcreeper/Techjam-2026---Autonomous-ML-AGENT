"""DIN block set -- train: BPR/BCE torch loop, early-stopped on valid primary."""
from pipeline.lib import ext
from pipeline.lib.din import device, fit_din


def train(model, lossfn, feats, bundle, cfg):
    e = ext.load(__file__, {"fb_dropout": 0.0})
    return fit_din(model, feats, cfg, device(), fb_drop=float(e.get("fb_dropout", 0.0) or 0.0))
