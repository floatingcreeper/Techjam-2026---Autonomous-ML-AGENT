"""DIN block set -- train: BPR/BCE torch loop, early-stopped on valid primary."""
from pipeline.lib.din import fit_din, device


def train(model, lossfn, feats, bundle, cfg):
    return fit_din(model, feats, cfg, device())
