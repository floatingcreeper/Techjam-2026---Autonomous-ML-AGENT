"""DIN block set -- infer: score a split's (target, history) pairs."""
from pipeline.lib.din import predict, device


def infer(model, feats, split):
    return predict(model, feats, split, device())
