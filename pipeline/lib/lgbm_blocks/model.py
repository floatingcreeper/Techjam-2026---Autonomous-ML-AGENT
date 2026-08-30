"""LightGBM block set -- model: a thin holder; the booster is fit in the train block."""


class GBMModel:
    def __init__(self):
        self.booster = None


def build_model(meta, cfg):
    return GBMModel()
