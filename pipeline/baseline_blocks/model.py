"""BASELINE model block -- the numpy FM. Ablation control for Lever D (model family)."""
from pipeline.lib.fm import FMModel


def build_model(meta, cfg):
    return FMModel(meta.dim, k=cfg.k, lr=cfg.lr, l2=cfg.l2, seed=cfg.seed)
