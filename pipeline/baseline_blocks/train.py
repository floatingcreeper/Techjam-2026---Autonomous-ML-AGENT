"""BASELINE train block -- delegates to the shared trainer, which dispatches batching on
the loss `.mode`. With the BCE (point) loss this reproduces baseline.py's FM loop exactly;
when the loss block is swapped for a group-mode ranking loss, group batching kicks in
automatically -- so Lever A is a single-block edit.
"""
from pipeline.lib.train_np import fit


def train(model, lossfn, feats, bundle, cfg):
    return fit(model, lossfn, feats, cfg)
