"""BASELINE loss block -- routes the objective through `make_loss(cfg)` (Lever A).

docs/RESEARCH.md §7: this block previously HARDCODED BCE and never read `cfg.loss_type`, so a
`{"loss_type": "bpr"}` config mutation silently trained plain BCE and returned metrics byte-identical
to the root node. That happened in 4 of 21 recorded runs; it cost the reference run +0.0021 on its
strongest lever and wrote a fabricated negative result ("BPR -> 0.6015, rejected") into memory.

Routing through `make_loss` makes Lever A a genuine config knob and is what `agent/blockspec.py`
declares as honoured for this block set.

The BCE path is mathematically unchanged, so the M0 gate still reproduces baseline.py exactly:
`make_loss(cfg)` with loss_type="bce" returns `PointLoss`, whose gradient is `g = (sigmoid(z) - y)/B`
with `mode = "point"` -- identical to the closure this block used to build.

Contract: build_loss(cfg) -> lossfn(z, batch) -> (loss_value, g), g = dL/dz per row.
The returned object also carries `.mode` (point | pair | group) so `train_np.fit` dispatches the
correct batching.
"""
from pipeline.lib.losses import make_loss


def build_loss(cfg):
    return make_loss(cfg)
