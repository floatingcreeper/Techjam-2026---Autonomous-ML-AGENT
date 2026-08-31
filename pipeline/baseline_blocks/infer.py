"""BASELINE infer block -- score a split in its native row order (aligned for submission).

Side effect: when asked for `test` (which the orchestrator now requests on EVERY node so the
submitted predictions come from the same trained instance as the validation predictions), this block
also writes `rand_scores.npy` for the random-exposure surface. `run_node` is frozen and can infer only
one extra split per invocation, so the second one is written here rather than by re-training.
See pipeline/lib/extra_infer.py.
"""
import numpy as np

from pipeline.lib import extra_infer


def infer(model, feats, split):
    out = np.asarray(model.predict(feats.X[split]))
    if split == "test" and extra_infer.can_infer_rand(feats):
        extra_infer.save(__file__, "rand", model.predict(feats.X["rand"]))
    return out
