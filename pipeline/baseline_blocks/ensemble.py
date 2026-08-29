"""BASELINE ensemble block -- passthrough (single model). Lever F rewrites this to
stack / blend / calibrate multiple base learners in the assembly phase.
"""
import numpy as np


def combine(base, cfg):
    """base: {member_id -> scores (N,)}. Default: identity for one member, mean otherwise."""
    arrs = list(base.values())
    if len(arrs) == 1:
        return np.asarray(arrs[0])
    return np.mean(np.stack([np.asarray(a) for a in arrs]), axis=0)
