"""Fixed block contracts for the solution-space pipeline.

The six agent-owned blocks (features / model / loss / train / infer / ensemble)
must honour the signatures documented here. This module is FROZEN -- the agent may
read it but never edit it. See the README.

Block signatures (all receive the read-only DataBundle and the node Cfg):
    build_features(bundle, cfg) -> FeatureSet
    build_model(meta, cfg)      -> model object exposing .logits / .apply_grad / .predict
    build_loss(cfg)             -> lossfn(z, batch) -> (loss_value, g)     # g = dL/dz per row
    train(model, lossfn, feats, bundle, cfg) -> model   # validation-best, early-stopped
    infer(model, feats, split)  -> np.ndarray aligned to bundle row order
    combine(base, cfg)          -> np.ndarray            # assembly phase only
"""
from __future__ import annotations
import json, hashlib
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path


@dataclass
class Cfg:
    """Flat, validated hyper-parameter bag -- every lever knob lives here.

    A *config* mutation just changes these values (cheap, no code). A *block* mutation
    additionally rewrites one block body. Unknown-to-a-block fields are simply ignored.
    """
    seed: int = 0
    # -- features (Lever B/C/D inputs) --
    use_seq: bool = False
    L: int = 50
    use_vstat: bool = False
    use_aux: bool = False
    # -- model (Lever B/D) --
    model_type: str = "fm"          # fm | deepfm | dcnv2 | din | bst
    k: int = 16
    # -- loss (Lever A) --
    loss_type: str = "bce"          # bce | bpr | softmax_ce | blend
    alpha: float = 0.5              # blend weight on the BPR (AUC) term
    tau: float = 1.0                # softmax temperature
    neg_ratio: int = 4              # negatives per positive for BPR
    lambdarank: bool = False        # weight pairs by |delta nDCG|
    group_filter: bool = False      # drop all-pos / all-neg groups from the loss
    # -- train --
    lr: float = 0.001
    l2: float = 1e-6
    epochs: int = 40
    batch: int = 8192
    patience: int = 4
    grad_clip: float = 0.0          # 0 = off (Reflector may enable)
    # -- multi-task (Lever C) --
    aux_tasks: tuple = ()           # subset of {click, like, follow, comment, forward, play_time}
    aux_weights: tuple = ()
    mtl_arch: str = "shared"        # shared | mmoe | ple
    # -- debias (Lever E) --
    ips: bool = False
    # -- ensemble (Lever F) --
    ensemble_members: tuple = ()

    # ---- serialisation / identity ----
    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=list))

    @classmethod
    def from_json(cls, path) -> "Cfg":
        d = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: (tuple(v) if isinstance(v, list) else v)
                      for k, v in d.items() if k in known})

    def replace(self, **kw) -> "Cfg":
        d = asdict(self); d.update(kw)
        return Cfg.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "Cfg":
        known = {f.name for f in fields(cls)}
        return cls(**{k: (tuple(v) if isinstance(v, list) else v)
                      for k, v in d.items() if k in known})

    def hash(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, default=list).encode()
        ).hexdigest()[:12]


@dataclass
class Meta:
    """Static description of the feature space handed to build_model."""
    dim: int                 # total flat embedding-index dimension (FM offset space)
    field_dims: list         # per-field vocab sizes (may be None for M0)
    n_fields: int


@dataclass
class FeatureSet:
    """What a features block returns; what model/train/infer consume."""
    X: dict                  # split -> int32 (N, F)
    y: dict                  # split -> float32 (N,)
    users: dict              # split -> int64 (N,)   user code per row (for grouping)
    meta: Meta
    seq: dict | None = None  # split -> SeqTensors  (Lever B)
    aux: dict | None = None  # split -> {task -> (N,)}  (Lever C)
    vstat: dict | None = None
