"""The honoured-config contract -- docs/RESEARCH.md §7 (BPR) / docs/SYSTEM.md §11.

The defect this closes
----------------------
`Cfg` has 26 fields, but how many of them reach an execution path depends entirely on which block set
is mounted. Before this module existed, a hypothesis of `{"loss_type": "bpr"}` against the baseline
blocks trained plain BCE and returned metrics byte-identical to the root node -- and the agent recorded
that as the scientific finding "BPR -> 0.6015, rejected". Measured: 4 of 21 recorded runs, and all 14
LightGBM nodes ever run scored exactly 0.60205 because `gbm.train_ranker` reads only `cfg.seed`.

What this module provides
-------------------------
1. `HONOURED` -- per block set, the `Cfg` fields and `cfg_ext` keys that provably reach an execution
   path. Derived by reading the block modules and every lib they delegate to; `tests/test_blockspec.py`
   re-derives them from source and fails if the declaration drifts.
2. `DOMAINS` -- allowed values/ranges, so an out-of-domain value (the live agent has emitted
   `loss_type="softmax"`, which `make_loss` does not accept) is caught BEFORE a training launch instead
   of silently running the baseline or crashing.
3. `validate_delta` -- classifies every key of a proposed mutation into effective / ineffective /
   not-honoured / invalid, with a machine-checkable reason string the Proposer can act on.

A mutation whose effective set is empty is a STRUCTURAL no-op: it provably cannot change execution, so
it is never trained and never enters the scientific record (docs/SYSTEM.md §12).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from pipeline.contracts import Cfg

CFG_FIELDS = {f.name for f in fields(Cfg)}

# `model_type` is a family LABEL that must follow the mounted blocks (agent/mutate.py sets it from
# `adopt_blockset`). Letting a hypothesis set it by hand desynchronises the label from the code, which
# is the bug that collapsed every node into one ensemble family (docs/SYSTEM.md §6 (model_type must follow the blocks)).
MANAGED_FIELDS = {"model_type"}


@dataclass(frozen=True)
class BlockSetSpec:
    name: str
    honoured: frozenset          # Cfg field names that reach an execution path
    ext_honoured: frozenset      # cfg_ext keys that reach an execution path
    note: str = ""


# ---------------------------------------------------------------------------------------------
# Declarations. Each entry cites the modules that read the field.
# ---------------------------------------------------------------------------------------------
_FM_HONOURED = frozenset({
    # baseline_blocks/model.py -> pipeline.lib.fm.FMModel
    "seed", "k", "lr", "l2",
    # baseline_blocks/loss.py -> pipeline.lib.losses.make_loss   (routed as of the Lever-A fix, docs/RESEARCH.md §7)
    "loss_type", "tau", "group_filter",
    # baseline_blocks/train.py -> pipeline.lib.train_np.fit
    "epochs", "batch", "patience", "grad_clip", "neg_ratio", "ips",
})

_DIN_HONOURED = frozenset({
    # din_blocks/model.py
    "k", "mtl_arch",
    # din_blocks/features.py + din.fit_din
    "aux_tasks", "aux_weights",
    # din.fit_din
    "seed", "lr", "l2", "epochs", "batch", "patience", "loss_type", "neg_ratio",
})

_LGBM_HONOURED = frozenset({
    # lgbm_blocks/train.py -> pipeline.lib.gbm.train_ranker
    "seed",
})

# Extension knobs (pipeline/lib/ext.py sidecar), used where the frozen Cfg cannot gain a field.
_DIN_EXT = frozenset({
    "use_fb",           # behavior-aware history: din_blocks/{features,model}.py -> DIN.fb
    "fb_dropout",       # mask history states to UNKNOWN during training (train/serve match)
})
_LGBM_EXT = frozenset({
    # gbm.train_ranker reads these from the sidecar
    "gbm_learning_rate", "gbm_num_boost_round", "gbm_num_leaves",
    "gbm_min_data_in_leaf", "gbm_feature_fraction", "gbm_lambda_l2", "gbm_bagging_fraction",
})

SPECS: dict[str, BlockSetSpec] = {
    "fm": BlockSetSpec("fm", _FM_HONOURED, frozenset(),
                       "numpy FM; loss is pluggable via make_loss"),
    "din": BlockSetSpec("din", _DIN_HONOURED, _DIN_EXT,
                        "DeepFM+DIN (torch); objective chosen inside fit_din; history length is the "
                        "cached SEQ_L, not cfg.L; use_fb adds behavior-aware history states"),
    "lgbm": BlockSetSpec("lgbm", _LGBM_HONOURED, _LGBM_EXT,
                         "LightGBM LambdaRank; tune via the gbm_* extension knobs"),
}

# Block sets whose training is nondeterministic at a fixed seed. Measured (docs/RESEARCH.md §4):
# fm and lgbm have std 0.00000 over 24 and 14 nodes; din has sigma ~ 0.00025 with range 0.0011.
# This is what decides whether multi-seed RE-TRAINING is worth a training pass (agent/reeval.py) as
# opposed to a free paired bootstrap (agent/stats.py).
STOCHASTIC_FAMILIES = frozenset({"din", "bst"})


def is_stochastic(model_type: str) -> bool:
    return model_type in STOCHASTIC_FAMILIES


# ---------------------------------------------------------------------------------------------
# Value domains
# ---------------------------------------------------------------------------------------------
AUX_TASK_NAMES = frozenset({"click", "like", "follow", "comment", "forward"})

# `make_loss` raises ValueError for anything else. Note "blend" is advertised in the frozen Cfg
# comment but has no implementation, and the live agent has emitted "softmax" -- both are rejected
# here instead of silently running the baseline objective.
LOSS_TYPES = frozenset({"bce", "bpr", "softmax_ce", "ips_bce"})

# `din_blocks/model.py` raises NotImplementedError for anything else.
MTL_ARCHS = frozenset({"shared"})

_NUMERIC: dict[str, tuple[float, float]] = {
    "k": (2, 256), "lr": (1e-6, 1.0), "l2": (0.0, 1.0), "epochs": (1, 300),
    "batch": (128, 262144), "patience": (1, 50), "grad_clip": (0.0, 1e3),
    "neg_ratio": (1, 128), "tau": (1e-3, 100.0), "alpha": (0.0, 1.0), "L": (1, 512),
    "seed": (0, 2**31 - 1),
    # LightGBM extension knobs -- a small, scientifically useful surface, not every LightGBM flag.
    "gbm_learning_rate": (1e-3, 0.5), "gbm_num_boost_round": (50, 3000),
    "gbm_num_leaves": (7, 1023), "gbm_min_data_in_leaf": (5, 5000),
    "gbm_feature_fraction": (0.3, 1.0), "gbm_lambda_l2": (0.0, 100.0),
    "gbm_bagging_fraction": (0.3, 1.0), "fb_dropout": (0.0, 0.95),
}
_ENUM: dict[str, frozenset] = {"loss_type": LOSS_TYPES, "mtl_arch": MTL_ARCHS}
_BOOL = {"use_seq", "use_vstat", "use_aux", "lambdarank", "group_filter", "ips", "use_fb"}


def _check_value(key: str, val):
    """-> None if acceptable, else a reason string."""
    if key in _ENUM:
        if val not in _ENUM[key]:
            return f"{key}={val!r} is not one of {sorted(_ENUM[key])}"
        return None
    if key in _BOOL:
        if not isinstance(val, bool):
            return f"{key} must be a boolean, got {type(val).__name__}"
        return None
    if key == "aux_tasks":
        if not isinstance(val, (list, tuple)):
            return "aux_tasks must be a list of task names"
        bad = [t for t in val if t not in AUX_TASK_NAMES]
        if bad:
            return f"unknown aux task(s) {bad}; available: {sorted(AUX_TASK_NAMES)}"
        if len(set(val)) != len(val):
            return f"aux_tasks contains duplicates: {list(val)}"
        return None
    if key == "aux_weights":
        if not isinstance(val, (list, tuple)):
            return "aux_weights must be a list of floats"
        if any((not isinstance(w, (int, float))) or isinstance(w, bool) or w < 0 for w in val):
            return "aux_weights must be non-negative numbers"
        return None
    if key in _NUMERIC:
        lo, hi = _NUMERIC[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return f"{key} must be numeric, got {type(val).__name__}"
        if not (lo <= val <= hi):
            return f"{key}={val} is outside the supported range [{lo}, {hi}]"
        return None
    return None


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------
@dataclass
class Validation:
    """Outcome of checking a proposed mutation against the block set that will execute it."""
    effective: dict                  # keys that will actually change execution
    ext_effective: dict              # cfg_ext keys that will actually change execution
    ineffective: dict                # honoured, but the value already equals the current one
    not_honoured: dict               # real knobs this block set provably ignores
    invalid: dict                    # unknown keys or out-of-domain values
    reasons: list                    # human/LLM-readable reasons, one per rejected key

    @property
    def has_effect(self) -> bool:
        return bool(self.effective) or bool(self.ext_effective)

    @property
    def is_clean(self) -> bool:
        return not self.invalid and not self.not_honoured

    def feedback(self, blockset: str) -> str:
        """A short, specific, machine-derived message for the Proposer (docs/SYSTEM.md §12)."""
        spec = SPECS.get(blockset)
        lines = list(self.reasons)
        if not self.has_effect:
            lines.append(
                "This mutation would not change execution at all, so it was NOT run "
                "(it would produce a result identical to its parent)."
            )
        if spec is not None:
            lines.append(f"Knobs honoured by block set {blockset!r}: {sorted(spec.honoured)}.")
        return " ".join(lines)


def validate_delta(blockset: str, current: Cfg, delta: dict,
                   ext_current: dict | None = None, ext_delta: dict | None = None) -> Validation:
    """Classify every key of a proposed mutation. Never raises.

    `blockset` is the family whose blocks will execute ("fm" | "din" | "lgbm"); for an adoption it is
    the ADOPTED set, because that is what will run.
    """
    spec = SPECS.get(blockset)
    honoured = spec.honoured if spec else CFG_FIELDS
    ext_honoured = spec.ext_honoured if spec else frozenset()

    v = Validation({}, {}, {}, {}, {}, [])
    cur = current.__dict__ if hasattr(current, "__dict__") else dict(current)
    ext_cur = dict(ext_current or {})

    for key, val in (delta or {}).items():
        if key in MANAGED_FIELDS:
            v.invalid[key] = val
            v.reasons.append(
                f"{key!r} is managed by the harness (it must follow the mounted block set); "
                f"use adopt_blockset to change model family."
            )
            continue
        if key not in CFG_FIELDS:
            # not a Cfg field -- maybe it belongs in the extension space for this block set
            if key in ext_honoured:
                bad = _check_value(key, val)
                if bad:
                    v.invalid[key] = val
                    v.reasons.append(bad)
                elif ext_cur.get(key) != val:
                    v.ext_effective[key] = val
                else:
                    v.ineffective[key] = val
                continue
            v.invalid[key] = val
            v.reasons.append(
                f"{key!r} is not a configuration field (it would be silently dropped). "
                f"Valid fields: {sorted(CFG_FIELDS - MANAGED_FIELDS)}."
            )
            continue
        bad = _check_value(key, val)
        if bad:
            v.invalid[key] = val
            v.reasons.append(bad)
            continue
        if key not in honoured:
            v.not_honoured[key] = val
            v.reasons.append(
                f"{key!r} is a real field but block set {blockset!r} never reads it, so changing it "
                f"has no effect{': ' + spec.note if spec and spec.note else ''}."
            )
            continue
        # tuple/list equivalence: Cfg.from_dict normalises lists to tuples
        a, b = cur.get(key), val
        if isinstance(a, tuple) and isinstance(b, list):
            b = tuple(b)
        if a == b:
            v.ineffective[key] = val
            v.reasons.append(f"{key!r} is already {a!r}; this changes nothing.")
        else:
            v.effective[key] = val

    for key, val in (ext_delta or {}).items():
        if key not in ext_honoured:
            v.not_honoured[key] = val
            v.reasons.append(f"extension knob {key!r} is not honoured by block set {blockset!r}.")
        elif (bad := _check_value(key, val)):
            v.invalid[key] = val
            v.reasons.append(bad)
        elif ext_cur.get(key) != val:
            v.ext_effective[key] = val
        else:
            v.ineffective[key] = val

    return v


def honoured_summary() -> str:
    """Rendered into the Proposer prompt so it stops guessing at knobs nothing reads."""
    out = []
    for name, spec in SPECS.items():
        ks = sorted(spec.honoured | spec.ext_honoured)
        out.append(f"  {name}: {', '.join(ks)}")
    return "\n".join(out)
