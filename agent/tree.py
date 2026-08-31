"""Search tree + best-first selection.

A `Node` now carries THREE independent statuses, because docs/SYSTEM.md §14 (research memory) measured what happens
when they are collapsed into one:

  status      -- tree/adoption bookkeeping (root | improved | no_gain | abandoned | duplicate).
                 Decides tree SHAPE. Says nothing about whether an effect is real.
  evidence    -- what the paired bootstrap says about the effect (confirmed | promising |
                 inconclusive | rejected). Decides what MEMORY tells the Proposer.
  noop_class  -- whether an intervention actually happened at all (STRUCTURAL_NOOP | EXACT_NOOP |
                 NEAR_NOOP | None). A structural/exact no-op is never scientific evidence.

Collapsing these is how the reference run came to tell its Proposer "REJECTED: multi-task DIN 0.6026"
about its own best model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# An intervention that provably could not change execution: the requested knobs are not honoured by
# the mounted block set, or the value already equals the current one, and no block was edited.
STRUCTURAL_NOOP = "STRUCTURAL_NOOP"
# Predictions came back bit-identical to the parent's. Meaningful only for deterministic families;
# for a stochastic family it is strong evidence the intervention never reached execution.
EXACT_NOOP = "EXACT_NOOP"
# Predictions are extremely similar but not identical. This IS legitimate evidence that the
# intervention had a negligible effect -- it is not a no-op.
NEAR_NOOP = "NEAR_NOOP"


@dataclass
class Node:
    id: str
    parent: str | None
    phase: int
    cfg: object                       # pipeline.contracts.Cfg
    block_dir: str
    lever: str = ""
    hypothesis: str = ""
    problem: str = ""                 # 1B: the Proposer's problem_identified diagnosis
    metrics: dict | None = None       # {GAUC, nDCG@5, primary_valid, ...}
    status: str = "pending"           # tree/adoption bookkeeping
    evidence: dict = field(default_factory=dict)      # paired-bootstrap verdict vs. its control
    portfolio: dict = field(default_factory=dict)     # rank_corr_to_best / pair_blend_gain / emc
    provenance: dict = field(default_factory=dict)    # intended vs. executed intervention
    ext: dict = field(default_factory=dict)           # cfg_ext sidecar
    noop_class: str | None = None
    informative: bool = False         # did this experiment yield research information? (docs/SYSTEM.md §16)

    def score(self) -> float:
        if self.metrics and self.metrics.get("primary_valid") is not None:
            return float(self.metrics["primary_valid"])
        return -math.inf

    def gauc(self):
        return (self.metrics or {}).get("GAUC")

    def ndcg(self):
        return (self.metrics or {}).get("nDCG@5")


class SearchTree:
    """Frontier = nodes with metrics that didn't fail. Selection is best-first with an epsilon
    exploration valve so the search can escape a local optimum."""

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.root: Node | None = None

    def add(self, node: Node):
        self.nodes[node.id] = node
        if node.status == "root":
            self.root = node

    def _viable(self):
        return [n for n in self.nodes.values()
                if n.metrics is not None and n.status in ("root", "improved", "no_gain")]

    def best(self) -> Node | None:
        v = self._viable()
        return max(v, key=lambda n: n.score()) if v else self.root

    def select(self, explore_p: float, rng, prefer_diverse: bool = False) -> Node:
        """Best-first with an epsilon exploration valve.

        `prefer_diverse` is the plateau-escalation hook (docs/SYSTEM.md §16): when the search has stopped producing
        information, expanding the most DECORRELATED viable node is more useful than expanding the
        best one again. It changes which parent is chosen; it never changes when the run stops.
        """
        v = self._viable()
        if not v:
            return self.root
        if len(v) > 1 and rng.random() < explore_p:
            if prefer_diverse:
                champ = max(v, key=lambda n: n.score())
                others = [n for n in v if n is not champ]
                if others:
                    return min(others, key=lambda n: n.portfolio.get("rank_corr_to_best", 1.0))
            return v[rng.integers(len(v))]
        return max(v, key=lambda n: n.score())
