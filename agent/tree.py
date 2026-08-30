"""Search tree + best-first selection."""
from __future__ import annotations
import math
from dataclasses import dataclass, field


@dataclass
class Node:
    id: str
    parent: str | None
    phase: int
    cfg: object                       # pipeline.contracts.Cfg
    block_dir: str
    lever: str = ""
    hypothesis: str = ""
    problem: str = ""                 # 1B: the Proposer's problem_identified diagnosis for this node
    metrics: dict | None = None       # {GAUC, nDCG@5, primary_valid, primary_unbiased}
    status: str = "pending"           # root | improved | no_gain | failed | abandoned

    def score(self) -> float:
        if self.metrics and self.metrics.get("primary_valid") is not None:
            return float(self.metrics["primary_valid"])
        return -math.inf


class SearchTree:
    """Frontier = nodes with metrics that didn't fail. Selection is best-first with an
    epsilon exploration valve so the search can escape a local optimum."""

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

    def select(self, explore_p: float, rng) -> Node:
        v = self._viable()
        if not v:
            return self.root
        if len(v) > 1 and rng.random() < explore_p:
            return v[rng.integers(len(v))]
        return max(v, key=lambda n: n.score())
