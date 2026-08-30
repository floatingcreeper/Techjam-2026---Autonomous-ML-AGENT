"""LLM driver interface + MockDriver.

The roles (Proposer/Coder/Ablation/Reflector) talk only to this interface, so the backend
is swappable. MockDriver replays a scripted list of moves -- it exercises the ENTIRE agent
loop offline, with no network and no Gemini credits, which is how M3/M4/M5 are tested.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    seconds: float = 0.0

    def __add__(self, o: "Usage") -> "Usage":
        return Usage(self.input_tokens + o.input_tokens,
                     self.output_tokens + o.output_tokens, self.model, self.seconds + o.seconds)


class LLMDriver(ABC):
    @abstractmethod
    def generate(self, *, role, system, user, schema, model, temperature=0.4, thinking=None):
        """Return (schema_instance, Usage)."""
        raise NotImplementedError


class MockDriver(LLMDriver):
    """moves: list of dicts, each {'hypothesis': Hypothesis, 'block_edit': BlockEdit|None}.
    Proposer calls advance to the next move; Coder reads the current move's block_edit."""

    def __init__(self, moves):
        self.moves = moves
        self.i = -1

    def generate(self, *, role, system, user, schema, model, temperature=0.4, thinking=None):
        from agent.llm.schemas import Hypothesis, BlockEdit, RecoveryAction, AblationRead
        u = Usage(input_tokens=80, output_tokens=60, model="mock")
        if schema is Hypothesis:
            self.i += 1
            if self.i >= len(self.moves):
                return (Hypothesis(problem_identified="scripted moves exhausted",
                                   lever="A", statement="scripted moves exhausted",
                                   rationale="mock", mutation_kind="config",
                                   config_delta_json="{}", expected_gain=0.0), u)
            return (self.moves[self.i]["hypothesis"], u)
        if schema is BlockEdit:
            be = self.moves[self.i].get("block_edit")
            if be is None:
                raise ValueError("MockDriver: coder asked but current move has no block_edit")
            return (be, u)
        if schema is RecoveryAction:
            mv = self.moves[self.i] if 0 <= self.i < len(self.moves) else {}
            return (mv.get("recovery", RecoveryAction(failure_class="other", action="abandon",
                                                      explanation="mock default")), u)
        if schema is AblationRead:
            return (AblationRead(top_axis="A", recommended_next=["B"]), u)
        raise ValueError(f"MockDriver: unsupported schema {schema}")
