"""Structured output schemas.

Gemini's response_schema guarantees valid JSON matching these Pydantic models, removing an
entire class of parse failures. config_delta is carried as a JSON-object *string* so the
open-ended Cfg key space works uniformly across Gemini and the MockDriver.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

Block = Literal["features", "model", "loss", "train", "infer", "ensemble"]
Lever = Literal["A", "B", "C", "D", "E", "F"]


class Hypothesis(BaseModel):
    problem_identified: str = Field(
        description="FIRST: the single most likely bottleneck right now -- cite concrete "
                    "numbers/levers from the history, not generic ML advice")
    lever: Lever
    statement: str = Field(description="what to try, one sentence")
    rationale: str = Field(description="why it should help; cite a playbook card / the metric")
    mutation_kind: Literal["config", "block"]
    target_block: Optional[Block] = None
    config_delta_json: str = Field(default="{}", description="JSON object of Cfg overrides")
    adopt_blockset: Optional[str] = Field(
        default=None,
        description="adopt a pre-built model-family block set wholesale, e.g. 'lgbm' or 'din'")
    expected_metric: Literal["GAUC", "nDCG@5", "both"] = "both"
    expected_gain: float = 0.0


class BlockEdit(BaseModel):
    target_block: Block
    new_source: str = Field(default="", description="full new source of that one block module; "
                            "empty if implementable is false")
    imports_used: list[str] = Field(default_factory=list)
    implementable: bool = True        # 1D: false = decline (with `reason`), emit no code
    reason: str = ""                  # 1D: one sentence -- why it can't be done in this one block
    notes: str = ""


class AblationRead(BaseModel):
    top_axis: Lever
    per_block_delta_json: str = "{}"
    recommended_next: list[str] = Field(default_factory=list)


class RecoveryAction(BaseModel):
    failure_class: Literal["code", "timeout", "numerical", "no_improve", "llm", "other"]
    action: Literal["patch_retry", "degrade", "abandon", "switch_lever"]
    patch_block: Optional[Block] = None
    new_source: Optional[str] = None
    config_delta_json: str = "{}"
    explanation: str = ""
