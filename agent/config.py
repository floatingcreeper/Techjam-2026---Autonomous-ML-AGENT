"""Central configuration.

Defaults live here; agent/config.yaml (optional) overrides them. Kept dependency-light:
if pyyaml is present and a yaml file is given, it is merged over the defaults.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Budget:
    max_iter: int = 50
    wall_clock_hours: float = 6.0
    per_iter_timeout_s: int = 900
    eps: float = 0.0002
    N: int = 6


@dataclass
class Phases:
    breadth_until: int = 12
    depth_until: int = 40
    ablation_every: int = 6
    explore_p: float = 0.15


@dataclass
class LLM:
    provider: str = "gemini"
    proposer: str = "gemini-3.1-pro-preview"
    reflector: str = "gemini-3.1-pro-preview"
    coder: str = "gemini-3.7-flash"
    ablation: str = "gemini-3.5-flash-lite"
    temperature: float = 0.4
    max_retries: int = 5
    max_llm_usd: float = 0.0          # 0 = no soft cap


@dataclass
class Config:
    data_dir: str = "./KuaiRand-Pure/data"
    cache_dir: str = "runs/_cache"
    runs_dir: str = "runs"
    seed: int = 0
    gpu: str = "auto"                 # auto | on | off
    budget: Budget = field(default_factory=Budget)
    phases: Phases = field(default_factory=Phases)
    llm: LLM = field(default_factory=LLM)

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        cfg = cls()
        if path and Path(path).exists():
            import yaml
            data = yaml.safe_load(Path(path).read_text()) or {}
            cfg = _merge(cfg, data)
        return cfg


def _merge(cfg: Config, data: dict) -> Config:
    for k, v in data.items():
        if not hasattr(cfg, k):
            continue
        cur = getattr(cfg, k)
        if isinstance(v, dict) and hasattr(cur, "__dataclass_fields__"):
            for kk, vv in v.items():
                if hasattr(cur, kk):
                    setattr(cur, kk, vv)
        else:
            setattr(cfg, k, v)
    return cfg
