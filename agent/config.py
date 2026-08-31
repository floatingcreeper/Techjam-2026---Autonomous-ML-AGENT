"""Central configuration.

Defaults live here; agent/config.yaml (optional) overrides them. Kept dependency-light:
if pyyaml is present and a yaml file is given, it is merged over the defaults.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Budget:
    """The OFFICIAL benchmark contract. Nothing in this dataclass is a target to be consumed.

    `eps` and `N` are the organizer's convergence rule, read verbatim from
    `baseline_scores.json -> convergence_rule` ({"epsilon": 0.002, "N": 3}).

    The repository previously shipped `N = 6`. the pre-consolidation design notes records that as a deliberate
    2A call ("do NOT inflate to 15-20"), but the effect was still a deviation in the LOOSENING
    direction: N=6 requires 7 scored nodes without a 0.002 improvement before converging, N=3 requires
    4. `docs/PROBLEM_STATEMENT.pdf` is an image-only scan and cannot arbitrate, so the JSON artifact
    is authoritative. Restored to 3. See docs/EN/SYSTEM.md §16.

    `max_iter` is a HARD CAP on executed experiments and `wall_clock_hours` a HARD BACKSTOP -- neither
    is something the agent should try to spend. Convergence is expected to fire first.
    """
    max_iter: int = 50                 # hard cap on EXECUTED experiments (not proposal attempts)
    wall_clock_hours: float = 6.0      # hard backstop
    per_iter_timeout_s: int = 900
    eps: float = 0.002                 # OFFICIAL convergence epsilon
    N: int = 3                         # OFFICIAL convergence window
    adopt_eps: float = 0.001           # tree-shape margin only; never a convergence criterion


@dataclass
class Research:
    """Internal research bookkeeping. MUST NOT postpone official convergence (docs/EN/SYSTEM.md §16).

    These knobs change WHAT is proposed and how evidence is graded. The only one that can end a run is
    `proposal_guard_limit`, and that is a liveness guard against an LLM that cannot emit anything
    executable -- not a competing definition of convergence.
    """
    # statistical adoption (docs/EN/SYSTEM.md §13) -- paired user-level bootstrap on saved predictions, no retraining
    bootstrap_B: int = 1000
    adopt_p: float = 0.90              # P(delta>0) to call an effect confirmed
    promising_p: float = 0.60          # lower edge of "promising"
    ens_eps: float = 0.0002            # EMC above this counts as research information
    diversity_corr: float = 0.90       # rank_corr below this counts as new ranking diversity
    # proposal hygiene (docs/EN/SYSTEM.md §12)
    max_reproposals: int = 3           # retries after a duplicate / no-op / invalid proposal
    proposal_guard_limit: int = 12     # abort only a pathological proposal loop
    # plateau escalation (docs/EN/SYSTEM.md §16) -- changes proposal policy, never the stopping rule
    plateau_after: int = 2             # consecutive uninformative experiments before escalating
    explore_p_escalated: float = 0.40
    # portfolio (docs/EN/SYSTEM.md §15)
    cv_folds: int = 5
    max_members: int = 4
    weight_step: float = 0.25
    # integrity (docs/EN/SYSTEM.md §8) -- a valid primary this high is a bug or a leak, never a model
    leak_tripwire_primary: float = 0.70


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
    usd_per_1k_input: float = 0.0
    usd_per_1k_output: float = 0.0


@dataclass
class Config:
    data_dir: str = "./KuaiRand-Pure/data"
    cache_dir: str = "runs/_cache"
    runs_dir: str = "runs"
    seed: int = 0
    gpu: str = "auto"                 # auto | on | off
    # F5 -- debug-first sample gate (torch nodes)
    debug_gate: bool = True
    debug_train_n: int = 20000
    debug_other_n: int = 10000
    debug_epochs: int = 2
    # F4 -- multi-seed re-eval. Reserved for STOCHASTIC families only (fm/lgbm have std 0.00000 at a
    # fixed seed, so re-seeding them measures nothing a paired bootstrap does not measure for free).
    recheck: bool = True
    recheck_seeds: tuple = (1, 2)
    recheck_top_k: int = 3
    # F3 -- cross-run champion resume
    resume: bool = False
    champion_dir: str = "runs/_champion"
    # cross-run evidence ledger (docs/EN/SYSTEM.md §14)
    ledger_path: str = "runs/_ledger.jsonl"
    use_ledger: bool = True
    # 3B -- Lever E unbiased-exposure eval, reported as a SECOND surface (docs/EN/RESEARCH.md §15)
    unbiased_eval: bool = False
    # structured event stream for the Research Console (docs/EN/SYSTEM.md §19)
    events: bool = True
    budget: Budget = field(default_factory=Budget)
    research: Research = field(default_factory=Research)
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
