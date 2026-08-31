"""Cross-run evidence ledger + ResearchInsight -- docs/EN/SYSTEM.md §14 (cross-run ledger).

Why this exists, measured
-------------------------
`Memory` is constructed per run directory and every run starts empty, so the agent can only ever see
one run's worth of evidence. docs/EN/RESEARCH.md §8 measured the cost: the reference run's single most confident
conclusion ("multi-task DIN helps, +0.00082") is contradicted when the 8 historical runs that contain
BOTH arms are pooled -- adding aux heads to DIN+BPR is worse in **8 of 8 runs**, paired mean -0.00037,
SE 0.00009, t = -4.16. The agent produced the strongest result in its own history by accident and
could not see it.

Compatibility, not just accumulation
------------------------------------
Pooling incompatible results is worse than not pooling. Every entry records `cache_version`,
`code_state`, the effective family, the block digest and the effective config, and `compatible()`
refuses to pool across a cache or code change.

What repetition means here
--------------------------
Repeated trainings evaluated on the SAME validation users are NOT independent datasets. They estimate
training stochasticity, not validation-sample uncertainty and not hidden-test generalisation. The
three are kept in separate fields and separately labelled everywhere they surface.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------- storage
def load(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def append_run(path, run_dir, tree, cache_version):
    """Append every EXECUTED experiment of this run to the durable ledger."""
    from agent import provenance
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cs = provenance.code_state()
    run_id = Path(run_dir).name
    n = 0
    with p.open("a", encoding="utf-8") as fh:
        for node in tree.nodes.values():
            if node.metrics is None or node.status in ("duplicate", "rejected_proposal"):
                continue
            fh.write(json.dumps({
                "run_id": run_id, "node_id": node.id, "parent_id": node.parent,
                "cache_version": int(cache_version), "code_state": cs,
                "family": node.cfg.model_type, "lever": node.lever,
                "loss_type": node.cfg.loss_type,
                "aux_tasks": list(node.cfg.aux_tasks or ()),
                "seed": int(node.cfg.seed),
                "blocks_digest": (node.provenance or {}).get("blocks_digest", ""),
                "cfg_hash": node.cfg.hash(),
                "arm": arm_key(node),
                "primary_valid": node.score(),
                "GAUC": node.gauc(), "nDCG@5": node.ndcg(),
                "primary_rand": (node.metrics or {}).get("primary_rand"),
                "status": node.status, "noop_class": node.noop_class,
                "evidence": node.evidence or {}, "portfolio": node.portfolio or {},
                "hypothesis": node.hypothesis,
            }, default=float) + "\n")
            n += 1
    return n


def arm_key(node) -> str:
    """The experimental ARM -- what makes two nodes the same experiment for pooling purposes.

    Deliberately excludes `seed`: two seeds of one arm are repetitions, not distinct arms.
    """
    aux = ",".join(sorted(node.cfg.aux_tasks or ())) or "-"
    return f"{node.cfg.model_type}|{node.cfg.loss_type}|aux={aux}"


def compatible(entry, cache_version, code_state) -> bool:
    return (int(entry.get("cache_version", -1)) == int(cache_version)
            and entry.get("code_state") == code_state)


# ---------------------------------------------------------------------------- insight
@dataclass
class ResearchInsight:
    """One evidence-graded scientific claim, generated rule-based from the ledger (no LLM call)."""
    claim: str
    scope: str                       # the conditions under which it was established
    control: str
    treatment: str
    delta_primary: float | None = None
    delta_GAUC: float | None = None
    delta_nDCG: float | None = None
    paired_se: float | None = None            # across training repetitions, NOT bootstrap
    p_gt0: float | None = None                # validation-sample uncertainty, from the bootstrap
    training_repetitions: int = 0
    n_paired_runs: int = 0
    counterevidence: list = field(default_factory=list)
    confidence: str = "inconclusive"
    next_test: str = ""

    def to_dict(self):
        return asdict(self)


def paired_arms(entries, cache_version, code_state, control_arm, treatment_arm):
    """Within-run paired comparison of two arms across the ledger.

    Pairing is what gave docs/EN/RESEARCH.md §8 its result: it removes confounding by code state, hardware and run
    conditions, which is exactly what makes unpaired pooling of historical rows uninterpretable.
    """
    usable = [e for e in entries if compatible(e, cache_version, code_state)]
    by_run: dict[str, dict[str, list]] = {}
    for e in usable:
        by_run.setdefault(e["run_id"], {}).setdefault(e["arm"], []).append(e)
    pairs = []
    for run_id, arms in by_run.items():
        if control_arm in arms and treatment_arm in arms:
            c = statistics.fmean(x["primary_valid"] for x in arms[control_arm])
            t = statistics.fmean(x["primary_valid"] for x in arms[treatment_arm])
            pairs.append((run_id, c, t, t - c))
    return pairs


def insight_from_pairs(pairs, control_arm, treatment_arm, scope="") -> ResearchInsight | None:
    if len(pairs) < 2:
        return None
    d = [p[3] for p in pairs]
    mean = statistics.fmean(d)
    sd = statistics.stdev(d)
    se = sd / (len(d) ** 0.5) if len(d) > 1 else float("nan")
    neg = sum(1 for x in d if x < 0)
    direction = "improves" if mean > 0 else "degrades"
    if se and se == se and abs(mean) > 2.5 * se:
        conf = "strong"
    elif se and se == se and abs(mean) > 1.5 * se:
        conf = "moderate"
    else:
        conf = "weak"
    return ResearchInsight(
        claim=f"{treatment_arm} {direction} on {control_arm} by {mean:+.5f} primary",
        scope=scope or f"paired within-run comparison, {len(pairs)} runs",
        control=control_arm, treatment=treatment_arm,
        delta_primary=round(mean, 6), paired_se=round(se, 6) if se == se else None,
        n_paired_runs=len(pairs), confidence=conf,
        counterevidence=[f"{p[0]}: {p[3]:+.5f}" for p in pairs if (p[3] > 0) != (mean > 0)],
        next_test=("isolate the individual auxiliary tasks" if "aux=" in treatment_arm
                   else "repeat under the other objective to test the interaction"),
    )


def summarise(entries, cache_version, code_state, max_arms: int = 12) -> dict:
    """Compact cross-run evidence for the Proposer context. Compatible entries only."""
    usable = [e for e in entries if compatible(e, cache_version, code_state)]
    if not usable:
        return {"compatible_entries": 0, "arms": {}, "note":
                f"{len(entries)} ledger entries exist but none match the current cache/code state; "
                f"they are deliberately NOT pooled."}
    arms: dict[str, list] = {}
    for e in usable:
        if e.get("noop_class") in ("STRUCTURAL_NOOP", "EXACT_NOOP"):
            continue
        arms.setdefault(e["arm"], []).append(e["primary_valid"])
    out = {}
    for arm, vals in sorted(arms.items(), key=lambda kv: -statistics.fmean(kv[1]))[:max_arms]:
        out[arm] = {"n": len(vals), "mean": round(statistics.fmean(vals), 6),
                    "sd": round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
                    "min": round(min(vals), 6), "max": round(max(vals), 6)}
    return {"compatible_entries": len(usable), "total_entries": len(entries), "arms": out}
