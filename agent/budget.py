"""Time-aware dynamic planning (AGENT_STRATEGY.md strategy 3) — PARTIAL, built early as a Phase 2
dependency: 06-Master-Prompts_1.md's propose prompt needs `budget_tier_instruction` filled in on
every call, so the minimum viable piece of Phase 3 has to exist before Phase 2 can run at all.
Still TODO for full Phase 3: threading a tier into agent/action_space.py's cost constraints (once
that module exists) and a real wall_clock_tier() once Q3 (rough iteration/time budget) is answered.

Three tiers by budget_fraction (0-100). Thresholds and wording match 06-Master-Prompts_1.md's own
comment exactly — if you change one, change the other, they must never drift apart.
"""

EARLY_STAGE = (
    "EARLY STAGE: prioritize cheap, fast, novel ideas. Prefer single, isolated changes "
    "(one feature, one hyperparameter, one architectural tweak) over compound changes. "
    "Do NOT propose ensembling, extensive cross-validation, or other expensive "
    "techniques yet. Goal is breadth: quickly learn which directions have signal."
)

MID_STAGE = (
    "MID STAGE: you should have signal on what works by now. Prioritize refining and "
    "combining the strongest directions from history rather than exploring entirely new "
    "ones, unless nothing so far has produced a meaningful gain."
)

LATE_STAGE = (
    "LATE STAGE: focus on squeezing out remaining gains from the current best approach. "
    "Now is the appropriate time for more expensive techniques (ensembling, careful "
    "hyperparameter search, multi-task refinements) if compute allows. Avoid starting "
    "any large new architectural direction this late."
)


def budget_tier_instruction(budget_fraction):
    """budget_fraction: 0-100. Returns the instruction text block for the propose prompt's
    BUDGET TIER section — this IS the "prompt-variant selection" mechanism from
    AGENT_STRATEGY.md §3, just implemented as one injected block inside a single template rather
    than three separate template files (matches 06-Master-Prompts_1.md's actual design)."""
    if budget_fraction < 40:
        return EARLY_STAGE
    if budget_fraction < 75:
        return MID_STAGE
    return LATE_STAGE


def iteration_budget_fraction(iteration_number, expected_total_iterations):
    """Fallback used until Q3 (rough iteration/time budget) is answered and a real wall-clock
    budget can be wired in — iteration-count-based, per AGENT_STRATEGY.md §3's stated default."""
    if not expected_total_iterations:
        return 0.0
    return 100.0 * min(iteration_number / expected_total_iterations, 1.0)


def wall_clock_tier(elapsed_s, budget_s):
    """Wall-clock version of budget_tier_instruction's threshold logic — folded in from Phase 3
    per the decision to fold its remainder into Phase 4 rather than keep it a separate phase. The
    math itself needs nothing else once a real budget exists; it's just not CALLED anywhere in the
    loop yet, since Q3 (rough iteration/time budget) is still unconfirmed and
    agent/hypothesis_agent.py uses iteration_budget_fraction() as the fallback. Swap the call site
    once Q3 is answered — this function is already correct and ready."""
    if not budget_s:
        return budget_tier_instruction(0.0)
    return budget_tier_instruction(100.0 * min(elapsed_s / budget_s, 1.0))
