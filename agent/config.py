"""Single source of truth for loop-wide config constants — do not duplicate these numbers
anywhere else (06-Master-Prompts_1.md's propose prompt explicitly calls this out for
convergence_N/convergence_epsilon). Values match this repo's established convention in
baseline_scores.json's own convergence_rule, and fm_official's std_over_5_seeds (~0.0008) is why
CONVERGENCE_EPSILON isn't set any tighter than 0.002 — see AGENT_STRATEGY.md's Stopping condition.
"""

CONVERGENCE_N = 3
CONVERGENCE_EPSILON = 0.002
