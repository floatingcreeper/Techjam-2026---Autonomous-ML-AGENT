# Results

**Honest estimate: 0.60363 ± 0.00138** (portfolio_cv, portfolio ['n1', 'root'] (w=[0.75, 0.25]))

Tuned/in-sample validation primary: 0.60376 (optimistic — the blend weights were tuned on these same users)

| | FM baseline | agent (honest) | agent (tuned) |
|---|---|---|---|
| primary | 0.6015 | 0.60363 | 0.60376 |
| delta | — | +0.00213 | +0.00226 |

## Benchmark contract

- Stop reason: **proposal_guard**
- Official convergence rule: eps=0.002, N=3 (source: baseline_scores.json) — satisfied: **False**
- Executed experiments: 2 / 50 (hard cap)
- Proposal attempts: 27 (12 rejected before training — duplicates, structural no-ops, invalid configs, unsupported capabilities)
- Wall-clock: 156s / 21600s backstop

Resource usage: {'input_tokens': 2320, 'output_tokens': 1740, 'wall_clock_s': 145.3, 'iters': 15}
