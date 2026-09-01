# Results

**Honest estimate: 0.60419 ± 0.00159** (portfolio_cv, portfolio ['n1', 'n3', 'n4', 'root'] (w=[0.25, 0.25, 0.25, 0.25]))

Tuned/in-sample validation primary: 0.60458 (optimistic — the blend weights were tuned on these same users)

| | FM baseline | agent (honest) | agent (tuned) |
|---|---|---|---|
| primary | 0.6015 | 0.60419 | 0.60458 |
| delta | — | +0.00269 | +0.00308 |

## Benchmark contract

- Stop reason: **official_convergence**
- Official convergence rule: eps=0.002, N=3 (source: baseline_scores.json) — satisfied: **True**
- Executed experiments: 4 / 50 (hard cap)
- Proposal attempts: 5 (1 rejected before training — duplicates, structural no-ops, invalid configs, unsupported capabilities)
- Wall-clock: 210s / 21600s backstop

Resource usage: {'input_tokens': 560, 'output_tokens': 420, 'wall_clock_s': 198.4, 'iters': 5}
