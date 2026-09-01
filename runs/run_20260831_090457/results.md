# Results

**Honest estimate: 0.60409 ± 0.00141** (portfolio_cv, portfolio ['n1', 'n2', 'n3', 'root'] (w=[0.25, 0.25, 0.25, 0.25]))

Tuned/in-sample validation primary: 0.60463 (optimistic — the blend weights were tuned on these same users)

| | FM baseline | agent (honest) | agent (tuned) |
|---|---|---|---|
| primary | 0.6015 | 0.60409 | 0.60463 |
| delta | — | +0.00259 | +0.00313 |

## Benchmark contract

- Stop reason: **official_convergence**
- Official convergence rule: eps=0.002, N=3 (source: baseline_scores.json) — satisfied: **True**
- Executed experiments: 4 / 50 (hard cap)
- Proposal attempts: 5 (1 rejected before training — duplicates, structural no-ops, invalid configs, unsupported capabilities)
- Wall-clock: 450s / 21600s backstop

Resource usage: {'input_tokens': 11183, 'output_tokens': 1410, 'wall_clock_s': 255.1, 'iters': 5}
