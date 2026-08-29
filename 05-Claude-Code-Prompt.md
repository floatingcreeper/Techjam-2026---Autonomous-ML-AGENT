# Claude Code Prompt — Integrate R&D-Agent Strategies into Our Autonomous ML Agent

Copy everything below the line into Claude Code as your first message in the repo.

---

## Context

I'm building an autonomous ML research agent for a hackathon (TikTok TechJam PS2). The agent must run the ML-engineering loop — read problem, inspect data, engineer features, train/tune, evaluate, reflect, repeat — **on its own**, on a recommender-systems benchmark (KuaiRand-Pure: 1.4M interactions, 27K users × 7.6K items), until it beats an official baseline on NDCG@10 and Recall@50. Full constraints:

- No external training data — only the provided KuaiRand splits.
- The agent develops on train + validation only; it never sees the hidden test set.
- A run is "converged" when validation score hasn't improved by more than ε over the last N iterations, or the compute budget is hit.
- We are scored on: (1) the delta over baseline at convergence, (2) how few manual interventions it took, (3) total LLM tokens + GPU-hours spent, (4) the quality/reasoning behind what the agent tried, and (5) whether it recovers from failures instead of crashing.
- We must produce a per-iteration log with: hypothesis, code diff, resulting metrics, and any error/recovery events — this is a required deliverable, not optional.

I want to bring in four specific design strategies from a paper we're using as a design reference: **Yang et al., "R&D-Agent: An LLM-Agent Framework Towards Autonomous Data Science," Microsoft Research/GenAI, 2025, arXiv:2505.14738.** That paper achieves SOTA on MLE-Bench with a 6-component framework; their own ablations show four of the six components matter most for a resource-constrained setting like ours. I'm asking for those four, not the full framework — the other two (a probabilistic cross-branch memory kernel, and full MCTS tree search) are more engineering than a hackathon timeline justifies, per their own ablation numbers.

## Related work — other architectures to weigh in your plan

R&D-Agent's own related-work section cites several other MLE-agent architectures. These are genuine design references (not just benchmarks) and directly relevant to the exploration-path-structuring decision (greedy vs. tree/chain vs. tree+MCTS) that R&D-Agent's ablation flagged as their single highest-impact component:

- **ML-Master** (Liu et al. 2025, arXiv:2506.16499) — tree search + MCTS. The closest open competitor to R&D-Agent (29.3% medal rate vs. R&D-Agent's 35.1%). If you think a lightweight branching structure is worth the complexity for us, this is the architecture to reference.
- **MLE-STAR** (Nam et al. 2025, arXiv:2506.15692) — **chain-based** (single-path, deep) search rather than tree-based. This is architecturally the closest match to a simple greedy loop, which is what I'd lean toward for a hackathon timeline — cite this if you recommend going the greedy route.
- **AIRA** (Toledo et al. 2025, arXiv:2507.02554) — another framework-level paper exploring the design space, useful for cross-checking whether R&D-Agent's four components we're borrowing are complete or missing something this paper considers important.
- **KompeteAI** (Kulibaba et al. 2025, arXiv:2508.10177) — emphasizes a fast iterative debug workflow specifically, which is a second, independent source validating strategy 1 below (debug-first-on-a-sample), not just R&D-Agent's word for it.
- **OpenHands** (Wang et al. 2024, arXiv:2407.16741) and **MLAB** (Huang et al. 2023, arXiv:2310.03302) — general-purpose/earlier baseline coding agents, scoring far lower than the specialized MLE agents above on MLE-Bench. Useful only as a "why not just use a generic coding agent" comparison point, not as an architecture source.

**Ask for your plan:** given these, make an explicit, justified recommendation on the greedy-vs-chain-vs-tree decision (currently an open `[TEAM DECIDE]` item), citing whichever of the above best supports the choice, and note what we give up by not doing full tree+MCTS like ML-Master.

## Your task

**Do not start writing implementation code yet.** First:

1. **Explore the current repo state.** Read whatever code, configs, notebooks, and docs already exist here. Summarize in a few sentences: what's already built (data loading? baseline reproduction? any agent loop scaffold?), what's stubbed, and what's entirely missing.
2. **Merge that understanding with the plan below** and produce a written implementation plan — a concrete, ordered task list with file paths, not prose essays. Flag anywhere the strategy below conflicts with or duplicates something that already exists in the repo, and propose how to reconcile it.
3. **Ask me clarifying questions** only where the repo state genuinely leaves something ambiguous (e.g., which LLM API is already wired up, what the current action space is, whether there's an existing logging module). Don't ask about things covered below.
4. Stop after the plan. I'll review it before you touch code.

## The four strategies to integrate

### 1. Debug-first coding workflow (highest priority — implement first)
Before any full training run, the agent must validate its own code on a small sample first:
- Sample ~10% of the training split (or a fixed small N, whichever is more appropriate for KuaiRand-Pure's scale — use your judgment and note the trade-off).
- Run the full pipeline (features → train → eval) on that sample with reduced epochs/iterations.
- Time it, and use that to estimate full-run duration before committing to the full run.
- Only proceed to the full-scale run if the sampled run executes cleanly and produces a plausible metric (not NaN, not a crashed process, not an obviously malformed output).
- If the sampled run fails, this is where the error→repair loop kicks in (see strategy 4) — don't burn a full training run's compute on code that doesn't even execute.

Design question for you to resolve in the plan: where does this slot into our existing loop structure (if one exists), or what's the minimal new module needed?

### 2. Structured reasoning/hypothesis pipeline (implement second — this doubles as our required run-log content)
Every iteration's "propose" step should force the LLM through explicit stages, not a single unstructured "improve the model" prompt:
- **Identify the core problem**: given current results and data characteristics, what's the most likely bottleneck right now? (e.g., "post-click signals are sparse," "no sequence features yet," "embedding dim may be too small for this sparsity level")
- **Form a specific, testable hypothesis**: not "try a better model" but "adding a 20-item user history sequence feature should improve NDCG@10 because ranking quality depends on recent intent."
- **State why**: the reasoning connecting the hypothesis to the identified problem.
- **Output an implementable action**: a concrete code-level change, not just an idea.

This structured output should map directly onto the required run-log schema: `{hypothesis, target_stage, reasoning, expected_effect}`. Design the prompt template and the parsing/validation of its output (reject and retry if the LLM doesn't return all four fields).

### 3. Time-aware dynamic planning (implement third — one config-level change)
The agent's behavior should shift as the run progresses:
- **Early iterations**: bias toward cheap, fast, novel ideas — small config changes, single features, no ensembling, no expensive techniques. Goal is breadth and quick signal.
- **Later iterations**: once a direction is clearly working, allow more expensive techniques (larger models, ensembling, more careful hyperparameter search) and bias toward refining the current best rather than exploring new directions.
- Implement this as a simple function of `elapsed_time / total_budget` (or `iteration / expected_total_iterations` if wall-clock budget isn't confirmed yet) that adjusts either (a) which prompt variant is used, or (b) a "budget tier" parameter passed into the propose step that constrains what kinds of actions are allowed.

Keep this simple — a single threshold or two, not a complex schedule. Note in the plan where the threshold(s) should live so they're easy to tune later.

### 4. Simplified aggregated evaluation + error/recovery (implement alongside whatever loop exists)
- **Recovery**: when a proposed change fails (crashes, times out, produces invalid output), feed the error back to the LLM, ask for a fix, cap retry attempts (e.g. 2–3), and if still failing, roll back to the last known-good state and let the next iteration try a different hypothesis rather than looping forever on the same broken idea. This must be logged as an error/recovery event per our schema.
- **Aggregated evaluation, simplified**: we don't need multi-branch merging — but before accepting a new "best" checkpoint, don't trust a single validation pass blindly if compute allows; note in the plan whether re-checking the top 1–2 candidates before finalizing convergence is worth the extra compute for us, given our budget is currently unconfirmed. Make this configurable/toggleable rather than a hard requirement, since our compute budget may not allow it.

## Explicitly out of scope — do not implement
- Any cross-branch/multi-trace memory sharing mechanism (their probabilistic interaction kernel). Use a flat, append-only history/log instead — a list of past `{hypothesis, diff, metrics}` fed into the propose prompt's context is sufficient.
- Full tree search or MCTS-style parallel branch exploration. If a branching structure doesn't already exist in the repo, default to a simple greedy loop (always extend the current best) unless the repo already has branching support worth preserving — flag this as a decision point rather than assuming.
- Retrieval-augmented generation / external knowledge bases. Not appropriate here — flag if you see anything RAG-like already in the repo and ask whether to keep it.

## Also make sure the plan covers (these aren't from the paper, they're our own hard requirements)
- Every LLM call must go through one wrapped client that logs input/output tokens — this feeds our required cost report.
- Every iteration must produce one structured log entry (JSONL) with: hypothesis, code diff, resulting metrics, error/recovery events, token cost, GPU time.
- A manual-intervention counter that a human increments explicitly — don't infer this automatically.
- The hidden test split must be physically unreachable by any code path the agent can execute.
- Crash-safe resume: if the process dies mid-run, we must be able to resume from the last completed iteration, not restart from zero.

## Output format for this first response
1. Repo summary (what exists / what's missing) — a few bullet points, not exhaustive.
2. A phased task list (e.g. Phase 1: debug-first workflow, Phase 2: reasoning pipeline, ...) with specific file paths to create or modify.
3. Explicit list of clarifying questions, if any remain after reading the repo.
4. Explicit list of decisions you made unilaterally and why, so I can veto any of them.

Do not write implementation code in this first response — plan only.
