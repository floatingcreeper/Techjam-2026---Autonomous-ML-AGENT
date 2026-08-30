> [!NOTE]
> This document is an **initial exploratory analysis** of teammate codebases. The refined integration
> decisions — including what was adopted, what was rejected, and why — are in
> [INTEGRATION.md](INTEGRATION.md). The code-level build specs for all adopted features are in
> [IMPLEMENTATION.md](IMPLEMENTATION.md). Where this document's proposals conflict with
> INTEGRATION.md, **INTEGRATION.md is authoritative**.

# Codebase Architecture Comparison & Integration Guide

This document provides AI agents (like Claude Code) with a preliminary architectural understanding of
three teammate codebases (`archives/aerin`, `archives/jx`, `archives/jon`) and our main
`kuairand-starter-kit` implementation. For the authoritative integration decisions, see
[INTEGRATION.md](INTEGRATION.md).

## 1. Aerin's Codebase Review

Aerin's codebase (`archives/aerin`) takes a heavily Machine Learning-centric approach. Instead of building an open-ended, LLM-driven code-writing agent, it functions as a **guarded public-validation experiment controller**.

**Key Architectural Features:**
*   **Deterministic Pipeline Execution:** Uses `research_agent.py` to loop through a hardcoded `REGISTRY` of predefined `Experiment` objects (e.g., `bpr_fm`, `fm_din_mtl_seed1`, `rank_blend`).
*   **Hand-Optimized ML Implementations:**
    *   `bpr_fm.py`: A Factorization Machine trained with Pairwise BPR loss, implemented entirely in NumPy. It features a highly optimized `IntraUserPairSampler` and manual gradient descent step calculations.
    *   `sequence_ranker.py`: A PyTorch-based Causal DIN-style Multi-task ranker that incorporates sequence behavior and optimizes for multiple auxiliary targets (click, like, follow, comment).
*   **Search Strategy:** Linear execution of hypotheses. It evaluates each experiment and implements a strict early-stopping convergence rule (`CONVERGENCE_N = 3` failures to improve by `EPSILON = 0.002`).

**Summary:** Aerin's codebase shines in its rigorous, handcrafted ML engineering but lacks the autonomy and dynamic search space exploration of an LLM-driven agent.

## 2. Aerin's Codebase Integration Proposal

The main implementation is a highly autonomous, hypothesis-driven best-first tree search. To maximize its potential, we should supply it with the best possible "blocks" to mutate and combine.

**Architectural Changes:**
*   **Adopt Aerin's BPR-FM Sampler:** Integrate Aerin's `IntraUserPairSampler` and NumPy gradient optimizations from `bpr_fm.py` into our `pipeline/lib/train_np.py`.
*   **Adopt Aerin's Multi-Task DIN Ranker:** Convert `sequence_ranker.py` into a fully adoptable block-set (e.g., `pipeline/lib/aerin_din_blocks/`). Ensure it adheres to the strict contract signatures defined in `pipeline/contracts.py`.

**Decision Making Rationale:**
*   Our current LLM agent dynamically generates code, but highly specific algorithmic optimizations (like vectorized NumPy pair sampling or multi-task PyTorch attention heads) are difficult for an LLM to invent from scratch in a single shot.
*   By exposing Aerin's handcrafted implementations as "block sets", the LLM orchestrator can freely adopt them, tune their hyperparameters (via `Cfg` deltas), and ensemble them with other models, drastically raising the ceiling of the solution space.

---

## 3. JX's Codebase Review

JX's codebase (`archives/jx`) focuses heavily on system wrapper design, single-path execution, and most notably, an elegant offline UI/dashboarding system.

**Key Architectural Features:**
*   **Monolithic File Replacement:** Instead of modifying modular pipeline blocks, JX's agent proposes full-file rewrites of `data.py` or `baseline.py`.
*   **Single-Path Execution Loop:** The loop follows a rigid Propose → Apply → Run → Score → Adopt-or-Discard sequence. It evaluates $N$ candidates per iteration, picks the best, and moves forward linearly. Regressions are strictly reverted.
*   **Zero-Dependency UI (`hypothesis-ledger.html`):** The crown jewel of JX's implementation. A standalone HTML file that acts as an interactive dashboard. You drag and drop the agent's `iteration_log.jsonl` into the browser (or connect it via the File System Access API), and it parses the JSON entirely client-side to render Chart.js graphs (Validation Primary vs. Iteration, GAUC vs. nDCG, Wall-clock, and Spend).

**Summary:** JX's architecture is a robust but rigid linear wrapper. However, its client-side logging dashboard represents a massive UX/DX upgrade over reading raw JSON lines.

## 4. JX's Codebase Integration Proposal

Our current implementation (`orchestrator.py`) already writes an extremely detailed `run_log.jsonl` that tracks hypotheses, configuration deltas, metrics, events, and token costs. Integrating JX's UI system is a low-effort, high-reward architectural enhancement.

**Architectural Changes:**
*   **Drop-in the Dashboard:** Copy `hypothesis-ledger.html` into the root of our repository.
*   **Schema Mapping:** Modify the JavaScript parsing logic inside `hypothesis-ledger.html` to map to our `run_log.jsonl` schema. 
    *   *Examples:* Map `r.metrics.valid.primary` to `r.metrics.primary_valid`; map `r.resource_usage.llm_input_tokens` to `r.cost.input_tokens`.
*   **Tree-Search Visual Adaptations:** Since our agent uses a **best-first tree search** (jumping between branches of the search tree), a standard linear line chart for validation scores will look like a zigzag. The charting logic must be updated to plot a `best-so-far` envelope line, or color-code nodes by their `parent_id` to visually represent the search tree branches.

**Decision Making Rationale:**
*   **Zero Backend Cost:** The HTML dashboard requires no web server, no React/Node.js dependencies, and no infrastructure. It maintains the lightweight nature of the starter kit.
*   **Observability:** Tree-search agents are notoriously difficult to debug from raw logs. A dashboard that visualizes cost burn-rate, time spent on failed recoveries, and the trajectory of the `best-so-far` metric gives human researchers instant insight into whether the agent is converging, stalling, or hallucinating.

---

## 5. Jonathan's Codebase Review

Jonathan's codebase (`archives/jon`) is a highly engineered, deterministic Python-based LLM agent inspired directly by recent academic papers (MLE-STAR, R&D-Agent). 

**Key Architectural Features:**
*   **Constrained Action Space:** Unlike our main implementation which allows freeform Python code generation, Jon restricts the LLM to a predefined `action_space.py`. The agent can currently only tune hyperparameters (e.g., `set_hyperparam` for `lr`, `batch_size`).
*   **Greedy Chain Search:** It strictly avoids tree search (MCTS) to save compute. It uses a single linear chain that only extends the current best solution.
*   **Debug-First Workflow (`debug_run.py`):** Before executing a full 40-epoch training run, it gates the execution with a 20,000-row sample. If the sample crashes, returns NaNs, or goes out of bounds, it instantly fails in 1 second and routes to error recovery, saving massive amounts of compute.
*   **Strict Data Guarding (`data_guard.py`):** It physically intercepts the dataset loader and deletes the `'test'` key from the dictionary before returning it to the agent, making it physically impossible for the agent to overfit to the hidden test set.
*   **Multi-Seed Re-evaluation (`reeval.py`):** Before adopting a new candidate as the global best, it re-runs the configuration across 2-3 different seeds. It requires the *mean* of those seeds to beat the current best, preventing statistical noise from hijacking the search direction.

**Summary:** Jonathan's architecture is an extremely safe, compute-efficient, and academically rigorous approach to agentic ML, sacrificing the high ceiling of freeform code generation for the reliability of constrained optimization.

## 6. Jonathan's Codebase Integration Proposal

While we want to keep our agent's ability to freely generate code, Jonathan's safety and compute-saving mechanisms are brilliant and should be ported over to our execution sandbox.

**Architectural Changes:**
*   **Adopt the Debug-First Sandbox:** Modify our `executor.py` to first run a fast-fail pass on a 20k-row sample (or 1 epoch) before running the full pipeline. If it fails, feed the traceback back to the Reflector immediately.
*   **Adopt the Data Guard:** Wrap our environment's data loader to physically strip out test-set labels during agent iterations.
*   **Adopt Multi-Seed Re-evaluation:** Update our orchestrator's node adoption logic. When a node beats the current best `primary` metric, trigger a multi-seed recheck. Only update the global best if the multi-seed mean also wins. 

**Decision Making Rationale:**
*   Our LLM tree search is powerful but prone to hallucinations and statistical noise. 
*   A debug-first workflow will save us hundreds of dollars in compute by instantly killing doomed code before it runs for 30 minutes. 
*   Multi-seed re-evaluation acts as a critical anchor against the agent "hacking" the validation set via lucky random seeds, which is a known flaw in autonomous ML pipelines.
