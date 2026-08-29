# Design & Rationale

Why the system is built the way it is. The [README](../README.md) covers *what* each file does
and *how* they connect; [MATH.md](MATH.md) covers the derivations; this doc covers the
**decisions** — the reasoning, the research it draws on, and the alternatives rejected.

---

## Optimizing the rubric

The single most important realization: this track scores an **agent**, not a model. Reading the
judging weights ([PROBLEM_STATEMENT.pdf](PROBLEM_STATEMENT.pdf)):

| Criterion | Weight | Who earns it |
|---|--:|---|
| Technical Execution | 35% | the score delta over baseline **and** robustness (recover, never crash/stall/diverge) |
| Innovation & Problem Insight | 20% | *what* the agent targeted and *why*; originality |
| Impact & Relevance | 20% | **Autonomy** — measured by the number of manual interventions |
| Feasibility & Practicality | 15% | tokens + wall-clock, **scored only among submissions that beat the baseline** |
| Presentation & Communication | 10% | final event |

Roughly **40 % of the score is agent behavior** (Autonomy + the robustness part of Technical
Execution + Feasibility), not model accuracy. Three consequences drove the whole design:

1. **The agent is at least as important as the model.** So we invested in the loop — best-first
   search, a failure taxonomy with recovery, a self-documenting run-log — as much as in the levers.
2. **Feasibility is gated behind beating the baseline.** An agent that stops after three cheap
   iterations to look efficient scores *worst* (it isn't scored at all until it clears the gate). So
   the agent must beat FM first, then be efficient — which killed any temptation to under-explore.
3. **The submission is the converged validation-best, and improvement need not be monotonic.** So the
   best-checkpoint is tracked and protected independently of the search path; the agent can take
   risky, non-monotonic bets and the submission never degrades.

**Design principles that follow:** hypothesis-first (every iteration logs a hypothesis + rationale —
this *is* the Innovation evidence and the search heuristic); a fixed trust boundary (below); a
best-checkpoint invariant; exploit the metric's structure; recover, don't crash; and one artifact
(the run-log) that is memory *and* deliverable.

---

## The two-layer thesis

We separate **what** the agent searches from **how** it searches.

- **Layer 1 — the solution space** (`pipeline/`): a small, contract-bound *pipeline* of six blocks
  (features / model / loss / train / infer / ensemble). The space of models is the space of block
  implementations. This is the "what."
- **Layer 2 — the agent** (`agent/`): a best-first tree search whose operators are LLM roles
  (Proposer / Coder / Reflector). This is the "how."

Keeping them separate means the agent's creativity is confined to well-defined, contract-bound
extension points, and the search machinery is model-agnostic — the same orchestrator drives an FM, a
LightGBM, or a DIN without knowing anything about them.

---

## The trust boundary

**Decision:** the data loader, the metric (`evaluate.py`), the submission checker, and the node runner
are **frozen** (SHA-256-pinned in `frozen.lock`); the agent owns **only the six block function
bodies**. This was chosen over the two obvious alternatives:

- *Maximal autonomy* (the agent writes the whole pipeline from scratch each run, à la AIDE/MLE-STAR):
  best "autonomy" optics, but far more failure modes, more tokens, and it lets the agent accidentally
  (or via a hallucinated "fix") alter the code the score depends on. Rejected for a competition where
  robustness and a trustworthy score matter.
- *Fully guided* (a human pipeline + the agent tweaks configs): safest, but the Autonomy criterion
  scores the *number of manual interventions* — a hand-held pipeline scores poorly there.

The middle path — a fixed, trusted harness plus agent-owned modeling code — maximizes the autonomy
that is actually judged while shrinking the failure surface. Guardrails make it enforceable: a static
import allowlist gates every agent-written block *before* it runs, and the frozen-file hash check
aborts the run if anything downstream of the boundary changed.

---

## The research context (what we borrowed)

The agent synthesizes ideas from the current autonomous-ML-agent literature, adapted to this rubric:

- **AIDE** (Weco AI, *AI-Driven Exploration in the Space of Code*) — framing ML engineering as a
  **tree search in the space of code**, with a Generator / Evaluator / Selector. Our node =
  runnable-solution and best-first selection come from here.
- **MLE-STAR** (Google, NeurIPS 2025) — the current SOTA; **retrieve a strong model, then refine one
  code block at a time guided by ablation, and adaptively ensemble.** Our block-level edits (not
  whole-file rewrites), the adopt-a-model-family move, and the Phase-3 ensemble are MLE-STAR-shaped.
- **AI-Scientist-v2** (Sakana) — a **best-first tree search** managed by an experiment-manager agent;
  our orchestrator-as-policy / roles-as-operators split mirrors this.
- **RD-Agent** (Microsoft) — the **hypothesis → implement → validate → feedback** loop; every
  iteration here is exactly that, and the loop is the unit of both progress and logging.
- **SELA / I-MCTS** — MCTS over pipeline configs; we deliberately did *not* follow this (next section).

RecSys methods for the levers: **BPR** (Rendle et al.) as an AUC surrogate; **softmax cross-entropy**
for LTR (Bruch et al. 2019) as an nDCG surrogate; **DIN** (Zhou et al.) target-attention over behavior;
**DeepFM** for the wide+deep memorization; **LightGBM LambdaRank**; and **IPS/SNIPS** for the planned
debiasing (Lever E).

---

## Why best-first, not MCTS

SELA and I-MCTS use Monte-Carlo tree search. We use **best-first + an ε exploration valve** instead,
because the binding constraint here is the **evaluation budget** (50 iterations / 6 h): each node
costs a real training run. MCTS spends evaluations on rollouts and backups to estimate values that,
with only ~50 total evaluations, never become reliable. Best-first exploitation (expand the current
best) plus a small random-exploration valve (escape local optima) plus **ablation-guided refinement**
(MLE-STAR) extracts more signal per evaluation — which is the scarce resource. We state this trade-off
explicitly rather than defaulting to the fanciest search.

---

## The loss↔metric insight (and what actually happened)

The organizers' strongest hint is that the baseline optimizes **pointwise logloss** but is scored on
**ranking** metrics. The clean thesis: since `primary = ½(GAUC + nDCG@5)`, and

- **BPR** is a smooth **AUC** surrogate (→ the GAUC half), and
- **softmax-CE / LambdaLoss** is an **nDCG** surrogate (→ the nDCG@5 half),

the aligned objective is a blend of the two, computed within each user's group. This is the highest-EV,
cheapest lever — only the loss changes on a proven backbone.

**What the experiments showed (and this honesty matters):** BPR wins (`0.6015 → 0.6036`), but
**softmax-CE loses** (0.5997, below baseline) — it overfits the high-cardinality ID embeddings on this
data. So the thesis is *half* right here: the pairwise surrogate helps, the listwise one overfits. This
is a genuine finding the agent surfaces, not a story we forced — and it is exactly the kind of insight
the Innovation criterion rewards. (Full argument in [MATH.md §3](MATH.md#3-ranking-losses-lever-a).)

The second big finding: a **pure-sequence DIN underperforms FM** (0.5895) because attention over video
history has no user identity; adding a **DeepFM part** (user×item memorization) lifts it to 0.6031.
Sequences help, but only *on top of* the FM signal, not instead of it.

---

## The lever design

The levers are ordered by expected value ÷ cost, and the search is seeded (via the Proposer's system
prompt and the scripted mock) to try the high-EV ones first:

| Lever | Idea | Why this priority |
|---|---|---|
| **A — loss alignment** | BPR / softmax-CE | cheapest (loss-only change on a proven backbone), directly attacks the metric mismatch |
| **B — sequences (DIN)** | attention over history + DeepFM | highest ceiling (the organizers' blank direction); more expensive (torch, GPU-preferred) |
| **D — model family** | LightGBM LambdaRank | complementary tree bias; individually weak but strong ensemble diversity |
| **F — ensemble** | rank-blend diverse families | compounds small complementary gains; the finisher |
| **C — multi-task**, **E — debias** | aux heads; unbiased-log guard | planned; `Cfg` fields reserved, not yet built |

**Dead-ends encoded as hard "don'ts"** (the organizers proved these don't help, so the Proposer is told
not to waste iterations on them): adding static user-side features, and raising the embedding size `k`
for its own sake. Pure user-side features are constant within a user and carry no within-user ranking
signal — a fact that also shapes the LightGBM feature set (item-centric).

---

## Node = blocks + cfg; config-vs-code mutations

**Decision:** a node is a full snapshot of the six block source files plus a `Cfg`, executed by a fixed
runner. Two mutation kinds keep it cheap and safe:

- a **config mutation** changes `Cfg` values only (near-zero tokens) — used for sweeps (`neg_ratio`,
  `lr`, `L`, `k`);
- a **block edit** rewrites exactly one block body (the Coder), gated by a static import/syntax check;
- a **block-set adoption** swaps in a pre-built model family (`lgbm`, `din`) wholesale — the realistic
  "adopt a known method" move.

Full snapshots (rather than diffs) make every node independently runnable and reproducible, and let two
nodes run concurrently without collision. Preferring config over code (MLE-STAR's granularity, taken
further) is what keeps token and wall-clock cost low — directly serving the Feasibility criterion.

---

## Alternatives considered / rejected

- **MCTS over configs** — rejected: doesn't amortize at a 50-evaluation budget (above).
- **Whole-pipeline generation each run** — rejected: too many failure modes; violates the trust
  boundary (above).
- **Live web search for models each iteration** (MLE-STAR) — deferred: a pre-curated lever "playbook"
  in the Proposer prompt is cheaper and more reliable, and encodes the organizer dead-ends so no
  iteration is wasted. Live search remains a drop-in fallback.
- **A single blended loss** (α·BPR + (1−α)·softmax-CE) — deprioritized once softmax-CE proved to
  overfit; the diversity is captured better by *ensembling* a BPR-FM with a DIN and a LightGBM than by
  blending two FM losses.
- **Trae (ByteDance) as the driver** — Trae is an IDE, not an API; the driver is the **Gemini API**
  behind a swappable `LLMDriver` interface (model ids live in `agent/config.py`).

---

## References

**Agents:** AIDE (arXiv:2502.13138) · MLE-STAR (arXiv:2506.15692, NeurIPS 2025) · AI-Scientist-v2
(arXiv:2504.08066) · RD-Agent (Microsoft) · SELA (arXiv:2410.17238).
**RecSys:** BPR (Rendle et al. 2009) · Softmax-CE for LTR (Bruch et al. 2019) · DIN (Zhou et al. 2018) ·
DeepFM (Guo et al. 2017) · LightGBM (Ke et al. 2017) · IPS/SNIPS counterfactual learning ·
KuaiRand (arXiv:2208.08696).

---

*Back to the [README](../README.md) · math in [MATH.md](MATH.md).*
