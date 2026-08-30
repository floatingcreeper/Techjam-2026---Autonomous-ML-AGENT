# Archive Integration — What We Adopted and Why

Six features pulled from three teammates' archives — `archives/aerin`, `archives/jx`, and
`archives/jon` — **all now fully implemented** in the main codebase. For each: what it is, **why it
earns its place**, and the concrete changes made. Every claim was checked against the real source, not
[COMPARE.md](COMPARE.md)'s initial summary; where the two disagree, this doc says so — and for the
Jon set (4–6) the disagreement is only about *effort and emphasis*, since COMPARE.md's review of Jon
is itself accurate.

See also: [README.md](../README.md) (what the system is) · [DESIGN.md](DESIGN.md) (why it's built this
way) · [IMPLEMENTATION.md](IMPLEMENTATION.md) (code-level build specs) ·
[COMPARE.md](COMPARE.md) (the initial exploratory analysis this revises).

---

## Decision at a glance

Five adopts and one stretch, in two families: **capability & observability** (1–3, from aerin/jx) and
**safety, rigor & compute-efficiency** (4–6, from jon). The unifying property survives the expansion:
**none of the six touch the frozen harness** — even Jon's safety guards are implemented *around* the
frozen runner, never through it, so the trust boundary that makes every run's score credible is preserved.

| # | Feature | Source | Verdict | Value | Effort | Risk | Contract |
|---|---|---|---|---|---|---|---|
| 1 | Multi-task heads — Lever C | `aerin/sequence_ranker.py` | **Adopt** | High | Moderate | Low–Med | None |
| 2 | Hypothesis-ledger dashboard | `jx/hypothesis-ledger.html` | **Adopt** | High | Low–Mod | Low | None |
| 3 | Cross-run champion resume | `jx/agent/controller.py` | **Stretch** | Medium | Low | Low–Med | None |
| 4 | Multi-seed re-eval of the submission | `jon/agent/reeval.py` | **Adopt** | High | Low–Mod | Low | None |
| 5 | Debug-first sample gate | `jon/agent/debug_run.py` | **Adopt** | Med–High | Moderate | Low | None |
| 6 | Test-label data guard | `jon/agent/data_guard.py` | **Adopt** | Medium | Low | Low | None |

---

## Where they attach

The system is two layers behind a hash-pinned trust boundary. Every proposed change lands *outside* that
boundary — in the agent, the solution space, or a read-only satellite viewer. Nothing above the dashed
line changes.

```
╔══════════════════════════════════════════════════════════════════╗
║  🔒 FROZEN HARNESS — hash-pinned, never edited                     ║
║     data.py · evaluate.py · submit.py · run_node.py · contracts.py ║
╚══════════════════════════════════════════════════════════════════╝
 ------------------------------ trust boundary ----------------------
   AGENT  (best-first tree search)
     orchestrator · roles · memory → run_log.jsonl · executor · datced (cache)
        └─ [F1] cache aux labels
        └─ [F3] persist + resume champion
        └─ [F4] multi-seed re-eval of the submission (finalize)
        └─ [F5] debug sample gate before full runs (executor)
        └─ [F6] withhold test labels from blocks (load_bundle)
   SOLUTION SPACE  (six blocks + lib)
     baseline_blocks · din_blocks · lgbm_blocks · lib/din.py
        └─ [F1] Lever C in din_blocks
   OBSERVABILITY  (read-only satellite)
        └─ [F2] dashboard reads run_log.jsonl
```

All six live below the trust boundary — even the safety guards (4–6) work *around* the frozen runner,
never through it.

---

## Feature 1 — Multi-task auxiliary heads (build Lever C)

**Verdict: Adopt.** Source: `archives/aerin/sequence_ranker.py`.

**What it is.** Add auxiliary supervision to the existing DIN model. The shared DIN trunk grows a second
head that predicts a user's *other* reactions to the target video — `is_click`, `is_like`, `is_follow`,
`is_comment` — while the primary head still predicts `long_view`. Training minimises
`primary_loss + Σ wₖ·bce(auxₖ)` on top of the DIN's existing BPR/BCE objective. This is Aerin's
`DINMultiTaskRanker` idea, implemented on our rails rather than copied. An optional extension carries
feedback-type-aware history embeddings (encode *how* the user reacted to each past video, not just which
video).

**Why adopt — the reasons:**

1. **It's the only Aerin idea that adds capability we don't already have.** Verified in source: the BPR
   sampler and single-task DIN both already exist in the main repo, and in more-vectorised form — so
   multi-task is the one non-redundant contribution.
2. **It fills a hole the codebase itself has already dug.** `Cfg` reserves `aux_tasks` / `aux_weights` /
   `mtl_arch`; `FeatureSet` reserves an `aux` field; `datced.py`'s own docstring lists "aux labels" as a
   planned cache extension; and the Proposer's system prompt *already lists* "C (multi-task)" as a lever
   it may propose. Four independent placeholders point at one missing implementation.
3. **It buys ensemble diversity, which is what actually wins here.** The Phase-3 rank-blend already relies
   on model-family diversity to beat any single learner. Auxiliary signals regularise the shared ID
   embeddings differently from the pure-ranking models, producing exactly the kind of complementary errors
   the blend exploits.
4. **Zero inference/submission impact.** Auxiliary heads supervise during training only; the primary head
   alone scores at inference, so nothing downstream of the model changes.

> **Honest caveat.** Aerin reports no headline score for the DIN-MTL and ran it small
> (`embedding_dim=8, epochs=5`). So this is a *hypothesis added to the search space*, not a guaranteed
> lift — which is precisely how the agent is designed to treat a new block: it tunes and adopts it
> empirically, or the ablation shows it doesn't help. Adding the lever is the win; proving the gain is the
> agent's job.

**Concrete changes — all agent-owned, contract frozen:**

| Where | Change |
|---|---|
| `pipeline/lib/aux_build.py` *(new)* | Read the aux columns straight from the raw logs — the frozen `data.load()` drops them — applying identical file-order + date-split logic so arrays stay row-aligned to the base cache. Write `{split}_aux.npy` (N×K). Aerin's `_iter_public_rows` is the working template for this read. |
| `pipeline/lib/seq_build.py` *(optional)* | In the existing history loop, also emit `{split}_seqfb.npy` — a per-step feedback code — for the optional feedback-aware history embedding. |
| `agent/datced.py` | Call `aux_build.build(...)` beside `gbm`/`seq_build`; bump `CACHE_VERSION` 4→5 (one rebuild). Add an alignment guard: assert per-split row counts equal `sizes[name]` and cross-check raw ids against the cached `{split}_vid.npy` / `_u.npy`. |
| `pipeline/lib/din.py` | `DIN.__init__` gains an optional aux head; `forward` returns `(primary, aux_logits)`; `fit_din` adds the weighted aux-BCE term, keeping the existing BPR/BCE primary path intact. |
| `pipeline/lib/din_blocks/features.py` | Populate `FeatureSet(..., aux={split:{task:arr}})` from the new cache — mirrors exactly how it already loads `seq`. |
| `pipeline/lib/din_blocks/model.py` | Pass `cfg.aux_tasks` into `DIN` so the head is built only when aux is requested. |
| `agent/roles/proposer.py` | One line noting Lever C is now adoptable (via `adopt_blockset:"din"` + `aux_tasks` config) so the agent actually reaches for it. |

**Scope guard.** Implement `mtl_arch="shared"` only (Aerin's approach). Leave `mmoe` / `ple` as future
work — don't over-build a routing architecture before the shared-trunk version has shown signal.

**Spec:** Effort Moderate (~5 files) · main risk = cache row-alignment · contract untouched · rollback via
config flag / block-set.

---

## Feature 2 — The hypothesis-ledger dashboard

**Verdict: Adopt.** Source: `archives/jx/agent-recsys/hypothesis-ledger.html`.

**What it is.** JX's single-file, fully client-side `hypothesis-ledger.html` — KPI cards, per-iteration
charts, a sortable metrics table, and an expandable hypothesis log — retargeted to our `run_log.jsonl`
schema and made tree-search-aware. No server, no build step, no new runtime dependency.

**Why adopt — the reasons:**

1. **Pure upside, zero duplication.** Today we emit only raw JSONL plus a static `results.md`. A
   client-side viewer is a large observability and presentation gain for no backend — it keeps the kit's
   lightweight, offline character.
2. **Tree searches are unreadable from raw logs.** A visual of the best-so-far trajectory, per-node spend,
   and branch structure tells a human at a glance whether the agent is converging, stalling, or burning
   budget on failed recoveries. That's direct Presentation-rubric value and a real debugging aid.
3. **It's already a finished, careful artifact** — theme-aware, accessible, with a File-System-Access
   "reload" flow. We adapt it, we don't build a dashboard from scratch.

**But it's more than COMPARE.md's "map two fields."** JX's viewer assumes a *linear* single-path run; ours
is a branching tree. The honest adaptation:

| Where | Change |
|---|---|
| Schema remap (≈6 fields) | `iteration`→`iter`; nested `metrics.valid.*`→flat `metrics.primary_valid` / `.GAUC` / `["nDCG@5"]`; `resource_usage.llm_input_tokens`→`cost.input_tokens`; top-level `wall_clock_s`→`cost.wall_clock_s`; status vocabulary `{ok,failed}`→`{root,improved,no_gain,abandoned,duplicate}`; no `timestamp` (sort falls back to insertion order — fine for us). |
| Drop the Test columns | We never score test during iteration (only at finalize), so JX's entire per-iteration Test half is dead for us and should be removed rather than rendered empty. |
| Make it tree-aware *(the real work)* | Records carry `parent_id` / `node_id` / `phase` / `lever`, and best-first jumps between branches — so a raw per-iteration line zig-zags. Replace it with a **best-so-far envelope**, and colour/group points by `lever` (or `parent_id`) so branches read as branches. Add a per-lever view — we already compute `ablation_best_by_lever`. |
| Offline + provider fixes | **Vendor Chart.js inline** — JX loads it from cdnjs, which breaks the charts with no network, against a kit that prides itself on deterministic offline runs. And fix the hardcoded Claude Sonnet `$2/$10` pricing to Gemini (our `LLM.provider`), or make the rate an input field. |

**Spec:** Effort Low–Moderate (1 file) · the real chunk is the tree-aware chart · read-only, zero coupling
to the run · contract untouched.

---

## Feature 3 — Cross-run champion resume

**Verdict: Stretch.** Source: `archives/jx/agent-recsys/agent/controller.py`.

**What it is.** Persist the best *validated* node across separate `agent.run` invocations, so a later run
resumes from the high-water mark instead of starting cold. Adapted to our tree: JX's version overwrites a
single mutable pipeline directory, whereas ours keeps the FM-reproduction root (the harness self-check) and
additionally seeds the prior champion into the tree as an expandable node.

**Why adopt — the reasons:**

1. **A real capability we lack.** Each run today mints a fresh `run_id` and a fresh tree, rediscovering
   gains from scratch. A long or interrupted hackathon session throws away its own progress.
2. **It hardens a guarantee the design already prizes.** The best-checkpoint invariant becomes *durable*
   rather than per-run — the validated high-water mark now survives the process.
3. **It fits cleanly without weakening any invariant.** Root still reproduces FM (the self-check stays);
   the champion is loaded *in addition*, as a node best-first can build on, and `finalize` updates it when
   a run beats it.

**Concrete changes:**

| Where | Change |
|---|---|
| `runs/_champion/` *(new)* | A stable snapshot: the champion's `blocks/`, `cfg.json`, and a `champion.json` (score, source run, iter, timestamp) — mirroring JX's `best/_best_meta.json`. |
| `agent/orchestrator.py` | After the root node, if a valid champion exists, materialise it as a node (parent = root) and `tree.add` it with a "resumed champion" log record. In `finalize`, overwrite the champion when `final_valid` beats it. |
| `agent/config.py` | A `resume: bool` / `champion_dir` knob (default on). |
| Staleness guard | Re-validate the champion under the current `CACHE_VERSION` before trusting its stored score — a cache change can move numbers. On mismatch, re-run it once or discard. |

**Why it's a stretch, not tier-1.** It only pays off across multiple sessions, it adds a little
persistent-state surface, and the re-validation step is needed to keep scores honest across cache changes.
Worth doing — after the higher-value items.

**Spec:** Effort Low · main risk = score staleness · payoff is multi-session only · contract untouched.

---

# Safety, rigor & compute-efficiency (from `archives/jon`)

The next three are a different *kind* of feature — not new capability, but **guards on an agent powerful
enough to fool itself**: statistical noise, wasted compute on doomed code, and accidental test-set access.
Two points of rigor before the details:

- **COMPARE.md's review of Jon (§5) is accurate** — verified claim-by-claim against source (`action_space.py`
  really does only execute `set_hyperparam`; `data_guard.py` really does `splits.pop('test')`; `reeval.py`
  really does require a seed-mean to beat best by ε; the loop really is a greedy linear chain, not a tree).
  This is a marked contrast with its Aerin sections. So the disagreement below is about **effort and
  emphasis**, not correctness.
- **Two of the three interact with our frozen `run_node.py`.** Jon calls his model in-process, so a sample
  gate or a test-strip is a one-liner for him. Ours runs blocks in a subprocess against a mmap'd cache
  behind a hash-pinned runner — so these guards must be built *around* the frozen boundary, which is more
  than the "just modify executor.py / wrap the loader" the proposal implies. None of them edit a frozen
  file; all stay agent-owned.

---

## Feature 4 — Multi-seed re-eval of the submission

**Verdict: Adopt** (the highest-value item of the Jon set). Source: `archives/jon/agent/reeval.py`.

**What it is.** Before committing a candidate as the final submission (and optionally as a new global best),
re-run it across 2–3 seeds and decide on the **seed-mean**, not a single lucky seed — Jon requires
`mean > best + ε`. Ports cleanly: our blocks already key on `cfg.seed`.

**Why adopt — the reasons:**

1. **Our decision thresholds sit *inside* the noise.** Per-seed FM std ≈ `0.0008` (`baseline_scores.json`);
   our adoption delta is `pv > parent + 1e-9` (`orchestrator._iterate`) and our convergence `eps` is
   `0.0002` (`config.py`) — both **far below** the noise floor. We currently accept "improvements" we can't
   distinguish from luck. This is the single strongest rigor finding in the whole doc, and COMPARE.md
   under-sells it.
2. **Our final pick is a max over ~50 noisy draws.** `tree.best()` returns the argmax single-seed valid
   primary across all nodes — textbook selection bias / winner's curse: the reported best is upward-biased
   and may not generalise to test. A seed-mean confirmation is the direct antidote.
3. **Cheapest high-value placement is at finalize.** Re-run the chosen best (and each ensemble member)
   across seeds and select the submission on the mean — this protects the *actually-scored artifact*
   without tripling search cost.
4. **It anchors the "trustworthy autonomy" story.** Guarding against seed-hacking is exactly the kind of
   robustness the rubric rewards, and a known failure mode of autonomous ML pipelines.

**Concrete changes:**

| Where | Change |
|---|---|
| `agent/reeval.py` *(new, ported)* | `recheck(run_fn, cfg, *, seeds, epsilon, current_best) -> (accept, mean, per_seed)`. Mirror Jon's **toggle** (default on) and **short-circuit** (skip the extra seeds if the single seed doesn't already beat best — extra seeds cost ~a full run each). |
| `agent/orchestrator.py` (`finalize` / `assemble`) | Before writing the submission, re-run `tree.best()` (and each ensemble member) under 2 extra seeds via `executor.run_node` with seed-overridden cfgs; select on the seed-mean valid primary; record per-seed scores in `resource_report.json`. |
| `agent/orchestrator.py` (optional) | Also gate global-best *updates* during search on a multi-seed mean — but finalize-time is the high-EV first step; do it first. |
| `agent/config.py` | `recheck: bool` (default on), `recheck_seeds` (e.g. `(1, 2)`); reuse `budget.eps` as ε. |

> **Caveat.** Extra seeds are real compute. Keep it toggle + short-circuit like Jon, and prefer
> finalize-time over per-iteration so cost stays bounded to the handful of nodes that actually matter.

**Spec:** Effort Low–Moderate · main cost = a few extra training runs (bounded) · contract untouched ·
highest value-for-us of the Jon set.

---

## Feature 5 — Debug-first sample gate

**Verdict: Adopt** (selective). Source: `archives/jon/agent/debug_run.py`.

**What it is.** Before a full training run, validate the candidate on a small fixed sample (Jon: 20k train
rows, 2 epochs) with a shallow sanity check — no NaN/Inf, metrics in `[0,1]`. A failing sample routes
straight to the Reflector instead of burning full-run compute.

**Why adopt — the reasons:**

1. **Real gap.** Our `executor.run_node` runs the *full* node with only a static gate (`check_imports`,
   syntax + import allowlist). Runtime failures — shape errors, NaNs — aren't caught until the full run.
2. **Value is concentrated on the expensive nodes.** DIN/torch trains ~80s/epoch on CPU; a crashing DIN
   node can waste minutes under the 900s timeout. A 1–2s sample gate kills those fast.
3. **It fits our Reflector recovery.** A failed sample feeds its traceback to `_recover` immediately — the
   same path a real failure already takes.

**But it's more than "modify executor.py":**

| Where | Change |
|---|---|
| `pipeline/debug_cache.py` *(new)* | Build a **row-consistent** subsample of the cache (`X`/`y`/`u`/`vid` **and** the gbm + seq sibling arrays) into a temp dir. This is the real work — our cache is a multi-array mmap; Jon's in-process FM has none of it. |
| `agent/executor.py` | `debug_gate(blocks, cfg)`: call the **existing frozen runner** with `--cache <debug> --cfg <capped-epochs>` (can't add a `--debug-sample` flag — `run_node.py` is hash-pinned). On failure return a `Failure` for `_recover`. |
| `agent/orchestrator.py` | In `_iterate`, run `debug_gate` before the full `run_node` — **gated on `model_type`** (torch/`din`/`bst` only). A debug pass taxes every *successful* node too, plus our subprocess+cache-load overhead, so cheap FM nodes should skip it. |
| `agent/config.py` | `debug_gate: bool`, `debug_train_n`, `debug_epochs`, and the model-type allowlist. |

> **Caveat.** A small sample under-exercises vocab-size / UNK-rate bugs that only appear at real scale
> (Jon flags this in his own docstring) — it's a crash/sanity gate, not a quality gate. And don't gate the
> cheap nodes, or the gate costs more than it saves.

**Spec:** Effort Moderate (the debug cache is the work) · value concentrated on torch nodes · works
*around* the frozen runner (contract untouched) · make it selective.

---

## Feature 6 — Test-label data guard

**Verdict: Adopt** (integrity guarantee; low urgency). Source: `archives/jon/agent/data_guard.py`.

**What it is.** Physically withhold the hidden-test *labels* from agent iterations, so no agent-written
block can read `bundle.y["test"]`.

**Why adopt — the reasons:**

1. **Real gap — and worse than for Jon/JX.** Our `datced.load_bundle` loads *all* splits including test, so
   `bundle.X["test"]` / `bundle.y["test"]` are handed to every block. Jon (`splits.pop('test')`) and JX both
   strip test; we don't. A hallucinated or adversarial block *could* read test labels.
2. **Cleaner and cheaper for us than Jon's version.** Tracing every consumer: **`test_y` is never
   legitimately read** — `run_node` scores valid only, and finalize's test path does *inference only*
   (`infer(..., "test")`) with the submission written from `test_u` / `test_vid`, never labels. So the
   minimal correct guard is to withhold **`test_y` specifically**, keeping test features + ids for the
   legitimate submission path.
3. **It strengthens the trust-boundary/autonomy story the repo already sells** — a *physical* guarantee, not
   a prompt instruction, and its value rises as we widen the agent's code latitude (Features 1 and beyond).

**Concrete changes:**

| Where | Change |
|---|---|
| `agent/datced.py` | `load_bundle` omits `y["test"]` (and, optionally, all of test) unless an env flag is set; `build_or_load` can simply stop caching `test_y`. `load_bundle` must tolerate the absent array. |
| `agent/orchestrator.py` | Set that env flag on the finalize `run_node(..., extra_split="test")` subprocess **only** — finalize scores test through the same frozen runner, so this is how the legitimate path still gets test *features* while iteration nodes never see test at all. |

> **Caveat.** Low urgency today — no current block reads test, and the Proposer is told data is frozen. Do
> it as a cheap, strictly-correct integrity guarantee, not a fire. (We must *never* score test locally — it's
> the hidden grader's job — so withholding `test_y` can never break a legitimate path.)

**Spec:** Effort Low · strictly correct · low urgency today · works around the frozen runner (contract
untouched).

---

## What we're explicitly not taking

Scope discipline is part of the proposal. These were considered and rejected — several of them are things
COMPARE.md recommends, which we disagree with after reading the actual source.

- **Aerin's `IntraUserPairSampler` + NumPy BPR gradients.** Our `train_np._fit_pair` is already fully
  vectorised (flat negative pool, no per-user Python loop); Aerin's loops over users every epoch. Adopting
  it is a lateral move at best. *(COMPARE.md recommends this; we disagree — verified in source.)*
- **Aerin's DIN as a parallel `aerin_din_blocks/`.** Redundant with the existing `din_blocks` (0.6031),
  off-contract, and unproven. We take its one new idea — multi-task — into the existing block set instead.
  *(COMPARE.md frames this as a wholesale adopt; we don't.)*
- **Aerin's `research_agent.py` (hardcoded linear REGISTRY) and `blend_experiment.py`.** Strictly inferior
  to, and redundant with, our orchestrator and its Phase-3 `assemble()`.
- **JX's linear whole-file-rewrite controller / sandbox.** Well-engineered, but architecturally a
  single-path rewriter — inferior to our tree search over a block contract. We take its dashboard and its
  resume idea, not its loop.
- **Jon's constrained `action_space` + greedy linear chain.** His agent can only `set_hyperparam` on a
  single hand-rolled FM and never branches — a far smaller solution space than our freeform block-editing
  tree search. We take his safety mechanisms (4–6), not his loop or action space.

---

## Recommended sequencing

Ordered by value-for-effort and natural pairing, across all six:

1. **Feature 2 — dashboard.** Self-contained, low-risk, immediately useful, and it makes every subsequent
   run easier to read while we build the rest.
2. **Feature 4 — multi-seed submission re-eval.** Cheap-ish and the highest-integrity item — it guards the
   *actually-scored artifact* against the seed noise our thresholds currently sit inside.
3. **Feature 1 — Lever C.** The real modeling upside. Do the cache row-alignment carefully and add the
   assertion guard before trusting any aux-trained score.
4. **Feature 5 — debug-first gate.** Pairs naturally with Feature 1: the DIN/torch nodes it protects are
   exactly the expensive ones Lever C introduces more of.
5. **Feature 6 — test-label guard.** Cheap integrity guarantee; low urgency, so fold it in whenever the
   `datced` cache is next touched (e.g. alongside Feature 1's `CACHE_VERSION` bump).
6. **Feature 3 — champion resume.** A stretch; take it only if runs will actually span multiple sessions.

One line ties all six together: **none of them cross the trust boundary.** Features 1/4/5/6 extend
agent-owned code and the cache, Feature 2 only reads the run log, Feature 3 only persists a validated
snapshot — and Jon's safety guards (4–6) are built *around* the frozen runner, never through it. The
hash-pinned harness — `data.py`, `evaluate.py`, `submit.py`, `run_node.py`, `contracts.py` — stays
byte-for-byte identical, so every run's score remains trustworthy.
