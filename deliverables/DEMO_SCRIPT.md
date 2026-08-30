# Demo Video Script — Autonomous ML Agent (TikTok TechJam 2026, PS2)

**Target length: 4:00.** Cut to 3:00 by dropping Scene 5 and shortening Scene 3.

---

## Pre-flight — do this BEFORE you hit record

The loop is too slow to demo live end-to-end. One iteration is 1–5 minutes, and seeding the
baseline alone takes ~60 s. **Pre-record the slow parts and cut to them.** Nothing below asks you
to wait on screen.

**1. Capture a loop run in advance** (background, while you prep everything else):

```bash
python -m agent.cli run --iterations 8 --ignore-convergence 2>&1 | tee demo_loop.log
```

You want a log containing at least: one `[hypothesis]`, one `[codegen] wrote`, one
`[debug_run] DebugResult(ok=True`, one `[decision]`, and — ideally — one `REVERT` from a failed
candidate. Scenes 4 and 5 are screen-captures of scrolling this log, not a live run.

**2. Regenerate the dashboard** so `runs/dashboard.html` is current:

```bash
python -m agent.viewer
```

**3. Have these open in tabs, pre-loaded:**

- Terminal (font size 16pt+ — judges watch on laptops)
- `runs/dashboard.html` in a browser
- `models/fm_bpr.py` scrolled to the `_step_pair` function
- `runs/experiment_log.jsonl` in an editor with word-wrap ON
- `README.md` architecture diagram

**4. Pre-run these so the output is instant on camera** (they're fast, but rehearse them):

```bash
python -m agent.cli status
python -m agent.cost_report
```

**5. Sanity-check the machine is quiet** — no background loop running, or your timings will lie:

```powershell
Get-Process python -ErrorAction SilentlyContinue
```

---

## Scene 1 — Cold open (0:00–0:20)

**Screen:** Terminal, cleared. Type and run:

```bash
python -m agent.cli status
```

**Narration:**

> "This is an autonomous ML agent. We gave it a recommender benchmark, a baseline to beat, and no
> further instructions. It writes its own hypotheses, writes its own model code, runs it, decides
> whether the result is real, and logs every step.
>
> It runs on a local model, on CPU, with numpy — no GPU, no ML framework, no autograd."

**On screen you should see:** current-best, the solution tree, and the cost report.

---

## Scene 2 — The task and the metric (0:20–0:45)

**Screen:** `README.md`, "The task" section. Highlight the metric line.

**Narration:**

> "KuaiRand-Pure — 1.4 million short-video interactions. For each user, rank the videos they were
> shown so the ones they actually watched to the end come first.
>
> The metric is the mean of GAUC and nDCG@5. Both are *within-user* rankings — they never compare
> one user against another. Remember that, because it turns out to be the whole story.
>
> Splits are by date, not shuffled: train on the past, evaluate on the future. Vocabularies are fit
> on train only. The agent never sees the test split — that's enforced structurally, not by
> convention."

**Screen at the last line:** jump to `agent/data_guard.py` — it's short enough to show whole.

---

## Scene 3 — Architecture (0:45–1:45)

**Screen:** the ASCII architecture diagram in `README.md`. Scroll slowly, or use a highlighter/
zoom as you name each stage.

**Narration:**

> "Every iteration runs the same pipeline.
>
> First, a **solution tree** decides what this iteration should even do — debug something broken,
> draft a new idea, or improve the current best. That's a chain-based greedy search, following
> MLE-STAR, rather than the full MCTS that ML-Master uses. We chose that deliberately for a
> hackathon compute budget.
>
> Then a **hypothesis agent** is forced through four stages — identify the bottleneck, state a
> testable hypothesis, justify it, and produce an implementable action. Never one unstructured
> 'improve the model' prompt.
>
> That goes down one of two paths. Cheap path: a **constrained action space** — validated
> hyperparameter changes only, no code risk. Expensive path: the **code generator** writes a
> complete new model module.
>
> Generated code is **statically analysed before it ever runs** — import allowlist, no file I/O, no
> eval, and it's rejected outright if it so much as mentions the test split.
>
> Then the key idea we took from R&D-Agent: **debug-first**. Before spending compute on 1.1 million
> rows, run on 20 thousand for two epochs. Does it crash? Is the number plausible? Did it actually
> score every row? Only then commit to the full run — and re-check the winner across multiple
> seeds before believing it."

---

## Scene 4 — One real iteration (1:45–2:40)

**Screen:** the pre-recorded `demo_loop.log`, scrolling. Pause on each stage as you name it.

**Narration:**

> "Here's a real iteration. The tree picks an operation. The hypothesis agent proposes — you can
> read its actual reasoning, not just its conclusion. The code generator writes a module to disk.
> Static analysis passes it. The debug run validates on the sample and estimates what the full run
> would cost. That estimate gates the expensive run.
>
> Then the decision step: does this beat the incumbent by more than the noise floor? Here it
> doesn't — so it's kept as a working node to build from, not promoted. That distinction matters:
> the agent is explicitly reasoning about whether a difference is real or just seed noise."

**Screen:** cut to `runs/experiment_log.jsonl`, one record expanded.

> "And every iteration writes this — hypothesis, code diff, metrics, error events, the decision and
> why, tokens spent, wall-clock. That's the required audit trail, and it's what the next iteration
> reads as history."

---

## Scene 5 — Failure recovery (2:40–3:10) *(cut this scene first if over time)*

**Screen:** a `REVERT` iteration in `demo_loop.log`, showing `error_events`.

**Narration:**

> "It fails constantly — that's expected, and it's designed for. Recovery happens at three levels.
>
> A bad candidate gets diagnosed, logged, marked buggy in the tree, and the loop moves on — and the
> diagnosis is attached to the node, so the next debug attempt sees the analysis, not just a
> traceback.
>
> A transient LLM timeout is retried with backoff.
>
> And if the whole process dies, state is written after every iteration, so a supervisor restarts
> it and it resumes from exactly where it stopped — losing at most the one iteration in flight."

**Screen:** flash `run_for_hours.ps1` and `agent/resume.py`.

---

## Scene 6 — Results, and what we actually learned (3:10–3:50)

**Screen:** the results table in `README.md`.

**Narration:**

> "We beat the official baseline: 0.5946 to 0.5974 on test.
>
> But the interesting part is *how*. The agent and we both spent a long time on feature
> engineering — item quality scores, target encoding, blending with a popularity model. All of it
> failed. One cheap offline sweep showed why: the factorization machine's video weight already *is*
> a learned per-video popularity score. We were feeding it information it had already extracted."

**Screen:** cut to `models/fm_bpr.py`, the `_step_pair` docstring.

> "What worked was changing the *objective*, not the inputs. The model was optimising pointwise
> log-loss — how likely is this row — while the metric only ever asks *within a user, is this video
> ranked above that one*. So we train on pairs from the same user instead. Zero new features. Same
> forward pass, same optimiser. Only the loss gradient changes.
>
> Every failed experiment is recorded in the code and fed back into the agent's own prompts — so
> neither we nor the loop rediscovers a dead end."

---

## Scene 7 — Close (3:50–4:00)

**Screen:** `python -m agent.cost_report` output.

**Narration:**

> "579 LLM calls, 1.3 million tokens, zero GPU-hours, one recorded manual intervention during
> autonomous operation.
>
> The honest limitation: the agent's own search never beat the model we hand-built from what its
> logs told us. Its real contribution was the audit trail — the negative results that told us where
> *not* to look."

---

## Delivery notes

- **Do not fake autonomy.** Judges score reasoning quality and failure recovery; a run that only
  shows successes reads as cherry-picked. Showing a `REVERT` and explaining the recovery is worth
  more than another green number.
- **Lead with the negative results.** "We proved three plausible ideas wrong and can show you why"
  is a stronger claim than a +0.003 delta, and it's the honest shape of the work.
- **State the delta in context.** 0.0028 sounds small until you say the baseline's own seed noise
  is 0.0008 and the whole hyperparameter space spans 0.0040. Say that out loud.
- Say **"local model, CPU, numpy only"** at least twice — zero GPU-hours is a scored criterion and
  it's easy for a viewer to miss.
- Keep the terminal at 16pt+ and don't scroll faster than you narrate.

## If you have 60 extra seconds

Show the crash-livelock bug as a war story: the tree referenced generated files that had been
archived away, `select()` kept returning the same dead node, and the supervisor restarted it eight
times in a row for zero iterations. It demonstrates real debugging of an autonomous system, and the
fix — prune dead code paths on load, and never let an import failure escape one iteration — is a
genuine design lesson about supervisors masking livelocks as crashes.
