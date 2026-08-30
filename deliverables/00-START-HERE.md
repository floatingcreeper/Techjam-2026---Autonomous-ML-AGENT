# Start here

Everything a teammate or judge needs, in one folder. **Read in this order.**

---

## 1. The headline

We beat the official baseline.

| | valid | test |
|---|---|---|
| `fm_official` — the official baseline | 0.6016 | 0.5946 |
| **our submission** — `fm_bpr` × 5-seed ensemble | **0.6039** | **0.5974** |

**+0.0028 test primary.** Verified across 5 seeds and validated by the official checker.

**The submission file is** [`submission/submission_ens_test.csv`](submission/submission_ens_test.csv).

**The honest asterisk:** the winning model was hand-built by reading the agent's logs. The agent's
own search never beat the model it was seeded with. Details in [`RESULTS.md`](RESULTS.md) — please
read that section before presenting, so nobody overclaims on camera.

---

## 2. Reading order

| Order | File | Why |
|---|---|---|
| 1 | [`RESULTS.md`](RESULTS.md) | The numbers, how to verify them, and what's real vs. noise |
| 2 | [`../README.md`](../README.md) | The project + full agent architecture. The main document. |
| 3 | [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) | Shot-by-shot demo video script with pre-flight checklist |
| 4 | [`evidence/SNAPSHOT.md`](evidence/SNAPSHOT.md) | What the run-log evidence is and when it was taken |

---

## 3. What's in this folder

```
deliverables/
├── 00-START-HERE.md      you are here
├── RESULTS.md            scores, verification commands, honest caveats
├── DEMO_SCRIPT.md        demo video script (has a pre-flight section — read it first)
├── submission/
│   ├── submission_ens_test.csv       ← THE SUBMISSION (fm_bpr × 5 seeds)
│   ├── submission_test.csv           comparison: fm_bpr, single model
│   └── submission_test_fm_v1.csv     comparison: the baseline, retrained in our environment
└── evidence/
    ├── SNAPSHOT.md               when this evidence was taken, and whether mid-run
    ├── experiment_log.jsonl      ★ REQUIRED DELIVERABLE — one record per iteration
    ├── solution_tree.json        the search tree: every node, status, score, parent
    ├── token_ledger.jsonl        every LLM call with tokens and latency
    ├── cost_report.txt           totals: calls, tokens, GPU-hours, by caller
    ├── manual_interventions.jsonl human interventions during autonomous operation
    ├── state.json                crash-safe resume state
    └── dashboard.html            open in a browser — human-readable run viewer
```

`evidence/` holds **point-in-time copies**. The live files stay in `runs/` because a running loop
writes to them continuously. Refresh the copies any time with:

```bash
python refresh_deliverables.py
```

Do this immediately before packaging or recording — and check `SNAPSHOT.md` says the loop was
*not* running, or you're shipping a half-finished log.

---

## 4. Verify it yourself — two commands

```bash
# Score our submission against the test labels
PYTHONIOENCODING=utf-8 python submit.py --score deliverables/submission/submission_ens_test.csv --split test
#   expect: GAUC 0.6644 | nDCG@5 0.5304 | primary 0.5974

# Reproduce the whole thing from scratch (~5 min, CPU only)
python ensemble_submission.py --seeds 5 --split test --out /tmp/check.csv
```

On Windows the `PYTHONIOENCODING=utf-8` prefix is needed — `submit.py` prints a `✓` that crashes
cp1252 consoles *after* validation has already succeeded.

---

## 5. Open items — not done, needs a human

Ordered by how much damage they do if ignored.

1. **Nobody has checked this against the actual judging rubric.** The README and this folder are
   structured around the five criteria written in `05-Claude-Code-Prompt.md` — which is our own
   prompt draft, not a primary source. **Someone must diff this against the official brief.**
2. **Possible metric mismatch.** `05-Claude-Code-Prompt.md` says the target is *NDCG@10 and
   Recall@50*. The actual kit (`evaluate.py`, `baseline_scores.json`, `submit.py`) scores
   *mean(GAUC, nDCG@5)*, and calls itself the fixed contract. We optimised the second. If the
   first is right, every number in this folder is against the wrong metric. **Verify this first.**
3. **The demo video does not exist yet.** `DEMO_SCRIPT.md` is written and ready to shoot.
4. **The intervention count is understated.** `manual_interventions.jsonl` records 1. The true
   number is far higher — we debugged the loop, rewrote its prompts, and hand-built the winning
   model. Report the honest figure; a judge who spots the gap discounts everything else.
5. **One clean autonomous run would strengthen the log.** The current `experiment_log.jsonl` is
   fragmented across restarts and hand-fixes. Even 10 unbroken iterations is better evidence for
   the autonomy and recovery criteria.
