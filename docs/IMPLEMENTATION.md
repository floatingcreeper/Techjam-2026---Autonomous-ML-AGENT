# Implementation Reference

Detailed, code-level build specs for the six features adopted from teammate archives, as described in
[INTEGRATION.md](INTEGRATION.md). **All six are now implemented.** This document is the *how*;
INTEGRATION.md is the *what & why*. For each feature: exact files, data flow, reference code, config,
verification steps, and edge cases.

> **Reference code, not a patch.** Snippets below are written against the verified current source
> (signatures, cache layout, contracts all checked) and are meant to be reviewed and tested, not
> pasted blind. Every path is real; `…` marks unchanged surrounding code.

See also: [README.md](../README.md) · [DESIGN.md](DESIGN.md) · [COMPARE.md](COMPARE.md).

---

## 0. Ground rules (read first)

**The trust boundary is absolute.** These five files are hash-pinned in `frozen.lock` and verified by
`guardrails.ensure_frozen()` at the top of every run — **do not edit them**:
`data.py`, `evaluate.py`, `submit.py`, `pipeline/run_node.py`, `pipeline/contracts.py`.
Every change below lives in **agent-owned** code (`agent/*`, `pipeline/lib/*`, `pipeline/*_blocks/*`,
new `pipeline/*.py`) or a standalone HTML file. If a step seems to need a frozen edit, it's the wrong
step — route *around* the frozen file (F5 and F6 show the pattern).

**`Cfg` already has the fields we need.** `pipeline/contracts.py` already declares `use_aux`,
`aux_tasks`, `aux_weights`, `mtl_arch` (for F1) and `seed` (for F4) — so F1 needs **no** contract
change. New *agent-side* knobs go in `agent/config.py`'s `Config`/`Budget` (not frozen).

**Testing harness (no LLM credits needed):**
- `python -m agent.run --smoke` → builds the cache, reproduces FM (`primary≈0.6015`), runs
  `ensure_frozen()`. This is the first gate after **any** cache or block change.
- `python -m agent.run --mock` → drives the whole loop with `MockDriver` replaying scripted moves
  from `tests/mock_moves.py`. Add a scripted move to exercise a new lever/path.
- `python -m agent.run --faults` → injects failures; use to verify F5's debug gate routes to recovery.

### Build order & dependencies

```
F2 dashboard ──(independent)────────────────────────────► ship anytime
F4 multi-seed ─(independent, finalize-only)─────────────► ship anytime
F1 Lever C ────(datced + CACHE_VERSION 4→5)─┐
                                            ├─► F5 debug-cache must include aux arrays
F6 test-guard ─(datced load-time only)──────┘   (do F1+F6 together — both edit datced.py)
F5 debug gate ─(needs full cache inventory; pairs with F1's torch nodes)
F3 champion ───(independent; touches orchestrator.run/finalize + config)
```

Recommended sequence (from INTEGRATION.md): **F2 → F4 → F1 → F5 → F6 → F3**. F1 and F6 both edit
`agent/datced.py`; land them in one change. F5's debug-cache inventory must be updated when F1 adds
the aux arrays.

### Cache coordination

| Feature | Cache impact | `CACHE_VERSION` bump? |
|---|---|---|
| F1 | Adds `cache/aux/{split}_aux.npy` + `_vid.npy` | **Yes, 4 → 5** (forces one rebuild) |
| F6 | Load-time only — `load_bundle` stops exposing `y["test"]` | No |
| F5 | Reads existing cache; builds a throwaway subsampled copy | No |
| F2, F3, F4 | None | No |

---

## Feature 1 — Multi-task auxiliary heads (Lever C)

**Goal.** Let the agent adopt a DIN that also predicts `is_click/like/follow/comment` as auxiliary
targets, via `adopt_blockset:"din"` + `config_delta {"aux_tasks":[...], "aux_weights":[...]}`.

**Files:** new `pipeline/lib/aux_build.py`; edit `agent/datced.py`, `pipeline/lib/din.py`,
`pipeline/lib/din_blocks/features.py`, `pipeline/lib/din_blocks/model.py`, `agent/roles/proposer.py`.
No frozen edits (`Cfg` already carries `aux_tasks/aux_weights/mtl_arch`; `adopt_blockset:"din"`
already routes to `din_blocks` in `mutate.materialize_child`).

**Data flow:** raw CSVs → `aux_build` caches per-row aux labels (aligned to `data.load` row order) →
`datced` builds it + asserts alignment → `din_blocks/features` puts them in `FeatureSet.aux` →
`din.fit_din` adds a weighted aux-BCE term on a shared-trunk aux head. Inference is unchanged (primary
head only), so no submission impact.

### Step 1 — `pipeline/lib/aux_build.py` (new)

The frozen `data.load()` drops the aux columns (it keeps only 7 fields per row), so we re-read the raw
logs. Alignment is guaranteed by mirroring `data.load`'s exact read order and date filter (imported
from the frozen `data.SPLITS`), then **asserted** in `datced` against the cached video ids.

```python
"""Lever C data — per-row auxiliary labels (is_click/like/follow/comment), cached.

The frozen data.load() exposes only long_view; these auxiliary targets live in the raw logs. We
reproduce data.load()'s file order + date filter EXACTLY so the arrays are row-aligned to the base
cache, then datced.build_or_load asserts that alignment against the cached video ids.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from data import SPLITS                      # frozen: date ranges only, safe to read

AUX_COLUMNS = {                              # task name -> raw CSV column (all binary in v1)
    "click": "is_click", "like": "is_like", "follow": "is_follow",
    "comment": "is_comment", "forward": "is_forward",
}
LOG_FILES = ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")


def build(data_dir, cache_dir, force=False):
    fc = Path(cache_dir) / "aux"
    if (fc / "meta.json").exists() and not force:
        return json.loads((fc / "meta.json").read_text())
    fc.mkdir(parents=True, exist_ok=True)
    tasks = list(AUX_COLUMNS)
    rows = {name: {"aux": [], "vid": []} for name in SPLITS}
    for fname in LOG_FILES:
        with open(Path(data_dir) / fname, newline="") as fh:
            for r in csv.DictReader(fh):
                date = int(r["date"])
                for name, (lo, hi) in SPLITS.items():
                    if lo <= date <= hi:                       # disjoint ranges -> exactly one split
                        rows[name]["aux"].append([1.0 if r[AUX_COLUMNS[t]] != "0" else 0.0
                                                  for t in tasks])
                        rows[name]["vid"].append(int(r["video_id"]))
                        break
    sizes = {}
    for name in SPLITS:
        A = np.asarray(rows[name]["aux"], np.float32).reshape(-1, len(tasks))
        np.save(fc / f"{name}_aux.npy", A)
        np.save(fc / f"{name}_vid.npy", np.asarray(rows[name]["vid"], np.int64))
        sizes[name] = len(A)
    meta = {"tasks": tasks, "sizes": sizes}
    (fc / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_aux(cache_dir, name):
    fc = Path(cache_dir) / "aux"
    meta = json.loads((fc / "meta.json").read_text())
    A = np.load(fc / f"{name}_aux.npy", mmap_mode="r")
    return {t: A[:, j] for j, t in enumerate(meta["tasks"])}
```

### Step 2 — `agent/datced.py` (edit)

Bump the version and build+verify aux beside the existing sibling caches.

```python
CACHE_VERSION = 5          # was 4 — adds cache/aux/*  (forces one rebuild)
…
def build_or_load(data_dir, cache_dir, force=False):
    …
    from pipeline.lib import seq_build
    seq_build.build(data_dir, str(cache), L=SEQ_L, force=True, splits=splits)
    # Lever C aux labels (re-reads raw logs; alignment asserted below)
    from pipeline.lib import aux_build
    aux_build.build(data_dir, str(cache), force=True)
    _assert_aux_aligned(str(cache))          # hard guard: aux rows == base rows, per split
    …

def _assert_aux_aligned(cache_dir):
    from pathlib import Path
    import numpy as np, json
    meta = json.loads((Path(cache_dir) / "aux" / "meta.json").read_text())
    for name in SPLITS:
        base_vid = np.load(Path(cache_dir) / f"{name}_vid.npy")
        aux_vid = np.load(Path(cache_dir) / "aux" / f"{name}_vid.npy")
        if base_vid.shape != aux_vid.shape or not np.array_equal(base_vid, aux_vid):
            raise RuntimeError(f"aux cache misaligned with base cache on split {name!r} "
                               f"(sizes {aux_vid.shape} vs {base_vid.shape}) — aux_build's read "
                               f"order diverged from data.load(); do not train on this cache.")
```

> **Why the assertion matters.** `aux_build` re-derives row order independently; a silent drift (a
> changed filter, an extra/missing row) would misattribute every aux label. Comparing to the cached
> `{split}_vid.npy` (raw video ids written by `datced` in `data.load` order) makes any drift a loud
> build-time failure, never a training-time corruption.

### Step 3 — `pipeline/lib/din.py` (edit): shared-trunk aux head

Split the current single-output MLP into a shared trunk + a primary head + an optional aux head. The
primary computation (`bias + lin + fm_cross + deep`) is **byte-for-byte preserved**; the aux head only
reads the shared hidden vector.

```python
class DIN(nn.Module):
    def __init__(self, V, fm_dim, k=16, att_hid=32, mlp_hid=64, n_aux=0):
        super().__init__()
        …                                          # embeddings unchanged
        self.att = nn.Sequential(nn.Linear(4 * k, att_hid), nn.ReLU(), nn.Linear(att_hid, 1))
        self.trunk = nn.Sequential(nn.Linear(3 * k, mlp_hid), nn.ReLU())   # was self.mlp -> 1
        self.head = nn.Linear(mlp_hid, 1)
        self.aux_head = nn.Linear(mlp_hid, n_aux) if n_aux else None

    def forward(self, tgt, seq, basex):
        …                                          # et, eh, attention, u, be, s, fm_cross, lin unchanged
        hid = self.trunk(torch.cat([u, et, s], -1))
        primary = self.bias + lin + fm_cross + self.head(hid).squeeze(-1)
        aux = self.aux_head(hid) if self.aux_head is not None else None
        return primary, aux                        # was: return <scalar>
```

`forward` now returns a tuple — update the two callers in the same file:

```python
def predict(model, feats, split, dev, bs=8192):
    …
        outs.append(model(t, s, x)[0].float().cpu().numpy())   # [0] = primary
    …

def fit_din(model, feats, cfg, dev):
    …
    aux_tr, aw = None, None
    tasks = list(getattr(cfg, "aux_tasks", ()) or ())
    if tasks:
        aux_tr = np.stack([np.asarray(feats.aux["train"][t]) for t in tasks], 1).astype(np.float32)
        w = cfg.aux_weights or tuple(0.1 for _ in tasks)
        aw = torch.as_tensor(np.asarray(w, np.float32), device=dev)      # (K,)

    def aux_loss(aux_logits, idx):
        if aux_logits is None or aux_tr is None:
            return 0.0
        tgt = torch.as_tensor(aux_tr[idx], device=dev)                   # (b, K)
        per = F.binary_cross_entropy_with_logits(aux_logits, tgt, reduction="none").mean(0)
        return (per * aw).sum()
    …
    # BPR path: zp, ap = fwd(pt);  zn, an = fwd(ng)
    #   loss = -F.logsigmoid(zp - zn).mean() + aux_loss(ap, pt) + aux_loss(an, ng)
    # BCE path: z, a = fwd(idx)
    #   loss = F.binary_cross_entropy_with_logits(z, yb) + aux_loss(a, idx)
```

(where `fwd(idx)` now returns `model(t, s, x)` = `(primary, aux)`).

### Step 4 — `pipeline/lib/din_blocks/{features,model}.py` (edit)

```python
# features.py — add aux to the FeatureSet (mirrors how seq is already loaded)
from pipeline.lib import seq_build, aux_build
def build_features(bundle, cfg):
    …
    aux = None
    if getattr(cfg, "aux_tasks", ()):
        aux = {name: aux_build.load_aux(bundle.cache_dir, name)
               for name in ("train", "valid", "test")}
    return FeatureSet(X=bundle.X, y=bundle.y, users=bundle.users,
                      meta=Meta(dim=m["V"], field_dims=bundle.dim, n_fields=5), seq=seq, aux=aux)

# model.py — build the aux head only when requested
def build_model(meta, cfg):
    n_aux = len(getattr(cfg, "aux_tasks", ()) or ())
    return DIN(meta.dim, meta.field_dims, k=cfg.k, n_aux=n_aux).to(device())
```

`din_blocks/{loss,train,infer}.py` are unchanged — the objective already lives inside `fit_din`.

### Step 5 — `agent/roles/proposer.py` (edit): make the lever discoverable

Add one line to `SYSTEM` so the Proposer actually reaches for it:

> `- Lever C (multi-task) is adoptable: adopt_blockset:"din" with config_delta`
> `{"aux_tasks":["click","like"],"aux_weights":[0.1,0.1]} adds auxiliary heads to the DIN.`

### Config (already in `Cfg`, no contract edit)

`aux_tasks: tuple` (subset of `click/like/follow/comment/forward`), `aux_weights: tuple` (same length),
`mtl_arch: str` — implement `"shared"` only; leave `mmoe`/`ple` raising `NotImplementedError` in
`build_model` as an explicit scope stop.

### Verification

1. `--smoke` (rebuilds cache at v5; the aux alignment assertion must pass; FM still `≈0.6015`).
2. Unit: `aux_build.build` then assert `sizes == meta["sizes"]` from base cache; assert every aux value ∈ {0,1}.
3. Add a `tests/mock_moves.py` move: `adopt_blockset:"din"`, `config_delta {"aux_tasks":["click","like"],"aux_weights":[0.1,0.1]}`; run `--mock`; confirm the node trains, scores, and logs without error.
4. Sanity: aux **off** (`aux_tasks=()`) must reproduce the current DIN number exactly (the primary path is unchanged).

### Edge cases / guards
- **Row alignment** — the `_assert_aux_aligned` guard is mandatory; never skip it.
- **`play_time` regression** is out of scope for v1 (binary aux only). Adding it means a regression head + MSE term — future work.
- **Weight scale** — default `aux_weights` small (0.1); large weights can drown the primary objective.
- **Rollback** — purely config: `aux_tasks=()` disables everything; no schema/contract change to revert.

---

## Feature 2 — Hypothesis-ledger dashboard

**Goal.** A client-side viewer over our `run_log.jsonl`, tree-aware and offline.

**Files:** new `dashboard/hypothesis-ledger.html` (adapt JX's file). No Python changes.

### Step 1 — Copy & remap the schema

Start from `archives/jx/agent-recsys/hypothesis-ledger.html`. Replace its field access (JX's linear
schema → our `_record()` schema from `agent/orchestrator.py`):

| JX field | Our field | Where |
|---|---|---|
| `r.iteration` | `r.iter` | labels, run-boundary detection |
| `r.metrics.valid.primary` | `r.metrics.primary_valid` | charts, table, KPIs |
| `r.metrics.valid.GAUC` / `["nDCG@5"]` | `r.metrics.GAUC` / `r.metrics["nDCG@5"]` | components chart |
| `r.metrics.test.*` | *(remove)* | we never score test per-iteration |
| `r.resource_usage.llm_input_tokens` | `r.cost.input_tokens` | spend chart, KPIs |
| `r.resource_usage.llm_output_tokens` | `r.cost.output_tokens` | spend chart |
| `r.wall_clock_s` | `r.cost.wall_clock_s` | wall-clock chart |
| `r.status ∈ {ok,failed}` | `r.status ∈ {root,improved,no_gain,abandoned,duplicate}` | pills, filters |
| `r.timestamp` | *(absent)* | sort falls back to file order — fine |

Success predicate becomes `r.metrics && r.metrics.primary_valid != null` (replaces `status==="ok"`).
Map status → pill class: `improved/root` = success (green), `no_gain` = neutral, `abandoned` = fail,
`duplicate` = muted.

### Step 2 — Make the primary chart tree-aware (the real work)

Our records carry `node_id`, `parent_id`, `phase`, `lever`. Replace JX's raw per-iteration line:

- **Best-so-far envelope**: `bestSoFar[i] = max(primary_valid over records 0..i)` — a monotone line
  that reads correctly even as the search jumps branches.
- **Scatter colored by `lever`** (A/B/C/D/F): plot each node's `primary_valid` at its `iter`, colored by
  lever, so branch families are visible. Keep the envelope as the trend line over the scatter.
- **Per-lever ablation bar**: we already emit `ablation_best_by_lever` in `resource_report.json`; add a
  small bar chart if that file is dropped in too (optional second input).

### Step 3 — Offline + provider fixes
- **Vendor Chart.js**: download `chart.umd.min.js` and inline it in a `<script>` (replace the cdnjs
  `<link>`), so charts work with no network. Keep the Google-Fonts `<link>` (it degrades gracefully).
- **Pricing**: change the `PRICE_IN=2, PRICE_OUT=10` constants (Claude Sonnet) to the Gemini rate, or
  expose them as editable inputs like the existing budget field.

### Verification
Run `--mock`, drag the produced `runs/<id>/run_log.jsonl` onto the page. Confirm: KPIs populate, the
best-so-far line is monotone, lever colors render, no console errors, and it works with the network
disabled.

### Edge cases
- Records with `metrics: null` (abandoned/duplicate) must not break charts — guard every metric access.
- The dashboard is **read-only**; it never writes, so there is zero coupling risk to a live run.

---

## Feature 3 — Cross-run champion resume

**Goal.** Persist the best validated node across `agent.run` invocations and seed it into a new run's
tree, without weakening the FM-reproduction root self-check.

**Files:** new `agent/champion.py`; edit `agent/orchestrator.py` (`run`, `finalize`), `agent/config.py`.

### Step 1 — `agent/champion.py` (new)

```python
"""Cross-run champion: a durable snapshot of the best validated node."""
from __future__ import annotations
import json, shutil, time
from pathlib import Path

def load(champion_dir):
    meta = Path(champion_dir) / "champion.json"
    return json.loads(meta.read_text()) if meta.exists() else None

def save(champion_dir, block_dir, cfg, primary_valid, cache_version, run_id, node_id):
    d = Path(champion_dir); (d / "blocks").mkdir(parents=True, exist_ok=True)
    for b in ("features","model","loss","train","infer","ensemble"):
        shutil.copy(Path(block_dir) / f"{b}.py", d / "blocks" / f"{b}.py")
    cfg.to_json(d / "cfg.json")
    (d / "champion.json").write_text(json.dumps({
        "primary_valid": float(primary_valid), "cache_version": cache_version,
        "run_id": run_id, "node_id": node_id, "cfg_hash": cfg.hash(),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
```

### Step 2 — `agent/orchestrator.py` (edit)

After the root node is added, seed the champion as an expandable node — but **re-validate it under the
current cache** rather than trusting its stored score (a `CACHE_VERSION` change can move numbers):

```python
from agent import champion
from agent.datced import CACHE_VERSION
…
    tree.add(root)
    …
    champ = champion.load(cfg.champion_dir) if cfg.resume else None
    if champ:
        node_dir, blocks, ccfg = mutate.materialize_named(run_dir, "champion",
                                                          champ_blocks=Path(cfg.champion_dir)/"blocks",
                                                          cfg=Cfg.from_json(Path(cfg.champion_dir)/"cfg.json"))
        res, wc = executor.run_node(blocks, node_dir, Path(node_dir)/"cfg.json",
                                    cfg.cache_dir, cfg.budget.per_iter_timeout_s)
        if not isinstance(res, executor.Failure):
            cnode = Node(id="champion", parent="root", phase=0, cfg=ccfg, block_dir=blocks,
                         lever="resume", hypothesis="resumed cross-run champion",
                         metrics={"GAUC": res["GAUC"], "nDCG@5": res["nDCG@5"],
                                  "primary_valid": res["primary_valid"], "primary_unbiased": None},
                         status="improved")            # "improved" -> counts as viable/expandable
            tree.add(cnode)
            mem.append(_record(0, cnode, diff="# resumed champion", events=[], cost={"wall_clock_s": wc}, signature=None))
            log(f"[champion] revalidated primary_valid={cnode.score():.4f}")
```

Add a small `mutate.materialize_named(run_dir, node_id, champ_blocks, cfg)` that snapshots
`champ_blocks/*` into `nodes/<id>/blocks` and writes `cfg.json` (mirror `materialize_root`).

In `finalize`, persist the champion when this run improved on it:

```python
    prev = champion.load(cfg.champion_dir) if cfg.resume else None
    if cfg.resume and (prev is None or final_valid > prev["primary_valid"]):
        champion.save(cfg.champion_dir, best.block_dir, best.cfg, final_valid,
                      CACHE_VERSION, run_id, best.id)
        log(f"[champion] updated -> {final_valid:.4f}")
```

### Step 3 — `agent/config.py` (edit)

```python
resume: bool = True
champion_dir: str = "runs/_champion"
```

### Verification
Run `--mock` twice. Second run must print `[champion] revalidated …`, the tree must contain a
`champion` node parented to root, and if the second run finds nothing better, `finalize` must keep the
prior champion. Delete `runs/_champion/` → behaves like a cold start.

### Edge cases
- **Stale cache**: if `champ["cache_version"] != CACHE_VERSION`, still re-validate (we always do) but
  log a warning; the re-validated score is the source of truth, never the stored one.
- **Dedup**: skip the `mem.seen(sig)` check for the champion node (it's intentionally a re-run).
- **`_viable` membership**: status `"improved"` is already in `SearchTree._viable`'s allow-set, so the
  champion is selectable by best-first with no `tree.py` change.

---

## Feature 4 — Multi-seed re-eval of the submission

**Goal.** Before writing the submission, confirm the chosen best (and ensemble members) on a seed-mean
rather than a single lucky seed — our adoption thresholds sit *below* the per-seed noise floor
(`eps=0.0002`, adopt-delta `1e-9`, seed std ≈ `0.0008`), and `tree.best()` is a max over ~50 draws.

**Files:** new `agent/reeval.py`; edit `agent/orchestrator.py` (`finalize`), `agent/config.py`.

### Step 1 — `agent/reeval.py` (new)

```python
"""Multi-seed confirmation: re-run a node's blocks under extra seeds; decide on the seed-MEAN.
Ports Jon's toggle + short-circuit (agent/reeval.py in archives/jon)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from agent import executor

def seed_scores(block_dir, cfg, cache_dir, timeout_s, out_root, seeds):
    out = []
    for s in seeds:
        c = cfg.replace(seed=int(s))                       # Cfg.replace exists (contracts.py)
        od = Path(out_root) / f"seed{s}"; od.mkdir(parents=True, exist_ok=True)
        cp = od / "cfg.json"; c.to_json(cp)
        res, _ = executor.run_node(block_dir, od, cp, cache_dir, timeout_s)
        if not isinstance(res, executor.Failure):
            out.append(float(res["primary_valid"]))
    return out

def confirm(block_dir, cfg, orig_primary, current_best, cache_dir, timeout_s, out_root,
            extra_seeds=(1, 2), eps=0.0002):
    if orig_primary <= current_best:                       # short-circuit: can't win, don't spend seeds
        return False, orig_primary, [orig_primary]
    prims = [orig_primary] + seed_scores(block_dir, cfg, cache_dir, timeout_s, out_root, extra_seeds)
    mean = float(np.mean(prims))
    return mean > current_best + eps, mean, prims
```

### Step 2 — `agent/orchestrator.finalize` (edit)

Before `best = tree.best()` selects the submission, re-rank the top candidates by seed-mean:

```python
def _multiseed_best(cfg, tree, run_dir, log):
    ranked = sorted(tree._viable(), key=lambda n: -n.score())[: cfg.recheck_top_k]
    scored = []
    for n in ranked:
        _, mean, prims = reeval.confirm(n.block_dir, n.cfg, n.score(), -1.0, cfg.cache_dir,
                                        cfg.budget.per_iter_timeout_s,
                                        Path(run_dir) / "reeval" / n.id, cfg.recheck_seeds, cfg.budget.eps)
        scored.append((mean, n, prims)); log(f"[reeval] {n.id} seeds={prims} mean={mean:.4f}")
    scored.sort(key=lambda t: -t[0])
    return scored[0][1] if scored else tree.best()

def finalize(cfg, run_dir, tree, mem, log):
    best = _multiseed_best(cfg, tree, run_dir, log) if cfg.recheck else tree.best()
    …
```

Record the per-seed scores in `resource_report.json` (add a `reeval` field) for the dashboard/write-up.
Apply the same `confirm` to each ensemble member inside `assemble()` before committing the blend
(optional, second pass).

### Step 3 — `agent/config.py` (edit)

```python
recheck: bool = True
recheck_seeds: tuple = (1, 2)      # extra seeds beyond the node's own
recheck_top_k: int = 3             # how many top nodes to re-rank
```
(`eps` reused from `Budget.eps`.)

### Verification
`--mock`: `finalize` must print `[reeval] …` lines with 3 seed scores per candidate and pick the
seed-mean winner. With `recheck=False`, behavior is identical to today. Confirm the short-circuit skips
extra seeds for a candidate that can't beat the reference.

### Edge cases
- **Cost bound**: extra seeds cost ~one full run each; keep `recheck_top_k` small (3) and prefer
  finalize-time over per-iteration. Torch (DIN) nodes are the expensive ones — the top-k cap bounds this.
- **Determinism**: re-running with the same seed must reproduce the same score; if not, a block has
  hidden nondeterminism — surface it, don't average over it.

---

## Feature 5 — Debug-first sample gate

**Goal.** Gate expensive (torch) nodes behind a fast sample run; a crash/NaN routes straight to the
Reflector instead of burning a full training run.

**Files:** new `pipeline/debug_cache.py`; edit `agent/executor.py`, `agent/orchestrator.py`,
`agent/config.py`. Works *around* the frozen `run_node.py` — we build a subsampled cache and call the
existing runner with `--cache <that>`; we never add a flag to `run_node.py`.

### Step 1 — `pipeline/debug_cache.py` (new)

Every cache array is **per-row and aligned**, so a shared row-index subsample per split is coherent
across the base, gbm, seq (and aux) caches. Global metas are copied with sizes patched.

```python
"""Build a small, row-consistent subsample of runs/_cache for a debug/smoke run."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PER_ROW = {                       # subdir ("" = base) -> per-row array stems
    "":    ["X", "y", "u", "vid"],
    "gbm": ["X", "y", "u"],
    "seq": ["seq", "slen", "tgt"],
    "aux": ["aux", "vid"],        # present only after Feature 1
}

def build(cache_dir, out_dir, n_train=20_000, n_other=10_000, seed=0):
    src, dst = Path(cache_dir), Path(out_dir)
    meta = json.loads((src / "meta.json").read_text())
    rng = np.random.default_rng(seed)
    idx = {}
    for split, size in meta["sizes"].items():
        n = n_train if split == "train" else n_other
        idx[split] = np.arange(size) if size <= n else rng.choice(size, n, replace=False)
    for sub, stems in PER_ROW.items():
        s = src / sub if sub else src
        d = dst / sub if sub else dst
        if not s.exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        for split in meta["sizes"]:
            for stem in stems:
                p = s / f"{split}_{stem}.npy"
                if p.exists():
                    np.save(d / f"{split}_{stem}.npy", np.asarray(np.load(p, mmap_mode="r"))[idx[split]])
        mp = s / "meta.json"                       # copy sibling meta, patch sizes
        if mp.exists():
            mm = json.loads(mp.read_text())
            if "sizes" in mm:
                mm["sizes"] = {k: int(len(idx[k])) for k in idx}
            (d / "meta.json").write_text(json.dumps(mm))
    m2 = dict(meta); m2["sizes"] = {k: int(len(idx[k])) for k in idx}
    (dst / "meta.json").write_text(json.dumps(m2))
    return str(dst)
```

### Step 2 — `agent/executor.py` (edit): the gate

```python
def debug_gate(blocks_dir, cfg, cache_dir, scratch_dir, n_train=20_000, n_other=10_000, epochs=2):
    """Fast sample run via the existing frozen runner. Returns a metrics dict or Failure."""
    from pipeline import debug_cache
    from pipeline.contracts import Cfg
    dbg_cache = debug_cache.build(cache_dir, Path(scratch_dir) / "cache", n_train, n_other, seed=cfg.seed)
    c = cfg.replace(epochs=min(int(cfg.epochs), epochs), patience=1)
    cp = Path(scratch_dir) / "cfg.json"; c.to_json(cp)
    res, _ = run_node(blocks_dir, Path(scratch_dir) / "out", cp, dbg_cache, timeout_s=180)
    if isinstance(res, Failure):
        return res
    pv = res.get("primary_valid")                  # run_node already rejects None/NaN as numerical
    if not (0.0 <= float(pv) <= 1.0):
        return Failure("numerical", f"debug primary_valid out of range: {pv}")
    return res
```

### Step 3 — `agent/orchestrator._iterate` (edit): call it selectively

Between materialising the child and the full `run_node`, gate only the expensive model types (a debug
pass taxes every *successful* node too, so cheap FM nodes skip it):

```python
    TORCH_MODELS = {"din", "bst"}
    if cfg.debug_gate and ncfg.model_type in TORCH_MODELS:
        dbg = executor.debug_gate(blocks, ncfg, cfg.cache_dir, Path(node_dir) / "_dbg")
        if isinstance(dbg, executor.Failure):
            res, wc2, events = _recover(cfg, driver, dbg, node_dir, blocks, ncfg, block_edit, log, it)
            if isinstance(res, executor.Failure):
                …  # log abandoned exactly as the existing failure branch does, return stall+1
    res, wc = executor.run_node(blocks, node_dir, Path(node_dir) / "cfg.json", …)   # full run
```

### Step 4 — `agent/config.py` (edit)

```python
debug_gate: bool = True
debug_train_n: int = 20_000
debug_other_n: int = 10_000
debug_epochs: int = 2
```

### Verification
`--faults` (injects a runtime crash): the debug gate must catch it in ~1–2s and route to `_recover`
*before* any full run. `--smoke`/`--mock`: an FM node must skip the gate (cheap); a `din` node must run
the gate then the full run. Confirm total wall-clock on a healthy `din` node ≈ gate + full (acceptable
overhead), and on a broken one ≈ gate only.

### Edge cases
- **Scope**: gate torch models only. Do not gate FM/LightGBM — the gate would cost more than it saves.
- **Sample blind spots** (Jon's own caveat): a small sample under-exercises vocab/UNK-scale bugs; the
  full run remains the first real signal there. This is a crash/sanity gate, not a quality gate.
- **aux coordination**: `PER_ROW` already includes `aux` (skipped when the dir is absent), so F5 works
  before or after F1.

---

## Feature 6 — Test-label data guard

**Goal.** Make it physically impossible for an agent block to read hidden-test labels.

**Files:** edit `agent/datced.py` only (load-time). No `CACHE_VERSION` bump, no frozen edit.

### The key fact

`y["test"]` is **never read anywhere** in the pipeline — `run_node` evaluates valid only; finalize's
test path is inference-only (`infer(..., "test")`) and the submission is written from `test_u`/`test_vid`
loaded directly from the cache. So we can drop `y["test"]` from the bundle unconditionally, with zero
behavioral change except closing the leak. Test *features* (`X["test"]`, `users["test"]`) stay — finalize
needs them for inference.

### Step — `agent/datced.py` (edit)

```python
def load_bundle(cache_dir):
    cache = Path(cache_dir)
    meta = json.loads((cache / "meta.json").read_text())
    X, y, users = {}, {}, {}
    for name in SPLITS:                                  # ("train","valid","test")
        X[name] = np.load(cache / f"{name}_X.npy", mmap_mode="r")
        users[name] = np.load(cache / f"{name}_u.npy", mmap_mode="r")
        if name != "test":                              # GUARD: never expose test labels to blocks
            y[name] = np.load(cache / f"{name}_y.npy", mmap_mode="r")
    return Bundle(X=X, y=y, users=users, dim=meta["dim"],
                  field_dims=meta.get("field_dims"), n_fields=meta["n_fields"], cache_dir=str(cache))
```

Any block that reaches for `feats.y["test"]` now raises `KeyError` — a loud, correct failure, which is
exactly the guard working. Optionally also stop writing `test_y.npy` in `build_or_load` (it's then never
cached at all); not required for the guard, and it's a `CACHE_VERSION` bump if you do it.

### Optional hardening (belt-and-suspenders)
If you want the bundle fully test-free during iteration (drop `X["test"]`/`users["test"]` too) while
finalize still runs test inference, gate on an env var the orchestrator sets **only** on the finalize
`run_node(..., extra_split="test")` subprocess:

```python
import os
def load_bundle(cache_dir, allow_test_features=None):
    if allow_test_features is None:
        allow_test_features = os.environ.get("KUAIRAND_FINALIZE") == "1"
    splits = SPLITS if allow_test_features else ("train", "valid")
    …  # y["test"] still never loaded, regardless
```
and in `orchestrator._rerun_test`, pass `env={**executor.utf8_env(), "KUAIRAND_FINALIZE": "1"}`. The
minimal `y["test"]`-drop above is sufficient on its own; this is only if you want defense in depth.

### Verification
`--smoke` and `--mock` must pass unchanged (nothing legitimately reads `y["test"]`). Add a negative test:
a block that reads `feats.y["test"]` must fail with `KeyError`. Finalize must still produce a valid
`submission_test.csv` that passes `submit.py --check`.

### Edge cases
- **Do not** withhold `test_u`/`test_vid` — the submission writer needs them.
- We must **never** locally score test (hidden), so withholding `y["test"]` can never break a legitimate
  path — that's what makes this strictly correct.

---

## Appendix A — Consolidated `agent/config.py` additions

```python
# Feature 3
resume: bool = True
champion_dir: str = "runs/_champion"
# Feature 4
recheck: bool = True
recheck_seeds: tuple = (1, 2)
recheck_top_k: int = 3
# Feature 5
debug_gate: bool = True
debug_train_n: int = 20_000
debug_other_n: int = 10_000
debug_epochs: int = 2
```
(F1 uses existing `Cfg` fields; F6 needs no config; F2 is a standalone file.)

## Appendix B — Acceptance matrix

| Feature | Primary gate | Must-not-regress check |
|---|---|---|
| F1 | `--smoke` rebuilds v5, aux alignment asserts, mock `din+aux` node trains | `aux_tasks=()` reproduces current DIN score exactly |
| F2 | Drag real `run_log.jsonl`; charts render offline | read-only — cannot affect a run |
| F3 | Second `--mock` prints `[champion] revalidated`; champion node in tree | delete `runs/_champion/` → cold start unchanged |
| F4 | `finalize` prints `[reeval]` with N seeds; picks seed-mean | `recheck=False` → identical to today |
| F5 | `--faults` caught in ~1–2s pre-full-run; FM nodes skip gate | healthy `din` node still trains + scores |
| F6 | block reading `y["test"]` → `KeyError`; submission still valid | `--smoke`/`--mock` pass unchanged |

## Appendix C — Cache file inventory (for F5 subsampling)

| Dir | Per-row arrays (subsample by shared row-index) | Global (copy, patch `sizes`) |
|---|---|---|
| `runs/_cache/` | `{split}_X`, `_y`, `_u`, `_vid` | `meta.json` |
| `runs/_cache/gbm/` | `{split}_X`, `_y`, `_u` | `meta.json` (`stat_cols`, `n_features`) |
| `runs/_cache/seq/` | `{split}_seq`, `_slen`, `_tgt` | `meta.json` (`V`, `UNK`, `L`) |
| `runs/_cache/aux/` *(F1)* | `{split}_aux`, `_vid` | `meta.json` (`tasks`) |

---

*All six features stay below the trust boundary. After each, run `python -m agent.run --smoke` — it
re-verifies `frozen.lock`, so an accidental frozen-file edit fails loudly before anything else.*
