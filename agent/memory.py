"""Append-only run-log = agent memory = deliverable.

`research_table()` synthesises the scientific state handed to the Proposer. docs/SYSTEM.md §14 records what the old rule-based version did: it bucketed a record as `confirmed` iff
`status == "improved"` and as **rejected otherwise**, so in `run_20260831_000142` -- a run with zero
`improved` nodes -- the Proposer was told from iteration 4 onward:

    REJECTED (below noise / regressed -- don't repeat):
      C: "Adopt multi-task learning ..." -> 0.6026     <-- the run's BEST model
      E: "Increase the negative sampling ratio ..."    <-- a no-op re-run of C

Two categories were being conflated: a TREE/ADOPTION status (did this beat its parent by
`adopt_eps`?) and SCIENTIFIC EVIDENCE (what does the data say about the effect?). They are now
separate. Buckets come from the paired-bootstrap evidence class (agent/stats.py), structural no-ops
are excluded from science entirely, and the current champion is always surfaced explicitly whatever
bucket it lands in.
"""
from __future__ import annotations

import json
from pathlib import Path

# Evidence classes, in the order the Proposer should read them.
CONFIRMED = "confirmed"
PROMISING = "promising"
INCONCLUSIVE = "inconclusive"
REJECTED = "rejected"
NO_EFFECT = "no_effect"
UNSUPPORTED = "unsupported_capability"

LEVER_NAMES = {"A": "loss", "B": "sequence/DIN", "C": "multi-task", "D": "model-family",
               "E": "debias/exposure", "F": "ensemble"}


class Memory:
    def __init__(self, run_dir: str):
        self.path = Path(run_dir) / "run_log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        self._seen: set[str] = set()

    # ------------------------------------------------------------------ ledger
    def append(self, rec: dict):
        self.records.append(rec)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        if rec.get("signature"):
            self._seen.add(rec["signature"])

    def seen(self, sig: str) -> bool:
        return sig in self._seen

    def note_seen(self, sig: str):
        self._seen.add(sig)

    def best(self) -> dict | None:
        cand = [r for r in self.records
                if r.get("metrics") and r["metrics"].get("primary_valid") is not None
                and r.get("status") in ("root", "improved", "no_gain")]
        return max(cand, key=lambda r: r["metrics"]["primary_valid"]) if cand else None

    def recall(self, lever: str | None = None, k: int = 12) -> list[dict]:
        rs = [r for r in self.records if lever is None or r.get("lever") == lever]
        return rs[-k:]

    # ------------------------------------------------------------------ scientific state
    def _scored(self):
        for r in self.records:
            m = r.get("metrics") or {}
            if m.get("primary_valid") is None:
                continue
            if r.get("noop_class") in ("STRUCTURAL_NOOP", "EXACT_NOOP"):
                continue                       # provably no intervention -> not a scientific result
            yield r

    def research_table(self, all_levers=("A", "B", "C", "D", "E", "F")) -> dict:
        """A compact, evidence-graded scientific state for the Proposer.

        Buckets are decided by the paired-bootstrap evidence class recorded on each node
        (`evidence.class`), NOT by tree adoption status. A node can be `inconclusive` and still be the
        champion; a node can be standalone-`rejected` and still earn a portfolio slot on its ensemble
        marginal contribution -- both are stated explicitly rather than collapsed into "rejected".
        """
        buckets = {CONFIRMED: [], PROMISING: [], INCONCLUSIVE: [], REJECTED: [], NO_EFFECT: []}
        per_lever: dict[str, float] = {}
        diverse, unsupported = [], []
        best_rec, best_pv = None, -1.0

        for r in self.records:
            # capability rejections are research information, not failures (docs/SYSTEM.md §14)
            for e in r.get("events") or []:
                if e.get("class") == "not_implementable":
                    unsupported.append(f"{r.get('lever', '?')}: {str(e.get('detail', ''))[:100]}")

        for r in self.records:
            if r.get("noop_class") in ("STRUCTURAL_NOOP", "EXACT_NOOP"):
                m = r.get("metrics") or {}
                buckets[NO_EFFECT].append(
                    f"{r.get('lever', '?')}: \"{str(r.get('hypothesis', ''))[:56]}\" -> "
                    f"{r.get('noop_class')} (no execution change; not evidence)")
                continue

        for r in self._scored():
            m = r["metrics"]
            pv = float(m["primary_valid"])
            lv = r.get("lever", "?")
            per_lever[lv] = max(per_lever.get(lv, 0.0), pv)
            if pv > best_pv:
                best_pv, best_rec = pv, r

            ev = r.get("evidence") or {}
            if not ev.get("control_id"):
                continue          # root / resumed champion: a control, not an experiment
            klass = ev.get("class", INCONCLUSIVE)
            d = ev.get("delta_primary")
            p = ev.get("p_gt0")
            stat = ""
            if d is not None and p is not None:
                stat = f"d={d:+.5f} P(d>0)={p:.2f}"
            elif d is not None:
                stat = f"d={d:+.5f}"
            line = (f"{lv}: \"{str(r.get('hypothesis', ''))[:56]}\" -> {pv:.5f}"
                    f"{'  [' + stat + ']' if stat else ''}"
                    f"{' vs ' + ev['control_id'] if ev.get('control_id') else ''}")
            buckets.setdefault(klass, []).append(line)

            port = r.get("portfolio") or {}
            emc, rc = port.get("emc"), port.get("rank_corr_to_best")
            if emc is not None and rc is not None and (emc > 0 or rc < 0.90):
                diverse.append(f"{r.get('node_id')} ({lv}, {r.get('config', {}).get('model_type')}): "
                               f"standalone {pv:.5f}, rank_corr {rc:.3f}, EMC {emc:+.5f}")

        untried = [f"{lv} ({LEVER_NAMES.get(lv, lv)})" for lv in all_levers if lv not in per_lever]
        tested_levers = sorted(per_lever)
        interactions = [f"{a}x{b}" for i, a in enumerate(tested_levers)
                        for b in tested_levers[i + 1:]]

        return {
            CONFIRMED: buckets[CONFIRMED][-5:],
            PROMISING: buckets[PROMISING][-5:],
            INCONCLUSIVE: buckets[INCONCLUSIVE][-5:],
            REJECTED: buckets[REJECTED][-5:],
            NO_EFFECT: buckets[NO_EFFECT][-4:],
            UNSUPPORTED: unsupported[-4:],
            "diverse_portfolio_candidates": diverse[-5:],
            "best_per_lever": {lv: round(p, 5) for lv, p in
                               sorted(per_lever.items(), key=lambda x: -x[1])},
            "untried_levers": untried,
            "tested_interactions": interactions[:8],
            "champion": ({"node_id": best_rec.get("node_id"), "lever": best_rec.get("lever"),
                          "primary_valid": round(best_pv, 5),
                          "model_type": (best_rec.get("config") or {}).get("model_type"),
                          "GAUC": best_rec["metrics"].get("GAUC"),
                          "nDCG@5": best_rec["metrics"].get("nDCG@5")}
                         if best_rec else None),
        }

    def resource_totals(self) -> dict:
        cost = [r.get("cost", {}) for r in self.records]
        return {
            "input_tokens": sum(c.get("input_tokens", 0) for c in cost),
            "output_tokens": sum(c.get("output_tokens", 0) for c in cost),
            "wall_clock_s": round(sum(c.get("wall_clock_s", 0.0) for c in cost), 1),
            "iters": len([r for r in self.records if r.get("iter", 0) > 0]),
        }
