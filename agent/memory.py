"""Append-only run-log = agent memory = deliverable."""
from __future__ import annotations
import json
from pathlib import Path


class Memory:
    def __init__(self, run_dir: str):
        self.path = Path(run_dir) / "run_log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        self._seen: set[str] = set()

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

    def resource_totals(self) -> dict:
        cost = [r.get("cost", {}) for r in self.records]
        return {
            "input_tokens": sum(c.get("input_tokens", 0) for c in cost),
            "output_tokens": sum(c.get("output_tokens", 0) for c in cost),
            "wall_clock_s": round(sum(c.get("wall_clock_s", 0.0) for c in cost), 1),
            "iters": len([r for r in self.records if r.get("iter", 0) > 0]),
        }
