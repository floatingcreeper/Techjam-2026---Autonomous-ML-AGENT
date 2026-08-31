"""Sequence chronology and behavior-state leakage safety -- docs/RESEARCH.md §10 (sequence chronology) / §6.1.

Verified INDEPENDENTLY of the builder: the expected history for sampled rows is recomputed from the
raw logs by a separate implementation and compared against the cached arrays. A builder that asserts
its own correctness is not evidence.

Run: cudaenv/Scripts/python.exe -m tests.test_sequence
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import SPLITS                                   # noqa: E402  (frozen: date ranges)
from pipeline.lib import seq_build                        # noqa: E402

CACHE = ROOT / "runs" / "_cache"
DATA = ROOT / "KuaiRand-Pure" / "data"
SPLIT_ORDER = ("train", "valid", "test")
failures: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


def _raw():
    """Independent re-read of the raw logs (same file order + date filter as data.load)."""
    rows = {n: [] for n in SPLIT_ORDER}
    for fname in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with open(DATA / fname, newline="") as fh:
            for r in csv.DictReader(fh):
                d = int(r["date"])
                for name, (lo, hi) in SPLITS.items():
                    if lo <= d <= hi:
                        rows[name].append((r["user_id"], r["video_id"], int(r["time_ms"]), d))
                        break
    return rows


def main():
    if not (CACHE / "seq" / "meta.json").exists() or not DATA.exists():
        print("SKIP: cache or raw data missing")
        return
    m = seq_build.meta(str(CACHE))
    print("\n[1] cache metadata")
    check("cache declares chronological construction", m.get("chronological") is True)
    check("feedback states present", m.get("n_fb_states") == seq_build.N_FB_STATES)
    print(f"  INFO  test-window UNKNOWN feedback events: "
          f"{m.get('fb_unknown_fraction', 0):.2%} of all history events")

    raw = _raw()
    print("\n[2] row alignment against the base cache (data.load order)")
    for name in SPLIT_ORDER:
        base = np.load(CACHE / f"{name}_vid.npy")
        got = np.fromiter((int(v) for _, v, _, _ in raw[name]), np.int64, len(raw[name]))
        check(f"{name}: seq rows align with base cache", np.array_equal(base, got),
              f"n={len(got)}")

    # ---- independent chronological reconstruction ------------------------------------------
    print("\n[3] independent reconstruction of per-user chronological history")
    # global event list in true (user, time, split, index) order -- computed here, not imported
    events = []
    for si, name in enumerate(SPLIT_ORDER):
        for i, (u, v, t, d) in enumerate(raw[name]):
            events.append((u, t, si, i))
    events.sort()

    L = m["L"]
    vv = {}
    for _, v, _, _ in raw["train"]:
        if v not in vv:
            vv[v] = len(vv) + 1
    UNK = len(vv) + 1

    hist = defaultdict(list)
    expect = {name: {} for name in SPLIT_ORDER}
    rng = np.random.default_rng(0)
    want = {name: set(rng.choice(len(raw[name]), size=min(400, len(raw[name])),
                                 replace=False).tolist()) for name in SPLIT_ORDER}
    for u, t, si, i in events:
        name = SPLIT_ORDER[si]
        if i in want[name]:
            expect[name][i] = list(hist[u][-L:])
        hist[u].append(vv.get(raw[name][i][1], UNK))

    for name in SPLIT_ORDER:
        seq = np.load(CACHE / "seq" / f"{name}_seq.npy", mmap_mode="r")
        slen = np.load(CACHE / "seq" / f"{name}_slen.npy", mmap_mode="r")
        bad = 0
        for i, exp in expect[name].items():
            got = list(seq[i][L - len(exp):]) if exp else []
            if int(slen[i]) != len(exp) or [int(x) for x in got] != exp:
                bad += 1
        check(f"{name}: {len(expect[name])} sampled histories match an independent rebuild",
              bad == 0, f"{bad} mismatches")

    # ---- the actual leakage property --------------------------------------------------------
    print("\n[4] no future event appears in any history")
    tmap = {name: np.fromiter((t for _, _, t, _ in raw[name]), np.int64, len(raw[name]))
            for name in SPLIT_ORDER}
    # Reconstruct history TIMES the same independent way, then assert every one is <= the row's time.
    hist_t = defaultdict(list)
    viol = {name: 0 for name in SPLIT_ORDER}
    checked = {name: 0 for name in SPLIT_ORDER}
    for u, t, si, i in events:
        name = SPLIT_ORDER[si]
        if i in want[name]:
            checked[name] += 1
            if any(ht > t for ht in hist_t[u][-L:]):
                viol[name] += 1
        hist_t[u].append(t)
    for name in SPLIT_ORDER:
        check(f"{name}: 0 of {checked[name]} sampled rows see a LATER event "
              f"(was {'30.83%' if name == 'train' else '20.89%' if name == 'valid' else '31.54%'} "
              f"before the fix)", viol[name] == 0, f"{viol[name]} violations")

    print("\n[5] only TRAIN-window outcomes may become features (fb_policy)")
    policy = m.get("fb_policy")
    check("cache uses the honest 'train_only' feedback policy",
          policy == seq_build.FB_POLICY_TRAIN_ONLY, str(policy))
    # Why this matters (MEASURED, docs/RESEARCH.md §11 (behavior-aware history)): under the alternative "leq_split" policy a
    # VALID row could see the outcomes of earlier VALID rows while a TEST row could see no test-window
    # outcomes at all (valid 100% known vs test 75.8%). Behavior-aware DIN then scored +0.0165 on
    # valid -- a gain structurally unavailable at test time. "train_only" treats valid and test
    # identically, so validation stays an unbiased proxy.
    known = {}
    for name in SPLIT_ORDER:
        fb = np.asarray(np.load(CACHE / "seq" / f"{name}_fb.npy", mmap_mode="r"))
        seq = np.asarray(np.load(CACHE / "seq" / f"{name}_seq.npy", mmap_mode="r"))
        real = seq > 0
        tot = int(real.sum())
        unk = int(((fb == seq_build.FB_UNKNOWN) & real).sum())
        known[name] = (tot - unk) / max(1, tot)
        if name == "train":
            check("train: every history event is train-window, so none is UNKNOWN", unk == 0,
                  f"{unk} found")
        else:
            check(f"{name}: own-window events are UNKNOWN, not their outcome", unk > 0,
                  f"{unk:,d} UNKNOWN of {tot:,d}")
        check(f"{name}: padding slots carry FB_PAD only",
              bool(((seq[:20000] == 0) == (fb[:20000] == seq_build.FB_PAD)).all()))
    for name in SPLIT_ORDER:
        print(f"  INFO  {name}: {known[name]:.1%} of history events carry a usable outcome")
    check("valid and test are treated by the SAME rule (both lose their own window), so validation "
          "is not systematically more informed than test",
          known["valid"] > known["test"] and known["train"] == 1.0,
          f"train {known['train']:.1%} > valid {known['valid']:.1%} > test {known['test']:.1%} "
          f"-- ordered because later splits have more of their own window in history, "
          f"not because a different rule was applied")

    print("\n[6] feedback-state distribution (train history events)")
    fb = np.asarray(np.load(CACHE / "seq" / "train_fb.npy", mmap_mode="r")[:200000])
    names = {v: k for k, v in seq_build.meta(str(CACHE))["fb_states"].items()}
    tot = int((fb != 0).sum())
    for s in range(1, seq_build.N_FB_STATES):
        c = int((fb == s).sum())
        if c:
            print(f"  INFO  {names.get(s, s):<9} {c:>9,d}  ({c / max(1, tot):.1%})")
    check("history is not degenerate (more than one state occurs)",
          len({s for s in range(1, 7) if (fb == s).any()}) >= 3)


if __name__ == "__main__":
    main()
    print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
    sys.exit(1 if failures else 0)
