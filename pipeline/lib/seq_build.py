"""Lever B data -- per-user behavior sequences, cached, TRUE chronological order.

The defect this replaces (docs/EN/RESEARCH.md §10 (sequence chronology))
-----------------------------------------------
The previous version claimed temporal safety was "structural: we process rows in global time order".
`data.load()` does not sort -- it appends rows in CSV order and filters by date -- and the KuaiRand
logs are not time-ordered within a user. Measured on `log_standard_4_22_to_5_08_pure.csv`: 47,742
contiguous user runs for 25,877 users and 18,763 per-user `time_ms` inversions. Replaying the old
loop, the fraction of rows whose "prior" history contained a LATER-dated item was:

    train 30.83%   valid 20.89%   test 31.54%

With ids-only histories that was a mild acausal exposure leak. It becomes a genuine LABEL leak the
moment feedback states are attached, which is exactly what `feedback` below does. So the ordering is
fixed first, and asserted.

What this module now guarantees
-------------------------------
1. Histories are built in true `(user, time_ms)` order, re-read from the raw logs the same way
   `aux_build` re-reads the aux columns, and asserted row-aligned against the base cache's
   `{split}_vid.npy`.
2. Arrays are written back in `data.load()` ROW ORDER, so every downstream cache stays aligned.
3. A row's own event is appended to its user's history only AFTER the snapshot, so no row can ever
   see its own outcome.
4. `{split}_fb.npy` carries the FEEDBACK STATE of each history event. Test-window outcomes are never
   used: a history event that falls in the test window is recorded as `FB_UNKNOWN`. Predicting one
   test row from another test row's outcome would be transductive use of hidden-test labels, so it
   is structurally impossible here rather than merely discouraged.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from data import SPLITS                       # frozen: date ranges only, safe to read

SPLIT_ORDER = ("train", "valid", "test")
LOG_FILES = ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")

# Feedback states for a HISTORY event. Deliberately small and interpretable (docs/EN/RESEARCH.md §11).
FB_PAD = 0          # padding
FB_SKIP = 1         # not clicked, or watched a negligible fraction
FB_SHORT = 2        # short view
FB_NORMAL = 3       # substantial view, but not a long_view
FB_LONG = 4         # long_view (the primary positive)
FB_EXPLICIT = 5     # explicit positive: like / comment / forward
FB_UNKNOWN = 6      # outcome not usable as a feature -- never a leaked label
N_FB_STATES = 7

# WHICH history outcomes a model may use as an input feature.
#
#   "train_only" (default, honest): only TRAIN-window events carry a state; valid- and test-window
#       events are FB_UNKNOWN. This is the only policy under which validation is an unbiased proxy
#       for test, and it was chosen after a MEASURED failure of the alternative (see below).
#   "leq_split": an event may carry a state if it precedes the scored row's split. Valid rows then
#       see earlier VALID outcomes while test rows see no test outcomes -- an asymmetry that inflates
#       validation. Retained only so the artifact can be reproduced.
#
# MEASURED (2026-08-31): under "leq_split", valid histories are 100% known-outcome but test histories
# only 75.8%. Behavior-aware DIN then scored 0.61925 on valid (+0.0165 over the same config without
# feedback states, ~18x the 0.0009 noise floor) -- a gain that is structurally unavailable at test
# time, because a submission scores all test rows at once with no feedback in between. Using one
# valid row's label to predict another is transductive leakage of the selection set.
FB_POLICY_TRAIN_ONLY = "train_only"
FB_POLICY_LEQ_SPLIT = "leq_split"

SHORT_RATIO = 0.10
NORMAL_RATIO = 0.50


def _state(row) -> int:
    """Feedback state from one raw log row. Uses only that row's OWN outcome, which is legitimate
    for a HISTORY event (it happened before the row being scored) and never for the current row."""
    if row["like"] or row["comment"] or row["forward"]:
        return FB_EXPLICIT
    if row["long_view"]:
        return FB_LONG
    if not row["click"]:
        return FB_SKIP
    ratio = row["play_ms"] / max(1.0, row["dur_ms"])
    if ratio >= NORMAL_RATIO:
        return FB_NORMAL
    if ratio >= SHORT_RATIO:
        return FB_SHORT
    return FB_SKIP


def _read_rows(data_dir):
    """Re-read the raw logs in `data.load()`'s exact file order + date filter.

    Mirrors aux_build.build so the produced arrays are row-aligned with the base cache; that
    alignment is then asserted, not assumed.
    """
    rows = {name: [] for name in SPLIT_ORDER}
    for fname in LOG_FILES:
        with open(Path(data_dir) / fname, newline="") as fh:
            for r in csv.DictReader(fh):
                date = int(r["date"])
                for name, (lo, hi) in SPLITS.items():
                    if lo <= date <= hi:                   # disjoint ranges -> exactly one split
                        rows[name].append({
                            "user": r["user_id"], "vid": r["video_id"],
                            "t": int(r["time_ms"]), "date": date,
                            "click": r["is_click"] != "0",
                            "like": r["is_like"] != "0",
                            "comment": r["is_comment"] != "0",
                            "forward": r["is_forward"] != "0",
                            "long_view": r["long_view"] != "0",
                            "play_ms": float(r["play_time_ms"] or 0.0),
                            "dur_ms": float(r["duration_ms"] or 0.0),
                        })
                        break
    return rows


def build(data_dir, cache_dir, L=30, force=False, splits=None,
          fb_policy=FB_POLICY_TRAIN_ONLY):
    fc = Path(cache_dir) / "seq"
    meta_p = fc / "meta.json"
    if meta_p.exists() and not force:
        m = json.loads(meta_p.read_text())
        if m.get("L") == L and m.get("chronological") and m.get("fb_policy") == fb_policy:
            return m
    fc.mkdir(parents=True, exist_ok=True)

    raw = _read_rows(data_dir)

    # Row-alignment guard against the base cache (which is written in data.load() order).
    for name in SPLIT_ORDER:
        base_p = Path(cache_dir) / f"{name}_vid.npy"
        if base_p.exists():
            base_vid = np.load(base_p)
            got = np.fromiter((int(r["vid"]) for r in raw[name]), np.int64, len(raw[name]))
            if base_vid.shape != got.shape or not np.array_equal(base_vid, got):
                raise RuntimeError(
                    f"seq_build row order diverged from data.load() on split {name!r} "
                    f"({got.shape} vs {base_vid.shape}); refusing to build a misaligned cache.")

    # video vocab from train: 1..V ; 0 = PAD ; V+1 = UNK (unseen in valid/test)
    vv = {}
    for r in raw["train"]:
        if r["vid"] not in vv:
            vv[r["vid"]] = len(vv) + 1
    V = len(vv)
    UNK = V + 1

    # ---- the fix: walk every event in TRUE (user, time) order -------------------------------
    # `order` is a global list of (user, time_ms, split_rank, row_index) so the walk is
    # deterministic and total even when two events share a timestamp.
    events = []
    for si, name in enumerate(SPLIT_ORDER):
        for i, r in enumerate(raw[name]):
            events.append((r["user"], r["t"], si, i))
    events.sort(key=lambda e: (e[0], e[1], e[2], e[3]))

    out = {name: {
        "seq": np.zeros((len(raw[name]), L), np.int32),
        "fb": np.zeros((len(raw[name]), L), np.int8),
        "slen": np.zeros(len(raw[name]), np.int32),
        "tgt": np.zeros(len(raw[name]), np.int32),
    } for name in SPLIT_ORDER}

    hist = defaultdict(lambda: deque(maxlen=L))      # user -> deque[(vid_code, fb_state, split_idx)]
    inversions = {name: 0 for name in SPLIT_ORDER}
    last_t = {}
    fb_unknown_events = 0
    total_hist_events = 0
    cross_split_dropped = 0

    for user, t, si, i in events:
        name = SPLIT_ORDER[si]
        r = raw[name][i]
        h = hist[user]
        # Filter by SPLIT as well as by time. `date` and `time_ms` disagree at the split boundary --
        # measured: 28 test-dated rows carry a timestamp earlier than the last valid row (the logs'
        # `date` is a local calendar day, the timestamps are epoch ms), and 5 valid rows of those
        # users fall after such a test row. Time order alone would therefore put a handful of
        # test-window events into valid histories. Filtering on split index makes "a train/valid row
        # never sees a test-window event" structural rather than dependent on a timestamp quirk.
        vis = [(c, f) for (c, f, s) in h if s <= si]
        cross_split_dropped += len(h) - len(vis)
        if vis:
            a = np.fromiter((c for c, _ in vis), np.int32, len(vis))
            b = np.fromiter((f for _, f in vis), np.int8, len(vis))
            out[name]["seq"][i, L - len(a):] = a         # left-pad
            out[name]["fb"][i, L - len(b):] = b
            out[name]["slen"][i] = len(a)
            total_hist_events += len(a)
            fb_unknown_events += int((b == FB_UNKNOWN).sum())
        out[name]["tgt"][i] = vv.get(r["vid"], UNK)
        # sanity: within a user the walk must be non-decreasing in time
        if user in last_t and t < last_t[user]:
            inversions[name] += 1
        last_t[user] = t
        # Which outcomes may become features. Under the default "train_only" policy only
        # train-window outcomes do, so valid and test are treated IDENTICALLY and validation stays an
        # honest proxy for test. See the FB_POLICY_* note at the top of this module for the
        # measurement that motivated it.
        if fb_policy == FB_POLICY_TRAIN_ONLY:
            usable = (name == "train")
        else:
            usable = (name != "test")
        state = _state(r) if usable else FB_UNKNOWN
        h.append((vv.get(r["vid"], UNK), state, si))     # append AFTER the snapshot

    if any(inversions.values()):                          # must be structurally impossible now
        raise RuntimeError(f"seq_build: time inversions after sorting: {inversions}")

    sizes = {}
    for name in SPLIT_ORDER:
        for stem in ("seq", "fb", "slen", "tgt"):
            np.save(fc / f"{name}_{stem}.npy", out[name][stem])
        sizes[name] = len(raw[name])

    meta = {"V": V, "UNK": UNK, "L": L, "sizes": sizes, "chronological": True,
            "fb_policy": fb_policy,
            "n_fb_states": N_FB_STATES,
            "fb_states": {"PAD": FB_PAD, "SKIP": FB_SKIP, "SHORT": FB_SHORT, "NORMAL": FB_NORMAL,
                          "LONG": FB_LONG, "EXPLICIT": FB_EXPLICIT, "UNKNOWN": FB_UNKNOWN},
            "fb_unknown_fraction": round(fb_unknown_events / max(1, total_hist_events), 5),
            "cross_split_events_dropped": int(cross_split_dropped)}
    meta_p.write_text(json.dumps(meta, indent=2))
    return meta


def load_split(cache_dir, name):
    fc = Path(cache_dir) / "seq"
    return (np.load(fc / f"{name}_tgt.npy"), np.load(fc / f"{name}_seq.npy"),
            np.load(fc / f"{name}_slen.npy"))


def load_fb(cache_dir, name):
    """Per-history-event feedback states (Lever B behavior-aware history, docs/EN/RESEARCH.md §11)."""
    return np.load(Path(cache_dir) / "seq" / f"{name}_fb.npy")


def meta(cache_dir):
    return json.loads((Path(cache_dir) / "seq" / "meta.json").read_text())
