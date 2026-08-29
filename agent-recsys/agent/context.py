"""
Builds the context handed to the LLM at each iteration. Kept in one place
so token cost is easy to reason about: static facts are computed once and
reused every call, rather than re-derived or re-explained each time.
"""
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Static knowledge carried over from the organizer's README (see the
# starter-kit README's "从哪里开始改" section). Hardcoded here because it's
# fixed, organizer-published information, not something to re-discover.
# Feeding this to the LLM up front is what stops it from re-proposing ideas
# the organizers already tested and ruled out.
# ---------------------------------------------------------------------------
KNOWN_DEAD_ENDS = """\
Already tested by the organizers -- do not re-propose these as novel ideas,
they are confirmed non-improvements (primary score flat or slightly down,
well within seed noise of ~0.0008):
  1. Adding more raw categorical feature columns (going from the base 5
     fields to the full 13-field CWM feature set: +music_id, +video_type,
     +upload_type, +5 user-side bucketed features). Result: 0.5950 -> 0.5940,
     i.e. no gain. Diagnosis: user_id x video_id crossing already captures
     most learnable signal; coarse user-side buckets are redundant once
     user_id itself is a feature; a purely user-side feature's linear term
     is also mathematically inert here, since ranking is within-user (any
     term constant across one user's own impressions cannot reorder them).
  2. Increasing FM embedding dimension k (8 / 16 / 32). Result: 0.5895 /
     0.5902 / 0.5887 -- essentially flat. Diagnosis: model capacity is not
     the bottleneck; ~1.14M training rows does not support much larger
     capacity anyway.
"""

KNOWN_HEADROOM_DIRECTIONS = """\
Organizer-ranked list of UNEXPLORED directions (most promising first) --
prefer these over re-tuning what's already been tried:
  1. Loss function change. Training currently uses pointwise logloss, but
     the scored metrics (GAUC, nDCG@5) are ranking metrics. Switching to a
     pairwise loss (e.g. BPR: sample a positive/negative pair per user, push
     score(pos) > score(neg)) or a listwise loss (softmax over one user's
     impressions) would align the training objective with what's actually
     scored. Organizers' own assessment: most likely direction to help.
  2. User history / sequence modeling. Current features use zero behavioral
     history -- each user has hundreds to thousands of training interactions
     that aren't used as a feature at all. DIN/SIM-style interest modeling
     (attention over a user's past interacted videos) is a fully open
     direction here.
  3. Multi-task learning. The logs carry 11 other feedback signals besides
     long_view (is_click, is_like, is_follow, is_comment, is_forward,
     play_time_ms, ...) that could be auxiliary training targets (ESMM-style
     shared + task-specific parameters) to help the sparse long_view signal.
  4. Watch-time modeling (censored regression). See CWM reference (a
     completed play means true watch time was truncated by video length --
     a one-sided/censored loss instead of squared error). Research-depth
     direction, higher implementation risk.
  5. Model architecture upgrades (DeepFM / DCN / xDeepFM). Explicitly lower
     priority than 1-4, since capacity was already tested and found not to
     be the bottleneck (see dead end #2 above).
  6. Time features / distribution drift (hourmin, date; train vs test
     drift).
  7. (Advanced) Unbiased validation using log_random_4_22_to_5_08_pure.csv
     (the randomized-exposure log) as an additional de-biased check that the
     model isn't just overfitting to biased production traffic.
"""


def load_baseline_reference(pipeline_dir: Path) -> dict:
    with open(pipeline_dir / "baseline_scores.json", encoding="utf-8") as f:
        return json.load(f)


def build_data_profile_text(baseline_ref: dict) -> str:
    comp = baseline_ref["test_set_composition"]
    fm = baseline_ref["scores"]["fm_official"]
    oracle = baseline_ref["scores"]["oracle_ceiling"]
    return (
        f"Dataset: KuaiRand-Pure. Label: long_view (0/1). Task: rank each "
        f"user's own logged impressions (not full-catalog retrieval). "
        f"Metrics: GAUC + nDCG@5, primary = mean of both.\n"
        f"Test set: {comp['users']} users -- {comp['all_negative_pct']}% have "
        f"zero positives (nDCG forced to 0, excluded from GAUC), "
        f"{comp['all_positive_pct']}% are all-positive (nDCG forced to 1, "
        f"excluded from GAUC), only {comp['discriminative_pct']}% actually "
        f"drive the GAUC number.\n"
        f"Official FM baseline (must beat): valid primary {fm['valid']['primary']:.4f}, "
        f"test primary {fm['test']['primary']:.4f} (5-seed std ~0.0008).\n"
        f"Oracle ceiling (perfect ranking, theoretical max): test primary "
        f"{oracle['test']['primary']:.4f} -- judge headroom against this, not 1.0."
    )


def format_history_summary(history: list, max_recent: int = 8) -> str:
    """Keeps the prompt bounded: full detail for the most recent iterations,
    a compact one-liner for everything older. This is the main lever for
    controlling per-call token cost as the run gets longer."""
    if not history:
        return "No iterations yet. This is iteration 0 (reproduce baseline)."

    lines = []
    older = history[:-max_recent]
    recent = history[-max_recent:]

    if older:
        lines.append(f"[{len(older)} earlier iterations omitted for brevity]")
        best_older = max(
            (h for h in older if h.get("status") == "ok"),
            key=lambda h: h["metrics"]["valid"]["primary"],
            default=None,
        )
        if best_older:
            lines.append(
                f"  best among them: iter {best_older['iteration']} "
                f"(\"{best_older['hypothesis'][:80]}\") "
                f"valid primary {best_older['metrics']['valid']['primary']:.4f}"
            )

    for h in recent:
        if h.get("status") == "ok":
            m = h["metrics"]
            lines.append(
                f"iter {h['iteration']}: \"{h['hypothesis']}\" (edited {h['target_file']}) "
                f"-> valid primary {m['valid']['primary']:.4f} "
                f"(GAUC {m['valid']['GAUC']:.4f}, nDCG@5 {m['valid']['nDCG@5']:.4f})"
            )
        else:
            lines.append(
                f"iter {h['iteration']}: \"{h['hypothesis']}\" (edited {h['target_file']}) "
                f"-> FAILED after {h.get('repair_attempts', 0)} repair attempt(s): "
                f"{h.get('final_error', 'unknown error')[:200]}"
            )
    return "\n".join(lines)


def format_sibling_candidates(sibling_candidates: list, candidate_idx: int, candidates_per_iteration: int) -> str:
    """Describes the other candidate(s) already tried for THIS SAME
    iteration slot, so a later candidate can choose to refine a near-miss
    instead of only ever pivoting to something unrelated. Distinct from
    format_history_summary(), which covers PAST iterations."""
    if candidates_per_iteration <= 1:
        return ""
    if not sibling_candidates:
        return (
            f"You are proposing candidate 1 of {candidates_per_iteration} for "
            f"THIS iteration. The other {candidates_per_iteration - 1} candidate(s) "
            f"run after you, in this same slot -- whichever candidate scores "
            f"highest is the one that's kept."
        )
    lines = [
        f"You are proposing candidate {candidate_idx + 1} of {candidates_per_iteration} "
        f"for THIS SAME iteration. The other candidate(s) already tried this "
        f"iteration (not past iterations -- see history above for those):"
    ]
    for j, c in enumerate(sibling_candidates):
        if c["status"] == "ok":
            lines.append(f"  candidate {j + 1}: \"{c['hypothesis']}\" -> valid primary {c['valid_primary']:.4f}")
        else:
            lines.append(f"  candidate {j + 1}: \"{c['hypothesis']}\" -> FAILED: {(c['error'] or '')[:150]}")
    lines.append(
        "Only the single best-scoring candidate this iteration is kept. You may "
        "either (a) refine the most promising candidate above with a concrete, "
        "different parameterization (e.g. different epoch count / sampling rate "
        "/ learning rate / loss weight) if you think it was close but "
        "under-tuned, or (b) propose a genuinely different hypothesis. Prefer "
        "(a) when a sibling candidate scored close to (or above) the current "
        "best but plausibly wasn't given a fair shot -- don't abandon a "
        "promising direction after only one untuned attempt."
    )
    return "\n".join(lines)


def build_prompt(
    pipeline_dir: Path,
    baseline_ref: dict,
    history: list,
    last_error: str | None = None,
    best_valid_primary: float | None = None,
    candidate_idx: int = 0,
    candidates_per_iteration: int = 1,
    sibling_candidates: list | None = None,
) -> str:
    # explicit encoding="utf-8": pathlib defaults to the OS locale codec,
    # which on Windows is cp1252 and cannot decode the Chinese-language
    # comments the organizer's original kit files still carry
    data_py = (pipeline_dir / "data.py").read_text(encoding="utf-8")
    baseline_py = (pipeline_dir / "baseline.py").read_text(encoding="utf-8")

    parts = [
        "You are the reflect+revise step of an autonomous ML research agent "
        "iterating on a recommender-systems pipeline. Propose exactly ONE "
        "concrete, testable change for this iteration.",
        "",
        "## Data profile",
        build_data_profile_text(baseline_ref),
        "",
        "## Already tried (do not repeat)",
        KNOWN_DEAD_ENDS,
        "## Suggested directions (organizer-ranked, not mandatory)",
        KNOWN_HEADROOM_DIRECTIONS,
        "",
        "## Iteration history so far",
        format_history_summary(history),
        "",
    ]

    sibling_text = format_sibling_candidates(sibling_candidates or [], candidate_idx, candidates_per_iteration)
    if sibling_text:
        parts += [
            "## This iteration's other candidates",
            sibling_text,
            "",
        ]

    parts += [
        "## Current best",
        (
            f"Best validation primary reached so far: {best_valid_primary:.4f}. "
            "The code below (data.py / baseline.py) IS that best-known state -- "
            "an iteration that doesn't beat this score is discarded and never "
            "becomes 'current', so you are always editing from the best point "
            "reached, never from a regression."
            if best_valid_primary is not None
            else "No successful iteration yet -- this is the first one "
            "(reproduce the baseline, or make your first real change)."
        ),
        "",
        "## Current data.py",
        "```python", data_py, "```",
        "",
        "## Current baseline.py",
        "```python", baseline_py, "```",
    ]

    if last_error:
        parts += [
            "",
            "## Previous attempt this iteration FAILED -- fix it",
            "Your last proposed change for THIS iteration raised an error. "
            "Fix the actual bug, don't abandon the hypothesis unless the "
            "traceback shows the whole approach is unworkable.",
            "```", last_error, "```",
        ]

    parts += [
        "",
        "## Required response format",
        "Respond with ONLY a JSON object (no markdown fences, no prose "
        "outside it), matching this shape exactly:",
        '{'
        '"hypothesis": "one or two sentences: what you are trying and why",'
        '"target_stage": "feature_engineering|model_architecture|training_strategy|evaluation_loop",'
        '"target_file": "data.py" or "baseline.py",'
        '"expected_valid_primary_delta": 0.004,'
        '"new_content": "the COMPLETE new content of target_file, as a single string"'
        '}',
        "",
        "expected_valid_primary_delta is YOUR OWN prediction: a signed number "
        "estimating how much validation primary will change versus the "
        "current best above (e.g. 0.004 if you expect a small gain, -0.002 "
        "if you're testing something you suspect may not help, 0.0 if "
        "genuinely unsure). This is compared against what actually happens "
        "so a calibration gap is visible over time -- an honest, reasoned "
        "estimate is more useful here than an optimistic one.",
        "",
        "Constraints on new_content: it must remain valid, self-contained "
        "Python using only numpy + the standard library (no torch/pandas/"
        "sklearn -- the whole kit is numpy-only by design so it stays fast "
        "and dependency-free). It MUST preserve the required interface: "
        "data.py must still expose load(data_dir), baseline.py must still "
        "expose run_fm(splits, verbose=False) -> {'valid': {...}} with "
        "GAUC/nDCG@5/primary keys, computed via evaluate.py's evaluate() "
        "(import it, do not reimplement scoring), AND "
        "train_and_predict(splits, predict_split='test') -> (metrics, "
        "scores) with the same metrics shape plus a 1-D scores array for "
        "predict_split -- this second function is only called once at the "
        "very end to build the submission file, but must keep working even "
        "if you replace the model entirely. Only rewrite the ONE file "
        "named in target_file; leave the other file's logic conceptually "
        "compatible with what's already there.",
        "",
        "Hidden test set: run_fm() is called every iteration with the "
        "TEST SPLIT REMOVED from splits -- it will not be present as a key "
        "at all, by design. Per the challenge rules, the hidden test set "
        "must never be touched during development, only scored once at the "
        "very end. Do not add any code to run_fm() that reads splits['test'] "
        "or otherwise tries to evaluate against it -- it isn't there during "
        "iteration, and reintroducing that access (even just to print it) "
        "would defeat the point. train_and_predict() is the one function "
        "allowed to use the test split, since it only ever runs once, "
        "after the whole iterative run has already stopped.",
        "",
        "Isolated change: touch only what your hypothesis is actually "
        "about. Every hyperparameter, constant, or code path NOT named in "
        "your hypothesis must be copied over unchanged from the current "
        "best file shown above -- do not also nudge the learning rate, "
        "epoch count, embedding size, or similar 'while I'm in there' "
        "tweaks alongside your real change. A failed iteration that "
        "changed five things at once tells us nothing about which of the "
        "five mattered; an isolated change makes the actual_delta a clean "
        "read on your specific hypothesis, and makes it possible for a "
        "later iteration to keep the part that worked.",
    ]
    return "\n".join(parts)
