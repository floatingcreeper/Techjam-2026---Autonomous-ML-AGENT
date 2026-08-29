"""
Runs once, after the controller loop stops. Takes best/ (the validation-best
pipeline snapshot the controller saved during the run -- NOT necessarily the
last iteration) and:
  1. calls baseline.train_and_predict() to get real prediction scores on the
     requested split, using the same best-validation checkpoint the run's
     metrics reflect (see the AGENT-AUTOMATION CONTRACT docstring on that
     function in pipeline/baseline.py -- this is what keeps finalize.py
     working even if the agent has replaced FM with a different model)
  2. writes them as a submission CSV in the official format
  3. runs the equivalent of `submit.py --check` against it before calling
     anything final

This is the ONE place in the whole codebase allowed to score against the
hidden test set (see agent/controller.py's docstring and
pipeline/_run_iteration.py's HIDDEN-TEST COMPLIANCE note -- every iteration
during the run itself is scored on the validation split only). When
`split == "test"`, the real test score computed here is patched into
best/_best_meta.json's test_primary field, which every iteration leaves as
None -- this is the only code path that ever fills it in.
"""
import importlib
import json
import sys
from pathlib import Path

from agent.controller import _load_best_meta, _write_best_meta


def _jsonable(obj):
    """Recursively converts numpy scalar types (float32/int64/etc -- what
    evaluate.py's GAUC/nDCG@5/primary commonly come back as) into plain
    Python types json.dumps can handle. Same helper as
    pipeline/_run_iteration.py's, duplicated rather than shared since that
    file runs in an isolated scratch copy with no access to this package."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def finalize(best_dir: Path, data_dir: str, out_csv: Path, split: str = "test") -> dict:
    sys.path.insert(0, str(best_dir))
    for mod in ("data", "baseline", "evaluate", "submit"):
        sys.modules.pop(mod, None)  # force a fresh import of the best/ snapshot's code

    data = importlib.import_module("data")
    baseline = importlib.import_module("baseline")
    submit = importlib.import_module("submit")

    splits = data.load(data_dir)
    metrics, scores = baseline.train_and_predict(splits, predict_split=split)

    submit.write_submission(str(out_csv), splits[split], scores)
    submit.read_submission(str(out_csv), splits[split])  # raises if misaligned/malformed -- this IS submit.py --check

    if split == "test" and isinstance(metrics, dict) and "test" in metrics and "primary" in metrics["test"]:
        prior_meta = _load_best_meta(best_dir)
        if prior_meta is not None:
            _write_best_meta(
                best_dir,
                valid_primary=prior_meta["valid_primary"],
                iteration=prior_meta.get("iteration"),
                hypothesis=prior_meta.get("hypothesis"),
                test_primary=metrics["test"]["primary"],
            )

    return _jsonable({
        "split": split,
        "submission_path": str(out_csv),
        "rows": len(splits[split]),
        "metrics": metrics,
    })


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_dir", default="best")
    ap.add_argument("--data_dir", default="../kuairand-starter-kit/KuaiRand-Pure/data")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    a = ap.parse_args()
    result = finalize(Path(a.best_dir), a.data_dir, Path(a.out), a.split)
    print(json.dumps(result, indent=2))
