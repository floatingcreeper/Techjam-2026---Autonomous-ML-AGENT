"""
HARNESS-OWNED FILE — not one of the two files the agent edits.

This is the stable contract between the controller and whatever data.py /
baseline.py currently look like. It must keep working no matter what the
agent changes inside data.py or baseline.py, which is why it only relies
on two entrypoints that are documented as invariants (see pipeline/README.md
in this same folder):

    data.load(data_dir) -> {'train': [...], 'valid': [...], 'test': [...]}
    baseline.run_fm(splits, verbose=False) -> {'valid': {...}}

HIDDEN-TEST COMPLIANCE — the single most important line in this file:

    data.load()'s 'test' split IS the challenge's hidden test set (its
    official baseline numbers match baseline_scores.json exactly). The
    challenge rules state four separate times that it must never be
    touched during development, only scored once at the very end. So this
    file POPS 'test' OUT OF splits before calling run_fm(). run_fm() never
    receives it -- it is not merely "not asked to score it," the rows are
    structurally absent, so a rewritten baseline.py cannot score against
    the hidden test set even if it tries to.

    A rewritten run_fm() that still expects a 'test' split will raise
    KeyError, BY DESIGN. The controller treats that like any other
    recoverable iteration error and retries with the traceback fed back
    (agent/context.py's prompt already tells the agent not to reach for
    splits['test'] inside run_fm()).

    agent/finalize.py is the ONE path allowed to see the test split, and
    it does not go through this file -- it calls data.load() directly,
    once, after the whole iterative run has already stopped.

    agent/sandbox.py independently rejects any output carrying a 'test'
    key as a hard compliance violation. That redundancy is deliberate:
    this exact stripping was silently lost once to a git checkout that
    restored a pre-fix commit, and nothing caught it until a manual audit.

Prints exactly one line of JSON to stdout: {'valid': {...},
'wall_clock_s': float}. The controller reads that one line rather than
parsing baseline.py's human-readable epoch-by-epoch prints, so it doesn't
break if the agent changes what baseline.py prints for a human to read.
"""
import sys
import json
import time

sys.path.insert(0, '.')


def _jsonable(obj):
    """Recursively converts numpy scalar types (float32/int64/etc, which
    evaluate.py's GAUC/nDCG@5/primary calculations commonly produce) into
    plain Python types json.dumps can handle."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar (float32, int64, ...)
        return obj.item()
    return obj


def main():
    data_dir = sys.argv[1]
    t0 = time.time()

    from data import load
    import baseline as B

    splits = load(data_dir)
    # THE enforcement point -- see this module's HIDDEN-TEST COMPLIANCE
    # note. Do not "simplify" this back to passing `splits` directly.
    iter_splits = {k: v for k, v in splits.items() if k != 'test'}
    result = B.run_fm(iter_splits, verbose=False)
    result['wall_clock_s'] = time.time() - t0
    print(json.dumps(_jsonable(result)))


if __name__ == '__main__':
    main()
