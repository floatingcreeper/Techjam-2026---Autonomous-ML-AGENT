"""
HARNESS-OWNED FILE — not one of the two files the agent edits.

This is the stable contract between the controller and whatever data.py /
baseline.py currently look like. It must keep working no matter what the
agent changes inside data.py or baseline.py, which is why it only relies
on two entrypoints that are documented as invariants (see pipeline/README.md
in this same folder):

    data.load(data_dir) -> {'train': [...], 'valid': [...], 'test': [...]}
    baseline.run_fm(splits, verbose=False) -> {'valid': {...}, 'test': {...}}

Prints exactly one line of JSON to stdout: {'valid': {...}, 'test': {...},
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
    result = B.run_fm(splits, verbose=False)
    result['wall_clock_s'] = time.time() - t0
    print(json.dumps(_jsonable(result)))


if __name__ == '__main__':
    main()
