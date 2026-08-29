"""Numpy-safe JSON serialization — shared by every module that writes a runs/*.jsonl file.

Found live (not by static review): evaluate.evaluate() returns numpy.float32 for GAUC/nDCG@5/
primary (numpy arithmetic internally), and plain json.dumps() cannot serialize that — the first
real accepted iteration in end-to-end testing crashed archivist.append_record() on exactly this.
Cannot fix it at the source (evaluate.py is a fixed, do-not-modify contract per CLAUDE.md), so the
fix belongs here, at the serialization boundary, applied everywhere a record might carry a numpy
scalar through (metrics dicts especially).
"""
import json


def json_default(o):
    """Pass as json.dumps(obj, default=json_default). Handles any numpy scalar (float32, float64,
    int64, bool_, ...) via the .item() method every numpy scalar type provides — duck-typed
    rather than importing numpy here, since this module has no other reason to depend on it."""
    if hasattr(o, 'item'):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def dumps(obj, **kwargs):
    return json.dumps(obj, default=json_default, **kwargs)
