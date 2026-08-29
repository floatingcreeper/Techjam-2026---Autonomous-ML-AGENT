"""Shared prompt-templating + JSON-response parsing helpers — extracted after code review found
the same ~30 lines duplicated near-verbatim across agent/hypothesis_agent.py, agent/coding_agent.py,
and agent/error_recovery.py. Three independent copies meant a fix to fence-stripping or the
missing-placeholder error message in one wouldn't apply to the other two — this module is the one
place that logic lives now.
"""
import json
import re

PLACEHOLDER_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def load_template(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def fill_template(template, mapping, *, source_name='template'):
    missing = []

    def _sub(m):
        key = m.group(1)
        if key not in mapping:
            missing.append(key)
            return m.group(0)
        return str(mapping[key])

    filled = PLACEHOLDER_RE.sub(_sub, template)
    if missing:
        raise KeyError(f"{source_name} referenced placeholders not supplied: {sorted(set(missing))}")
    return filled


def strip_fences(text):
    t = text.strip()
    if t.startswith('```'):
        t = re.sub(r'^```(?:json)?\s*', '', t)
        t = re.sub(r'```\s*$', '', t)
    return t.strip()


def parse_json(text):
    """Returns (obj, error) — error is None on success, else a human-readable string."""
    try:
        return json.loads(strip_fences(text)), None
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
