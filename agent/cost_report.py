"""Resource-consumption report — reads runs/token_ledger.jsonl (every LLM call, logged
unconditionally by agent/llm_client.py, success or failure) and summarizes it. This is what makes
the "total LLM tokens + GPU-hours spent" scored dimension checkable at any point, not just totted
up once at the end of a run.

GPU-hours: always 0.0 here — this repo is CPU-only (numpy, no torch/CUDA). Logged explicitly as a
real field, not omitted, so the report has something to point at rather than a silent gap.
Wall-clock LLM time is tracked per-call (`latency_s` in the ledger) and summed as a proxy.

Usage: `python -m agent.cost_report` any time, from anywhere in the loop.
"""
import json
import os
from collections import defaultdict

LEDGER_PATH = os.environ.get('AGENT_TOKEN_LEDGER', os.path.join('runs', 'token_ledger.jsonl'))


def load_ledger(path=LEDGER_PATH):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize(path=LEDGER_PATH):
    records = load_ledger(path)
    by_caller = defaultdict(lambda: {'calls': 0, 'input_tokens': 0, 'output_tokens': 0,
                                      'latency_s': 0.0, 'failed': 0})
    total = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'latency_s': 0.0, 'failed': 0,
             'gpu_hours': 0.0}

    for r in records:
        caller = r.get('caller', 'unknown')
        b = by_caller[caller]
        b['calls'] += 1
        total['calls'] += 1
        if r.get('ok'):
            b['input_tokens'] += r.get('input_tokens', 0)
            b['output_tokens'] += r.get('output_tokens', 0)
            total['input_tokens'] += r.get('input_tokens', 0)
            total['output_tokens'] += r.get('output_tokens', 0)
        else:
            b['failed'] += 1
            total['failed'] += 1
        b['latency_s'] += r.get('latency_s', 0.0)
        total['latency_s'] += r.get('latency_s', 0.0)

    return {'total': total, 'by_caller': dict(by_caller)}


def format_report(path=LEDGER_PATH):
    s = summarize(path)
    t = s['total']
    lines = [
        f"Resource consumption ({path}):",
        f"  LLM calls: {t['calls']} ({t['failed']} failed)",
        f"  Tokens: {t['input_tokens']:,} in / {t['output_tokens']:,} out "
        f"({t['input_tokens'] + t['output_tokens']:,} total)",
        f"  Wall-clock LLM time: {t['latency_s']:.1f}s",
        f"  GPU-hours: {t['gpu_hours']:.2f} (CPU-only repo - always 0)",
    ]
    if s['by_caller']:
        lines.append("  By caller:")
        for caller, b in sorted(s['by_caller'].items()):
            lines.append(f"    {caller:24s} calls={b['calls']:3d}  in={b['input_tokens']:6,}  "
                          f"out={b['output_tokens']:6,}  latency={b['latency_s']:6.1f}s  "
                          f"failed={b['failed']}")
    return "\n".join(lines)


if __name__ == '__main__':
    print(format_report())
