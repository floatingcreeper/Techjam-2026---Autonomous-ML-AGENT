"""Phase 6 — human-facing run/log viewer (AGENT_STRATEGY.md). Reads the files every other phase
already writes (runs/experiment_log.jsonl, token_ledger.jsonl, state.json,
manual_interventions.jsonl) and renders ONE self-contained, dependency-free runs/dashboard.html —
no server, no external library, works fully offline (even the progression chart is hand-rolled
inline SVG, not a charting library — matches the "static file, no server" decision in
AGENT_STRATEGY.md Phase 6/decision #15).

Regenerated as the last step of every agent/archivist.py call (see its _regenerate_dashboard()),
so a human can just leave the file open in a browser and refresh. Can also be run standalone any
time: `python -m agent.viewer`.

This is a plain local file, not published through Claude's Artifact tool — it's a live,
continuously-regenerated operational view the user opens themselves, not a one-off deliverable
handed to someone else, so it gets a full standalone HTML document here (its own <!DOCTYPE>/
<html>/<head>/<body>), unlike Artifact pages which get that wrapper automatically.
"""
import html
import json
import os

from agent import cost_report, manual_intervention
from agent.config import CONVERGENCE_EPSILON, CONVERGENCE_N

EXPERIMENT_LOG_PATH = os.path.join('runs', 'experiment_log.jsonl')
STATE_PATH = os.path.join('runs', 'state.json')
OUTPUT_PATH = os.path.join('runs', 'dashboard.html')
BASELINE_SCORES_PATH = 'baseline_scores.json'


def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _baseline_reference():
    d = _load_json(BASELINE_SCORES_PATH, {}) or {}
    scores = d.get('scores', {})
    return {
        'fm_official': (scores.get('fm_official') or {}).get('valid', {}),
        'oracle_ceiling': (scores.get('oracle_ceiling') or {}).get('valid', {}),
    }


def _esc(s):
    return html.escape(str(s)) if s is not None else ''


def _svg_chart(records, baseline, width=760, height=200):
    """Best-primary-so-far over iterations, with dashed reference lines for fm_official and
    oracle_ceiling. Only ACCEPTED iterations move the line — a rejected iteration doesn't change
    current-best, so it shouldn't look like it did."""
    if not records:
        return '<p class="muted">No iterations logged yet.</p>'

    pad_l, pad_r, pad_t, pad_b = 44, 24, 16, 24
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    best_so_far = []
    running_best = None
    for r in records:
        m = (r.get('metrics') or {}).get('valid') or {}
        if r.get('accepted') and 'primary' in m:
            running_best = m['primary']
        best_so_far.append(running_best)

    accepted_vals = [v for v in best_so_far if v is not None]
    if not accepted_vals:
        return '<p class="muted">No accepted iterations yet — nothing to chart.</p>'

    ref = {'fm_official': baseline['fm_official'].get('primary'),
           'oracle': baseline['oracle_ceiling'].get('primary')}
    all_vals = accepted_vals + [v for v in ref.values() if v is not None]
    lo, hi = min(all_vals + [0.0]), max(all_vals) * 1.05
    n = len(records)

    def x_of(i):
        return pad_l + (i / max(n - 1, 1)) * plot_w

    def y_of(v):
        if hi == lo:
            return pad_t + plot_h / 2
        return pad_t + plot_h - ((v - lo) / (hi - lo)) * plot_h

    path_parts, dots = [], []
    for i, v in enumerate(best_so_far):
        if v is None:
            continue
        x, y = x_of(i), y_of(v)
        path_parts.append(f"{'M' if not path_parts else 'L'}{x:.1f},{y:.1f}")
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" class="point">'
                     f'<title>iter {records[i]["iteration"]}: {v:.4f}</title></circle>')

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'aria-label="best validation primary over iterations">',
             f'<rect x="0" y="0" width="{width}" height="{height}" class="chart-bg"/>']
    for label, val, cls in (('fm_official', ref['fm_official'], 'ref-baseline'),
                             ('oracle ceiling', ref['oracle'], 'ref-oracle')):
        if val is not None:
            y = y_of(val)
            parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
                          f'class="{cls}"/>')
            parts.append(f'<text x="{width - pad_r}" y="{y - 3:.1f}" class="ref-label" '
                          f'text-anchor="end">{_esc(label)} {val:.4f}</text>')
    if path_parts:
        parts.append(f'<path d="{" ".join(path_parts)}" class="best-line" fill="none"/>')
    parts.extend(dots)
    parts.append(f'<text x="{pad_l}" y="{height - 4}" class="axis-label">iter 1</text>')
    parts.append(f'<text x="{width - pad_r}" y="{height - 4}" text-anchor="end" '
                 f'class="axis-label">iter {n}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def _iteration_rows(records):
    rows = []
    for r in records:
        hyp = r.get('hypothesis') or {}
        m = (r.get('metrics') or {}).get('valid') or {}
        errs = r.get('error_events') or []
        badge = 'accepted' if r.get('accepted') else 'rejected'
        primary = f"{m['primary']:.4f}" if 'primary' in m else '—'
        err_html = ''
        if errs:
            last = errs[-1]
            reason = last.get('fix_description') or last.get('error_text') or ''
            err_html = f'<div class="err">{_esc(reason)}</div>'
        tok = r.get('token_cost') or {}
        tok_str = f"{tok.get('input_tokens', 0):,} in / {tok.get('output_tokens', 0):,} out"
        rows.append(f'''
        <tr class="{badge}">
          <td>{r['iteration']}</td>
          <td><span class="badge {badge}">{badge}</span></td>
          <td>{_esc(hyp.get('target_stage', '—'))}</td>
          <td class="statement">{_esc(hyp.get('statement', '(hypothesis generation failed)'))}
            {err_html}</td>
          <td class="num">{primary}</td>
          <td class="num">{tok_str}</td>
          <td class="num">{r.get('wall_clock_s', 0):.0f}s</td>
        </tr>''')
    return ''.join(rows)


def render(*, log_path=EXPERIMENT_LOG_PATH, state_path=STATE_PATH):
    records = _load_jsonl(log_path)
    state = _load_json(state_path, {'last_completed_iteration': 0, 'current_best': None}) or {}
    baseline = _baseline_reference()
    cost = cost_report.summarize()
    interventions = manual_intervention.all_events()

    current_best = state.get('current_best')
    cb_primary = current_best['primary'] if current_best else None
    fm_primary = baseline['fm_official'].get('primary')
    oracle_primary = baseline['oracle_ceiling'].get('primary')

    stale = 0
    for r in reversed(records):
        if r.get('accepted'):
            break
        stale += 1

    def rel(v, ref):
        return f"{v - ref:+.4f}" if v is not None and ref is not None else '—'

    t = cost['total']
    caller_rows = ''.join(
        f"<tr><td>{_esc(c)}</td><td class='num'>{b['calls']}</td>"
        f"<td class='num'>{b['input_tokens']:,}</td><td class='num'>{b['output_tokens']:,}</td>"
        f"<td class='num'>{b['latency_s']:.1f}s</td><td class='num'>{b['failed']}</td></tr>"
        for c, b in sorted(cost['by_caller'].items())
    ) or '<tr><td colspan="6" class="muted">No LLM calls logged yet.</td></tr>'

    interventions_rows = ''.join(
        f"<tr><td>{_esc(e.get('timestamp'))}</td><td>{_esc(e.get('reason'))}</td></tr>"
        for e in interventions
    ) or '<tr><td colspan="2" class="muted">None recorded.</td></tr>'

    iteration_rows = _iteration_rows(records) or \
        '<tr><td colspan="7" class="muted">No iterations logged yet.</td></tr>'

    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Run Dashboard</title>
<style>
  :root {{
    --bg: #f7f7f5; --panel: #ffffff; --text: #1c1c1c; --muted: #6b6b6b; --border: #e2e2e0;
    --accent: #2f6f4f; --accept-bg: #e3f0e8; --reject-bg: #f6e6e3;
    --accept: #2f6f4f; --reject: #a4453a; --ref: #9a9a9a;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #16181b; --panel: #1e2125; --text: #e8e8e6; --muted: #9a9a98; --border: #33373c;
      --accent: #6fbf95; --accept-bg: #223129; --reject-bg: #35211f;
      --accept: #6fbf95; --reject: #e08a7d; --ref: #7a7a78;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--text); margin: 0; padding: 24px;
    font: 14px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  h2 {{ font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted);
       margin: 28px 0 8px; }}
  .sub {{ color: var(--muted); margin-bottom: 20px; }}
  .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
            padding: 16px 20px; margin-bottom: 16px; overflow-x: auto; }}
  .stat-row {{ display: flex; gap: 28px; flex-wrap: wrap; }}
  .stat {{ min-width: 140px; }}
  .stat .label {{ color: var(--muted); font-size: 12px; }}
  .stat .value {{ font-size: 20px; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border);
            vertical-align: top; }}
  th {{ color: var(--muted); font-weight: 500; font-size: 12px; text-transform: uppercase; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .statement {{ max-width: 420px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px;
            font-weight: 600; white-space: nowrap; }}
  .badge.accepted {{ background: var(--accept-bg); color: var(--accept); }}
  .badge.rejected {{ background: var(--reject-bg); color: var(--reject); }}
  tr.rejected td {{ color: var(--muted); }}
  .err {{ color: var(--reject); font-size: 12px; margin-top: 2px; }}
  .muted {{ color: var(--muted); }}
  .chart-bg {{ fill: var(--panel); }}
  .best-line {{ stroke: var(--accent); stroke-width: 2; }}
  .point {{ fill: var(--accent); }}
  .ref-baseline {{ stroke: var(--ref); stroke-dasharray: 4 3; }}
  .ref-oracle {{ stroke: var(--ref); stroke-dasharray: 1 3; }}
  .ref-label, .axis-label {{ fill: var(--muted); font-size: 10px; }}
</style>
</head>
<body>

<h1>Autonomous ML Agent &mdash; Run Dashboard</h1>
<div class="sub">KuaiRand-Pure FM tuning loop &middot; regenerated after every iteration &middot;
  {len(records)} iteration(s) logged &middot; last completed: {state.get('last_completed_iteration', 0)}</div>

<div class="panel">
  <div class="stat-row">
    <div class="stat"><div class="label">Current-best primary (valid)</div>
      <div class="value">{f'{cb_primary:.4f}' if cb_primary is not None else '—'}</div></div>
    <div class="stat"><div class="label">vs fm_official baseline</div>
      <div class="value">{rel(cb_primary, fm_primary)}</div></div>
    <div class="stat"><div class="label">vs oracle ceiling</div>
      <div class="value">{rel(cb_primary, oracle_primary)}</div></div>
    <div class="stat"><div class="label">Convergence</div>
      <div class="value">{stale} / {CONVERGENCE_N} stale (&epsilon;={CONVERGENCE_EPSILON})</div></div>
    <div class="stat"><div class="label">Manual interventions</div>
      <div class="value">{manual_intervention.count()}</div></div>
  </div>
</div>

<div class="panel">{_svg_chart(records, baseline)}</div>

<h2>Iterations</h2>
<div class="panel">
  <table>
    <thead><tr><th>#</th><th>Status</th><th>Stage</th><th>Hypothesis</th>
      <th class="num">Primary</th><th class="num">Tokens</th><th class="num">Wall-clock</th></tr></thead>
    <tbody>{iteration_rows}</tbody>
  </table>
</div>

<h2>Resource consumption</h2>
<div class="panel">
  <div class="stat-row">
    <div class="stat"><div class="label">LLM calls</div>
      <div class="value">{t['calls']} ({t['failed']} failed)</div></div>
    <div class="stat"><div class="label">Tokens</div>
      <div class="value">{t['input_tokens']:,} in / {t['output_tokens']:,} out</div></div>
    <div class="stat"><div class="label">LLM wall-clock</div>
      <div class="value">{t['latency_s']:.0f}s</div></div>
    <div class="stat"><div class="label">GPU-hours</div>
      <div class="value">0.00 (CPU-only)</div></div>
  </div>
  <table style="margin-top:12px">
    <thead><tr><th>Caller</th><th class="num">Calls</th><th class="num">In</th>
      <th class="num">Out</th><th class="num">Latency</th><th class="num">Failed</th></tr></thead>
    <tbody>{caller_rows}</tbody>
  </table>
</div>

<h2>Manual interventions</h2>
<div class="panel">
  <table>
    <thead><tr><th>When</th><th>Reason</th></tr></thead>
    <tbody>{interventions_rows}</tbody>
  </table>
</div>

</body>
</html>
'''


def regenerate(*, output_path=OUTPUT_PATH, **kwargs):
    doc = render(**kwargs)
    d = os.path.dirname(output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write(doc)
    return output_path


if __name__ == '__main__':
    path = regenerate()
    print(f"Wrote {path}")
