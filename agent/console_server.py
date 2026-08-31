"""Research Console server -- docs/SYSTEM.md §19 (Research Console).

Serves `dashboard/research-console.html` and the real artifacts of a run. Standard library only
(no Flask, no build step), so it runs anywhere the agent runs.

    python -m agent.console_server                 # newest run, opens a browser
    python -m agent.console_server --run <run_id>  # a specific run
    python -m agent.console_server --port 8djust   # custom port

Endpoints
    /                       the console
    /api/runs               every run under runs/ with its headline numbers
    /api/events?run=<id>&since=<seq>   incremental event stream (this is what makes LIVE work)
    /api/log?run=<id>       run_log.jsonl (per-node detail: diffs, provenance, config)
    /api/report?run=<id>    resource_report.json

It only ever READS files the agent wrote. There is no second source of truth, and the server never
computes a metric of its own.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PAGE = ROOT / "dashboard" / "research-console.html"


def _runs():
    out = []
    if not RUNS.exists():
        return out
    for d in sorted(RUNS.glob("run_*"), reverse=True):
        ev = d / "events.jsonl"
        rep = d / "resource_report.json"
        item = {"id": d.name, "has_events": ev.exists(), "mtime": d.stat().st_mtime}
        if rep.exists():
            try:
                r = json.loads(rep.read_text())
                b = r.get("benchmark", {})
                item.update({
                    "stop_reason": r.get("stop_reason"),
                    "experiments": b.get("experiments_executed"),
                    "proposals": b.get("proposal_attempts"),
                    "final_tuned": r.get("final_valid_tuned"),
                    "final_honest": (r.get("final_valid_honest") or {}).get("estimate"),
                })
            except (OSError, ValueError):
                pass
        out.append(item)
    return out


def _read_events(run_id, since=0):
    p = RUNS / run_id / "events.jsonl"
    out = []
    if not p.exists():
        return out
    try:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue                 # a run mid-write leaves one partial line
                if e.get("seq", 0) > since:
                    out.append(e)
    except OSError:
        pass
    return out


def _read_jsonl(run_id, name):
    p = RUNS / run_id / name
    out = []
    if p.exists():
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            pass
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                 # keep the agent's own logs readable
        pass

    def _send(self, body, ctype="application/json", code=200):
        data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        run = (q.get("run") or [None])[0]
        try:
            if u.path in ("/", "/index.html"):
                if not PAGE.exists():
                    return self._send(b"research-console.html not found", "text/plain", 404)
                return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/runs":
                return self._send({"runs": _runs()})
            if u.path == "/api/events":
                since = int((q.get("since") or ["0"])[0])
                return self._send({"run": run, "events": _read_events(run, since)})
            if u.path == "/api/log":
                return self._send({"run": run, "records": _read_jsonl(run, "run_log.jsonl")})
            if u.path == "/api/report":
                p = RUNS / (run or "") / "resource_report.json"
                if p.exists():
                    return self._send(json.loads(p.read_text()))
                return self._send({})
            return self._send(b"not found", "text/plain", 404)
        except Exception as e:                 # a UI request must never take the server down
            return self._send({"error": repr(e)}, code=500)


def main():
    ap = argparse.ArgumentParser(description="Research Console (live + replay)")
    ap.add_argument("--port", type=int, default=8712)
    ap.add_argument("--run", default=None, help="run id to open (default: newest)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    runs = _runs()
    run = a.run or (runs[0]["id"] if runs else "")
    url = f"http://127.0.0.1:{a.port}/" + (f"?run={run}" if run else "")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Research Console -> {url}")
    print(f"  serving {len(runs)} run(s) from {RUNS}")
    print("  Ctrl-C to stop")
    if not a.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
