"""The one wrapped LLM client (AGENT_STRATEGY.md hard requirement: every LLM call goes through
here, and every call is logged with input/output tokens for the cost report).

Talks to a local Ollama server's NATIVE /api/chat endpoint — not the OpenAI-compatible shim —
because Ollama's native response always includes `prompt_eval_count`/`eval_count`, i.e. exact
input/output token counts, with no extra tokenizer bookkeeping needed on our side. If we ever
swap providers, only this file changes: every call site uses the same `call(...)` shape.

No third-party HTTP dependency (stdlib `urllib` only) — this repo's only dependency beyond the
stdlib is numpy (see CLAUDE.md); adding `requests` just for this would break that.
"""
import json
import os
import time
import urllib.error
import urllib.request

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("AGENT_LLM_MODEL", "qwen2.5-coder:7b")
LEDGER_PATH = os.environ.get("AGENT_TOKEN_LEDGER", os.path.join("runs", "token_ledger.jsonl"))


class LLMError(RuntimeError):
    """Raised when the Ollama call fails, times out, or returns something unusable.
    Callers (hypothesis_agent, error_recovery, ...) are expected to let this propagate into
    the loop's error/recovery path rather than catching it locally."""


def _append_ledger(record):
    d = os.path.dirname(LEDGER_PATH)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def call(system, messages, *, json_mode=False, temperature=0.2, caller="unknown", timeout=120):
    """One LLM call.

    system:      system prompt (str)
    messages:    [{"role": "user"|"assistant", "content": str}, ...] — Anthropic-shaped on
                 purpose, so a future provider swap doesn't touch every call site.
    json_mode:   pass True to make Ollama constrain output to valid JSON (qwen2.5-coder:7b is
                 small enough that this matters a lot for the structured hypothesis pipeline —
                 don't rely on prompt wording alone for that).
    caller:      free-text tag (e.g. "hypothesis_agent", "error_recovery") — logged per call so
                 the cost report can break spend down by agent stage.

    Returns (text, usage) where usage = {"input_tokens": int, "output_tokens": int}.
    Every call — success or failure — is appended to runs/token_ledger.jsonl.
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}] + list(messages),
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"

    t0 = time.time()
    record = {"ts": t0, "caller": caller, "model": MODEL, "json_mode": json_mode}
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        record.update(ok=False, error=str(e), latency_s=time.time() - t0)
        _append_ledger(record)
        raise LLMError(f"Ollama call failed (host={OLLAMA_HOST}, model={MODEL}): {e}") from e

    text = body.get("message", {}).get("content", "")
    usage = {
        "input_tokens": body.get("prompt_eval_count", 0),
        "output_tokens": body.get("eval_count", 0),
    }
    record.update(ok=True, latency_s=time.time() - t0, chars_out=len(text), **usage)
    _append_ledger(record)

    if not text:
        raise LLMError(f"Ollama returned an empty message body: {body!r}")
    return text, usage


if __name__ == "__main__":
    # Smoke test: `python -m agent.llm_client` — confirms the Ollama link works end to end
    # (server reachable, model loaded, tokens logged) before anything else in the loop depends
    # on it.
    text, usage = call(
        system="You are a terse test assistant.",
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        caller="smoke_test",
    )
    print(f"response: {text!r}")
    print(f"usage:    {usage}")
    print(f"ledger:   {os.path.abspath(LEDGER_PATH)}")
