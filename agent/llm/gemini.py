"""GeminiDriver -- google-genai backend with structured
output, retry/backoff, and token accounting. google is imported lazily so the MockDriver
path needs nothing installed. Requires env GEMINI_API_KEY for live runs.
"""
from __future__ import annotations
import os, time
from agent.llm.driver import LLMDriver, Usage, LLMUnavailable


class GeminiDriver(LLMDriver):
    def __init__(self, max_retries: int = 5, api_key: str | None = None):
        self.max_retries = max_retries
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = None

    def _client_(self):
        if self._client is None:
            from google import genai
            if not self._api_key:
                raise LLMUnavailable("GEMINI_API_KEY not set")
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(self, *, role, system, user, schema, model, temperature=0.4, thinking=None):
        from google.genai import types
        client = self._client_()
        gc = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        if thinking is not None:
            gc.thinking_config = types.ThinkingConfig(thinking_budget=thinking)

        last = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                r = client.models.generate_content(model=model, contents=user, config=gc)
                obj = r.parsed
                if obj is None:
                    raise ValueError("Gemini returned no parseable structured output")
                um = getattr(r, "usage_metadata", None)
                usage = Usage(
                    input_tokens=int(getattr(um, "prompt_token_count", 0) or 0),
                    output_tokens=int(getattr(um, "candidates_token_count", 0) or 0),
                    model=model, seconds=round(time.time() - t0, 2),
                )
                return obj, usage
            except Exception as e:                       # transient API / parse errors
                last = e
                time.sleep(min(2 ** attempt, 30))
        raise LLMUnavailable(f"Gemini failed after {self.max_retries} retries: {last}")
