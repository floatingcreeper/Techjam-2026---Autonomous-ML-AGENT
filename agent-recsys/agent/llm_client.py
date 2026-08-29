"""
The one place that calls out to an LLM. Two implementations:

  - AnthropicClient  -- real calls via the Anthropic API. Requires
                         ANTHROPIC_API_KEY in the environment. Swap this
                         class (or add a sibling) if you want a different
                         provider -- controller.py only depends on the
                         propose(prompt) -> Proposal interface below, not
                         on which vendor answers it.
  - DryRunClient      -- no network calls, no API key. Cycles through a
                         short fixed list of legitimate, hand-written
                         hypotheses. Its only purpose is to prove the
                         controller/sandbox/logging/convergence machinery
                         works end-to-end before you spend real LLM tokens
                         on it. Use --dry-run to select it (see run_agent.py).

Both return a Proposal with token counts populated (DryRunClient reports 0
tokens, since no LLM was actually called), so agent/finalize.py's resource
summary is accurate either way.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Proposal:
    hypothesis: str
    target_stage: str
    target_file: str          # "data.py" or "baseline.py"
    new_content: str
    input_tokens: int = 0
    output_tokens: int = 0
    expected_delta: float = 0.0   # model's own predicted change in valid primary


class LLMError(Exception):
    """A recoverable failure: the API call itself succeeded, but the
    response was unusable (truncated, malformed JSON, ...). Worth retrying
    with the traceback fed back to the model -- a different generation
    might well come out clean."""
    pass


class LLMFatalError(LLMError):
    """An unrecoverable failure: the API call itself never completed
    (network down, bad/expired key, rate limit, quota exhausted, an
    Anthropic-side 5xx, an unknown model name, ...). Retrying immediately
    with no backoff can't fix any of these -- it just repeats the same
    failure (burning real tokens too, for anything that reached the
    server) up to MAX_REPAIR_ATTEMPTS more times. The controller treats
    this as a signal to stop the whole run rather than keep retrying."""
    pass


# Claude Sonnet 5 pricing, USD per million tokens. Used for the running
# cost estimate and the --max_cost_usd kill switch in controller.py -- keep
# this in sync with reality if pricing changes.
PRICE_PER_MTOK_IN = 2.0
PRICE_PER_MTOK_OUT = 10.0


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1e6) * PRICE_PER_MTOK_IN + (output_tokens / 1e6) * PRICE_PER_MTOK_OUT


class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-5"):
        try:
            import anthropic
        except ImportError as e:
            raise LLMError(
                "the 'anthropic' package is not installed -- run "
                "`pip install anthropic` (or `pip install anthropic "
                "--break-system-packages` in this environment)"
            ) from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Export it before running "
                "with a real LLM, or pass --dry-run to test the harness "
                "without one."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def propose(self, prompt: str, iteration: int = 0) -> Proposal:
        # 8192 was too small: the response has to hold a full-file rewrite
        # (baseline.py is 200+ commented lines) INSIDE a JSON string, with
        # every newline/quote escaped -- that alone can exceed 8192 tokens
        # before the model even gets to close the string. Claude Sonnet 5's
        # standard (non-beta) ceiling is 128K output tokens, so 32000 gives
        # a large real-world margin without needing any beta header. Raising
        # this cap does not cost more by itself -- Anthropic bills for
        # tokens actually generated, not for max_tokens.
        #
        # max_tokens=32000 is large enough that the SDK refuses to run it as
        # a plain non-streaming call: a single HTTP request that might take
        # longer than ~10 minutes to complete is fragile (proxies/load
        # balancers can kill a long-idle connection before the full
        # response arrives), so the SDK requires streaming instead, which
        # keeps the connection alive by sending data incrementally. We still
        # just want the whole response at the end, so we open a stream and
        # collect it into the same shape a non-streaming call would give us.
        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=32000,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for _ in stream.text_stream:
                    pass  # drain the stream; we only need the assembled final message
                resp = stream.get_final_message()
        except Exception as e:  # noqa: BLE001 - the API call itself failed (network
            # down, bad/expired key, rate limit, quota exhausted, a 5xx from
            # Anthropic, an unknown model name, ...). None of these are fixed
            # by retrying immediately with no backoff -- see LLMFatalError's
            # docstring. Distinct from the block below, which only handles
            # the call SUCCEEDING but returning unusable content.
            raise LLMFatalError(f"{type(e).__name__}: {e}") from e
        text = "".join(block.text for block in resp.content if block.type == "text")
        if resp.stop_reason == "max_tokens":
            # The response was cut off mid-generation -- json.loads() would
            # just report "unterminated string" here, which reads like a
            # generic malformed-JSON bug. Surface the real cause instead so
            # it's obvious this is a length problem, not a prompting bug.
            raise LLMError(
                "LLM response was truncated: it hit the max_tokens=32000 "
                "cap before finishing its JSON output (stop_reason="
                "'max_tokens'). This usually means the proposed file "
                "rewrite is unusually large. The controller will treat "
                "this like any other failed iteration and retry with the "
                "traceback fed back to the model.\n"
                f"--- partial response (last 2000 chars) ---\n{text[-2000:]}"
            )
        data = _parse_json_response(text)
        try:
            expected_delta = float(data.get("expected_valid_primary_delta", 0.0))
        except (TypeError, ValueError):
            expected_delta = 0.0
        return Proposal(
            hypothesis=data["hypothesis"],
            target_stage=data["target_stage"],
            target_file=data["target_file"],
            new_content=data["new_content"],
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            expected_delta=expected_delta,
        )


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    # tolerate the model wrapping the JSON in a markdown fence anyway
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        # strict=False: when the model is asked to embed a full source file
        # (new_content) inside a JSON string, it sometimes pastes a literal
        # newline/tab byte instead of the escaped "\n"/"\t" sequence --
        # totally readable code, but technically invalid *strict* JSON
        # (control characters like raw \n are only legal inside a JSON
        # string when escaped). strict=False is a documented json.loads
        # option that allows those raw control characters inside strings,
        # which is exactly this failure mode -- it does not weaken any
        # other part of JSON parsing (structure, quoting, etc. are still
        # enforced).
        return json.loads(text, strict=False)
    except json.JSONDecodeError as e:
        raise LLMError(f"LLM response was not valid JSON: {e}\n---\n{text[:2000]}") from e


class DryRunClient:
    """No API calls. Cycles through a short list of real, sensible edits so
    the controller can be exercised end-to-end without spending LLM tokens.
    Each entry is (hypothesis, target_stage, target_file, content_fn) where
    content_fn(current_pipeline_dir) -> new file content."""

    def __init__(self, pipeline_dir: Path):
        self._pipeline_dir = pipeline_dir
        self._plan = [
            self._reproduce_baseline,
            self._bump_lr,
            self._add_tab_x_video_cross,
        ]

    def propose(self, prompt: str, iteration: int = 0) -> Proposal:
        # Keyed by iteration, not by call count -- a retry within the same
        # iteration (see controller.py's repair-attempt loop) re-proposes
        # the SAME plan step rather than skipping ahead. Past the end of
        # the fixed plan, repeats the last entry.
        fn = self._plan[min(iteration, len(self._plan) - 1)]
        hypothesis, target_stage, target_file, content = fn()
        return Proposal(
            hypothesis=hypothesis,
            target_stage=target_stage,
            target_file=target_file,
            new_content=content,
            input_tokens=0,
            output_tokens=0,
        )

    def _reproduce_baseline(self):
        # Iteration 0: no change -- just confirms the harness reproduces the
        # official baseline before anything starts iterating.
        content = (self._pipeline_dir / "baseline.py").read_text(encoding="utf-8")
        return (
            "Reproduce the official baseline unmodified as iteration 0, "
            "to establish the fixed reference point before iterating.",
            "training_strategy", "baseline.py", content,
        )

    def _bump_lr(self):
        content = (self._pipeline_dir / "baseline.py").read_text(encoding="utf-8")
        # cheap, low-risk config-style change: slightly higher lr + more patience
        content = content.replace("lr=0.001, epochs=40, bs=8192, patience=4",
                                   "lr=0.0015, epochs=40, bs=8192, patience=5")
        return (
            "Try a slightly higher learning rate (0.001 -> 0.0015) with one "
            "extra epoch of patience, as a cheap first probe before any "
            "structural change.",
            "training_strategy", "baseline.py", content,
        )

    def _add_tab_x_video_cross(self):
        content = (self._pipeline_dir / "data.py").read_text(encoding="utf-8")
        # Adds one explicit cross feature (tab x dur_bucket) as a 6th field.
        # This is a real, testable feature-engineering change (distinct
        # from the already-ruled-out "just add more raw columns" dead end,
        # since it's an explicit interaction term, not a raw column).
        content = content.replace(
            "FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']",
            "FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'tab_x_dur']",
        )
        content = content.replace(
            "        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]",
            "        _db = str(int(np.searchsorted(edges, x[5])))\n"
            "        return [x[1], x[2], x[3], x[4], _db, f\"{x[4]}_{_db}\"]",
        )
        return (
            "Add an explicit tab x duration-bucket cross feature (distinct "
            "from the already-tested 'just add more raw columns' dead end "
            "-- this is a specific interaction term, not a new raw field).",
            "feature_engineering", "data.py", content,
        )
