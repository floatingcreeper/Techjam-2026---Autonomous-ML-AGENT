"""
The one place that calls out to an LLM. Three implementations:

  - AnthropicClient  -- real calls via the Anthropic API. Requires
                         ANTHROPIC_API_KEY in the environment. Swap this
                         class (or add a sibling) if you want a different
                         provider -- controller.py only depends on the
                         propose(prompt) -> Proposal interface below, not
                         on which vendor answers it.
  - OllamaClient      -- local inference via a running Ollama server
                         (`ollama serve`, with the model already pulled).
                         No API key, no per-token cost, no network egress
                         beyond localhost -- but a small local model is
                         meaningfully weaker at this task than Claude
                         Sonnet, and much slower at this workload's prompt
                         sizes on an 8GB-class GPU. See run_agent.py's
                         --local_model / --ollama_host / --ollama_num_ctx
                         flags and the README's "Running against a local
                         Ollama model" section before reaching for this.
  - DryRunClient      -- no network calls, no API key. Cycles through a
                         short fixed list of legitimate, hand-written
                         hypotheses. Its only purpose is to prove the
                         controller/sandbox/logging/convergence machinery
                         works end-to-end before you spend real LLM tokens
                         on it. Use --dry-run to select it (see run_agent.py).

All three return a Proposal with token counts populated (DryRunClient
reports 0 tokens, since no LLM was actually called), so
agent/finalize.py's resource summary is accurate either way. Each client
also exposes .estimate_cost(input_tokens, output_tokens) -> float so
controller.py's cost accounting and --max_cost_usd kill switch stay
correct per-provider -- OllamaClient's and DryRunClient's are both
hardcoded to 0.0 (no per-token billing for either), only AnthropicClient's
reflects real spend.
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


# Rough chars-per-token divisor for the OllamaClient preflight check below.
# Deliberately on the low side (English prose is nearer 4.0; source code,
# which dominates this prompt, tokenizes denser) so the estimate errs
# toward OVER-counting -- a false "your prompt may not fit" warning costs
# nothing, while an under-count lets a silently truncated prompt through,
# which is the exact failure this guard exists to catch.
_CHARS_PER_TOKEN = 3.5

# How much of the context window to keep free for the model's own answer.
# A full-file rewrite of baseline.py is ~31KB of source embedded inside a
# JSON string, so ~9-10K tokens before escaping overhead. num_ctx has to
# hold prompt AND completion; reserving less than this means the model
# runs out of window mid-rewrite and emits truncated JSON.
_OLLAMA_OUTPUT_RESERVE_TOKENS = 10000


def _estimate_tokens(text: str) -> int:
    """Approximate token count from character length. Used only for the
    OllamaClient context-window preflight -- never for billing or for the
    resource summary, both of which use real counts reported by the
    provider."""
    return int(len(text) / _CHARS_PER_TOKEN)


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

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return estimate_cost(input_tokens, output_tokens)


class OllamaClient:
    """Local inference via a running Ollama server's REST API. Requires
    `ollama serve` already running and the model already pulled
    (`ollama pull <model>`) -- this class does not start a server or pull a
    model itself, it just calls one that's already there.

    Expected performance on an 8GB-class consumer GPU (e.g. RTX 5060), for
    THIS workload specifically: this prompt runs ~15-20K input tokens
    (data profile, dead ends/headroom directions, bounded history, both
    full pipeline files) and asks for a full-file rewrite back, easily
    10K+ output tokens for baseline.py. Published RTX 5060 Ollama
    benchmarks for 7-8B models cluster around 30-70 tok/s, but that's
    measured at short-to-moderate context that fits entirely in 8GB VRAM;
    every one of those benchmarks also notes that pushing past roughly an
    8K-token context window on an 8GB card forces KV-cache spillover to
    CPU, which is a much bigger throughput hit than the model-size
    difference alone. At this prompt's actual size, expect generation
    closer to the low tens of tokens/sec than the high end of that range,
    i.e. low-to-mid tens of MINUTES per candidate, not seconds -- roughly
    an order of magnitude slower wall-clock than the Anthropic API for the
    same run. A 7-8B model is also simply weaker than Claude Sonnet at
    reliably emitting fence-free JSON and at coherently rewriting an
    entire 500-800 line file without corrupting logic unrelated to its
    hypothesis (the "isolated change" constraint is harder for a small
    model to honor at this file size) -- expect a higher failed/repaired
    iteration rate and likely smaller, noisier score improvements than the
    same run against a hosted frontier model. See the README for the
    concrete recommendation (a coding-tuned model, --candidates_per_iteration 1,
    and why) before running this unattended for hours.
    """

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 timeout_s: int = 1800, num_ctx: int = 32768, temperature: float = 0.2):
        try:
            import requests
        except ImportError as e:
            raise LLMError(
                "the 'requests' package is not installed -- run "
                "`pip install requests` (or `pip install requests "
                "--break-system-packages` in this environment)"
            ) from e
        self._requests = requests
        self._model = model
        self._host = host.rstrip("/")
        # Generous default: a small model at this prompt's size (~15-20K
        # input tokens) running on an 8GB-class GPU can genuinely take
        # 10-20+ minutes per call once KV cache spills past VRAM (see class
        # docstring) -- a short timeout would just misreport slow-but-
        # working generation as a dead server.
        self._timeout_s = timeout_s
        self._num_ctx = num_ctx
        # Ollama's default temperature (0.8) is tuned for open-ended chat,
        # not for "reproduce this ~800-line file exactly except one named
        # function" -- a lower temperature biases decoding toward the most
        # likely (closer to verbatim-copy) continuation, which is what this
        # task actually wants. This is a real but partial mitigation, not a
        # fix: the observed failure mode (a small model silently omitting
        # whole helper functions from a "complete" rewrite) is a capability/
        # attention limitation at this file size, not primarily a sampling-
        # randomness one -- see the class docstring.
        self._temperature = temperature

    def propose(self, prompt: str, iteration: int = 0) -> Proposal:
        # PREFLIGHT: Ollama silently TRUNCATES a prompt longer than num_ctx
        # rather than erroring. There is no field in the response saying it
        # happened, so an over-long prompt shows up only as inexplicably
        # bad output -- and the part most likely to be cut is the tail,
        # which is where this prompt keeps its response-format spec and its
        # constraints. Failing loudly here is strictly better than spending
        # 20 minutes of local generation on a prompt the model only half
        # received.
        est_prompt_tokens = _estimate_tokens(prompt)
        needed = est_prompt_tokens + _OLLAMA_OUTPUT_RESERVE_TOKENS
        if needed > self._num_ctx:
            raise LLMFatalError(
                f"prompt does not fit the context window: ~{est_prompt_tokens} "
                f"estimated prompt tokens + {_OLLAMA_OUTPUT_RESERVE_TOKENS} "
                f"reserved for the response = ~{needed}, but num_ctx is "
                f"{self._num_ctx}. Ollama would silently truncate the prompt "
                f"rather than tell you. Either raise --ollama_num_ctx to at "
                f"least {needed} (costs VRAM: roughly 56 KiB per token of KV "
                f"cache for a 7B model, so ~{needed * 56 / 1024 / 1024:.1f} GB "
                f"on top of the model weights -- an 8GB card will not hold "
                f"both and will offload layers to CPU), or use a model with a "
                f"larger native context window, or shrink the prompt."
            )
        try:
            resp = self._requests.post(
                f"{self._host}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    # Constrains decoding to valid JSON -- Ollama enforces
                    # this at the token-sampling level, not just via the
                    # prompt instruction. Meaningfully cuts the malformed-
                    # output failure rate a small model would otherwise
                    # have; a large hosted model tends to follow the plain
                    # "respond with ONLY a JSON object" instruction
                    # reliably enough not to need this, but it costs
                    # nothing to also set here.
                    "format": "json",
                    "options": {"num_ctx": self._num_ctx, "temperature": self._temperature},
                },
                timeout=self._timeout_s,
            )
            resp.raise_for_status()
        except self._requests.exceptions.ConnectionError as e:
            raise LLMFatalError(
                f"could not reach Ollama at {self._host} -- is `ollama serve` "
                f"running, and is --ollama_host pointing at the right "
                f"address/port? ({e})"
            ) from e
        except Exception as e:  # noqa: BLE001 - mirrors AnthropicClient:
            # any transport-level failure here (timeout, connection reset,
            # a 5xx from the Ollama server, a model name that was never
            # `ollama pull`-ed, ...) is not fixed by retrying instantly
            # with no backoff -- see LLMFatalError's docstring.
            raise LLMFatalError(f"{type(e).__name__}: {e}") from e

        body = resp.json()
        # POSTFLIGHT: the estimate above is a heuristic, so also check what
        # actually happened. Ollama reports real counts; if prompt plus
        # completion filled the whole window, something was cut even though
        # the estimate said it would fit.
        real_prompt_tokens = body.get("prompt_eval_count", 0)
        real_output_tokens = body.get("eval_count", 0)
        if real_prompt_tokens and real_prompt_tokens + real_output_tokens >= self._num_ctx:
            print(
                f"  WARNING: context window saturated -- {real_prompt_tokens} prompt "
                f"+ {real_output_tokens} response tokens against num_ctx="
                f"{self._num_ctx}. The prompt and/or the response was truncated, "
                f"so treat this iteration's result as unreliable. Raise "
                f"--ollama_num_ctx if VRAM allows."
            )
        text = body.get("response", "")
        if not text.strip():
            raise LLMError(
                "Ollama returned an empty response. Likely the model hit "
                "--ollama_num_ctx before producing any output (this "
                "prompt is large -- try a model with a larger native "
                "context window, or raise --ollama_num_ctx if VRAM "
                "allows), or generation was interrupted server-side."
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
            # Ollama reports these directly in the response body -- kept
            # for the same resource-summary reporting the other clients
            # provide, even though there's no $ cost attached to them here.
            input_tokens=body.get("prompt_eval_count", 0),
            output_tokens=body.get("eval_count", 0),
            expected_delta=expected_delta,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0  # local inference -- no per-token billing


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

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0  # no LLM was actually called

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
