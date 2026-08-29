"""Structured reasoning/hypothesis pipeline (AGENT_STRATEGY.md strategy 2).

Fills agent/prompts/hypothesize.md (sourced from 06-Master-Prompts_1.md's Propose Prompt) with the
current loop state, calls the one wrapped LLM client (agent/llm_client.py), and parses/validates
the response against the required schema:

    {problem_identified, hypothesis: {statement, target_stage, reasoning, expected_effect},
     implementation_sketch}

On an invalid/incomplete response, retries ONCE (per 06-Master-Prompts_1.md's explicit guidance —
not the earlier draft's "cap 2 retries") by continuing the same conversation with the validation
error appended, then gives up. Never raises on a bad LLM *response* — that's reported as
`HypothesisResult(ok=False, ...)` for the caller to log as a `hypothesis_generation_failed` event.
A genuine `agent.llm_client.LLMError` (Ollama unreachable, timeout, ...) still propagates — that's
an infrastructure failure, not a hypothesis-quality one, and belongs in error_recovery's territory.
"""
import json
import os

from agent import llm_client
from agent.budget import budget_tier_instruction, iteration_budget_fraction
from agent.config import CONVERGENCE_EPSILON, CONVERGENCE_N
from agent.prompt_utils import fill_template, load_template, parse_json
from data import EXTRA_FIELDS, FIELDS

PROMPT_PATH = os.path.join(os.path.dirname(__file__), 'prompts', 'hypothesize.md')
BASELINE_SCORES_PATH = 'baseline_scores.json'

REQUIRED_TARGET_STAGES = {'features', 'model', 'training', 'sampling', 'eval_postprocessing'}

# AGENT_STRATEGY.md decision #9 is resolved as of v0.11: data.encode_with_extra_fields() is now
# the one shared join-logic implementation (see data.py), and agent/action_space.py's
# `toggle_field` action makes EXTRA_FIELDS genuinely addable/removable, not just a description —
# imported directly from data.py now rather than duplicated as a hardcoded string.
FIELDS_IN_USE = FIELDS
FIELDS_TOGGLEABLE = (
    f"{', '.join(EXTRA_FIELDS)} — real columns, genuinely addable/removable via the "
    f"toggle_field action (agent/action_space.py), not just a description."
)
# Present in the raw CSV logs but NOT currently read into data.load()'s row tuples at all (only
# date/user_id/video_id/author_id/tab/duration_ms/long_view are, see data.py's load()) — stated
# explicitly so the model doesn't assume these are already reachable via some x[N] index today.
FEEDBACK_SIGNALS = (
    "is_like, is_follow, is_comment, is_forward, is_hate, play_time_ms, profile_stay_time, "
    "comment_stay_time, is_profile_enter — present in the raw interaction-log CSVs but NOT "
    "currently loaded by data.load() into the row tuples; using any of these requires first "
    "adding them there and updating every positional x[N] read downstream of that change."
)


class HypothesisResult:
    def __init__(self, ok, data=None, error=None, attempts=0, total_usage=None):
        self.ok = ok
        self.data = data
        self.error = error
        self.attempts = attempts
        self.total_usage = total_usage or {'input_tokens': 0, 'output_tokens': 0}

    def __repr__(self):
        if self.ok:
            return (f"HypothesisResult(ok=True, attempts={self.attempts}, "
                    f"statement={self.data['hypothesis']['statement']!r})")
        return f"HypothesisResult(ok=False, error={self.error!r}, attempts={self.attempts})"


# ---------------------------------------------------------------------------
# state assembly
# ---------------------------------------------------------------------------

def _load_baseline_reference(path=BASELINE_SCORES_PATH):
    with open(path, encoding='utf-8') as fh:
        d = json.load(fh)
    return d['scores']['fm_official']['valid']


def _stale_count(history):
    """Executed iterations since the last 'kept' (accepted) entry, counting from the end. Only
    counts entries that actually completed a real training run (h['executed']) — a hypothesis
    rejected before training (not-implementable, guardrail-rejected, duplicate-skipped,
    hypothesis-generation-failed) is skipped over, not counted and not treated as a break, since
    it provides no evidence either way about whether the search has stalled. Found live: without
    this filter, a short run of pre-training rejections inflated stale_count (and, more
    seriously, falsely triggered agent/orchestrator.py's convergence check) after essentially no
    real search had happened."""
    n = 0
    for h in reversed(history):
        if h.get('kept'):
            break
        if not h.get('executed'):
            continue
        n += 1
    return n


def _format_entry(h):
    if h.get('kept'):
        outcome = 'kept'
    else:
        reason = h.get('reason')
        outcome = f"discarded — {reason}" if reason else "discarded"
    return (f"- iter {h['iteration']}: [{h['target_stage']}] \"{h['statement']}\" -> "
            f"primary={h['primary']:.4f} (GAUC={h['gauc']:.4f}, nDCG@5={h['ndcg5']:.4f}) ({outcome})")


def format_history(history, window=8):
    """history: list of dicts, most-recent-last, each shaped:
       {'iteration': int, 'target_stage': str, 'statement': str,
        'gauc': float, 'ndcg5': float, 'primary': float, 'kept': bool}
    (a minimal projection of the full experiment_log.jsonl record — Phase 5's archivist is
    expected to produce entries in this shape, or project down to it, once it exists).

    Returns the {{ history_block }} text: the last `window` entries plus the single best-ever
    entry (by primary) if it isn't already in that window — per 06-Master-Prompts_1.md's own
    token-budget guidance."""
    if not history:
        return "(no iterations yet — this is the first)"
    recent = history[-window:]
    best = max(history, key=lambda h: h['primary'])
    lines = []
    if best not in recent:
        lines.append(_format_entry(best) + "  [best-ever]")
        lines.append("...")
    lines.extend(_format_entry(h) for h in recent)
    return "\n".join(lines)


def build_state(splits, *, current_best, iteration_number, expected_total_iterations,
                 history, elapsed_time=None, total_budget=None):
    """splits: agent.data_guard.load_train_valid()'s output (train/valid only — n_users/n_items
    are computed from it directly rather than trusted from any secondhand figure).
    current_best: {'gauc', 'ndcg5', 'primary', 'summary'} or None for iteration 0.
    history: see format_history()'s docstring for the expected shape.
    Returns the full {{ }}-placeholder mapping agent/prompts/hypothesize.md needs."""
    baseline = _load_baseline_reference()
    n_users = len({x[1] for x in splits['train']})
    n_items = len({x[2] for x in splits['train']})
    n_interactions = len(splits['train']) + len(splits.get('valid', []))

    budget_fraction = iteration_budget_fraction(iteration_number, expected_total_iterations)

    return {
        'n_users': n_users, 'n_items': n_items, 'n_interactions': n_interactions,
        'baseline_gauc': f"{baseline['GAUC']:.4f}",
        'baseline_ndcg5': f"{baseline['nDCG@5']:.4f}",
        'baseline_primary': f"{baseline['primary']:.4f}",
        'feature_list': f"always in use: {', '.join(FIELDS_IN_USE)}. Toggleable via "
                         f"toggle_field: {FIELDS_TOGGLEABLE}",
        'feedback_signals': FEEDBACK_SIGNALS,
        'iteration_number': iteration_number,
        'elapsed_time': elapsed_time if elapsed_time is not None else f"iteration {iteration_number}",
        'total_budget': total_budget if total_budget is not None else
            f"{expected_total_iterations} iterations (wall-clock budget not yet confirmed — "
            f"iteration-count fallback per AGENT_STRATEGY.md Q3)",
        'budget_fraction': f"{budget_fraction:.0f}",
        # NOTE (found on the very first real end-to-end run): filling these with "0.0000" when
        # there's no accepted candidate yet reads to the model as "the model is broken" rather
        # than "no baseline exists yet" — it built its problem_identified around exactly that
        # misreading. Use an explicit N/A string instead of a formatted zero.
        'current_best_gauc': f"{current_best['gauc']:.4f}" if current_best else "N/A (no accepted candidate yet)",
        'current_best_ndcg5': f"{current_best['ndcg5']:.4f}" if current_best else "N/A (no accepted candidate yet)",
        'current_best_primary': f"{current_best['primary']:.4f}" if current_best else "N/A (no accepted candidate yet)",
        'current_best_summary': current_best['summary'] if current_best else
            "none yet — this is the first iteration, nothing has been tried",
        'stale_count': _stale_count(history),
        'convergence_N': CONVERGENCE_N,
        'convergence_epsilon': CONVERGENCE_EPSILON,
        'history_window': 8,
        'history_block': format_history(history, window=8),
        'budget_tier_instruction': budget_tier_instruction(budget_fraction),
    }


# ---------------------------------------------------------------------------
# validate (templating and JSON parsing now live in agent/prompt_utils.py — see its docstring
# for why: this file, agent/coding_agent.py, and agent/error_recovery.py each had their own
# near-identical copy until code review flagged the duplication)
# ---------------------------------------------------------------------------

def _validate(obj):
    """Returns None if valid, else a human-readable error string."""
    if not isinstance(obj, dict):
        return "response is not a JSON object"
    if not obj.get('problem_identified'):
        return "missing/empty 'problem_identified'"
    hyp = obj.get('hypothesis')
    if not isinstance(hyp, dict):
        return "missing/malformed 'hypothesis' object"
    for field in ('statement', 'target_stage', 'reasoning', 'expected_effect'):
        if not hyp.get(field):
            return f"missing/empty 'hypothesis.{field}'"
    if hyp['target_stage'] not in REQUIRED_TARGET_STAGES:
        return (f"hypothesis.target_stage={hyp['target_stage']!r} not one of "
                f"{sorted(REQUIRED_TARGET_STAGES)}")
    if not obj.get('implementation_sketch'):
        return "missing/empty 'implementation_sketch'"
    return None


# ---------------------------------------------------------------------------
# the actual propose call
# ---------------------------------------------------------------------------

def propose(state, *, llm_call=llm_client.call, max_retries=1, template_path=PROMPT_PATH,
            caller='hypothesis_agent', temperature=0.8):
    """Runs the propose prompt once, retries once more on an invalid/incomplete response (per
    06-Master-Prompts_1.md's guidance), then gives up. Returns a HypothesisResult; a genuine
    agent.llm_client.LLMError (Ollama unreachable, timeout, ...) propagates unchanged — that's
    infrastructure failure, not something a prompt retry can fix.

    temperature=0.8 (not llm_client.call's low default of 0.2): found live that at low
    temperature, qwen2.5-coder:7b reliably converges on the same "obvious" hyperparameter change
    (lr=0.01) across multiple iterations even with different history context each time — a
    real diversity problem, not just a prompt-wording one. The retry-once mechanism above still
    catches a malformed/incomplete response regardless of temperature; json_mode's grammar
    constraint (not temperature) is what keeps output schema-valid, so raising this doesn't
    trade away validity for variety."""
    template = load_template(template_path)
    system = fill_template(template, state, source_name='hypothesize.md')
    messages = [{'role': 'user', 'content': "Propose your next iteration's hypothesis now."}]
    total_usage = {'input_tokens': 0, 'output_tokens': 0}

    attempts = 0
    err = None
    obj = None
    while True:
        attempts += 1
        text, usage = llm_call(system=system, messages=messages, json_mode=True, caller=caller,
                                temperature=temperature)
        total_usage['input_tokens'] += usage.get('input_tokens', 0)
        total_usage['output_tokens'] += usage.get('output_tokens', 0)

        obj, err = parse_json(text)
        if err is None:
            err = _validate(obj)
        if err is None:
            return HypothesisResult(ok=True, data=obj, attempts=attempts, total_usage=total_usage)
        if attempts > max_retries:
            break
        messages.append({'role': 'assistant', 'content': text})
        messages.append({'role': 'user', 'content':
            f"Your last response was invalid: {err}. Return ONLY the corrected JSON object, "
            f"matching the schema exactly — no other text."})

    return HypothesisResult(ok=False, error=err, attempts=attempts, total_usage=total_usage)
