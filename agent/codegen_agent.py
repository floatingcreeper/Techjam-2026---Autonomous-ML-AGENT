"""Generates real, runnable model modules — the agent writing actual code, not mutating a config.

Three operations, matching agent/solution_tree.py's AIDE-style node expansions:
    draft   - write a brand-new model module implementing a hypothesis from scratch
    debug   - a previous module didn't run; rewrite it so it does, keeping the same idea
    improve - a previous module ran; refine it to score better

Safety flow for every generated module, in order:
    LLM output -> strip fences -> agent/code_guardrail.py static analysis -> (retry with the
    rejection reasons fed back, up to MAX_ATTEMPTS) -> write to models/generated/ -> import.

Nothing is written to disk before it passes static analysis, and nothing outside
models/generated/ is ever written at all. Generated modules cannot do file I/O (enforced by the
guardrail), so an executing module can only ever see the `splits` dict handed to it — which
agent/data_guard.py already stripped the test split from. That's the whole containment argument:
generated code is powerful (any numpy model it can express) but reaches nothing it shouldn't.
"""
import importlib.util
import os
import re

from agent import llm_client
from agent.code_guardrail import check_source
from agent.prompt_utils import fill_template, load_template

PROMPT_PATH = os.path.join(os.path.dirname(__file__), 'prompts', 'write_model.md')
GENERATED_DIR = os.path.join('models', 'generated')
MAX_ATTEMPTS = 3          # 1 initial + 2 retries with guardrail feedback

OPERATION_INSTRUCTIONS = {
    'draft': (
        "OPERATION: DRAFT A NEW SOLUTION\n"
        "Write a complete new model module from scratch implementing the idea below. You are not "
        "editing anything - this is a fresh, independent solution. Favour a design that is simple "
        "enough to run correctly the first time; a working simple model beats a broken clever one."
    ),
    'debug': (
        "OPERATION: DEBUG AN EXISTING SOLUTION\n"
        "The module below was generated for this same idea but FAILED TO RUN. Diagnose the error "
        "and rewrite the module so it executes correctly. Keep the underlying idea intact - you "
        "are fixing execution, not changing the hypothesis. Return the COMPLETE corrected module, "
        "not a diff or a fragment."
    ),
    'improve': (
        "OPERATION: IMPROVE A WORKING SOLUTION\n"
        "The module below RUNS CORRECTLY and achieved the score shown. Rewrite it to score higher, "
        "guided by the idea below. Preserve what is working; change what the idea calls for. "
        "Return the COMPLETE improved module."
    ),
}


class GenerationResult:
    def __init__(self, ok, source=None, module_path=None, reasons=None, attempts=0,
                 total_usage=None):
        self.ok = ok
        self.source = source
        self.module_path = module_path
        self.reasons = reasons or []
        self.attempts = attempts
        self.total_usage = total_usage or {'input_tokens': 0, 'output_tokens': 0}

    def __repr__(self):
        if self.ok:
            return f"GenerationResult(ok=True, module_path={self.module_path!r}, attempts={self.attempts})"
        return f"GenerationResult(ok=False, reasons={self.reasons}, attempts={self.attempts})"


def _strip_code_fences(text):
    """Models wrap code in ```python fences despite instructions not to; strip them rather than
    burning a retry on something this mechanical."""
    t = text.strip()
    fence = re.match(r'^```(?:python|py)?\s*\n(.*?)\n?```\s*$', t, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    # A truncated response can open a fence without closing it.
    if t.startswith('```'):
        return re.sub(r'^```(?:python|py)?\s*\n?', '', t).strip()
    return t


def _slug(text, maxlen=40):
    s = re.sub(r'[^a-z0-9]+', '_', (text or 'solution').lower()).strip('_')
    return (s[:maxlen] or 'solution').rstrip('_')


def _context_block(operation, parent_source, parent_primary, error):
    if operation == 'debug':
        return (f"THE MODULE THAT FAILED:\n```\n{parent_source}\n```\n\n"
                f"THE ERROR IT PRODUCED:\n{error}")
    if operation == 'improve':
        score = f"{parent_primary:.4f}" if parent_primary is not None else "unknown"
        return (f"THE WORKING MODULE TO IMPROVE (current primary = {score}):\n"
                f"```\n{parent_source}\n```")
    return ("This is a fresh draft - there is no existing module to build on. The reference "
            "baseline for comparison is a Factorization Machine (baseline.FM) over the 5 fields "
            "in data.FIELDS, scoring primary ~0.60.")


def generate(operation, hypothesis, *, iteration, parent_source=None, parent_primary=None,
             error=None, llm_call=llm_client.call, generated_dir=GENERATED_DIR,
             max_attempts=MAX_ATTEMPTS, caller='codegen_agent'):
    """Generates a model module. Returns a GenerationResult.

    On a guardrail rejection, retries with the specific rejection reasons appended to the
    conversation (same retry-with-feedback pattern as agent/hypothesis_agent.py, just validating
    code safety instead of JSON schema). Never writes an unsafe module to disk.
    """
    if operation not in OPERATION_INSTRUCTIONS:
        raise ValueError(f"unknown operation {operation!r}")

    system = fill_template(load_template(PROMPT_PATH), {
        'operation_instruction': OPERATION_INSTRUCTIONS[operation],
        'target_stage': hypothesis.get('target_stage', 'model'),
        'statement': hypothesis['statement'],
        'reasoning': hypothesis.get('reasoning', ''),
        'implementation_sketch': hypothesis.get('implementation_sketch', ''),
        'context_block': _context_block(operation, parent_source, parent_primary, error),
    }, source_name='write_model.md')

    messages = [{'role': 'user', 'content': 'Write the complete module now.'}]
    total_usage = {'input_tokens': 0, 'output_tokens': 0}
    reasons = []

    for attempt in range(1, max_attempts + 1):
        # json_mode is deliberately OFF here: this returns Python source, not JSON. The guardrail
        # (not a grammar constraint) is what makes the output safe to use.
        text, usage = llm_call(system=system, messages=messages, json_mode=False,
                                temperature=0.6, caller=caller)
        total_usage['input_tokens'] += usage.get('input_tokens', 0)
        total_usage['output_tokens'] += usage.get('output_tokens', 0)

        source = _strip_code_fences(text)
        reasons = check_source(source)
        if not reasons:
            os.makedirs(generated_dir, exist_ok=True)
            path = os.path.join(generated_dir,
                                 f"iter{iteration:03d}_{operation}_{_slug(hypothesis['statement'])}.py")
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(source)
            return GenerationResult(ok=True, source=source, module_path=path, attempts=attempt,
                                     total_usage=total_usage)

        if attempt < max_attempts:
            messages.append({'role': 'assistant', 'content': text})
            messages.append({'role': 'user', 'content':
                "Your module was REJECTED by static analysis for these reasons:\n"
                + '\n'.join(f"  - {r}" for r in reasons)
                + "\n\nFix every one of them and return the COMPLETE corrected module. "
                  "Return only Python source - no fences, no prose."})

    return GenerationResult(ok=False, reasons=reasons, attempts=max_attempts,
                             total_usage=total_usage)


def load_module(module_path):
    """Imports a generated module from its file path and returns it. The caller is expected to
    have generated it through generate() (i.e. it already passed static analysis) — this does not
    re-check safety, it just imports."""
    name = 'generated_' + os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load generated module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'train'):
        raise ImportError(f"{module_path} defines no train() — should have been caught by the "
                           f"guardrail's REQUIRED_FUNCTION check")
    return module
