"""
CLI entrypoint for the autonomous agent.

Dry run (no API key needed, proves the harness works):
    python3 run_agent.py --dry-run --max_iterations 3

Real run (needs ANTHROPIC_API_KEY exported, or pip install anthropic first):
    python3 run_agent.py --max_iterations 50

Local run against Ollama instead of the Anthropic API (no API key, no
per-token cost -- needs `ollama serve` running with the model already
pulled, e.g. `ollama pull qwen2.5-coder:7b`; see the README's "Running
against a local Ollama model" section for what to expect on an 8GB-class
GPU before running this unattended for hours):
    python3 run_agent.py --local_model qwen2.5-coder:7b --candidates_per_iteration 1

By default, once the run stops, this ALSO finalizes the best checkpoint
into a submission CSV automatically -- the hidden test set is scored
exactly once, right here, using the validation-best snapshot the run just
converged to (see agent/finalize.py). Pass --skip_finalize to turn that off
and run it yourself later:
    python3 agent/finalize.py --best_dir best \\
        --data_dir ../kuairand-starter-kit/KuaiRand-Pure/data \\
        --out submission.csv --split test

--data_dir defaults to ../kuairand-starter-kit/KuaiRand-Pure/data, i.e. a
sibling of this agent-recsys folder (matches the layout after moving
kuairand-starter-kit into TECHJAM/). Pass --data_dir explicitly if your
data lives somewhere else.
"""
import argparse
import json
from pathlib import Path

from agent import context as ctx
from agent import controller
from agent import finalize as finalize_mod
from agent.llm_client import AnthropicClient, DryRunClient, LLMError, OllamaClient
from agent.sandbox import HiddenTestViolation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline_dir", default="pipeline")
    ap.add_argument("--data_dir", default="../kuairand-starter-kit/KuaiRand-Pure/data")
    ap.add_argument("--log_path", default="logs/iteration_log.jsonl")
    ap.add_argument("--best_dir", default="best")
    ap.add_argument("--max_iterations", type=int, default=50)
    ap.add_argument("--max_wall_clock_hours", type=float, default=6.0)
    ap.add_argument("--max_cost_usd", type=float, default=4.5,
                     help="stop the run once estimated LLM spend reaches this many dollars "
                          "(uses Claude Sonnet 5's published per-token rate). Pass 0 to disable.")
    ap.add_argument("--max_consecutive_failures", type=int, default=3,
                     help="stop the run after this many failed iterations in a row "
                          "(counts iterations that exhausted their repair attempts, or a "
                          "fatal API error -- not each individual retry)")
    ap.add_argument("--candidates_per_iteration", type=int, default=2,
                     help="independent proposals to try per iteration slot, keeping "
                          "whichever one scores best (the others are discarded, not "
                          "logged as separate iterations). Raises token spend roughly "
                          "proportionally -- the cost kill switch still applies. Pass 1 "
                          "to reproduce the old single-candidate-per-iteration behavior.")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--dry-run", action="store_true",
                     help="use a fixed local hypothesis list instead of calling an LLM")
    ap.add_argument("--local_model", default=None,
                     help="use a local Ollama model instead of the Anthropic API, e.g. "
                          "--local_model qwen2.5-coder:7b. Requires `ollama serve` running "
                          "with the model already pulled (`ollama pull qwen2.5-coder:7b`). "
                          "No API key, no per-token cost, and --max_cost_usd has no effect -- "
                          "but expect a slower, weaker run than the Anthropic API on this "
                          "workload's prompt sizes; see the README before using this "
                          "unattended for hours.")
    ap.add_argument("--ollama_host", default="http://localhost:11434",
                     help="Ollama server address (only used with --local_model)")
    ap.add_argument("--ollama_num_ctx", type=int, default=32768,
                     help="context window, in tokens, to request from Ollama (only used "
                          "with --local_model). This prompt runs roughly 15-20K input "
                          "tokens -- lower this only if your GPU can't hold that much KV "
                          "cache; raising it further only helps if the model's own native "
                          "context window is at least that large too.")
    ap.add_argument("--ollama_temperature", type=float, default=0.2,
                     help="sampling temperature for the local Ollama model (only used with "
                          "--local_model). Lower than Ollama's own default (0.8) since this "
                          "task wants the model to reproduce an existing file closely, not "
                          "write open-ended text.")
    ap.add_argument("--submission_out", default="submission.csv",
                     help="where the auto-finalize step writes the submission CSV (ignored with --skip_finalize)")
    ap.add_argument("--skip_finalize", action="store_true",
                     help="don't automatically finalize a submission when the run stops -- "
                          "run agent/finalize.py yourself later instead")
    args = ap.parse_args()

    pipeline_dir = Path(args.pipeline_dir)
    log_path = Path(args.log_path)
    best_dir = Path(args.best_dir)

    if args.dry_run:
        client = DryRunClient(pipeline_dir)
        print("running in --dry-run mode: no LLM calls, using a fixed hypothesis list")
    elif args.local_model:
        try:
            client = OllamaClient(
                model=args.local_model, host=args.ollama_host, num_ctx=args.ollama_num_ctx,
                temperature=args.ollama_temperature,
            )
        except LLMError as e:
            print(f"error: {e}")
            raise SystemExit(1)
        print(
            f"running against a local Ollama model ({args.local_model} @ {args.ollama_host}, "
            f"num_ctx={args.ollama_num_ctx}): no API key, no per-token cost, --max_cost_usd "
            f"has no effect -- but expect this to run much slower and produce noisier, "
            f"likely smaller improvements than the Anthropic API on this workload's prompt "
            f"sizes. See the README's \"Running against a local Ollama model\" section for "
            f"what to expect before leaving this running for hours."
        )
        if args.candidates_per_iteration > 1:
            print(
                f"note: --candidates_per_iteration is {args.candidates_per_iteration} -- "
                f"each candidate is a separate full LLM call, and a local 7-8B model at "
                f"this prompt's size can take tens of minutes per call on an 8GB-class GPU. "
                f"Consider --candidates_per_iteration 1 for a local run unless you're "
                f"deliberately trading wall-clock for a better per-iteration hit rate."
            )
    else:
        try:
            client = AnthropicClient(model=args.model)
        except LLMError as e:
            print(f"error: {e}")
            raise SystemExit(1)

    try:
        result = controller.run(
            pipeline_dir=pipeline_dir,
            log_path=log_path,
            best_dir=best_dir,
            llm_client=client,
            data_dir=args.data_dir,
            max_iterations=args.max_iterations,
            max_wall_clock_hours=args.max_wall_clock_hours,
            max_cost_usd=(args.max_cost_usd or None),
            max_consecutive_failures=args.max_consecutive_failures,
            candidates_per_iteration=args.candidates_per_iteration,
        )
    except HiddenTestViolation as e:
        # A compliance stop, not a crash -- print it as such rather than
        # dumping a traceback that reads like an unhandled bug.
        print("\nRUN STOPPED -- HIDDEN TEST SET COMPLIANCE VIOLATION")
        print(f"{e}")
        print(
            "\nNothing was logged for this iteration and best/ is untouched. "
            "Fix pipeline/_run_iteration.py (it must pop 'test' out of splits "
            "before calling run_fm()), then re-run."
        )
        raise SystemExit(2)

    summary_path = Path("logs/resource_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result["resource_summary"], f, indent=2)

    print(f"\nbest validation primary reached: {result['best_valid_primary']:.4f}")
    print(f"estimated total spend: ${result['total_cost_usd']:.2f}")
    print(f"manual interventions: {result['manual_interventions']}")
    print(f"iteration log: {log_path}")
    print(f"best pipeline snapshot: {best_dir}/")
    print(f"resource summary: {summary_path}")

    if args.skip_finalize:
        print(f"\n--skip_finalize set: run `python agent/finalize.py --best_dir {best_dir} "
              f"--data_dir \"{args.data_dir}\" --out {args.submission_out} --split test` yourself when ready.")
        return

    # Autonomously designate and produce the final submission -- the ONE
    # sanctioned, one-time touch of the hidden test set (see
    # agent/finalize.py's docstring). Wrapped so a finalize-side failure
    # (e.g. a malformed train_and_predict() in whatever the agent last
    # wrote) is reported clearly rather than erasing the fact that the run
    # itself already completed and best/ is intact.
    print(f"\nfinalizing submission from {best_dir}/ (this is the one point the hidden test set is scored)...")
    try:
        finalize_result = finalize_mod.finalize(
            best_dir=best_dir, data_dir=args.data_dir,
            out_csv=Path(args.submission_out), split="test",
        )
    except Exception as e:  # noqa: BLE001 - report, don't mask the run's own success
        print(f"finalize FAILED: {type(e).__name__}: {e}")
        print(f"the run itself completed successfully -- retry finalizing manually with "
              f"`python agent/finalize.py --best_dir {best_dir} --data_dir \"{args.data_dir}\" "
              f"--out {args.submission_out} --split test` once fixed.")
        return

    test_metrics = finalize_result["metrics"].get("test", {})
    official = None
    try:
        official = ctx.load_baseline_reference(best_dir)["scores"]["fm_official"]["test"]["primary"]
    except (FileNotFoundError, KeyError):
        pass  # best/baseline_scores.json missing or reshaped -- print without the comparison rather than crash here

    print(f"submission written: {finalize_result['submission_path']} ({finalize_result['rows']} rows)")
    if "primary" in test_metrics:
        line = f"hidden test primary: {test_metrics['primary']:.4f}"
        if official is not None:
            delta = test_metrics["primary"] - official
            line += f"  (official baseline {official:.4f}, delta {'+' if delta >= 0 else ''}{delta:.4f})"
        print(line)


if __name__ == "__main__":
    main()
