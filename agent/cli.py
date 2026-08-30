"""The actual entrypoint for running the loop.

    python -m agent.cli run --iterations 20
    python -m agent.cli status
    python -m agent.cli note-intervention "restarted a stuck run, no code changes"

There is no visual dashboard yet (AGENT_STRATEGY.md Phase 6, not built) — `status` and
`python -m agent.cost_report` are the only ways to inspect a run right now.
"""
import argparse

from agent import cost_report, decision, manual_intervention, resume
from agent.orchestrator import run_loop
from agent.solution_tree import SolutionTree
from models import fm_bpr, fm_v1

# Which model variant the loop treats as its starting point: it is what seed_baseline() trains to
# set the incumbent, and what a config-only candidate runs against. fm_bpr is the default because
# it measures better than fm_v1 on both splits (valid 0.6031 vs 0.6015, test 0.5970 vs 0.5953),
# and starting the search from the weaker model is what 75 prior iterations already did.
MODELS = {'fm_bpr': fm_bpr, 'fm_v1': fm_v1}


def cmd_run(args):
    if args.git_snapshot:
        # Opt-in only: an autonomous loop writing to the user's real git history is a side effect
        # they should choose deliberately. Off unless this flag is passed. See agent/decision.py.
        decision.GIT_SNAPSHOT = True
        print("[git] --git-snapshot: accepted solutions will be committed to git")
    model = MODELS[args.model]
    print(f"[model] starting from {model.__name__}")
    current_best, history = run_loop(
        args.data_dir, expected_total_iterations=args.iterations,
        max_iterations=args.iterations, seed=args.seed, verbose=not args.quiet,
        max_hours=args.max_hours, ignore_convergence=args.ignore_convergence,
        model=model)
    print()
    if current_best:
        print(f"Final current-best: primary={current_best['primary']:.4f} "
              f"(GAUC={current_best['gauc']:.4f}, nDCG@5={current_best['ndcg5']:.4f})")
        print(f"  {current_best['summary']}")
        print(f"  config: {current_best['config']}")
    else:
        print("No candidate was ever accepted.")
    print(f"History now has {len(history)} iteration(s) total, logged to "
          f"runs/experiment_log.jsonl")
    print()
    print(cost_report.format_report())


def cmd_status(args):
    state = resume.load_state()
    cb = state.get('current_best')
    print(f"Last completed iteration: {state.get('last_completed_iteration', 0)}")
    if cb:
        print(f"Current-best: primary={cb['primary']:.4f} (GAUC={cb['gauc']:.4f}, "
              f"nDCG@5={cb['ndcg5']:.4f})")
        print(f"  {cb['summary']}")
    else:
        print("Current-best: none yet")
    print(f"Manual interventions recorded: {manual_intervention.count()}")
    print()
    print("Solution tree:")
    print(SolutionTree.load().render())
    print()
    print(cost_report.format_report())


def cmd_note_intervention(args):
    manual_intervention.record(args.reason)
    print(f"Recorded (total now {manual_intervention.count()}).")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)

    run_ap = sub.add_parser('run', help='run the loop')
    run_ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    run_ap.add_argument('--iterations', type=int, default=20,
                         help='both the convergence-fraction denominator and the hard cap on '
                              'how many iterations this call will run')
    run_ap.add_argument('--seed', type=int, default=0)
    run_ap.add_argument('--model', default='fm_bpr', choices=sorted(MODELS),
                         help='model variant the loop starts from and seeds its incumbent with '
                              '(default: fm_bpr, the best measured variant)')
    run_ap.add_argument('--max-hours', type=float, default=None,
                         help='wall-clock budget in hours, checked between iterations (never '
                              'interrupts one mid-run). Pair with a large --iterations for '
                              '"run for N hours" semantics; re-run the same command to continue.')
    run_ap.add_argument('--ignore-convergence', action='store_true',
                         help='keep iterating even once the convergence rule says the search has '
                              'stalled. Use with --max-hours to fill a time budget.')
    run_ap.add_argument('--quiet', action='store_true')
    run_ap.add_argument('--git-snapshot', action='store_true',
                         help='git-commit each accepted solution (off by default - the loop does '
                              'not touch your git history unless you ask for it)')
    run_ap.set_defaults(func=cmd_run)

    status_ap = sub.add_parser('status', help='show current-best + resource consumption')
    status_ap.set_defaults(func=cmd_status)

    note_ap = sub.add_parser('note-intervention', help='record a human intervention (never automatic)')
    note_ap.add_argument('reason')
    note_ap.set_defaults(func=cmd_note_intervention)

    a = ap.parse_args()
    a.func(a)
