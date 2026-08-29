"""The actual entrypoint for running the loop.

    python -m agent.cli run --iterations 20
    python -m agent.cli status
    python -m agent.cli note-intervention "restarted a stuck run, no code changes"

There is no visual dashboard yet (AGENT_STRATEGY.md Phase 6, not built) — `status` and
`python -m agent.cost_report` are the only ways to inspect a run right now.
"""
import argparse

from agent import cost_report, manual_intervention, resume
from agent.orchestrator import run_loop


def cmd_run(args):
    current_best, history = run_loop(
        args.data_dir, expected_total_iterations=args.iterations,
        max_iterations=args.iterations, seed=args.seed, verbose=not args.quiet)
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
    run_ap.add_argument('--quiet', action='store_true')
    run_ap.set_defaults(func=cmd_run)

    status_ap = sub.add_parser('status', help='show current-best + resource consumption')
    status_ap.set_defaults(func=cmd_status)

    note_ap = sub.add_parser('note-intervention', help='record a human intervention (never automatic)')
    note_ap.add_argument('reason')
    note_ap.set_defaults(func=cmd_note_intervention)

    a = ap.parse_args()
    a.func(a)
