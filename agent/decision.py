"""The commit/revert decision tree.

Every candidate that comes out of an iteration is run through ONE explicit, inspectable decision
tree rather than a scattering of `if accept:` checks across the orchestrator. Each decision
records the exact path it took (`decision.path`), so a human reading runs/experiment_log.jsonl can
see *why* something was kept or thrown away, not just the verdict.

    was the generated code safe? (agent/code_guardrail.py)
    |-- no  -> REJECT_UNSAFE   never executed; nothing to revert, nothing to debug
    +-- yes
        did the debug-sample run execute cleanly? (agent/debug_run.py)
        |-- no  -> REVERT      mark node BUGGY; tree may spend a DEBUG child on it
        +-- yes
            did the full-scale run execute cleanly?
            |-- no  -> REVERT  mark node BUGGY (real-scale-only failure)
            +-- yes
                are the metrics plausible? (finite, in [0,1])
                |-- no  -> REVERT   mark node BUGGY; a NaN score is a broken model, not a bad one
                +-- yes
                    does single-seed primary beat the incumbent AT ALL?
                    |-- no  -> KEEP_NODE   node is WORKING and improvable, but not the new best
                    +-- yes
                        does the multi-seed mean beat incumbent by > epsilon? (agent/reeval.py)
                        |-- no  -> KEEP_NODE   inside the noise floor; not a real improvement
                        +-- yes -> COMMIT      new current-best

The KEEP_NODE outcome is the part that only makes sense once there's a tree (agent/solution_tree.py).
In the old chain design, "didn't beat current-best" and "throw it away" were the same thing. In a
tree, a solution that runs correctly but scores slightly below the incumbent is still a legitimate
base to IMPROVE from later - it is marked WORKING and kept, it just doesn't move current-best.
Only genuinely broken candidates become BUGGY, and only BUGGY nodes are eligible for DEBUG.

Note on "commit": COMMIT here means "accept as the new current-best solution" (tree/state level).
Writing an actual git commit is a separate, optional side effect controlled by GIT_SNAPSHOT (off
by default) - see snapshot_to_git().
"""
import os
import subprocess

from agent.config import CONVERGENCE_EPSILON

COMMIT = 'COMMIT'
KEEP_NODE = 'KEEP_NODE'
REVERT = 'REVERT'
REJECT_UNSAFE = 'REJECT_UNSAFE'

# Off by default: an autonomous loop writing git commits is a side effect on the user's real
# history, and the loop's own runs/ artifacts already record everything needed to reproduce a
# result. Turn on explicitly (agent/cli.py --git-snapshot) if you want per-accept git snapshots.
GIT_SNAPSHOT = False


class Decision:
    def __init__(self, outcome, reason, path):
        self.outcome = outcome
        self.reason = reason
        self.path = path            # list of the decision-tree branches taken, in order

    @property
    def accepted(self):
        return self.outcome == COMMIT

    @property
    def is_buggy(self):
        """Whether the solution-tree node should be marked BUGGY (and so become DEBUG-eligible)."""
        return self.outcome == REVERT

    def to_dict(self):
        return {'outcome': self.outcome, 'reason': self.reason, 'path': self.path}

    def __repr__(self):
        return f"Decision({self.outcome}: {self.reason})"


def decide(*, code_safe=True, code_reasons=None, debug_ok=True, debug_reason=None,
           run_ok=True, run_error=None, metrics=None, incumbent_primary=None,
           reeval_mean=None, epsilon=CONVERGENCE_EPSILON):
    """Walks the decision tree above. Every parameter has a permissive default so a config-only
    candidate (no generated code) can skip the code-safety branch naturally.

    incumbent_primary: current-best's primary, or None if nothing has been accepted yet (in which
    case any working candidate with valid metrics wins by definition - there's nothing to beat).
    reeval_mean: multi-seed mean primary from agent/reeval.py, or None if recheck was skipped.
    """
    path = []

    if not code_safe:
        path.append('code_safe=no')
        reasons = '; '.join(code_reasons or []) or 'failed static analysis'
        return Decision(REJECT_UNSAFE, f"generated code rejected by static analysis: {reasons}",
                         path)
    path.append('code_safe=yes')

    if not debug_ok:
        path.append('debug_run=failed')
        return Decision(REVERT, f"debug-sample run failed: {debug_reason}", path)
    path.append('debug_run=ok')

    if not run_ok:
        path.append('full_run=failed')
        return Decision(REVERT, f"full-scale run failed: {run_error}", path)
    path.append('full_run=ok')

    valid = metrics or {}
    primary = valid.get('primary')
    if primary is None or primary != primary or not (0.0 <= float(primary) <= 1.0):
        path.append('metrics_plausible=no')
        return Decision(REVERT, f"implausible metrics (primary={primary!r})", path)
    path.append('metrics_plausible=yes')
    primary = float(primary)

    if incumbent_primary is None:
        path.append('incumbent=none')
        return Decision(COMMIT, f"first working solution (primary={primary:.4f})", path)

    if primary <= incumbent_primary:
        path.append('beats_incumbent=no')
        return Decision(KEEP_NODE,
                         f"runs correctly but primary={primary:.4f} does not beat incumbent "
                         f"{incumbent_primary:.4f} - kept as a working node to improve from",
                         path)
    path.append('beats_incumbent=yes')

    mean = reeval_mean if reeval_mean is not None else primary
    if mean <= incumbent_primary + epsilon:
        path.append('multiseed_beats_epsilon=no')
        return Decision(KEEP_NODE,
                         f"multi-seed mean {mean:.4f} is within the noise floor of incumbent "
                         f"{incumbent_primary:.4f} (epsilon={epsilon}) - not a real improvement",
                         path)
    path.append('multiseed_beats_epsilon=yes')
    return Decision(COMMIT,
                     f"multi-seed mean {mean:.4f} beats incumbent {incumbent_primary:.4f} "
                     f"by more than epsilon={epsilon}",
                     path)


def snapshot_to_git(message, *, enabled=None,
                    paths=('models/generated', 'runs/solution_tree.json', 'runs/state.json',
                            'runs/experiment_log.jsonl')):
    """Optional: git-commit an accepted solution. OFF unless explicitly enabled - see GIT_SNAPSHOT.
    Returns the commit sha, or None if disabled/failed. Never raises: a git problem must not take
    down an otherwise-successful autonomous run.

    `-f` is REQUIRED here, not a shortcut. Everything worth snapshotting is deliberately
    gitignored for ordinary work — .gitignore has `runs/*` and `models/generated/*.py`, since these
    are run artifacts nobody wants in a normal commit. Found live: without -f this function could
    never once have produced a commit. `git add runs/solution_tree.json` exits 1 ("The following
    paths are ignored by one of your .gitignore files"), check=True raised, the bare `except`
    swallowed it, and it returned None looking exactly like "snapshots are off". Opting in via
    --git-snapshot is precisely the statement that you want these artifacts committed anyway.

    Missing paths are skipped rather than failing the whole snapshot: on the first accepted
    iteration, runs/experiment_log.jsonl does not exist yet (the archivist writes it after
    run_iteration returns), and a named-but-absent pathspec is a hard error to `git add`.
    """
    if not (GIT_SNAPSHOT if enabled is None else enabled):
        return None
    try:
        present = [p for p in paths if os.path.exists(p)]
        if not present:
            print("  [git] nothing to snapshot (none of the artifact paths exist yet)")
            return None
        subprocess.run(['git', 'add', '-f', *present], check=True, capture_output=True, text=True)
        staged = subprocess.run(['git', 'diff', '--cached', '--quiet'], capture_output=True)
        if staged.returncode == 0:
            # Nothing actually changed since the last snapshot — `git commit` would exit 1 here,
            # which is not a failure worth reporting as one.
            return None
        subprocess.run(['git', 'commit', '-m', message], check=True, capture_output=True, text=True)
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True,
                              text=True)
        return out.stdout.strip()
    except Exception as e:  # noqa: BLE001 - git failures are never worth crashing the loop over,
        # but they are worth SAYING, since a silent None is indistinguishable from "disabled" and
        # that is exactly how the ignored-path failure above went unnoticed.
        detail = getattr(e, 'stderr', None) or e
        print(f"  [git] snapshot failed, continuing anyway: {detail}")
        return None
