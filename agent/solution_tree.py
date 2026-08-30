"""AIDE-style solution tree (Jiang et al., "AIDE: AI-Driven Exploration in the Space of Code") —
each node is a COMPLETE candidate solution, and the tree is navigated greedily by three operations:

    DRAFT   - write a new solution from scratch (a new root-level branch)
    DEBUG   - child of a BUGGY node: same idea, fixed so it actually runs
    IMPROVE - child of a WORKING node: refine a solution that already runs

This replaces the pure chain/greedy structure AGENT_STRATEGY.md chose in v0.4 (which cited
MLE-STAR's single-path design and deferred branching as a stretch goal). The reversal is
deliberate and requested. It also earns its keep for a specific reason that only became visible
once real code generation existed: a 7B model writing a whole model implementation fails *often*,
and a chain has nowhere to put a failed-but-promising attempt except the bin. A tree gives it a
DEBUG child instead — which is far cheaper than drafting a fresh idea, since a solution that
almost ran usually needs one small fix.

Where the "efficient computation" comes from (the reason AIDE's tree is cheap, not expensive):
  - DEBUG and IMPROVE both start from an existing solution's code, so the model edits rather than
    re-derives; most nodes cost one LLM call plus one training run, not a fresh exploration.
  - Selection is greedy, never exhaustive: exactly one node is expanded per iteration, chosen by
    the policy in select(). There is no rollout, no backpropagation, no parallel branch fan-out
    (that's ML-Master's MCTS, explicitly still out of scope - see AGENT_STRATEGY.md's
    search-strategy section).
  - MAX_DEBUG_DEPTH caps how much compute a single broken idea can absorb before the tree
    abandons that branch and moves on.

Persisted to runs/solution_tree.json so the tree survives a crash/resume like everything else.
"""
import json
import os

from agent.json_utils import json_default

TREE_PATH = os.path.join('runs', 'solution_tree.json')

# A branch that has absorbed this many DEBUG attempts without ever reaching 'working' is
# abandoned - the idea itself is probably not implementable by this model, and further repair
# attempts are throwing compute at a dead branch.
#
# This is a budget on the whole BRANCH (every debug node descending from the original draft), not
# on any single node's chain depth. Found live and it cost a whole 20-iteration run: the check
# used to be `node.debug_depth < MAX_DEBUG_DEPTH` on the candidate node alone, and debug_depth is
# chain depth (parent's + 1). Once a depth-2 node existed, every debug child it spawned came out
# at depth 3 and was excluded - but the depth-2 PARENT stayed at 2 forever, stayed eligible
# forever, and select()'s `max(debug_depth, id)` kept re-picking it. Iterations 5-20 of that run
# were the same node debugged 16 times against the same error. Counting attempts per branch is
# what actually makes the cap fire.
MAX_DEBUG_DEPTH = 3
# Draft at least this many independent solutions before settling into pure greedy IMPROVE, so the
# tree gets some breadth instead of over-committing to whatever the first working idea happened
# to be. This is AIDE's "maximize diversity early" instinct, kept deliberately small for budget.
MIN_DRAFTS = 2

WORKING = 'working'
BUGGY = 'buggy'


class Node:
    def __init__(self, node_id, parent_id, operation, summary, config=None, code_path=None,
                 status=BUGGY, metrics=None, primary=None, error=None, debug_depth=0,
                 iteration=None):
        self.id = node_id
        self.parent_id = parent_id
        self.operation = operation          # draft | debug | improve | config
        self.summary = summary              # the hypothesis statement behind this node
        self.config = config or {}
        self.code_path = code_path          # None for config-only nodes (set_hyperparam etc.)
        self.status = status                # working | buggy
        self.metrics = metrics
        self.primary = primary              # valid primary, or None if buggy
        self.error = error                  # failure text, if buggy
        self.debug_depth = debug_depth      # consecutive debug attempts in this branch
        self.iteration = iteration

    def to_dict(self):
        return {'id': self.id, 'parent_id': self.parent_id, 'operation': self.operation,
                'summary': self.summary, 'config': self.config, 'code_path': self.code_path,
                'status': self.status, 'metrics': self.metrics, 'primary': self.primary,
                'error': self.error, 'debug_depth': self.debug_depth, 'iteration': self.iteration}

    @classmethod
    def from_dict(cls, d):
        return cls(d['id'], d['parent_id'], d['operation'], d['summary'], d.get('config'),
                   d.get('code_path'), d.get('status', BUGGY), d.get('metrics'), d.get('primary'),
                   d.get('error'), d.get('debug_depth', 0), d.get('iteration'))

    def __repr__(self):
        p = f"{self.primary:.4f}" if self.primary is not None else "n/a"
        return (f"Node(id={self.id}, parent={self.parent_id}, op={self.operation}, "
                f"status={self.status}, primary={p})")


class SolutionTree:
    def __init__(self, nodes=None, next_id=0):
        self.nodes = nodes or {}
        self._next_id = next_id

    # ---------------- persistence ----------------

    @classmethod
    def load(cls, path=TREE_PATH):
        if not os.path.exists(path):
            return cls()
        with open(path, encoding='utf-8') as fh:
            d = json.load(fh)
        nodes = {int(k): Node.from_dict(v) for k, v in d['nodes'].items()}
        # Drop a code_path that no longer exists on disk. Found live, and it killed a whole
        # overnight run: models/generated/ was archived away while runs/solution_tree.json kept
        # pointing at the moved files, so select() returned ('improve', node #58) and
        # orchestrator's load_module() raised FileNotFoundError OUTSIDE any try — the process
        # died, the supervisor restarted it, select() picked the same node, and it died again.
        # Eight identical restarts, zero iterations. A node whose code is gone is not a code node
        # any more: keeping its config (and its score) but clearing code_path degrades it to a
        # config-only node, which debuggable() already skips and which `improve` correctly runs
        # against the default model variant instead of a missing file.
        for n in nodes.values():
            if n.code_path and not os.path.exists(n.code_path):
                n.code_path = None
        return cls(nodes, d.get('next_id', max(nodes, default=-1) + 1))

    def save(self, path=TREE_PATH):
        """Atomic write (temp + os.replace), same reasoning as agent/resume.py: a crash mid-write
        must never leave a half-written tree."""
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump({'nodes': {str(k): v.to_dict() for k, v in self.nodes.items()},
                       'next_id': self._next_id}, fh, indent=2, default=json_default)
        os.replace(tmp, path)

    # ---------------- construction ----------------

    def add(self, *, parent_id, operation, summary, config=None, code_path=None, iteration=None):
        parent = self.nodes.get(parent_id)
        # debug_depth accumulates only along consecutive debug attempts; any other operation
        # resets it, because a fresh draft/improve is not "still stuck on the same broken idea".
        depth = (parent.debug_depth + 1) if (parent and operation == 'debug') else 0
        node = Node(self._next_id, parent_id, operation, summary, config=config,
                    code_path=code_path, debug_depth=depth, iteration=iteration)
        self.nodes[node.id] = node
        self._next_id += 1
        return node

    def mark_working(self, node_id, metrics, primary):
        n = self.nodes[node_id]
        n.status, n.metrics, n.primary, n.error = WORKING, metrics, float(primary), None

    def mark_buggy(self, node_id, error):
        n = self.nodes[node_id]
        n.status, n.error, n.primary, n.metrics = BUGGY, error, None, None

    # ---------------- queries ----------------

    def working(self):
        return [n for n in self.nodes.values() if n.status == WORKING]

    def best(self):
        w = self.working()
        return max(w, key=lambda n: n.primary) if w else None

    def _children(self, node_id):
        return [n for n in self.nodes.values() if n.parent_id == node_id]

    def _has_working_descendant(self, node_id):
        """Whether any node below `node_id` ever reached WORKING — i.e. this branch's problem was
        already solved by a descendant, so the broken ancestor needs no further repair."""
        return any(c.status == WORKING or self._has_working_descendant(c.id)
                   for c in self._children(node_id))

    def branch_root(self, node):
        """The non-debug ancestor a debug chain hangs off — the draft/improve/config node that
        originally introduced this idea. A node that isn't itself a debug attempt is its own root.
        Guards against a cycle in a hand-edited/corrupted tree file rather than hanging."""
        seen = set()
        while node.operation == 'debug' and node.parent_id is not None:
            if node.id in seen:
                break
            seen.add(node.id)
            parent = self.nodes.get(node.parent_id)
            if parent is None:
                break
            node = parent
        return node

    def branch_debug_attempts(self, node):
        """How many DEBUG attempts this node's whole branch has already absorbed — every debug
        node anywhere in the subtree below its branch root, not just along one chain. This is what
        MAX_DEBUG_DEPTH is a budget on."""
        root = self.branch_root(node)
        count, stack = 0, [root.id]
        seen = {root.id}
        while stack:
            for child in self._children(stack.pop()):
                if child.id in seen:
                    continue
                seen.add(child.id)
                if child.operation == 'debug':
                    count += 1
                stack.append(child.id)
        return count

    def debuggable(self):
        """Buggy nodes still worth spending a DEBUG child on. Excludes, in order:
          - config-only nodes (no code_path): a bad hyperparameter value isn't repairable by
            editing code, it just gets superseded by the next proposal.
          - branches that already spent MAX_DEBUG_DEPTH debug attempts: hopeless, stop feeding
            them compute. Measured per BRANCH (see branch_debug_attempts) — measuring it per node
            is what caused the 16-iteration livelock documented at MAX_DEBUG_DEPTH above.
          - branches where a descendant ALREADY reached WORKING: the fix has been found; the
            broken ancestor is history, not an open problem. (Found while unit-testing select():
            without this check the tree kept re-selecting a buggy parent whose debug child had
            already succeeded, burning an LLM call and a training run per iteration to re-fix
            something already fixed.)
        """
        return [n for n in self.nodes.values()
                if n.status == BUGGY and n.code_path
                and self.branch_debug_attempts(n) < MAX_DEBUG_DEPTH
                and not self._has_working_descendant(n.id)]

    # ---------------- the selection policy ----------------

    def select(self):
        """Decide what to do next. Returns (operation, node_or_None):

            ('debug',   node)  - a broken-but-promising solution is worth one cheap fix attempt
            ('draft',   None)  - not enough independent solutions yet; explore
            ('improve', node)  - refine the current best working solution

        Ordering rationale (this is the "which one to go towards" logic):
          1. DEBUG first. A node that failed to run is the cheapest possible win - the idea is
             already written, it just doesn't execute, and one targeted fix often converts a dead
             node into a working one. Leaving it broken wastes the LLM call that produced it.
             Capped by MAX_DEBUG_DEPTH so a hopeless branch can't absorb unlimited compute.
          2. DRAFT while the tree is still narrow (< MIN_DRAFTS working solutions). Committing to
             greedy refinement of a single early solution risks locking onto a local optimum
             before any real alternative has been tried.
          3. IMPROVE the best working node otherwise. Pure greedy exploitation, AIDE-style: the
             strongest solution so far is the most promising base to build on.
        """
        d = self.debuggable()
        if d:
            # Prefer the deepest-progressed broken branch (most invested), tie-broken by newest.
            return 'debug', max(d, key=lambda n: (n.debug_depth, n.id))
        if len(self.working()) < MIN_DRAFTS:
            return 'draft', None
        return 'improve', self.best()

    # ---------------- rendering ----------------

    def render(self):
        """Indented text view of the tree, for logs and the dashboard."""
        children = {}
        for n in self.nodes.values():
            children.setdefault(n.parent_id, []).append(n)
        lines = []

        def walk(parent_id, depth):
            for n in sorted(children.get(parent_id, []), key=lambda x: x.id):
                p = f"{n.primary:.4f}" if n.primary is not None else "  --  "
                mark = 'OK ' if n.status == WORKING else 'BUG'
                lines.append(f"{'  ' * depth}[{mark}] #{n.id} {p} ({n.operation}) {n.summary[:60]}")
                walk(n.id, depth + 1)

        walk(None, 0)
        return '\n'.join(lines) if lines else '(empty tree)'
