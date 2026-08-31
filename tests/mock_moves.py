"""Scripted moves for the MockDriver -- exercises the whole loop offline (no Gemini credits).
Used by `python -m agent.run --mock` / `--faults`.

Deliberately includes moves that must be REJECTED BEFORE TRAINING, so `--mock` exercises the
docs/RESEARCH.md §7 (BPR) / docs/SYSTEM.md §11/docs/SYSTEM.md §12/docs/SYSTEM.md §12 machinery rather than only the happy path:
  * an unhonoured/invalid config knob  -> effective-config validation
  * a repeat of an earlier experiment  -> content-based deduplication

The MockDriver advances one move per Proposer call, so a rejected proposal naturally consumes the
next move as its re-proposal -- which is exactly the loop the real agent runs.

Note the Lever-A move is now a PURE CONFIG mutation. Before the docs/SYSTEM.md §11 fix that was a silent no-op
(baseline_blocks/loss.py hardcoded BCE and ignored cfg.loss_type, so it trained the baseline and
scored 0.60147, identical to root). It now genuinely trains BPR and scores ~0.6036.
"""
from agent.llm.schemas import BlockEdit, Hypothesis, RecoveryAction

# A loss block routed through the tested ranking-loss library. Still used by the recovery path and by
# the block-edit coverage move.
LOSS_SRC = '''from pipeline.lib.losses import make_loss


def build_loss(cfg):
    return make_loss(cfg)
'''

# A syntactically-valid loss block that crashes at runtime -- used to test recovery.
CRASH_LOSS = '''def build_loss(cfg):
    def lossfn(z, batch):
        raise RuntimeError("injected runtime failure")
    return lossfn
'''

BPR_DELTA = '{"loss_type":"bpr","neg_ratio":4,"epochs":15}'


def _H(**kw):
    kw.setdefault("problem_identified", "(scripted mock move -- see tests/mock_moves.py)")
    return Hypothesis(**kw)


_BPR_WIN = {"hypothesis": _H(lever="A",
                             statement="Switch the pointwise logloss to within-user BPR (an AUC surrogate)",
                             rationale="BPR aligns the loss with GAUC",
                             mutation_kind="config", config_delta_json=BPR_DELTA,
                             expected_metric="both", expected_gain=0.003)}


def build_fault_moves():
    """Exercise the recovery policy: a crashing block the Reflector PATCHES, then one it ABANDONS.
    The run must never crash and must keep its best checkpoint."""
    return [
        _BPR_WIN,
        {"hypothesis": _H(lever="A", statement="[fault-injection] loss block that raises at runtime",
                          rationale="verify code-error recovery", mutation_kind="block",
                          target_block="loss", config_delta_json='{"epochs":12}'),
         "block_edit": BlockEdit(target_block="loss", new_source=CRASH_LOSS, imports_used=[],
                                 notes="will crash"),
         "recovery": RecoveryAction(failure_class="code", action="patch_retry", patch_block="loss",
                                    new_source=LOSS_SRC, config_delta_json="{}",
                                    explanation="replace the broken loss with the tested library loss")},
        {"hypothesis": _H(lever="A", statement="[fault-injection] unrecoverable loss block",
                          rationale="verify route-around (abandon)", mutation_kind="block",
                          target_block="loss", config_delta_json='{"epochs":11}'),
         "block_edit": BlockEdit(target_block="loss", new_source=CRASH_LOSS, imports_used=[],
                                 notes="will crash"),
         "recovery": RecoveryAction(failure_class="code", action="abandon",
                                    explanation="cannot fix; route around and keep current best")},
    ]


def build_moves():
    return [
        # 1. REJECTED before training: `num_leaves` is not a Cfg field (it would be silently dropped)
        #    and `aux_tasks` is not honoured by the fm block set.
        {"hypothesis": _H(lever="D",
                          statement="Tune LightGBM leaf count directly from the fm baseline",
                          rationale="deeper trees may fit the ranking better",
                          mutation_kind="config",
                          config_delta_json='{"num_leaves":127,"aux_tasks":["click"]}',
                          expected_metric="both", expected_gain=0.002)},
        # 2. The Lever-A win, as a pure config mutation (the docs/SYSTEM.md §11 fix in action).
        _BPR_WIN,
        # 3. Listwise alternative -- a real, informative NEGATIVE result (~0.5997).
        {"hypothesis": _H(lever="A", statement="Test listwise softmax-CE (an nDCG surrogate) instead of BPR",
                          rationale="softmax-CE is an nDCG surrogate; check whether listwise beats pairwise here",
                          mutation_kind="config",
                          config_delta_json='{"loss_type":"softmax_ce","epochs":15}',
                          expected_metric="nDCG@5", expected_gain=0.002)},
        # 4. A complementary model family -- weak standalone, valuable in the portfolio.
        {"hypothesis": _H(lever="D",
                          statement="Adopt a LightGBM LambdaRank learner on engineered item/author features",
                          rationale="a tree-based model has a complementary inductive bias to the embedding FM; "
                                    "even if individually weaker it should blend well (Lever F)",
                          mutation_kind="config", adopt_blockset="lgbm",
                          config_delta_json='{}',
                          expected_metric="both", expected_gain=0.001)},
        # 5. Sequence modelling.
        {"hypothesis": _H(lever="B",
                          statement="Adopt a DeepFM+DIN model: target-attention over the user's video history",
                          rationale="behavior sequences are the organizers' #1 unexplored direction; attention "
                                    "captures interest while the FM part keeps user x item memorization",
                          mutation_kind="config", adopt_blockset="din",
                          config_delta_json='{"k":16,"batch":2048,"loss_type":"bpr",'
                                            '"neg_ratio":2,"epochs":4,"patience":2}',
                          expected_metric="both", expected_gain=0.005)},
        # 6. REJECTED before training: byte-identical to move 5 -- content-based dedup (docs/SYSTEM.md §12).
        {"hypothesis": _H(lever="B",
                          statement="Adopt DeepFM+DIN with target attention over user history",
                          rationale="re-proposing the same architecture",
                          mutation_kind="config", adopt_blockset="din",
                          config_delta_json='{"k":16,"batch":2048,"loss_type":"bpr",'
                                            '"neg_ratio":2,"epochs":4,"patience":2}',
                          expected_metric="both", expected_gain=0.005)},
        # 7. Multi-task heads.
        {"hypothesis": _H(lever="C",
                          statement="Adopt DIN with multi-task auxiliary heads (click/like)",
                          rationale="auxiliary engagement signals regularise the shared embeddings and add "
                                    "ensemble diversity (Lever C)",
                          mutation_kind="config", adopt_blockset="din",
                          config_delta_json='{"k":16,"batch":2048,"loss_type":"bpr",'
                                            '"neg_ratio":2,"epochs":4,"patience":2,'
                                            '"aux_tasks":["click","like"],"aux_weights":[0.1,0.1]}',
                          expected_metric="both", expected_gain=0.003)},
        # 8. Block-edit coverage (Coder + check_imports path).
        {"hypothesis": _H(lever="A", statement="Rewrite the loss block to route through the loss library",
                          rationale="make the objective a first-class config knob",
                          mutation_kind="block", target_block="loss",
                          config_delta_json='{"epochs":13}'),
         "block_edit": BlockEdit(target_block="loss", new_source=LOSS_SRC,
                                 imports_used=["pipeline.lib.losses"], notes="ranking loss library")},
    ]
