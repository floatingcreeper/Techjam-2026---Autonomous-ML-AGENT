"""Scripted moves for the MockDriver -- exercises the whole loop offline (no Gemini credits).
Used by `python -m agent.run --mock` and the M4/M5 integration tests.
"""
from agent.llm.schemas import Hypothesis, BlockEdit, RecoveryAction

# The Lever-A edit: route the loss block through the tested ranking-loss library so that
# loss_type becomes a config knob (bpr / softmax_ce / bce).
LOSS_SRC = '''from pipeline.lib.losses import make_loss


def build_loss(cfg):
    return make_loss(cfg)
'''


def _H(**kw):
    # problem_identified is a required Hypothesis field (1B); scripted moves auto-fill it.
    kw.setdefault("problem_identified", "(scripted mock move -- see tests/mock_moves.py)")
    return Hypothesis(**kw)


# A syntactically-valid loss block that crashes at runtime -- used to test recovery.
CRASH_LOSS = '''def build_loss(cfg):
    def lossfn(z, batch):
        raise RuntimeError("injected runtime failure")
    return lossfn
'''

_BPR_WIN = {"hypothesis": _H(lever="A",
                             statement="Switch the pointwise logloss to within-user BPR (an AUC surrogate)",
                             rationale="BPR aligns the loss with GAUC",
                             mutation_kind="block", target_block="loss",
                             config_delta_json='{"loss_type":"bpr","neg_ratio":4,"lr":0.001,"epochs":15,"patience":4}',
                             expected_metric="both", expected_gain=0.003),
            "block_edit": BlockEdit(target_block="loss", new_source=LOSS_SRC,
                                    imports_used=["pipeline.lib.losses"], notes="ranking loss")}


def build_fault_moves():
    """M5: exercise the recovery policy. A crashing block that the Reflector PATCHES, then a
    crashing block it ABANDONS (routes around). The run must never crash and must keep its best."""
    return [
        _BPR_WIN,
        {"hypothesis": _H(lever="A", statement="[fault-injection] loss block that raises at runtime",
                          rationale="verify code-error recovery", mutation_kind="block",
                          target_block="loss", config_delta_json='{}'),
         "block_edit": BlockEdit(target_block="loss", new_source=CRASH_LOSS, imports_used=[],
                                 notes="will crash"),
         "recovery": RecoveryAction(failure_class="code", action="patch_retry", patch_block="loss",
                                    new_source=LOSS_SRC, config_delta_json="{}",
                                    explanation="replace the broken loss with the tested library loss")},
        {"hypothesis": _H(lever="A", statement="[fault-injection] unrecoverable loss block",
                          rationale="verify route-around (abandon)", mutation_kind="block",
                          target_block="loss", config_delta_json='{}'),
         "block_edit": BlockEdit(target_block="loss", new_source=CRASH_LOSS, imports_used=[],
                                 notes="will crash"),
         "recovery": RecoveryAction(failure_class="code", action="abandon",
                                    explanation="cannot fix; route around and keep current best")},
    ]


def build_moves():
    return [
        {"hypothesis": _H(lever="A",
                          statement="Switch the pointwise logloss to within-user BPR (an AUC surrogate)",
                          rationale="GAUC/nDCG are ranking metrics; BPR optimizes pairwise order, aligning the loss with GAUC",
                          mutation_kind="block", target_block="loss",
                          config_delta_json='{"loss_type":"bpr","neg_ratio":4,"lr":0.001,"epochs":15,"patience":4}',
                          expected_metric="both", expected_gain=0.003),
         "block_edit": BlockEdit(target_block="loss", new_source=LOSS_SRC,
                                 imports_used=["pipeline.lib.losses"], notes="use ranking-loss library")},
        {"hypothesis": _H(lever="A", statement="Test listwise softmax-CE (nDCG surrogate) instead of BPR",
                          rationale="softmax-CE is an nDCG surrogate; check whether listwise beats pairwise on this data",
                          mutation_kind="config",
                          config_delta_json='{"loss_type":"softmax_ce","epochs":15,"patience":4}',
                          expected_metric="nDCG@5", expected_gain=0.002)},
        {"hypothesis": _H(lever="D",
                          statement="Adopt a LightGBM LambdaRank learner on engineered item/author features",
                          rationale="a tree-based model has a complementary inductive bias to the embedding FM; "
                                    "even if individually weaker, it should blend well in assembly (Lever F)",
                          mutation_kind="config", adopt_blockset="lgbm",
                          config_delta_json='{"model_type":"lgbm"}',
                          expected_metric="both", expected_gain=0.001)},
        {"hypothesis": _H(lever="B",
                          statement="Adopt a DeepFM+DIN model: target-attention over the user's video history + FM base features",
                          rationale="behavior sequences are the organizers' #1 unexplored direction; attention captures "
                                    "interest while the FM part keeps user x item memorization",
                          mutation_kind="config", adopt_blockset="din",
                          config_delta_json='{"model_type":"din","k":16,"batch":2048,"loss_type":"bpr",'
                                            '"neg_ratio":2,"lr":0.001,"epochs":4,"patience":2}',
                          expected_metric="both", expected_gain=0.005)},
        {"hypothesis": _H(lever="C",
                          statement="Adopt DIN with multi-task auxiliary heads (click/like) for regularisation",
                          rationale="auxiliary engagement signals regularise the shared embeddings and add "
                                    "ensemble diversity (Lever C)",
                          mutation_kind="config", adopt_blockset="din",
                          config_delta_json='{"model_type":"din","k":16,"batch":2048,"loss_type":"bpr",'
                                            '"neg_ratio":2,"lr":0.001,"epochs":4,"patience":2,'
                                            '"aux_tasks":["click","like"],"aux_weights":[0.1,0.1]}',
                          expected_metric="both", expected_gain=0.003)},
    ]
