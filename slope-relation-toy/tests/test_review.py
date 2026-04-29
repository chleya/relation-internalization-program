from src.agents import GenericReviewBaseline, MajorityActionBaseline, SlopeRelationChainModel
from src.evaluate import review_consistency, review_score
from src.review import score_review_plan


def test_review_plan_scores_full_relation_chain():
    agent = SlopeRelationChainModel()
    context = {
        "rainfall": "high",
        "drainage": "poor",
        "anchoring": "none",
        "toe_excavation": "no",
        "monitoring": "sparse",
        "weather_label": "calm",
        "contractor_report": "normal",
    }
    scored = score_review_plan(agent.review_plan(context))
    assert scored["total"] == 14
    assert scored["passed"]


def test_review_score_rejects_empty_baseline():
    assert review_score(MajorityActionBaseline()) == 0.0


def test_review_score_rejects_generic_review():
    scored = score_review_plan(GenericReviewBaseline().review_plan({}))
    assert scored["total"] < 10
    assert not scored["passed"]


def test_review_consistency_checks_action_points():
    assert review_consistency(SlopeRelationChainModel()) == 1.0
    assert review_consistency(GenericReviewBaseline()) < 1.0
