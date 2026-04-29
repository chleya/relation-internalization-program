from src.agents import LearnedLinkBaseline, SlopeRelationChainModel, StructuralMemoryBaseline
from src.evaluate import edit_success, irrelevant_link_rejection, relation_audit


def test_relation_chain_edits_drainage_link():
    agent = SlopeRelationChainModel()
    context = {
        "rainfall": "high",
        "drainage": "good",
        "anchoring": "none",
        "toe_excavation": "no",
        "monitoring": "dense",
        "weather_label": "calm",
        "contractor_report": "normal",
    }
    assert agent.act(context) == "monitor"
    assert agent.edit_link("Drainage -> PorePressureDown", False)
    assert agent.act(context) == "drain"


def test_relation_audit_and_edit_metric():
    assert relation_audit(SlopeRelationChainModel()) == 1.0
    assert edit_success(SlopeRelationChainModel()) == 1.0


def test_structural_memory_has_no_relation_edit_or_audit():
    agent = StructuralMemoryBaseline()
    assert relation_audit(agent) == 0.0
    assert edit_success(agent) == 0.0


def test_learned_links_can_edit_and_audit_after_observations():
    agent = LearnedLinkBaseline(min_support=1)
    agent.observe(
        {
            "rainfall": "high",
            "drainage": "good",
            "anchoring": "none",
            "toe_excavation": "no",
            "monitoring": "dense",
        },
        "monitor",
        "monitor",
        1.0,
    )
    assert relation_audit(agent) == 1.0
    assert agent.edit_link("Drainage -> PorePressureDown", False)
    assert irrelevant_link_rejection(agent) == 1.0
