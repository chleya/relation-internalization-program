from src.b6_3_structural_necessity.env import make_b63_episode
from src.b6_3_structural_necessity.trace_repair import detect_trace_conflict, repair_trace_or_request_inspection


def test_wrong_trace_triggers_conflict_and_downgrade():
    episode = make_b63_episode({"b6_3": {"grid_size": 8}}, 0, "wrong_trace")
    model_input = episode["model_input"]
    conflict = detect_trace_conflict(model_input, {})
    repair = repair_trace_or_request_inspection(model_input, conflict, {})
    assert conflict["trace_conflict_detected"] is True
    assert conflict["trace_confidence_after"] < conflict["trace_confidence_before"]
    assert repair["repair_action"] in {"repair", "inspect", "abstain", "downgrade"}


def test_hidden_public_state_cue_can_break_state_repair_path():
    episode = make_b63_episode({"b6_3": {"grid_size": 8}}, 0, "hide_public_state_cue")
    conflict = detect_trace_conflict(episode["model_input"], {})
    assert conflict["state_supported_region"] is not None
    assert conflict["repair_source"] in {"history", "feedback", "none", "state"}
