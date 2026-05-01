from src.b51_decision_diversity import compute_decision_diversity, compute_shortcut_rates


def test_decision_diversity_and_shortcut_rates():
    records = [
        {"record_kind": "closed_loop_policy", "inspect_skipped": 0, "predicted_intervention_action_type": "apply_local_push", "needs_inspection": 1, "episode_type": "inspect_needed"},
        {"record_kind": "closed_loop_policy", "inspect_skipped": 1, "predicted_intervention_action_type": "do_nothing", "needs_inspection": 0, "episode_type": "no_action"},
    ]
    diversity = compute_decision_diversity(records, {})
    shortcuts = compute_shortcut_rates(records, {})
    assert 0.0 <= diversity["decision_diversity_score"] <= 1.0
    assert 0.0 <= shortcuts["inspect_always_rate"] <= 1.0
