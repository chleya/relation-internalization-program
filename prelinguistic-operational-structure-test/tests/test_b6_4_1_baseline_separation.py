from src.b6_4_1_transfer_hardening.runner import run_b6_4_1_transfer_hardening


def test_visual_hard_drops_state_and_trace_only():
    summary, _records, _metrics = run_b6_4_1_transfer_hardening({"b6_4_1": {"episodes_per_condition": 2, "conditions": ["clean_reference", "visual_remap_hard"]}}, seed=0)
    visual = row(summary, "visual_remap_hard", "b64_1_transfer_policy")
    state = row(summary, "visual_remap_hard", "state_only")
    trace = row(summary, "visual_remap_hard", "trace_only")
    assert float(visual["hard_transfer_score"]) > float(state["hard_transfer_score"])
    assert float(visual["hard_transfer_score"]) > float(trace["hard_transfer_score"])
    assert float(visual["visual_hard_state_only_drop"]) > 0.20
    assert float(visual["visual_hard_trace_only_drop"]) > 0.20


def test_mask_hard_drops_mask_only():
    summary, _records, _metrics = run_b6_4_1_transfer_hardening({"b6_4_1": {"episodes_per_condition": 2, "conditions": ["clean_reference", "mask_visibility_remap_hard"]}}, seed=0)
    policy = row(summary, "mask_visibility_remap_hard", "b64_1_transfer_policy")
    mask = row(summary, "mask_visibility_remap_hard", "mask_only")
    assert float(policy["hard_transfer_score"]) > float(mask["hard_transfer_score"])
    assert float(policy["mask_hard_mask_only_drop"]) > 0.20


def test_combined_hard_keeps_baseline_gap():
    summary, _records, metrics = run_b6_4_1_transfer_hardening({"b6_4_1": {"episodes_per_condition": 4, "conditions": ["clean_reference", "combined_remap_hard"]}}, seed=0)
    policy = row(summary, "combined_remap_hard", "b64_1_transfer_policy")
    state = row(summary, "combined_remap_hard", "state_only")
    assert float(policy["hard_transfer_score"]) > float(state["hard_transfer_score"])
    assert metrics["shortcut_equivalent_hard_remap_count"] == 0


def row(summary, condition, policy_name):
    return next(row for row in summary if row["condition"] == condition and row["policy_name"] == policy_name)
