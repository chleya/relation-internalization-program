from src.b6_4_transfer_generalization.runner import run_b6_4_transfer


def test_state_only_degrades_on_visual_remap():
    summary, _records, _metrics = run_b6_4_transfer({"b6_4": {"episodes_per_condition": 2, "conditions": ["clean_reference", "visual_remap"]}}, seed=0)
    clean_state = row(summary, "clean_reference", "state_only")
    visual_state = row(summary, "visual_remap", "state_only")
    assert float(clean_state["transfer_score"]) > float(visual_state["transfer_score"])


def test_mask_only_degrades_when_mask_visibility_remapped():
    summary, _records, _metrics = run_b6_4_transfer({"b6_4": {"episodes_per_condition": 2, "conditions": ["clean_reference", "mask_visibility_remap"]}}, seed=0)
    clean_mask = row(summary, "clean_reference", "mask_only")
    remap_mask = row(summary, "mask_visibility_remap", "mask_only")
    assert float(clean_mask["transfer_score"]) > float(remap_mask["transfer_score"])


def test_b64_beats_shortcuts_on_combined_remap():
    summary, _records, _metrics = run_b6_4_transfer({"b6_4": {"episodes_per_condition": 2, "conditions": ["clean_reference", "combined_remap"]}}, seed=0)
    b64 = row(summary, "combined_remap", "b64_transfer_policy")
    state = row(summary, "combined_remap", "state_only")
    mask = row(summary, "combined_remap", "mask_only")
    assert float(b64["transfer_score"]) > float(state["transfer_score"])
    assert float(b64["transfer_score"]) > float(mask["transfer_score"])


def row(summary, condition, policy_name):
    return next(row for row in summary if row["condition"] == condition and row["policy_name"] == policy_name)
