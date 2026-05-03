from src.b6_4_2_combined_remap_refinement.failure_attribution import attribute_failures
from src.b6_4_2_combined_remap_refinement.runner import run_b6_4_2_combined_refinement


def test_b642_failure_attribution_identifies_source():
    summary, records, _ = run_b6_4_2_combined_refinement({"b6_4_2": {"episodes_per_condition": 2}}, seed=0)
    attribution = attribute_failures(summary, records)
    assert attribution["combined_failure_source"]
    assert attribution["failure_attribution_confidence"] > 0.0
    assert attribution["pair_scores"]
    assert attribution["triple_scores"]
