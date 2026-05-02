from __future__ import annotations

from collections import Counter
from typing import Any

from .combined_remap_env import ABLATION_CONDITIONS, PAIRWISE_CONDITIONS, TRIPLE_CONDITIONS


def attribute_failures(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    policy_rows = [row for row in summary if row["policy_name"] == "b64_2_combined_policy"]
    by_condition = {row["condition"]: row for row in policy_rows}
    pair_scores = {condition: float(by_condition.get(condition, {}).get("combined_refinement_score", 0.0)) for condition in PAIRWISE_CONDITIONS}
    triple_scores = {condition: float(by_condition.get(condition, {}).get("combined_refinement_score", 0.0)) for condition in TRIPLE_CONDITIONS}
    ablation_gaps = {
        condition: float(by_condition.get(condition, {}).get("combined_oracle_gap", 0.0))
        for condition in ABLATION_CONDITIONS
    }
    source_counts = Counter(row.get("transfer_source", "none") for row in records if row["policy_name"] == "b64_2_combined_policy")
    largest_ablation = max(ablation_gaps.items(), key=lambda item: item[1]) if ablation_gaps else ("none", 0.0)
    weakest_pair = min(pair_scores.items(), key=lambda item: item[1]) if pair_scores else ("none", 0.0)
    weakest_triple = min(triple_scores.items(), key=lambda item: item[1]) if triple_scores else ("none", 0.0)
    return {
        "pair_scores": pair_scores,
        "triple_scores": triple_scores,
        "ablation_oracle_gaps": ablation_gaps,
        "combined_failure_source": infer_failure_source(weakest_pair[0], weakest_triple[0], largest_ablation[0]),
        "weakest_pair": weakest_pair[0],
        "weakest_triple": weakest_triple[0],
        "largest_mechanism_gap": largest_ablation[0],
        "transfer_source_counts": dict(source_counts),
        "failure_attribution_confidence": min(1.0, 0.35 + largest_ablation[1] + (1.0 - weakest_triple[1]) * 0.5),
    }


def infer_failure_source(weakest_pair: str, weakest_triple: str, largest_ablation: str) -> str:
    if "dynamics_delay" in weakest_pair or "dynamics_delay" in weakest_triple or "delayed_credit" in largest_ablation:
        return "dynamics_delay_credit_interaction"
    if "mask_indirect" in weakest_pair or "mask_indirect" in weakest_triple or "candidate_search" in largest_ablation:
        return "mask_indirect_candidate_search_interaction"
    if "fallback_risk" in largest_ablation:
        return "fallback_risk_under_multi_cue_remap"
    if "trace_repair" in largest_ablation:
        return "trace_repair_under_multi_cue_remap"
    return "multi_cue_interaction"
