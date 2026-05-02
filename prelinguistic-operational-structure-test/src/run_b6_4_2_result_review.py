from __future__ import annotations

from .b6_4_2_combined_remap_refinement.result_review import review_b6_4_2_results


def main() -> None:
    review = review_b6_4_2_results()
    print(f"b6_4_2_submit_ready_as_diagnostic={str(review['submit_ready_as_diagnostic']).lower()}")
    print(f"b6_4_2_combined_failure_source={review['combined_failure_source']}")
    print(f"b6_4_2_combined_oracle_gap={review['combined_oracle_gap']:.3f}")
    print(f"b6_4_2_gate_judgment={review['gate_judgment']}")


if __name__ == "__main__":
    main()
