from __future__ import annotations

from .b6_4_transfer_generalization.result_review import review_b6_4_results


def main() -> None:
    review = review_b6_4_results()
    print(f"b6_4_submit_ready_as_diagnostic={str(review['submit_ready_as_diagnostic']).lower()}")
    print(f"b6_4_mean_transfer_score={review['transfer_evidence']['mean_b64_transfer_score']:.3f}")
    print(f"b6_4_mean_baseline_transfer_gap={review['transfer_evidence']['mean_baseline_transfer_gap']:.3f}")


if __name__ == "__main__":
    main()
