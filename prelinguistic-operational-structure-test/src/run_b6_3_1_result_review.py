from __future__ import annotations

from .b6_3_1_refinement.result_review import review_b6_3_1_results


def main() -> None:
    review = review_b6_3_1_results()
    print(f"b6_3_1_submit_ready_as_diagnostic={str(review['submit_ready_as_diagnostic']).lower()}")
    print(f"b6_3_1_feedback_required_drop={review['feedback_history']['freeze_feedback_drop_on_feedback_required']:.3f}")
    print(f"b6_3_1_history_required_drop={review['feedback_history']['remove_history_drop_on_history_required']:.3f}")
    print(f"b6_3_1_credit_buffer_required_drop={review['delayed_credit']['drop_under_disable_credit_buffer_on_delay5_required']:.3f}")


if __name__ == "__main__":
    main()
