from __future__ import annotations

from .b6_4_1_transfer_hardening.result_review import review_b6_4_1_results


def main() -> None:
    review = review_b6_4_1_results()
    print(f"b6_4_1_submit_ready_as_diagnostic={str(review['submit_ready_as_diagnostic']).lower()}")
    print(f"b6_4_1_shortcut_equivalent_hard_remap_count={len(review['shortcut_equivalent_hard_remaps'])}")


if __name__ == "__main__":
    main()

