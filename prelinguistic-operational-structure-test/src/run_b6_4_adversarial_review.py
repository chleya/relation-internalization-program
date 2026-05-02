from __future__ import annotations

from .b6_4_transfer_generalization.adversarial_review import build_b6_4_adversarial_review


def main() -> None:
    review = build_b6_4_adversarial_review()
    print(f"b6_4_adversarial_decision={review['decision']}")
    print(f"b6_4_shortcut_equivalent_remap_count={review['summary']['shortcut_equivalent_remap_count']}")
    print(f"b6_4_second_pass_needed={str(review['second_pass_needed']).lower()}")


if __name__ == "__main__":
    main()
