from __future__ import annotations

from .b6_4_1_transfer_hardening.adversarial_review import build_b6_4_1_adversarial_review


def main() -> None:
    review = build_b6_4_1_adversarial_review()
    print(f"b6_4_1_adversarial_decision={review['decision']}")
    print(f"b6_4_1_hidden_answer_cue_count={review['hidden_cue_audit']['hidden_answer_cue_count']}")
    print(f"b6_4_1_remaining_blocker_count={len(review['remaining_blockers'])}")


if __name__ == "__main__":
    main()
