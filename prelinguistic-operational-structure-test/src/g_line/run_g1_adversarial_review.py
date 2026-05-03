from __future__ import annotations

from .g1_adversarial_review import build_g1_adversarial_review


def main() -> None:
    review = build_g1_adversarial_review()
    print(f"g1_adversarial_decision={review['decision']}")
    print(f"g1_leakage_issue_found={str(review['leakage_issue_found']).lower()}")
    print(f"g1_metric_issue_found={str(review['metric_issue_found']).lower()}")
    print(f"g1_weak_pressure_count={review['weak_pressure_count']}")


if __name__ == "__main__":
    main()
