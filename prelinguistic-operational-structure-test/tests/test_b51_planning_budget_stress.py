from src.b51_planning_budget_stress import compute_budget_stress_retention


def test_budget_stress_retention():
    assert compute_budget_stress_retention(1.0, 0.7) == 0.7
    assert compute_budget_stress_retention(0.0, 1.0) == 0.0
