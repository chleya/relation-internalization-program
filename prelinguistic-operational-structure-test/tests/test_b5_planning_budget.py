from src.b5_planning_budget import make_budget_record, planning_budget_compliant


def test_planning_budget_compliance():
    config = {"b5": {"planning_budget": {"max_candidate_inspections": 2, "max_candidate_interventions": 4, "max_rollout_evaluations": 6}}}
    assert planning_budget_compliant(make_budget_record(2, 4, 6), config)
    assert not planning_budget_compliant(make_budget_record(3, 4, 6), config)
