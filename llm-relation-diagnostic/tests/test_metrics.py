from __future__ import annotations

from src.cases import (
    make_audit_correctness_stress_cases,
    make_budgeted_inspect_stress_cases,
    make_cases,
    make_local_edit_behavior_stress_cases,
)
from src.metrics import evaluate_solver, summarize_records
from src.metrics import answers_by_query_match
from src.solvers import (
    EditComplianceSolver,
    EditNoBehaviorSolver,
    GlobalMappingSolver,
    MissingnessTemplateSolver,
    OutcomeAuditSolver,
    RelationOracleSolver,
    ReverseAuditSolver,
    SurfaceAuditSolver,
    parse_json_response,
)


GATES = {
    "random_symbol_transfer": 1.0,
    "support_conditioned_binding": 1.0,
    "counterfactual_use": 1.0,
    "local_edit_locality": 1.0,
    "audit_correctness": 1.0,
    "missing_observation_uncertainty": 1.0,
    "budgeted_inspect": 1.0,
}


def _summary_for(solver):
    records, _ = evaluate_solver(solver, make_cases(seed=0), seed=0)
    return summarize_records(records, GATES)[0]


def test_oracle_passes_all_gates() -> None:
    row = _summary_for(RelationOracleSolver())
    assert row["llm_relation_gated_score"] == 1.0


def test_global_mapping_fails_support_conditioned_binding() -> None:
    row = _summary_for(GlobalMappingSolver())
    assert row["support_conditioned_binding"] == 0.0
    assert row["llm_relation_gated_score"] == 0.0


def test_edit_compliance_fails_local_edit_locality() -> None:
    row = _summary_for(EditComplianceSolver())
    assert row["local_edit_locality"] == 0.0
    assert row["llm_relation_gated_score"] == 0.0


def test_missingness_template_fails_specific_inspection() -> None:
    row = _summary_for(MissingnessTemplateSolver())
    assert row["missing_observation_uncertainty"] == 0.0
    assert row["budgeted_inspect"] == 0.0
    assert row["llm_relation_gated_score"] == 0.0


def test_surface_audit_fails_exact_audit_link() -> None:
    row = _summary_for(SurfaceAuditSolver())
    assert row["audit_correctness"] == 0.0
    assert row["budgeted_inspect"] == 0.0
    assert row["llm_relation_gated_score"] == 0.0


def test_parse_json_response_handles_fenced_json() -> None:
    parsed = parse_json_response('```json\n{"answer": "z9=ru", "uncertain": false}\n```')
    assert parsed["answer"] == "z9=ru"
    assert parsed["uncertain"] is False


def test_strict_profile_preserves_gate_expectations() -> None:
    records, _ = evaluate_solver(RelationOracleSolver(), make_cases(seed=0, profile="strict"), seed=0)
    row = summarize_records(records, GATES)[0]
    assert row["llm_relation_gated_score"] == 1.0
    assert all(record["case_id"].endswith("_strict") for record in records)


def test_budgeted_inspect_stress_oracle_passes() -> None:
    records, _ = evaluate_solver(
        RelationOracleSolver(),
        make_budgeted_inspect_stress_cases(seed=0, profile="strict"),
        seed=0,
    )
    row = summarize_records(records, {"budgeted_inspect": 1.0})[0]
    assert row["budgeted_inspect"] == 1.0
    assert row["llm_relation_gated_score"] == 1.0


def test_budgeted_inspect_stress_exposes_first_missing_template() -> None:
    records, _ = evaluate_solver(
        MissingnessTemplateSolver(),
        make_budgeted_inspect_stress_cases(seed=0, profile="strict"),
        seed=0,
    )
    row = summarize_records(records, {"budgeted_inspect": 1.0})[0]
    assert 0.0 < row["budgeted_inspect"] < 1.0
    assert row["llm_relation_gated_score"] == 0.0


def test_local_edit_behavior_stress_oracle_passes() -> None:
    records, _ = evaluate_solver(
        RelationOracleSolver(),
        make_local_edit_behavior_stress_cases(seed=0, profile="strict"),
        seed=0,
    )
    row = summarize_records(records, {"local_edit_locality": 1.0})[0]
    assert row["local_edit_locality"] == 1.0
    assert row["llm_relation_gated_score"] == 1.0


def test_local_edit_behavior_stress_catches_edit_compliance() -> None:
    records, _ = evaluate_solver(
        EditComplianceSolver(),
        make_local_edit_behavior_stress_cases(seed=0, profile="strict"),
        seed=0,
    )
    row = summarize_records(records, {"local_edit_locality": 1.0})[0]
    assert row["local_edit_locality"] == 0.0
    assert row["llm_relation_gated_score"] == 0.0


def test_local_edit_behavior_stress_catches_no_behavior_edit() -> None:
    records, _ = evaluate_solver(
        EditNoBehaviorSolver(),
        make_local_edit_behavior_stress_cases(seed=0, profile="strict"),
        seed=0,
    )
    row = summarize_records(records, {"local_edit_locality": 1.0})[0]
    assert row["local_edit_locality"] == 0.0
    assert row["llm_relation_gated_score"] == 0.0


def test_answers_by_query_accepts_value_only_answers() -> None:
    response = {"answers_by_query": {"target_chain": "sa"}}
    expected = {"answers_by_query": {"target_chain": "q2=sa"}}
    assert answers_by_query_match(response, expected)


def test_audit_correctness_stress_oracle_passes() -> None:
    records, _ = evaluate_solver(
        RelationOracleSolver(),
        make_audit_correctness_stress_cases(seed=0, profile="strict"),
        seed=0,
    )
    row = summarize_records(records, {"audit_correctness": 1.0})[0]
    assert row["audit_correctness"] == 1.0
    assert row["llm_relation_gated_score"] == 1.0


def test_audit_correctness_stress_catches_surface_audit() -> None:
    records, _ = evaluate_solver(
        SurfaceAuditSolver(),
        make_audit_correctness_stress_cases(seed=0, profile="strict"),
        seed=0,
    )
    row = summarize_records(records, {"audit_correctness": 1.0})[0]
    assert row["audit_correctness"] == 0.0
    assert row["llm_relation_gated_score"] == 0.0


def test_audit_correctness_stress_catches_reversed_links() -> None:
    records, _ = evaluate_solver(
        ReverseAuditSolver(),
        make_audit_correctness_stress_cases(seed=0, profile="strict"),
        seed=0,
    )
    row = summarize_records(records, {"audit_correctness": 1.0})[0]
    assert row["audit_correctness"] < 1.0
    assert row["llm_relation_gated_score"] == 0.0


def test_audit_correctness_stress_catches_outcome_audit() -> None:
    records, _ = evaluate_solver(
        OutcomeAuditSolver(),
        make_audit_correctness_stress_cases(seed=0, profile="strict"),
        seed=0,
    )
    row = summarize_records(records, {"audit_correctness": 1.0})[0]
    assert row["audit_correctness"] < 1.0
    assert row["llm_relation_gated_score"] == 0.0
