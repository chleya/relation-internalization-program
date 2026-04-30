from __future__ import annotations

from statistics import mean
from typing import Any

from .cases import DiagnosticCase
from .solvers import DiagnosticSolver


GATE_NAMES = [
    "random_symbol_transfer",
    "support_conditioned_binding",
    "counterfactual_use",
    "local_edit_locality",
    "audit_correctness",
    "missing_observation_uncertainty",
    "budgeted_inspect",
]


def _norm(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "")


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _list_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def answer_match(response: dict[str, Any], expected: dict[str, Any]) -> bool:
    answers = expected.get("answers")
    if not answers:
        return True
    actual = _norm(response.get("answer", ""))
    return any(actual == _norm(answer) for answer in answers)


def _answer_value_variants(answer: Any) -> set[str]:
    text = str(answer)
    variants = {_norm(text)}
    if "=" in text:
        variants.add(_norm(text.split("=", 1)[1]))
    return variants


def inspect_match(response: dict[str, Any], expected: dict[str, Any]) -> bool:
    if "inspect" not in expected:
        return True
    return _norm(response.get("inspect", "")) == _norm(expected["inspect"])


def uncertain_match(response: dict[str, Any], expected: dict[str, Any]) -> bool:
    if "uncertain" not in expected:
        return True
    return _as_bool(response.get("uncertain")) is expected["uncertain"]


def audit_links_match(response: dict[str, Any], expected: dict[str, Any]) -> bool:
    links = expected.get("audit_links")
    if expected.get("exact_audit_links"):
        actual_links = [_norm(item) for item in _list_values(response.get("audit_links"))]
        return set(actual_links) == {_norm(link) for link in _list_values(links)}
    if not links:
        return True
    actual_links = [_norm(item) for item in _list_values(response.get("audit_links"))]
    return all(any(_norm(link) in actual for actual in actual_links) for link in links)


def query_sets_match(response: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key in ["affected_queries", "unchanged_queries"]:
        if key not in expected:
            continue
        actual = {_norm(item) for item in _list_values(response.get(key))}
        wanted = {_norm(item) for item in expected[key]}
        if actual != wanted:
            return False
    return True


def answers_by_query_match(response: dict[str, Any], expected: dict[str, Any]) -> bool:
    wanted = expected.get("answers_by_query")
    if not wanted:
        return True
    actual = response.get("answers_by_query")
    if not isinstance(actual, dict):
        return False
    for query_id, expected_answer in wanted.items():
        if _norm(actual.get(query_id, "")) not in _answer_value_variants(expected_answer):
            return False
    return set(_norm(key) for key in actual.keys()) == set(_norm(key) for key in wanted.keys())


def score_response(case: DiagnosticCase, response: dict[str, Any]) -> dict[str, float]:
    checks = {
        "answer_match": answer_match(response, case.expected),
        "inspect_match": inspect_match(response, case.expected),
        "uncertain_match": uncertain_match(response, case.expected),
        "audit_links_match": audit_links_match(response, case.expected),
        "query_sets_match": query_sets_match(response, case.expected),
        "answers_by_query_match": answers_by_query_match(response, case.expected),
    }
    passed = all(checks.values())
    return {
        **{name: 1.0 if value else 0.0 for name, value in checks.items()},
        "case_score": 1.0 if passed else 0.0,
    }


def evaluate_solver(solver: DiagnosticSolver, cases: list[DiagnosticCase], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    for case in cases:
        try:
            response = solver.solve(case)
            error = ""
        except Exception as exc:  # pragma: no cover - exercised only for live providers
            response = {"error": f"{type(exc).__name__}: {exc}"}
            error = response["error"]
        scores = score_response(case, response)
        records.append(
            {
                "solver": solver.name,
                "seed": seed,
                "case_id": case.case_id,
                "gate": case.gate,
                "case_score": scores["case_score"],
                "answer_match": scores["answer_match"],
                "inspect_match": scores["inspect_match"],
                "uncertain_match": scores["uncertain_match"],
                "audit_links_match": scores["audit_links_match"],
                "query_sets_match": scores["query_sets_match"],
                "answers_by_query_match": scores["answers_by_query_match"],
                "error": error,
            }
        )
        raw_records.append(
            {
                "solver": solver.name,
                "seed": seed,
                "case_id": case.case_id,
                "gate": case.gate,
                "expected": case.expected,
                "response": response,
                "scores": scores,
            }
        )
    return records, raw_records


def summarize_records(records: list[dict[str, Any]], gates: dict[str, float]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    gate_names = list(gates.keys()) if gates else GATE_NAMES
    solvers = sorted({record["solver"] for record in records})
    for solver in solvers:
        rows = [record for record in records if record["solver"] == solver]
        gate_scores = {}
        for gate in gate_names:
            gate_rows = [row for row in rows if row["gate"] == gate]
            gate_scores[gate] = mean(row["case_score"] for row in gate_rows) if gate_rows else 0.0
        thresholds_pass = all(gate_scores[name] >= gates.get(name, 1.0) for name in gate_names)
        summary.append(
            {
                "solver": solver,
                "n": len(rows),
                **gate_scores,
                "mean_case_score": mean(row["case_score"] for row in rows) if rows else 0.0,
                "llm_relation_gated_score": mean(gate_scores.values()) if thresholds_pass and gate_scores else 0.0,
            }
        )
    return summary
