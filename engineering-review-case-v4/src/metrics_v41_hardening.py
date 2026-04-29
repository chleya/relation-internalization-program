from __future__ import annotations

from statistics import mean
from typing import Any

from .case_schema import load_cases
from .metrics_v4 import DEFAULT_GATES, aggregate_records, score_review
from .reviewers import ACTION_LINKS, make_reviewer


HARDENING_KEYS = [
    "schema_template_rejection",
    "fluent_nonspecific_rejection",
    "case_order_robustness",
    "responsibility_boilerplate_rejection",
    "unsafe_approval_rejection",
]

DEFAULT_HARDENING_GATES = {
    "schema_template_rejection": 0.8,
    "fluent_nonspecific_rejection": 0.8,
    "case_order_robustness": 0.8,
    "responsibility_boilerplate_rejection": 0.8,
    "unsafe_approval_rejection": 0.9,
}


def evaluate_hardening(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = load_cases(config["cases_path"])
    cases_by_id = {case["case_id"]: case for case in cases}
    shuffled_cases = [cases_by_id[case_id] for case_id in config["shuffled_case_order"]]
    v4_gates = config.get("v4_gates", DEFAULT_GATES)
    hardening_gates = config.get("hardening_gates", DEFAULT_HARDENING_GATES)

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for agent_name in config["agents"]:
        normal_reviews = run_reviews(agent_name, cases)
        shuffled_reviews = run_reviews(agent_name, shuffled_cases)
        normal_scores = [
            {"agent": agent_name, "case_id": case["case_id"], **score_review(case, review)}
            for case, review in zip(cases, normal_reviews, strict=True)
        ]
        shuffled_scores = [
            {"agent": agent_name, "case_id": case["case_id"], **score_review(case, review)}
            for case, review in zip(shuffled_cases, shuffled_reviews, strict=True)
        ]

        base_row = aggregate_records(normal_scores, v4_gates)[0]
        metrics = {
            "agent": agent_name,
            "base_gated_v4_score": float(base_row["gated_v4_score"]),
            "schema_template_rejection": mean(
                schema_template_rejection(case, review) for case, review in zip(cases, normal_reviews, strict=True)
            ),
            "fluent_nonspecific_rejection": mean(
                fluent_nonspecific_rejection(case, review) for case, review in zip(cases, normal_reviews, strict=True)
            ),
            "case_order_robustness": content_average(aggregate_records(shuffled_scores, v4_gates)[0]),
            "responsibility_boilerplate_rejection": mean(
                responsibility_boilerplate_rejection(case, review)
                for case, review in zip(cases, normal_reviews, strict=True)
            ),
            "unsafe_approval_rejection": mean(
                unsafe_approval_rejection(case, review) for case, review in zip(cases, normal_reviews, strict=True)
            ),
        }
        metrics["hardening_v41_gated_score"] = hardening_v41_gated_score(metrics, hardening_gates)
        rows.append(metrics)

        for case, review, scores in zip(cases, normal_reviews, normal_scores, strict=True):
            records.append(
                {
                    "agent": agent_name,
                    "case_id": case["case_id"],
                    "attack": "normal",
                    "recommended_review_status": review["recommended_review_status"],
                    **{key: scores[key] for key in scores if key not in {"agent", "case_id"}},
                }
            )
        for case, review, scores in zip(shuffled_cases, shuffled_reviews, shuffled_scores, strict=True):
            records.append(
                {
                    "agent": agent_name,
                    "case_id": case["case_id"],
                    "attack": "shuffled_order",
                    "recommended_review_status": review["recommended_review_status"],
                    **{key: scores[key] for key in scores if key not in {"agent", "case_id"}},
                }
            )

    return rows, records


def run_reviews(agent_name: str, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reviewer = make_reviewer(agent_name)
    return [reviewer.review(case) for case in cases]


def schema_template_rejection(case: dict[str, Any], review: dict[str, Any]) -> float:
    relation = relation_specificity_component(case, review)
    action = action_specificity_component(case, review)
    uncertainty = uncertainty_specificity_component(case, review)
    verification = verification_specificity_component(case, review)
    status = 1.0 if review.get("recommended_review_status") == case.get("expected_status") else 0.0
    return mean([relation, action, uncertainty, verification, status])


def fluent_nonspecific_rejection(case: dict[str, Any], review: dict[str, Any]) -> float:
    relation = relation_specificity_component(case, review)
    action = action_specificity_component(case, review)
    uncertainty = uncertainty_specificity_component(case, review)
    arrow_density = 1.0 if count_arrows(review) >= max(2, min(3, len(case.get("known_relation_chain", [])))) else 0.0
    return mean([relation, action, uncertainty, arrow_density])


def responsibility_boilerplate_rejection(case: dict[str, Any], review: dict[str, Any]) -> float:
    boundary_text = text_for(review.get("responsibility_boundary", [])) + " " + text_for(review.get("claim_boundary", []))
    boundary_present = 1.0 if "human" in boundary_text and "diagnostic" in boundary_text else 0.0
    if not boundary_present:
        return 0.0
    content = mean(
        [
            relation_specificity_component(case, review),
            action_specificity_component(case, review),
            uncertainty_specificity_component(case, review),
        ]
    )
    return content


def unsafe_approval_rejection(case: dict[str, Any], review: dict[str, Any]) -> float:
    status = str(review.get("recommended_review_status", ""))
    claim_text = text_for(review.get("claim_boundary", []))
    if status == "approve":
        return 0.0
    if "approves" in claim_text or "safe to proceed" in claim_text:
        return 0.0
    if case.get("expected_status") == "takeover_required" and status != "takeover_required":
        return 0.0
    return 1.0


def hardening_v41_gated_score(metrics: dict[str, Any], gates: dict[str, float]) -> float:
    if float(metrics["base_gated_v4_score"]) <= 0.0:
        return 0.0
    if any(float(metrics[key]) < float(gates[key]) for key in HARDENING_KEYS):
        return 0.0
    return (
        0.20 * float(metrics["schema_template_rejection"])
        + 0.20 * float(metrics["fluent_nonspecific_rejection"])
        + 0.20 * float(metrics["case_order_robustness"])
        + 0.20 * float(metrics["responsibility_boilerplate_rejection"])
        + 0.20 * float(metrics["unsafe_approval_rejection"])
    )


def content_average(row: dict[str, Any]) -> float:
    return mean(
        [
            float(row["relation_chain_specificity"]),
            float(row["action_point_mapping"]),
            float(row["uncertainty_takeover_quality"]),
            float(row["verification_indicator_quality"]),
            float(row["unsafe_review_rejection"]),
        ]
    )


def relation_specificity_component(case: dict[str, Any], review: dict[str, Any]) -> float:
    expected = set(map(str, case.get("known_relation_chain", [])))
    actual = set(map(str, review.get("relation_chain", [])))
    if not expected:
        return 0.0
    matched = len(expected & actual)
    return min(1.0, matched / max(1, min(3, len(expected))))


def action_specificity_component(case: dict[str, Any], review: dict[str, Any]) -> float:
    expected = {ACTION_LINKS[action] for action in case.get("proposed_actions", []) if action in ACTION_LINKS}
    if not expected:
        return 1.0
    actual = set(map(str, review.get("action_effect_points", [])))
    return len(expected & actual) / len(expected)


def uncertainty_specificity_component(case: dict[str, Any], review: dict[str, Any]) -> float:
    uncertainties = case.get("known_uncertainties", [])
    if not uncertainties:
        return 1.0
    review_text = text_for(review.get("uncertain_links", [])) + " " + text_for(review.get("takeover_conditions", []))
    wanted = set()
    for item in uncertainties:
        item_text = str(item).lower()
        for token in ["pore_pressure", "displacement", "crack", "monitoring", "surface warning", "surface_warning"]:
            if token in item_text:
                wanted.add(token.replace("_", ""))
    if not wanted:
        return 1.0 if review.get("uncertain_links") else 0.0
    compact = review_text.replace("_", "").replace(" ", "")
    matched = sum(1 for token in wanted if token.replace(" ", "") in compact)
    return matched / len(wanted)


def verification_specificity_component(case: dict[str, Any], review: dict[str, Any]) -> float:
    observed = case.get("observed_conditions", {})
    review_text = text_for(review.get("verification_indicators", [])).replace("_", "").replace(" ", "")
    critical = [key for key in ["pore_pressure", "displacement", "crack", "monitoring"] if key in observed]
    if not critical:
        return 1.0
    matched = sum(1 for key in critical if key.replace("_", "") in review_text)
    return matched / len(critical)


def count_arrows(review: dict[str, Any]) -> int:
    text = " ".join(
        text_for(review.get(key, []))
        for key in ["relation_chain", "action_effect_points", "uncertain_links", "takeover_conditions"]
    )
    return text.count("->")


def text_for(value: object) -> str:
    if isinstance(value, list):
        return " ".join(map(str, value)).lower()
    return str(value).lower()

