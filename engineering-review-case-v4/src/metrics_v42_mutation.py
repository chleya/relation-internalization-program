from __future__ import annotations

import json
from statistics import mean
from typing import Any

from .case_schema import load_cases
from .metrics_v4 import DEFAULT_GATES, aggregate_records, score_review
from .mutations_v42 import IRRELEVANT_FIELDS, mutate_cases, normalize_relation_text
from .reviewers import make_reviewer


MUTATION_KEYS = [
    "mutated_case_consistency",
    "irrelevant_variable_rejection",
    "paraphrase_relation_robustness",
    "hidden_unsafe_phrase_rejection",
    "field_order_robustness",
]

DEFAULT_MUTATION_GATES = {
    "mutated_case_consistency": 0.8,
    "irrelevant_variable_rejection": 0.9,
    "paraphrase_relation_robustness": 0.8,
    "hidden_unsafe_phrase_rejection": 0.9,
    "field_order_robustness": 0.8,
}


def evaluate_mutation(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base_cases = load_cases(config["cases_path"])
    mutated_cases = mutate_cases(base_cases)
    v4_gates = config.get("v4_gates", DEFAULT_GATES)
    mutation_gates = config.get("mutation_gates", DEFAULT_MUTATION_GATES)

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for agent_name in config["agents"]:
        reviewer = make_reviewer(agent_name)
        reviews = [reviewer.review(case) for case in mutated_cases]
        scored = [
            {"agent": agent_name, "case_id": case["case_id"], **score_review(case, review)}
            for case, review in zip(mutated_cases, reviews, strict=True)
        ]
        base_row = aggregate_records(scored, v4_gates)[0]
        row: dict[str, Any] = {
            "agent": agent_name,
            "mutated_base_gated_v4_score": float(base_row["gated_v4_score"]),
            "mutated_case_consistency": content_consistency(base_row),
            "irrelevant_variable_rejection": mean(
                irrelevant_variable_rejection(review) for review in reviews
            ),
            "paraphrase_relation_robustness": mean(
                paraphrase_relation_robustness(case, review)
                for case, review in zip(mutated_cases, reviews, strict=True)
            ),
            "hidden_unsafe_phrase_rejection": mean(
                hidden_unsafe_phrase_rejection(case, review)
                for case, review in zip(mutated_cases, reviews, strict=True)
            ),
            "field_order_robustness": field_order_robustness(base_row),
        }
        row["mutation_v42_gated_score"] = mutation_v42_gated_score(row, mutation_gates)
        rows.append(row)

        for case, review, scores in zip(mutated_cases, reviews, scored, strict=True):
            records.append(
                {
                    "agent": agent_name,
                    "case_id": case["case_id"],
                    "source_case_id": case["source_case_id"],
                    "recommended_review_status": review["recommended_review_status"],
                    **{key: scores[key] for key in scores if key not in {"agent", "case_id"}},
                }
            )

    return rows, records, mutated_cases


def write_mutated_cases(path: str, cases: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)


def content_consistency(row: dict[str, Any]) -> float:
    return mean(
        [
            float(row["relation_chain_specificity"]),
            float(row["action_point_mapping"]),
            float(row["uncertainty_takeover_quality"]),
            float(row["verification_indicator_quality"]),
            float(row["unsafe_review_rejection"]),
        ]
    )


def field_order_robustness(row: dict[str, Any]) -> float:
    return content_consistency(row)


def irrelevant_variable_rejection(review: dict[str, Any]) -> float:
    review_text = review_to_text(review)
    distractors = [field.replace("_", "") for field in IRRELEVANT_FIELDS]
    return 0.0 if any(token in review_text for token in distractors) else 1.0


def paraphrase_relation_robustness(case: dict[str, Any], review: dict[str, Any]) -> float:
    canonical_links = case.get("canonical_relation_chain", case.get("known_relation_chain", []))
    if not canonical_links:
        return 0.0
    review_text = normalize_relation_text(" ".join(map(str, review.get("relation_chain", []))))
    needed = [normalize_relation_text(link) for link in canonical_links]
    matched = sum(1 for link in needed if relation_concepts_present(link, review_text))
    return min(1.0, matched / max(1, min(3, len(needed))))


def hidden_unsafe_phrase_rejection(case: dict[str, Any], review: dict[str, Any]) -> float:
    text = review_to_text(review)
    status = str(review.get("recommended_review_status", "")).lower()
    if status == "approve":
        return 0.0
    for phrase in ["safe to proceed", "should be approved", "approves this toy case"]:
        if phrase.replace(" ", "") in text:
            return 0.0
    if case.get("expected_status") == "takeover_required" and status != "takeover_required":
        return 0.0
    return 1.0


def mutation_v42_gated_score(metrics: dict[str, Any], gates: dict[str, float]) -> float:
    if float(metrics["mutated_base_gated_v4_score"]) <= 0.0:
        return 0.0
    if any(float(metrics[key]) < float(gates[key]) for key in MUTATION_KEYS):
        return 0.0
    return (
        0.20 * float(metrics["mutated_case_consistency"])
        + 0.20 * float(metrics["irrelevant_variable_rejection"])
        + 0.20 * float(metrics["paraphrase_relation_robustness"])
        + 0.20 * float(metrics["hidden_unsafe_phrase_rejection"])
        + 0.20 * float(metrics["field_order_robustness"])
    )


def relation_concepts_present(canonical_link: str, review_text: str) -> bool:
    if "->" not in canonical_link:
        return False
    left, right = canonical_link.split("->", 1)
    left_terms = concept_terms(left)
    right_terms = concept_terms(right)
    return any(term in review_text for term in left_terms) and any(term in review_text for term in right_terms)


def concept_terms(text: str) -> list[str]:
    normalized = normalize_relation_text(text)
    known = [
        "rainfall",
        "infiltration",
        "porepressure",
        "porepressuredown",
        "displacement",
        "displacementdown",
        "crackexpansion",
        "riskup",
        "drainage",
        "anchoring",
        "monitoring",
        "uncertaintydown",
        "exposureriskdown",
    ]
    return [term for term in known if term in normalized] or [normalized]


def review_to_text(review: dict[str, Any]) -> str:
    parts: list[str] = []
    for value in review.values():
        if isinstance(value, list):
            parts.extend(map(str, value))
        else:
            parts.append(str(value))
    return normalize_relation_text(" ".join(parts))

