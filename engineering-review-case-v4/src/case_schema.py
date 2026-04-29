from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_CASE_KEYS = {
    "case_id",
    "title",
    "site_context",
    "observed_conditions",
    "proposed_actions",
    "known_relation_chain",
    "known_uncertainties",
    "expected_status",
    "expected_review_points",
    "unsafe_review_patterns",
}

ALLOWED_STATUSES = {"approve_with_conditions", "revise", "takeover_required", "reject"}


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("case file must contain a list")
    for case in data:
        validate_case(case)
    return data


def validate_case(case: dict[str, Any]) -> None:
    missing = REQUIRED_CASE_KEYS - set(case)
    if missing:
        raise ValueError(f"case {case.get('case_id', '<unknown>')} missing keys: {sorted(missing)}")

    if case["expected_status"] not in ALLOWED_STATUSES:
        raise ValueError(f"invalid expected_status: {case['expected_status']}")

    if "approve" == case["expected_status"]:
        raise ValueError("plain approve status is not allowed in V4")

    if not isinstance(case["known_relation_chain"], list) or not case["known_relation_chain"]:
        raise ValueError("known_relation_chain must be a non-empty list")

    if not all("->" in str(link) for link in case["known_relation_chain"]):
        raise ValueError("known_relation_chain entries must contain relation arrows")


def observed_variables(case: dict[str, Any]) -> list[str]:
    return sorted(case.get("observed_conditions", {}).keys())

