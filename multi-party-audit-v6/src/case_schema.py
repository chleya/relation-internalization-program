from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_CASE_KEYS = {
    "case_id",
    "title",
    "reviews",
    "expected_material_disagreement",
    "expected_route",
    "minority_risk_review_id",
}


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("audit cases must be a list")
    for case in cases:
        validate_case(case)
    return cases


def validate_case(case: dict[str, Any]) -> None:
    missing = REQUIRED_CASE_KEYS - set(case)
    if missing:
        raise ValueError(f"case {case.get('case_id', '<unknown>')} missing {sorted(missing)}")
    if len(case["reviews"]) < 2:
        raise ValueError("V6 cases require at least two reviews")
    ids = {review.get("review_id") for review in case["reviews"]}
    if case["minority_risk_review_id"] not in ids:
        raise ValueError("minority_risk_review_id must reference an existing review")

