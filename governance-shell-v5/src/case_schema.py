from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {
    "case_id",
    "review_status",
    "relation_chain",
    "uncertain_links",
    "takeover_conditions",
    "verification_indicators",
    "responsibility_boundary",
    "claim_boundary",
    "expected_route",
    "expected_auto_approval",
}


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("governance cases must be a list")
    for case in cases:
        validate_case(case)
    return cases


def validate_case(case: dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - set(case)
    if missing:
        raise ValueError(f"case {case.get('case_id', '<unknown>')} missing {sorted(missing)}")
    if case["review_status"] == "approve":
        raise ValueError("plain approve is not allowed in V5")
    if case["expected_auto_approval"] is not False:
        raise ValueError("V5 toy governance does not allow automatic approval")

