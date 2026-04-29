from __future__ import annotations

REVIEW_ITEMS = [
    "variables",
    "relation_chain",
    "action_point",
    "verification_indicators",
    "failure_conditions",
    "human_takeover_conditions",
    "responsibility_chain",
]


def score_item(plan: dict[str, object], key: str) -> int:
    value = plan.get(key)
    if value is None or value == "" or value == []:
        return 0
    if key == "relation_chain":
        if not isinstance(value, list):
            return 0
        linked = [item for item in value if isinstance(item, str) and "->" in item]
        return 2 if len(linked) >= 3 else int(len(linked) > 0)
    if key == "action_point":
        if not isinstance(value, str):
            return 0
        return 2 if "->" in value else 1
    if isinstance(value, list):
        return 2 if len(value) >= 2 else 1
    return 2


def score_review_plan(plan: dict[str, object]) -> dict[str, object]:
    item_scores = {key: score_item(plan, key) for key in REVIEW_ITEMS}
    total = sum(item_scores.values())
    return {
        "item_scores": item_scores,
        "total": total,
        "max_total": 14,
        "passed": total >= 10,
    }
