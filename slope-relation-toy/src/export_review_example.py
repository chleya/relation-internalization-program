from __future__ import annotations

import json
from pathlib import Path

from .agents import GenericReviewBaseline, SlopeRelationChainModel
from .review import score_review_plan


def main() -> None:
    context = {
        "rainfall": "high",
        "drainage": "poor",
        "anchoring": "none",
        "toe_excavation": "no",
        "monitoring": "sparse",
        "weather_label": "calm",
        "contractor_report": "normal",
    }
    relation_agent = SlopeRelationChainModel()
    generic_agent = GenericReviewBaseline()
    relation_plan = relation_agent.review_plan(context)
    generic_plan = generic_agent.review_plan(context)
    output = {
        "context": context,
        "relation_chain": {
            "plan": relation_plan,
            "review_score": score_review_plan(relation_plan),
        },
        "generic_review": {
            "plan": generic_plan,
            "review_score": score_review_plan(generic_plan),
        },
    }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/review_example.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value["review_score"] for key, value in output.items() if key != "context"}, indent=2))


if __name__ == "__main__":
    main()
