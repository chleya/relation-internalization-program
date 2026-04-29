from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier

from .data import RELATIONS, RESOURCES, TEXTURES, WETS, encode_context
from .model import hidden_states


@dataclass
class ExtractedRule:
    condition: dict[str, str]
    outcome: str
    confidence: float
    probe_relation: str


class ExtractedRelationTable:
    def __init__(self, rules: list[ExtractedRule]):
        self.rules = rules

    def infer_resource(self, context: dict[str, str]) -> str:
        for rule in self.rules:
            if all(context[key] == value for key, value in rule.condition.items()):
                return rule.outcome
        return "neutral"

    def act(self, context: dict[str, str]) -> str:
        return "eat" if self.infer_resource(context) == "food" else "avoid"

    def edit_rule(self, condition: dict[str, str], new_outcome: str) -> bool:
        for rule in self.rules:
            if rule.condition == condition:
                rule.outcome = new_outcome
                return True
        self.rules.append(
            ExtractedRule(
                condition=dict(condition),
                outcome=new_outcome,
                confidence=1.0,
                probe_relation=f"{condition.get('texture')}_{condition.get('wet')}",
            )
        )
        return True

    def describe_relations(self) -> list[dict[str, object]]:
        return [
            {
                "condition": rule.condition,
                "outcome": rule.outcome,
                "confidence": rule.confidence,
                "probe_relation": rule.probe_relation,
            }
            for rule in self.rules
        ]


def canonical_context(texture: str, wet: str) -> dict[str, str]:
    return {"texture": texture, "wet": wet, "color": "red", "odor": "strong"}


def extract_relation_table(
    model: MLPClassifier,
    relation_probe,
) -> ExtractedRelationTable:
    rules: list[ExtractedRule] = []
    for texture in TEXTURES:
        for wet in WETS:
            context = canonical_context(texture, wet)
            x = encode_context(context).reshape(1, -1)
            hidden = hidden_states(model, x)
            relation_probs = relation_probe.predict_proba(hidden)[0]
            relation_index = int(np.argmax(relation_probs))
            resource_index = int(model.predict(x)[0])
            rules.append(
                ExtractedRule(
                    condition={"texture": texture, "wet": wet},
                    outcome=RESOURCES[resource_index],
                    confidence=float(relation_probs[relation_index]),
                    probe_relation=RELATIONS[relation_index],
                )
            )
    return ExtractedRelationTable(rules)


def table_resource_accuracy(table: ExtractedRelationTable, contexts: list[dict[str, str]], labels: np.ndarray) -> float:
    predictions = [RESOURCES.index(table.infer_resource(context)) for context in contexts]
    return float(np.mean(np.asarray(predictions) == labels))


def relation_alignment(table: ExtractedRelationTable) -> float:
    aligned = 0
    rules = table.describe_relations()
    for rule in rules:
        condition = rule["condition"]
        expected = f"{condition['texture']}_{condition['wet']}"
        aligned += int(rule["probe_relation"] == expected)
    return aligned / len(rules) if rules else 0.0
