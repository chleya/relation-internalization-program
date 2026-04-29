from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import COLORS, ODORS, RESOURCES, TEXTURES, WETS, canonical_context, make_record, resource_rule
from .features import dataset_tensors, encode_context, label_name
from .models import EditPressureModel, ExplicitTableOracle


@dataclass
class ExtractedRule:
    condition: dict[str, str]
    outcome: str
    confidence: float


class ExtractedRelationTable:
    def __init__(self, rules: list[ExtractedRule]) -> None:
        self.rules = {(rule.condition["texture"], rule.condition["wet"]): rule for rule in rules}

    def infer_resource(self, context: dict[str, str]) -> str:
        rule = self.rules[(context["texture"], context["wet"])]
        return rule.outcome

    def act(self, context: dict[str, str]) -> str:
        return "eat" if self.infer_resource(context) == "food" else "avoid"

    def edit_rule(self, condition: dict[str, str], new_outcome: str) -> bool:
        key = (condition["texture"], condition["wet"])
        if key not in self.rules:
            return False
        old = self.rules[key]
        self.rules[key] = ExtractedRule(condition=old.condition, outcome=new_outcome, confidence=old.confidence)
        return True

    def to_rows(self) -> list[dict[str, object]]:
        return [
            {"texture": key[0], "wet": key[1], "outcome": rule.outcome, "confidence": rule.confidence}
            for key, rule in sorted(self.rules.items())
        ]


def predict_resource_from_model(model, context: dict[str, str], support=None, edit=None) -> tuple[str, float]:
    if isinstance(model, ExplicitTableOracle):
        return model.predict_resource(context), 1.0
    with torch.no_grad():
        x = encode_context(context).view(1, -1)
        if isinstance(model, EditPressureModel) and support is None:
            support = canonical_support()
        if support is not None or edit is not None:
            logits = model(x, support=support, edit=edit)
        else:
            logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        return label_name(idx), float(probs[idx].item())


def extract_table_from_model(model, nuisance_average: bool = True) -> ExtractedRelationTable:
    rules = []
    for texture in TEXTURES:
        for wet in WETS:
            votes = {resource: 0.0 for resource in RESOURCES}
            contexts = []
            if nuisance_average:
                for color in COLORS:
                    for odor in ODORS:
                        contexts.append(canonical_context(texture, wet, color, odor))
            else:
                contexts.append(canonical_context(texture, wet))
            for context in contexts:
                outcome, confidence = predict_resource_from_model(model, context)
                votes[outcome] += confidence
            outcome = max(votes, key=votes.get)
            confidence = votes[outcome] / max(1, len(contexts))
            rules.append(ExtractedRule(condition={"texture": texture, "wet": wet}, outcome=outcome, confidence=confidence))
    return ExtractedRelationTable(rules)


def canonical_support(regime: str = "base"):
    records = []
    for texture in TEXTURES:
        for wet in WETS:
            records.append(make_record(canonical_context(texture, wet), regime))
    return dataset_tensors(records)


def table_alignment(table: ExtractedRelationTable, regime: str = "base") -> float:
    hits = []
    for texture in TEXTURES:
        for wet in WETS:
            context = canonical_context(texture, wet)
            hits.append(1.0 if table.infer_resource(context) == resource_rule(context, regime) else 0.0)
    return sum(hits) / len(hits)
