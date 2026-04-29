from dataclasses import dataclass, field
import random

from .agents_base import BaseAgent
from .features import FEATURE_ORDER, RESOURCES, condition_key, enumerate_candidate_conditions, matches_condition


@dataclass
class Rule:
    condition: dict
    counts: dict = field(default_factory=lambda: {resource: 0 for resource in RESOURCES})
    support: int = 0
    confidence: float = 0.0
    edited: bool = False

    def update(self, resource: str) -> None:
        if self.edited:
            return
        if self.support > 0 and self.winner() != resource and self.confidence >= 0.7:
            self.counts = {key: int(value * 0.1) for key, value in self.counts.items()}
            self.support = sum(self.counts.values())
        self.counts[resource] += 1
        self.support += 1
        self.confidence = max(self.counts.values()) / max(self.support, 1)

    def probability(self, resource: str) -> float:
        total = self.support + len(RESOURCES)
        return (self.counts.get(resource, 0) + 1) / total

    def winner(self) -> str:
        return max(self.counts, key=self.counts.get)


class RelationInternalizationModel(BaseAgent):
    name = "relation"

    def __init__(
        self,
        max_order: int = 2,
        min_support: int = 3,
        specificity_weight: float = 1.0,
        confidence_weight: float = 1.0,
        seed: int = 0,
        candidate_keys: list[str] | None = None,
        name: str | None = None,
    ):
        self.max_order = max_order
        self.min_support = min_support
        self.specificity_weight = specificity_weight
        self.confidence_weight = confidence_weight
        self.seed = seed
        self.candidate_keys = candidate_keys or ["texture", "wet"]
        if name is not None:
            self.name = name
        self.rules = {
            condition_key(condition): Rule(condition=condition)
            for condition in enumerate_candidate_conditions(max_order=max_order, keys=self.candidate_keys)
        }

    def act(self, context: dict) -> str:
        return "eat" if self.infer_resource(context) == "food" else "avoid"

    def observe(self, context: dict, action: str, resource: str, reward: float) -> None:
        self.update_rules(context, resource)

    def infer_resource(self, context: dict) -> str:
        edited_matches = [rule for rule in self.matching_rules(context) if rule.edited]
        if edited_matches:
            return max(edited_matches, key=lambda rule: (len(rule.condition), rule.confidence)).winner()
        scores = {resource: 0.0 for resource in RESOURCES}
        usable = [rule for rule in self.matching_rules(context) if rule.support >= self.min_support]
        if not usable:
            usable = self.matching_rules(context)
        for rule in usable:
            specificity = self.rule_specificity(rule) ** self.specificity_weight
            confidence = max(rule.confidence, 1.0 / len(RESOURCES)) ** self.confidence_weight
            for resource in RESOURCES:
                scores[resource] += confidence * specificity * rule.probability(resource)
        return max(scores, key=scores.get)

    def rule_specificity(self, rule: Rule) -> float:
        return float(len(rule.condition))

    def matching_rules(self, context: dict) -> list[Rule]:
        return [rule for rule in self.rules.values() if matches_condition(context, rule.condition)]

    def update_rules(self, context: dict, resource: str) -> None:
        for rule in self.matching_rules(context):
            rule.update(resource)

    def edit_rule(self, condition: dict, new_outcome: str) -> bool:
        key = condition_key(condition)
        rule = self.rules.get(key)
        if rule is None or new_outcome not in RESOURCES:
            return False
        rule.counts = {resource: 0 for resource in RESOURCES}
        rule.counts[new_outcome] = max(rule.support, self.min_support, 20)
        rule.support = rule.counts[new_outcome]
        rule.confidence = 1.0
        rule.edited = True
        return True


    def describe_relations(self) -> list[dict]:
        rows = []
        for rule in self.rules.values():
            if rule.support >= self.min_support or rule.edited:
                rows.append(
                    {
                        "condition": dict(rule.condition),
                        "outcome": rule.winner(),
                        "counts": dict(rule.counts),
                        "support": rule.support,
                        "confidence": rule.confidence,
                        "edited": rule.edited,
                    }
                )
        return sorted(rows, key=lambda r: (-len(r["condition"]), -r["confidence"], str(r["condition"])))

    def shuffle_rule_outcomes(self, seed: int = 0) -> bool:
        supported = [
            rule
            for rule in self.rules.values()
            if rule.support >= self.min_support and len(rule.condition) == self.max_order and not rule.edited
        ]
        if len(supported) < 2:
            return False
        winners = [rule.winner() for rule in supported]
        rng = random.Random(seed)
        shuffled = winners[:]
        for _ in range(10):
            rng.shuffle(shuffled)
            if shuffled != winners:
                break
        for rule, outcome in zip(supported, shuffled):
            rule.counts = {resource: 0 for resource in RESOURCES}
            rule.counts[outcome] = max(rule.support, self.min_support, 20)
            rule.support = rule.counts[outcome]
            rule.confidence = 1.0
        return True


class WideRelationInternalizationModel(RelationInternalizationModel):
    name = "wide_relation"

    def __init__(self, **kwargs):
        kwargs["candidate_keys"] = FEATURE_ORDER
        kwargs["name"] = self.name
        super().__init__(**kwargs)


class RobustWideRelationInternalizationModel(WideRelationInternalizationModel):
    name = "robust_wide_relation"

    def __init__(self, nuisance_keys: list[str] | None = None, nuisance_penalty: float = 0.15, **kwargs):
        self.nuisance_keys = set(nuisance_keys or ["color", "odor"])
        self.nuisance_penalty = nuisance_penalty
        kwargs["name"] = self.name
        super().__init__(**kwargs)

    def rule_specificity(self, rule: Rule) -> float:
        specificity = float(len(rule.condition))
        nuisance_count = sum(1 for key in rule.condition if key in self.nuisance_keys)
        if nuisance_count:
            specificity *= self.nuisance_penalty**nuisance_count
        return specificity
