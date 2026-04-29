from __future__ import annotations

from collections import Counter, defaultdict

from .env import ACTIONS, evaluate_chain


class BaseAgent:
    name = "base"

    def act(self, context: dict[str, str]) -> str:
        raise NotImplementedError

    def observe(self, context: dict[str, str], action: str, optimal_action: str, reward: float) -> None:
        pass

    def describe_relations(self) -> list[dict[str, object]]:
        return []

    def edit_link(self, link: str, enabled: bool) -> bool:
        return False

    def review_plan(self, context: dict[str, str]) -> dict[str, object]:
        return {}


class MajorityActionBaseline(BaseAgent):
    name = "majority"

    def __init__(self):
        self.counts: Counter[str] = Counter()

    def act(self, context: dict[str, str]) -> str:
        return self.counts.most_common(1)[0][0] if self.counts else "monitor"

    def observe(self, context: dict[str, str], action: str, optimal_action: str, reward: float) -> None:
        self.counts[optimal_action] += 1


class SurfaceCueBaseline(BaseAgent):
    name = "surface"

    def __init__(self):
        self.table: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.global_counts: Counter[str] = Counter()

    def act(self, context: dict[str, str]) -> str:
        key = (context["weather_label"], context["contractor_report"])
        if self.table[key]:
            return self.table[key].most_common(1)[0][0]
        return self.global_counts.most_common(1)[0][0] if self.global_counts else "monitor"

    def observe(self, context: dict[str, str], action: str, optimal_action: str, reward: float) -> None:
        key = (context["weather_label"], context["contractor_report"])
        self.table[key][optimal_action] += 1
        self.global_counts[optimal_action] += 1


class StructuralMemoryBaseline(BaseAgent):
    name = "structural_memory"

    def __init__(self):
        self.table: dict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
        self.global_counts: Counter[str] = Counter()

    def _key(self, context: dict[str, str]) -> tuple[str, str, str, str, str]:
        return (
            context["rainfall"],
            context["drainage"],
            context["anchoring"],
            context["toe_excavation"],
            context["monitoring"],
        )

    def act(self, context: dict[str, str]) -> str:
        key = self._key(context)
        if self.table[key]:
            return self.table[key].most_common(1)[0][0]
        return self.global_counts.most_common(1)[0][0] if self.global_counts else "monitor"

    def observe(self, context: dict[str, str], action: str, optimal_action: str, reward: float) -> None:
        self.table[self._key(context)][optimal_action] += 1
        self.global_counts[optimal_action] += 1


class LearnedLinkBaseline(BaseAgent):
    name = "learned_links"

    def __init__(self, min_support: int = 10):
        self.min_support = min_support
        self.drainage_good_high_rain: Counter[str] = Counter()
        self.anchored_excavation: Counter[str] = Counter()
        self.links = {
            "Drainage -> PorePressureDown": False,
            "Anchoring -> DisplacementDown": False,
            "WeatherLabel -> RiskUp": False,
            "ContractorReport -> RiskUp": False,
            "Drainage -> CrackDown": False,
        }

    def _refresh_links(self) -> None:
        drain_support = sum(self.drainage_good_high_rain.values())
        if drain_support >= self.min_support:
            self.links["Drainage -> PorePressureDown"] = self.drainage_good_high_rain["monitor"] >= self.drainage_good_high_rain["drain"]

        anchor_support = sum(self.anchored_excavation.values())
        if anchor_support >= self.min_support:
            self.links["Anchoring -> DisplacementDown"] = self.anchored_excavation["monitor"] >= self.anchored_excavation["anchor"]

    def act(self, context: dict[str, str]) -> str:
        state = evaluate_chain(
            context,
            drainage_effective=self.links["Drainage -> PorePressureDown"],
            anchoring_effective=self.links["Anchoring -> DisplacementDown"],
        )
        return state.optimal_action

    def observe(self, context: dict[str, str], action: str, optimal_action: str, reward: float) -> None:
        if context["rainfall"] == "high" and context["drainage"] == "good" and context["toe_excavation"] == "no":
            self.drainage_good_high_rain[optimal_action] += 1
        if context["anchoring"] == "present" and context["toe_excavation"] == "yes" and context["rainfall"] == "low":
            self.anchored_excavation[optimal_action] += 1
        self._refresh_links()

    def edit_link(self, link: str, enabled: bool) -> bool:
        editable = {"Drainage -> PorePressureDown", "Anchoring -> DisplacementDown"}
        if link not in editable:
            return False
        self.links[link] = enabled
        return True

    def describe_relations(self) -> list[dict[str, object]]:
        return [
            {"link": "Rainfall -> Infiltration", "enabled": True},
            {"link": "Infiltration -> PorePressure", "enabled": True},
            {"link": "Drainage -> PorePressureDown", "enabled": self.links["Drainage -> PorePressureDown"]},
            {"link": "PorePressure -> Displacement", "enabled": True},
            {"link": "Anchoring -> DisplacementDown", "enabled": self.links["Anchoring -> DisplacementDown"]},
            {"link": "ToeExcavation -> StabilityDown", "enabled": True},
            {"link": "Displacement -> CrackExpansion", "enabled": True},
            {"link": "CrackExpansion -> RiskUp", "enabled": True},
            {"link": "WeatherLabel -> RiskUp", "enabled": self.links["WeatherLabel -> RiskUp"]},
            {"link": "ContractorReport -> RiskUp", "enabled": self.links["ContractorReport -> RiskUp"]},
            {"link": "Drainage -> CrackDown", "enabled": self.links["Drainage -> CrackDown"]},
        ]

    def review_plan(self, context: dict[str, str]) -> dict[str, object]:
        return SlopeRelationChainModelReviewMixin.review_plan_from_links(context, self.links)


class GenericReviewBaseline(BaseAgent):
    name = "generic_review"

    def act(self, context: dict[str, str]) -> str:
        return "drain"

    def review_plan(self, context: dict[str, str]) -> dict[str, object]:
        return {
            "variables": ["rainfall", "slope risk"],
            "relation_chain": ["rainfall affects slope safety", "support improves stability"],
            "action": "drain",
            "action_point": "strengthen drainage and support",
            "verification_indicators": ["site condition"],
            "failure_conditions": ["risk remains high"],
            "human_takeover_conditions": [],
            "responsibility_chain": ["project team"],
        }


class SlopeRelationChainModel(BaseAgent):
    name = "relation_chain"

    def __init__(self):
        self.links = {
            "Drainage -> PorePressureDown": True,
            "Anchoring -> DisplacementDown": True,
        }

    def act(self, context: dict[str, str]) -> str:
        state = evaluate_chain(
            context,
            drainage_effective=self.links["Drainage -> PorePressureDown"],
            anchoring_effective=self.links["Anchoring -> DisplacementDown"],
        )
        return state.optimal_action

    def edit_link(self, link: str, enabled: bool) -> bool:
        if link not in self.links:
            return False
        self.links[link] = enabled
        return True

    def describe_relations(self) -> list[dict[str, object]]:
        return [
            {"link": "Rainfall -> Infiltration", "enabled": True},
            {"link": "Infiltration -> PorePressure", "enabled": True},
            {"link": "Drainage -> PorePressureDown", "enabled": self.links["Drainage -> PorePressureDown"]},
            {"link": "PorePressure -> Displacement", "enabled": True},
            {"link": "Anchoring -> DisplacementDown", "enabled": self.links["Anchoring -> DisplacementDown"]},
            {"link": "ToeExcavation -> StabilityDown", "enabled": True},
            {"link": "Displacement -> CrackExpansion", "enabled": True},
            {"link": "CrackExpansion -> RiskUp", "enabled": True},
            {"link": "StopWork -> ExposureRiskDown", "enabled": True},
            {"link": "Monitoring -> UncertaintyDown", "enabled": True},
        ]

    def review_plan(self, context: dict[str, str]) -> dict[str, object]:
        return SlopeRelationChainModelReviewMixin.review_plan_from_links(context, self.links)


class SlopeRelationChainModelReviewMixin:
    @staticmethod
    def review_plan_from_links(context: dict[str, str], links: dict[str, bool]) -> dict[str, object]:
        state = evaluate_chain(
            context,
            drainage_effective=links["Drainage -> PorePressureDown"],
            anchoring_effective=links["Anchoring -> DisplacementDown"],
        )
        action_points = {
            "drain": "Drainage -> PorePressureDown",
            "anchor": "Anchoring -> DisplacementDown",
            "stop_work": "StopWork -> ExposureRiskDown",
            "monitor": "Monitoring -> UncertaintyDown",
        }
        indicators = {
            "drain": ["pore_pressure", "infiltration"],
            "anchor": ["displacement", "crack"],
            "stop_work": ["exposure", "displacement", "crack"],
            "monitor": ["displacement", "crack", "pore_pressure"],
        }
        action = state.optimal_action
        return {
            "variables": [
                "rainfall",
                "infiltration",
                "pore_pressure",
                "displacement",
                "crack",
                "risk",
            ],
            "relation_chain": [
                "Rainfall -> Infiltration",
                "Infiltration -> PorePressure",
                "PorePressure -> Displacement",
                "Displacement -> CrackExpansion",
                "CrackExpansion -> RiskUp",
            ],
            "action": action,
            "action_point": action_points[action],
            "verification_indicators": indicators[action],
            "failure_conditions": [
                "pore_pressure does not decrease",
                "displacement accelerates",
                "crack expands",
            ],
            "human_takeover_conditions": [
                "rapid displacement increase",
                "new through-crack",
                "monitoring uncertainty remains high",
            ],
            "responsibility_chain": [
                "monitoring engineer",
                "geotechnical reviewer",
                "site commander",
            ],
            "risk": state.risk,
        }


def make_agent(name: str) -> BaseAgent:
    if name == "majority":
        return MajorityActionBaseline()
    if name == "surface":
        return SurfaceCueBaseline()
    if name == "structural_memory":
        return StructuralMemoryBaseline()
    if name == "learned_links":
        return LearnedLinkBaseline()
    if name == "generic_review":
        return GenericReviewBaseline()
    if name == "relation_chain":
        return SlopeRelationChainModel()
    raise ValueError(f"Unknown agent: {name}")
