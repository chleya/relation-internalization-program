from __future__ import annotations

from collections import Counter, defaultdict

from .env_temporal import TemporalStep, default_delay_map, evaluate_temporal_chain
from .env_uncertainty import UNKNOWN, ObservedTemporalStep, automatic_action_from_observation


class BaseUncertaintyAgent:
    name = "base_uncertainty"

    def reset(self) -> None:
        pass

    def observe_sequence(self, seq: list[ObservedTemporalStep]) -> None:
        pass

    def act(self, seq: list[ObservedTemporalStep], t: int) -> str:
        raise NotImplementedError

    def edit_delay(self, link: str, new_delay: int) -> bool:
        return False

    def audit_uncertainty(self, seq: list[ObservedTemporalStep], t: int) -> list[str]:
        return []


class SurfaceTemporalV3Agent(BaseUncertaintyAgent):
    name = "surface_temporal"

    def __init__(self):
        self.table: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
        self.global_counts: Counter[str] = Counter()

    def observe_sequence(self, seq: list[ObservedTemporalStep]) -> None:
        for step in seq:
            key = (step.surface_warning, step.weather_label, step.contractor_report)
            self.table[key][step.optimal_action] += 1
            self.global_counts[step.optimal_action] += 1

    def act(self, seq: list[ObservedTemporalStep], t: int) -> str:
        step = seq[t]
        key = (step.surface_warning, step.weather_label, step.contractor_report)
        if self.table[key]:
            return self.table[key].most_common(1)[0][0]
        return self.global_counts.most_common(1)[0][0] if self.global_counts else "monitor"


class StructuralMemoryTemporalV3Agent(BaseUncertaintyAgent):
    name = "structural_memory_temporal"

    def __init__(self):
        self.table: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.global_counts: Counter[str] = Counter()

    def key(self, step: ObservedTemporalStep) -> tuple[str, ...]:
        return (
            step.rainfall,
            step.drainage,
            step.anchoring,
            step.toe_excavation,
            step.monitoring,
            step.pore_pressure,
            step.displacement,
            step.crack,
        )

    def observe_sequence(self, seq: list[ObservedTemporalStep]) -> None:
        for step in seq:
            self.table[self.key(step)][step.optimal_action] += 1
            self.global_counts[step.optimal_action] += 1

    def act(self, seq: list[ObservedTemporalStep], t: int) -> str:
        key = self.key(seq[t])
        if self.table[key]:
            return self.table[key].most_common(1)[0][0]
        return self.global_counts.most_common(1)[0][0] if self.global_counts else "monitor"


class InstantRelationChainV3Agent(BaseUncertaintyAgent):
    name = "instant_relation_chain"

    def act(self, seq: list[ObservedTemporalStep], t: int) -> str:
        step = seq[t]
        if step.rainfall == "high" and step.drainage == "poor":
            return "drain"
        if step.toe_excavation == "yes" and step.anchoring == "none":
            return "anchor"
        if step.monitoring == "sparse" and step.displacement == "high":
            return "stop_work"
        return "monitor"


class DelayedRelationChainV3Agent(BaseUncertaintyAgent):
    name = "delayed_relation_chain"

    def __init__(self, delay_map: dict[str, int] | None = None):
        self.delay_map = delay_map.copy() if delay_map is not None else default_delay_map(1)

    def edit_delay(self, link: str, new_delay: int) -> bool:
        if link not in self.delay_map:
            return False
        self.delay_map[link] = int(new_delay)
        return True

    def _critical_uncertainty(self, seq: list[ObservedTemporalStep], t: int) -> bool:
        step = seq[t]
        if set(step.conflicting_fields) & {"pore_pressure", "displacement", "crack"}:
            return True
        if set(step.delayed_fields) & {"pore_pressure", "displacement", "crack"}:
            return True
        if "pore_pressure" in step.missing_fields:
            if t > 0 and "pore_pressure" in seq[t - 1].missing_fields:
                return True
            upstream_rain = t - self.delay_map["Rainfall -> PorePressure"]
            if (upstream_rain >= 0 and seq[upstream_rain].rainfall in {"high", UNKNOWN}) or step.displacement in {"high", UNKNOWN}:
                return True
        if "displacement" in step.missing_fields:
            if t > 0 and "displacement" in seq[t - 1].missing_fields:
                return True
            if step.pore_pressure in {"high", UNKNOWN} or step.toe_excavation == "yes" or step.crack in {"yes", UNKNOWN}:
                return True
        if "crack" in step.missing_fields:
            upstream_disp = t - self.delay_map["Displacement -> Crack"]
            if step.displacement in {"high", UNKNOWN} or (upstream_disp >= 0 and seq[upstream_disp].displacement in {"high", UNKNOWN}):
                return True
        if "monitoring" in step.missing_fields:
            upstream_disp = t - self.delay_map["Displacement -> Crack"]
            if step.displacement in {"high", UNKNOWN} or (upstream_disp >= 0 and seq[upstream_disp].displacement in {"high", UNKNOWN}):
                return True
        return False

    def act(self, seq: list[ObservedTemporalStep], t: int) -> str:
        if self._critical_uncertainty(seq, t):
            return "takeover"
        simulated = [
            TemporalStep(
                t=step.t,
                rainfall="low" if step.rainfall == UNKNOWN else step.rainfall,
                drainage=step.drainage,
                anchoring=step.anchoring,
                toe_excavation=step.toe_excavation,
                monitoring="dense" if step.monitoring == UNKNOWN else step.monitoring,
                surface_warning=step.surface_warning,
                weather_label=step.weather_label,
                contractor_report=step.contractor_report,
            )
            for step in seq
        ]
        evaluate_temporal_chain(simulated, delay_map=self.delay_map)
        return simulated[t].optimal_action

    def audit_uncertainty(self, seq: list[ObservedTemporalStep], t: int) -> list[str]:
        step = seq[t]
        if not self._critical_uncertainty(seq, t) and not step.takeover_required:
            return []
        fields = sorted(set(step.missing_fields) | set(step.noisy_fields) | set(step.delayed_fields) | set(step.conflicting_fields))
        links = step.uncertain_links
        return [
            f"takeover required at t={t}",
            f"uncertain fields: {', '.join(fields) if fields else 'none'}",
            *[f"uncertain link: {link}" for link in links],
            f"downstream action affected: {step.true_optimal_action}",
        ]


class LearnedDelayedLinksV3Agent(DelayedRelationChainV3Agent):
    name = "learned_delayed_links"

    def __init__(self):
        super().__init__(default_delay_map(1))
        self.candidate_delays = [1, 2, 3]
        self.candidate_links = [
            "Rainfall -> PorePressure",
            "PorePressure -> Displacement",
            "Displacement -> Crack",
            "Drainage -> PorePressureDown",
            "Anchoring -> DisplacementDown",
        ]
        self.delay_votes: dict[str, Counter[int]] = {link: Counter() for link in self.candidate_links}
        self.min_support = 20

    def observe_sequence(self, seq: list[ObservedTemporalStep]) -> None:
        clean = [
            TemporalStep(
                t=step.t,
                rainfall="low" if step.rainfall == UNKNOWN else step.rainfall,
                drainage=step.drainage,
                anchoring=step.anchoring,
                toe_excavation=step.toe_excavation,
                monitoring="dense" if step.monitoring == UNKNOWN else step.monitoring,
                surface_warning=step.surface_warning,
                weather_label=step.weather_label,
                contractor_report=step.contractor_report,
                pore_pressure="low" if step.pore_pressure == UNKNOWN else step.pore_pressure,
                displacement="low" if step.displacement == UNKNOWN else step.displacement,
                crack="no" if step.crack == UNKNOWN else step.crack,
            )
            for step in seq
        ]
        for link in self.candidate_links:
            for delay in self.candidate_delays:
                self.delay_votes[link][delay] += self._score_link_delay(clean, link, delay)
        self._refresh_delay_map()

    def _score_link_delay(self, seq: list[TemporalStep], link: str, delay: int) -> int:
        score = 0
        for t in range(delay, len(seq)):
            prev = seq[t - delay]
            cur = seq[t]
            if link == "Rainfall -> PorePressure":
                predicted = prev.rainfall == "high"
                actual = cur.pore_pressure == "high"
            elif link == "PorePressure -> Displacement":
                predicted = prev.pore_pressure == "high" or cur.toe_excavation == "yes"
                actual = cur.displacement == "high"
            elif link == "Displacement -> Crack":
                predicted = prev.displacement == "high" and prev.monitoring == "sparse"
                actual = cur.crack == "yes"
            elif link == "Drainage -> PorePressureDown":
                predicted = prev.drainage == "good"
                actual = cur.pore_pressure == "low"
            elif link == "Anchoring -> DisplacementDown":
                predicted = prev.anchoring == "present"
                actual = cur.displacement == "low"
            else:
                return 0
            score += 1 if predicted == actual else -1
        return score

    def _refresh_delay_map(self) -> None:
        for link, counter in self.delay_votes.items():
            if sum(abs(value) for value in counter.values()) >= self.min_support:
                self.delay_map[link] = counter.most_common(1)[0][0]


class UncertaintyAwareDelayedLinksAgent(LearnedDelayedLinksV3Agent):
    name = "uncertainty_aware_delayed_links"


class OvercautiousTakeoverAgent(BaseUncertaintyAgent):
    name = "overcautious_takeover"

    def act(self, seq: list[ObservedTemporalStep], t: int) -> str:
        step = seq[t]
        uncertainty = set(step.missing_fields) | set(step.noisy_fields) | set(step.delayed_fields) | set(step.conflicting_fields)
        if uncertainty:
            return "takeover"
        return automatic_action_from_observation(step)

    def audit_uncertainty(self, seq: list[ObservedTemporalStep], t: int) -> list[str]:
        step = seq[t]
        uncertainty = set(step.missing_fields) | set(step.noisy_fields) | set(step.delayed_fields) | set(step.conflicting_fields)
        if not uncertainty:
            return []
        return ["takeover: generic uncertainty"]


def make_uncertainty_agent(name: str) -> BaseUncertaintyAgent:
    if name == "surface_temporal":
        return SurfaceTemporalV3Agent()
    if name == "structural_memory_temporal":
        return StructuralMemoryTemporalV3Agent()
    if name == "instant_relation_chain":
        return InstantRelationChainV3Agent()
    if name == "delayed_relation_chain":
        return DelayedRelationChainV3Agent()
    if name == "learned_delayed_links":
        return LearnedDelayedLinksV3Agent()
    if name == "uncertainty_aware_delayed_links":
        return UncertaintyAwareDelayedLinksAgent()
    if name == "overcautious_takeover":
        return OvercautiousTakeoverAgent()
    raise ValueError(f"Unknown V3 agent: {name}")
