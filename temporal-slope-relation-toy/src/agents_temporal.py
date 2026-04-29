from __future__ import annotations

from collections import Counter, defaultdict

from .env_temporal import TemporalStep, default_delay_map, evaluate_temporal_chain


class BaseTemporalAgent:
    name = "base"

    def reset(self) -> None:
        pass

    def observe_sequence(self, seq: list[TemporalStep]) -> None:
        pass

    def act(self, history: list[TemporalStep], t: int) -> str:
        raise NotImplementedError

    def edit_delay(self, link: str, new_delay: int) -> bool:
        return False

    def temporal_audit(self) -> list[str]:
        return []

    def audit_temporal_relations(self) -> list[str]:
        return self.temporal_audit()


class SurfaceTemporalAgent(BaseTemporalAgent):
    name = "surface_temporal"

    def __init__(self):
        self.table: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.global_counts: Counter[str] = Counter()

    def observe_sequence(self, seq: list[TemporalStep]) -> None:
        for step in seq:
            key = (step.surface_warning, step.weather_label, step.contractor_report)
            self.table[key][step.optimal_action] += 1
            self.global_counts[step.optimal_action] += 1

    def act(self, history: list[TemporalStep], t: int) -> str:
        step = history[t]
        key = (step.surface_warning, step.weather_label, step.contractor_report)
        if self.table[key]:
            return self.table[key].most_common(1)[0][0]
        return self.global_counts.most_common(1)[0][0] if self.global_counts else "monitor"


class StructuralMemoryTemporalAgent(BaseTemporalAgent):
    name = "structural_memory_temporal"

    def __init__(self):
        self.table: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.global_counts: Counter[str] = Counter()

    def key(self, step: TemporalStep) -> tuple[str, ...]:
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

    def observe_sequence(self, seq: list[TemporalStep]) -> None:
        for step in seq:
            self.table[self.key(step)][step.optimal_action] += 1
            self.global_counts[step.optimal_action] += 1

    def act(self, history: list[TemporalStep], t: int) -> str:
        key = self.key(history[t])
        if self.table[key]:
            return self.table[key].most_common(1)[0][0]
        return self.global_counts.most_common(1)[0][0] if self.global_counts else "monitor"


class InstantRelationChainAgent(BaseTemporalAgent):
    name = "instant_relation_chain"

    def act(self, history: list[TemporalStep], t: int) -> str:
        step = history[t]
        if step.rainfall == "high" and step.drainage == "poor":
            return "drain"
        if step.toe_excavation == "yes" and step.anchoring == "none":
            return "anchor"
        if step.monitoring == "sparse" and step.toe_excavation == "yes":
            return "stop_work"
        return "monitor"

    def temporal_audit(self) -> list[str]:
        return ["Rainfall[t] -> PorePressure[t]", "PorePressure[t] -> Displacement[t]"]


class DelayedRelationChainAgent(BaseTemporalAgent):
    name = "delayed_relation_chain"

    def __init__(self, rainfall_delay: int = 1, delay_map: dict[str, int] | None = None):
        self.delay_map = delay_map.copy() if delay_map is not None else default_delay_map(rainfall_delay)

    def act(self, history: list[TemporalStep], t: int) -> str:
        simulated = [TemporalStep(**step.as_context()) for step in history]
        evaluate_temporal_chain(simulated, delay_map=self.delay_map)
        return simulated[t].optimal_action

    def edit_delay(self, link: str, new_delay: int) -> bool:
        if link not in self.delay_map:
            return False
        self.delay_map[link] = int(new_delay)
        return True

    def temporal_audit(self) -> list[str]:
        d1 = self.delay_map["Rainfall -> PorePressure"]
        d2 = self.delay_map["PorePressure -> Displacement"]
        d3 = self.delay_map["Displacement -> Crack"]
        d_drain = self.delay_map["Drainage -> PorePressureDown"]
        d_anchor = self.delay_map["Anchoring -> DisplacementDown"]
        return [
            f"Rainfall[t-{d1}] -> PorePressure[t]",
            f"Drainage[t-{d_drain}] -> PorePressureDown[t]",
            f"PorePressure[t-{d2}] -> Displacement[t]",
            f"Anchoring[t-{d_anchor}] -> DisplacementDown[t]",
            f"Displacement[t-{d3}] + Monitoring[t-{d3}] -> Crack[t]",
        ]


class LearnedDelayedLinksAgent(DelayedRelationChainAgent):
    name = "learned_delayed_links"

    def __init__(self):
        super().__init__(delay_map=default_delay_map(1))
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

    def observe_sequence(self, seq: list[TemporalStep]) -> None:
        for link in self.candidate_links:
            for delay in self.candidate_delays:
                self.delay_votes[link][delay] += self._score_link_delay(seq, link, delay)
        self._refresh_delay_map()

    def _score_link_delay(self, seq: list[TemporalStep], link: str, delay: int) -> int:
        score = 0
        for t in range(delay, len(seq)):
            prev = seq[t - delay]
            cur = seq[t]
            if link == "Rainfall -> PorePressure":
                if t - self.delay_map["Drainage -> PorePressureDown"] >= 0:
                    drain_override = seq[t - self.delay_map["Drainage -> PorePressureDown"]].drainage == "good"
                else:
                    drain_override = False
                predicted = prev.rainfall == "high" and not drain_override
                actual = cur.pore_pressure == "high"
            elif link == "PorePressure -> Displacement":
                if t - self.delay_map["Anchoring -> DisplacementDown"] >= 0:
                    anchor_override = seq[t - self.delay_map["Anchoring -> DisplacementDown"]].anchoring == "present"
                else:
                    anchor_override = False
                predicted = (prev.pore_pressure == "high" or cur.toe_excavation == "yes") and not anchor_override
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


def make_agent(name: str) -> BaseTemporalAgent:
    if name == "surface_temporal":
        return SurfaceTemporalAgent()
    if name == "structural_memory_temporal":
        return StructuralMemoryTemporalAgent()
    if name == "instant_relation_chain":
        return InstantRelationChainAgent()
    if name == "delayed_relation_chain":
        return DelayedRelationChainAgent()
    if name == "learned_delayed_links":
        return LearnedDelayedLinksAgent()
    raise ValueError(f"Unknown agent: {name}")
