from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .env import ACTIONS, PHYSICAL_ACTIONS, TRUE_LINKS, ProcessWorld, Transition, active_facts, outcome_facts


PROCESS_VARIABLES = {"rainfall", "load", "drainage", "support", "pore_pressure", "displacement", "risk"}
EFFECT_VARIABLES = {"load", "drainage", "support", "pore_pressure", "displacement", "risk"}


class BaseAgent:
    name = "base"

    def reset(self) -> None:
        pass

    def act(self, state: dict[str, str]) -> str:
        raise NotImplementedError

    def observe(self, transition: Transition) -> None:
        pass

    def describe_relations(self) -> list[dict[str, Any]]:
        return []

    def counterfactual(self, state: dict[str, str], action: str) -> dict[str, str]:
        return dict(state)

    def edit_relation(self, cause: str, effect: str) -> bool:
        return False


class RandomAgent(BaseAgent):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, state: dict[str, str]) -> str:
        return self.rng.choice(ACTIONS)


class ShortcutAgent(BaseAgent):
    name = "shortcut"

    def act(self, state: dict[str, str]) -> str:
        return "add_support" if state.get("warning") == "warning" else "noop"


class PassiveMemoryAgent(BaseAgent):
    name = "passive_memory"

    def __init__(self) -> None:
        self.stats: dict[tuple[tuple[str, str], ...], Counter[str]] = defaultdict(Counter)

    def act(self, state: dict[str, str]) -> str:
        key = self._key(state)
        if key not in self.stats:
            return "noop"
        return self.stats[key].most_common(1)[0][0]

    def observe(self, transition: Transition) -> None:
        key = self._key(transition.before)
        if transition.reward > 0:
            self.stats[key][transition.action] += 1

    def _key(self, state: dict[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(state.items()))


@dataclass
class LinkStats:
    positive: int = 0
    total: int = 0
    edited_effect: str | None = None

    @property
    def confidence(self) -> float:
        if self.total == 0:
            return 0.0
        return self.positive / self.total


class RelationAgent(BaseAgent):
    name = "relation_agent"

    def __init__(
        self,
        min_support: int = 4,
        min_confidence: float = 0.65,
        candidate_links: set[tuple[str, str]] | None = None,
        ignored_features: set[str] | None = None,
        explore: bool = True,
        decay: float = 1.0,
        action_costs: dict[str, float] | None = None,
    ) -> None:
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.candidate_links = set(candidate_links or TRUE_LINKS)
        self.ignored_features = set(ignored_features or set())
        self.explore = explore
        self.decay = decay
        self.action_costs = action_costs or {
            "improve_drainage": 0.1,
            "add_support": 0.1,
            "reduce_load": 0.1,
            "inspect": 0.05,
            "noop": 0.0,
        }
        self.links: dict[tuple[str, str], LinkStats] = defaultdict(LinkStats)
        self.steps = 0
        self.action_counts = Counter()
        self.explore_schedule = ["improve_drainage", "add_support", "reduce_load", "noop", "inspect"]

    def reset(self) -> None:
        self.links.clear()
        self.steps = 0
        self.action_counts.clear()

    def act(self, state: dict[str, str]) -> str:
        if self.explore and self.steps < len(self.explore_schedule) * 8:
            action = self.explore_schedule[self.steps % len(self.explore_schedule)]
            self.steps += 1
            self.action_counts[action] += 1
            return action

        scored = []
        for action in ["improve_drainage", "add_support", "reduce_load", "noop", "inspect"]:
            predicted = self.counterfactual(state, action)
            risk_score = 1.0 if predicted.get("risk") == "high" else 0.0
            cost = self.action_costs.get(action, 0.0)
            scored.append((risk_score + cost, action))
        action = min(scored)[1]
        self.action_counts[action] += 1
        return action

    def observe(self, transition: Transition) -> None:
        if self.decay < 1.0:
            for stats in self.links.values():
                stats.positive *= self.decay
                stats.total *= self.decay
        causes = active_facts(transition.before, transition.action, self.ignored_features) | active_facts(
            transition.after, transition.action, self.ignored_features
        )
        outcomes = outcome_facts(transition.after, self.ignored_features)
        for cause, effect in self.candidate_links:
            if cause in causes:
                stats = self.links[(cause, effect)]
                stats.total += 1
                if effect in outcomes:
                    stats.positive += 1

    def describe_relations(self) -> list[dict[str, Any]]:
        rows = []
        for (cause, effect), stats in sorted(self.links.items()):
            if stats.total >= self.min_support:
                rows.append(
                    {
                        "cause": cause,
                        "effect": stats.edited_effect or effect,
                        "raw_effect": effect,
                        "support": stats.total,
                        "confidence": round(stats.confidence, 3),
                        "edited": stats.edited_effect is not None,
                    }
                )
        return rows

    def learned_links(self) -> set[tuple[str, str]]:
        learned = set()
        for (cause, effect), stats in self.links.items():
            if stats.total >= self.min_support and stats.confidence >= self.min_confidence:
                learned.add((cause, stats.edited_effect or effect))
        return learned

    def counterfactual(self, state: dict[str, str], action: str) -> dict[str, str]:
        out = dict(state)
        facts = active_facts(out, action, self.ignored_features)

        for _ in range(3):
            changed = False
            for cause, effect in sorted(self.learned_links(), key=link_priority):
                if cause in facts:
                    key, value = effect.split("=", 1)
                    if out.get(key) != value:
                        out[key] = value
                        facts = active_facts(out, action, self.ignored_features)
                        changed = True
            if not changed:
                break

        if "displacement=normal" in facts:
            out["risk"] = "low"
        elif "displacement=high" in facts:
            out["risk"] = "high"
        return out

    def edit_relation(self, cause: str, effect: str) -> bool:
        edited = False
        for (link_cause, link_effect), stats in self.links.items():
            if link_cause == cause:
                stats.edited_effect = effect
                edited = True
        return edited

    def active_exploration_score(self) -> float:
        tried = sum(1 for action in PHYSICAL_ACTIONS if self.action_counts[action] > 0)
        return tried / len(PHYSICAL_ACTIONS)


class DiscoveryRelationAgent(BaseAgent):
    name = "discovery_relation_agent"

    def __init__(self, min_support: int = 6, min_confidence: float = 0.93, min_action_trials: int = 8) -> None:
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_action_trials = min_action_trials
        self.cause_counts = Counter()
        self.positive_counts = Counter()
        self.action_counts = Counter()

    def reset(self) -> None:
        self.cause_counts.clear()
        self.positive_counts.clear()
        self.action_counts.clear()

    def act(self, state: dict[str, str]) -> str:
        explore = self._adaptive_exploration_action()
        if explore is not None:
            self.action_counts[explore] += 1
            return explore

        scored = []
        for action in ACTIONS:
            predicted = self.counterfactual(state, action)
            risk_score = 1.0 if predicted.get("risk") == "high" else 0.0
            cost = 0.12 if action in PHYSICAL_ACTIONS else 0.05 if action == "inspect" else 0.0
            scored.append((risk_score + cost, action))
        action = min(scored)[1]
        self.action_counts[action] += 1
        return action

    def _adaptive_exploration_action(self) -> str | None:
        under_sampled = [action for action in ["improve_drainage", "add_support", "reduce_load", "noop"] if self.action_counts[action] < self.min_action_trials]
        if not under_sampled:
            return None
        return min(under_sampled, key=lambda action: (self.action_counts[action], action))

    def observe(self, transition: Transition) -> None:
        causes = active_facts(transition.before, transition.action) | active_facts(transition.after, transition.action)
        outcomes = outcome_facts(transition.after)
        changed_keys = {key for key, value in transition.after.items() if transition.before.get(key) != value}
        for cause in causes:
            self.cause_counts[cause] += 1
            for effect in outcomes:
                effect_key = effect.split("=", 1)[0]
                if cause.startswith("action=") and effect_key not in changed_keys:
                    continue
                if self._is_candidate_link(cause, effect):
                    self.positive_counts[(cause, effect)] += 1

    def _is_candidate_link(self, cause: str, effect: str) -> bool:
        effect_key = effect.split("=", 1)[0]
        if effect_key not in EFFECT_VARIABLES:
            return False
        cause_keys = fact_keys(cause)
        if effect_key in cause_keys:
            return False
        if "action" in cause_keys:
            action = cause.split("=", 1)[1]
            return action in PHYSICAL_ACTIONS
        if effect_key in {"load", "drainage", "support"}:
            return False
        if effect in {"pore_pressure=normal", "displacement=normal", "risk=low"}:
            values = fact_values(cause)
            if values and values <= {"poor", "absent"}:
                return False
        return bool(cause_keys) and cause_keys <= PROCESS_VARIABLES

    def learned_links(self) -> set[tuple[str, str]]:
        learned = set()
        for (cause, effect), positive in self.positive_counts.items():
            support = self.cause_counts[cause]
            if support >= self.min_support and support > 0 and positive / support >= self.min_confidence:
                learned.add((cause, effect))
        return learned

    def describe_relations(self) -> list[dict[str, Any]]:
        rows = []
        for cause, effect in sorted(self.learned_links()):
            support = self.cause_counts[cause]
            rows.append(
                {
                    "cause": cause,
                    "effect": effect,
                    "support": support,
                    "confidence": round(self.positive_counts[(cause, effect)] / support, 3),
                    "discovered": True,
                }
            )
        return rows

    def counterfactual(self, state: dict[str, str], action: str) -> dict[str, str]:
        out = dict(state)
        facts = active_facts(out, action)
        for _ in range(4):
            changed = False
            for cause, effect in sorted(self.learned_links(), key=link_priority):
                if cause in facts:
                    key, value = effect.split("=", 1)
                    if out.get(key) != value:
                        out[key] = value
                        facts = active_facts(out, action)
                        changed = True
            if not changed:
                break
        if "displacement=normal" in facts:
            out["risk"] = "low"
        elif "displacement=high" in facts:
            out["risk"] = "high"
        return out

    def edit_relation(self, cause: str, effect: str) -> bool:
        if self.cause_counts[cause] < self.min_support:
            return False
        for link_cause, link_effect in list(self.learned_links()):
            if link_cause == cause:
                self.positive_counts[(link_cause, link_effect)] = 0
        self.positive_counts[(cause, effect)] = self.cause_counts[cause]
        return True

    def active_exploration_score(self) -> float:
        tried = sum(1 for action in PHYSICAL_ACTIONS if self.action_counts[action] >= self.min_action_trials)
        return tried / len(PHYSICAL_ACTIONS)

    def adaptive_exploration_score(self) -> float:
        return self.active_exploration_score()


class UncertaintyDiscoveryAgent(DiscoveryRelationAgent):
    name = "uncertainty_discovery_agent"

    def __init__(
        self,
        min_support: int = 6,
        min_confidence: float = 0.93,
        min_action_trials: int = 8,
        inspect_on_uncertainty: bool = True,
    ) -> None:
        super().__init__(min_support=min_support, min_confidence=min_confidence, min_action_trials=min_action_trials)
        self.inspect_on_uncertainty = inspect_on_uncertainty

    def act(self, state: dict[str, str]) -> str:
        if self.inspect_on_uncertainty and self.uncertain_relation_links(state):
            self.action_counts["inspect"] += 1
            return "inspect"
        return super().act(state)

    def uncertain_relation_links(self, state: dict[str, str]) -> list[str]:
        links = []
        unknown = {key for key, value in state.items() if value == "unknown"} | (PROCESS_VARIABLES - set(state))

        if {"rainfall", "drainage", "pore_pressure"} & unknown:
            links.append("rainfall/drainage -> pore_pressure")
        if {"pore_pressure", "load", "support", "displacement"} & unknown:
            links.append("pore_pressure/load/support -> displacement")
        if {"displacement", "risk"} & unknown:
            links.append("displacement -> risk")

        if state.get("pore_pressure") == "normal" and state.get("displacement") == "high" and state.get("support") != "absent":
            links.append("pore_pressure/support -> displacement conflict")
        if state.get("displacement") == "normal" and state.get("risk") == "high":
            links.append("displacement -> risk conflict")
        if state.get("displacement") == "high" and state.get("risk") == "low":
            links.append("displacement -> risk conflict")
        if state.get("rainfall") == "high" and state.get("drainage") == "poor" and state.get("pore_pressure") == "normal":
            links.append("rainfall/drainage -> pore_pressure conflict")

        return sorted(set(links))

    def uncertainty_audit(self, state: dict[str, str]) -> list[dict[str, str]]:
        return [{"link": link, "reason": "missing_or_conflicting_observation", "recommended_action": "inspect"} for link in self.uncertain_relation_links(state)]


def fact_keys(fact: str) -> set[str]:
    keys = set()
    for part in fact.split("&"):
        if "=" in part:
            key, _ = part.split("=", 1)
            keys.add(key)
    return keys


def fact_values(fact: str) -> set[str]:
    values = set()
    for part in fact.split("&"):
        if "=" in part:
            _, value = part.split("=", 1)
            values.add(value)
    return values


def link_priority(link: tuple[str, str]) -> tuple[int, str, str]:
    cause, effect = link
    protective_terms = ["drainage=good", "support=present", "action=improve_drainage", "action=add_support", "action=reduce_load"]
    protective_effects = ["normal", "low", "good", "present"]
    is_protective = any(term in cause for term in protective_terms) or any(effect.endswith(f"={value}") for value in protective_effects)
    return (1 if is_protective else 0, cause, effect)


def make_agent(name: str, seed: int = 0) -> BaseAgent:
    if name == "random":
        return RandomAgent(seed)
    if name == "shortcut":
        return ShortcutAgent()
    if name == "passive_memory":
        return PassiveMemoryAgent()
    if name == "relation_agent":
        return RelationAgent()
    if name == "relation_no_explore":
        agent = RelationAgent(explore=False)
        agent.name = "relation_no_explore"
        return agent
    if name == "discovery_relation_agent":
        return DiscoveryRelationAgent()
    if name == "uncertainty_discovery_agent":
        return UncertaintyDiscoveryAgent()
    raise ValueError(f"unknown agent: {name}")
