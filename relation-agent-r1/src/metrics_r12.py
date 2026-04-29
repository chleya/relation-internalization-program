from __future__ import annotations

import random
from statistics import mean
from typing import Any

from .agents import BaseAgent, DiscoveryRelationAgent, RelationAgent, make_agent
from .env import ProcessWorld, Transition
from .metrics import action_success, train_agent
from .metrics_r11 import FALSE_CANDIDATE_LINKS


NEW_SUPPORT_PRESSURE_LINKS = {
    ("support=present", "pore_pressure=normal"),
    ("action=add_support", "pore_pressure=normal"),
}


def make_r12_agent(name: str, seed: int, config: dict[str, Any]) -> BaseAgent:
    if name == "discovery_relation_agent":
        agent_config = config["discovery_agent"]
        return DiscoveryRelationAgent(
            min_support=int(agent_config["min_support"]),
            min_confidence=float(agent_config["min_confidence"]),
            min_action_trials=int(agent_config["min_action_trials"]),
        )
    return make_agent(name, seed)


def add_persistent_nuisance(state: dict[str, str], rng: random.Random, correlation: float) -> dict[str, str]:
    out = dict(state)
    risk_high = out["risk"] == "high"
    correlated = rng.random() < correlation
    alarm = risk_high if correlated else not risk_high
    out["season"] = "storm" if alarm else "calm"
    out["contractor"] = "alarm" if alarm else "clear"
    return out


def train_with_persistent_nuisance(agent: BaseAgent, seed: int, steps: int, correlation: float, mode: str = "base") -> None:
    env = ProcessWorld(seed=seed, mode=mode)
    rng = random.Random(seed + 55_000)
    for _ in range(steps):
        state = add_persistent_nuisance(env.reset(), rng, correlation)
        env.state = dict(state)
        action = agent.act(state)
        transition = env.step(action)
        after = dict(transition.after)
        after["season"] = state["season"]
        after["contractor"] = state["contractor"]
        transition = Transition(before=state, action=transition.action, after=after, reward=transition.reward)
        env.state = dict(after)
        agent.observe(transition)


def nuisance_action_success(agent: BaseAgent, seed: int, steps: int, correlation: float) -> float:
    env = ProcessWorld(seed=seed + 60_000, mode="base")
    rng = random.Random(seed + 65_000)
    hits = []
    for _ in range(steps):
        state = add_persistent_nuisance(env.reset(), rng, correlation)
        action = agent.act(state)
        hits.append(1.0 if action in env.optimal_actions(state) else 0.0)
    return mean(hits)


def unmarked_nuisance_rejection(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_r12_agent(agent_name, seed, config)
    train_with_persistent_nuisance(agent, seed, int(config["train_steps"]), float(config["nuisance"]["train_correlation"]))
    action_score = nuisance_action_success(agent, seed, int(config["test_steps"]), float(config["nuisance"]["test_correlation"]))
    if isinstance(agent, DiscoveryRelationAgent):
        learned = agent.learned_links()
        nuisance_links = {link for link in learned if "season=" in link[0] or "contractor=" in link[0]}
        link_score = 1.0 if not nuisance_links else 0.0
        return mean([action_score, link_score])
    return action_score


def train_in_mode(agent: BaseAgent, seed: int, steps: int, mode: str) -> None:
    env = ProcessWorld(seed=seed, mode=mode)
    for _ in range(steps):
        state = env.reset()
        action = agent.act(state)
        transition = env.step(action)
        agent.observe(transition)


def new_link_discovery(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_r12_agent(agent_name, seed, config)
    train_in_mode(agent, seed, int(config["new_link_train_steps"]), mode="new_support_pressure_link")
    state = {
        "rainfall": "high",
        "load": "low",
        "drainage": "poor",
        "support": "absent",
        "pore_pressure": "high",
        "displacement": "normal",
        "risk": "low",
        "warning": "normal",
    }
    predicted = agent.counterfactual(state, "add_support")
    predicted_score = 1.0 if predicted.get("pore_pressure") == "normal" else 0.0
    if isinstance(agent, DiscoveryRelationAgent):
        link_score = 1.0 if agent.learned_links() & NEW_SUPPORT_PRESSURE_LINKS else 0.0
        return mean([predicted_score, link_score])
    return predicted_score


def adaptive_exploration(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_r12_agent(agent_name, seed, config)
    train_in_mode(agent, seed, int(config["train_steps"]), mode="base")
    if hasattr(agent, "adaptive_exploration_score"):
        return agent.adaptive_exploration_score()
    return 0.0


def discovered_relation_precision(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_r12_agent(agent_name, seed, config)
    if not isinstance(agent, DiscoveryRelationAgent):
        return 0.0
    train_with_persistent_nuisance(agent, seed, int(config["train_steps"]), float(config["nuisance"]["train_correlation"]))
    learned = agent.learned_links()
    if not learned:
        return 0.0
    false_links = {link for link in learned if "season=" in link[0] or "contractor=" in link[0]} | (learned & FALSE_CANDIDATE_LINKS)
    return 1.0 - (len(false_links) / len(learned))


def discovery_action_success(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_r12_agent(agent_name, seed, config)
    if agent_name == "discovery_relation_agent":
        train_in_mode(agent, seed, int(config["train_steps"]), mode="base")
    else:
        train_agent(agent, seed, int(config["train_steps"]))
    return action_success(agent, seed, int(config["test_steps"]), mode="base")


def discovery_r12_gated_score(metrics: dict[str, float], gates: dict[str, float]) -> float:
    required = [
        metrics["unmarked_nuisance_rejection"] >= gates["unmarked_nuisance_rejection"],
        metrics["new_link_discovery"] >= gates["new_link_discovery"],
        metrics["adaptive_exploration"] >= gates["adaptive_exploration"],
        metrics["discovered_relation_precision"] >= gates["discovered_relation_precision"],
        metrics["discovery_action_success"] >= gates["discovery_action_success"],
    ]
    if not all(required):
        return 0.0
    return mean(metrics[key] for key in gates)


def evaluate_discovery_agent(agent_name: str, seed: int, config: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "unmarked_nuisance_rejection": unmarked_nuisance_rejection(agent_name, seed, config),
        "new_link_discovery": new_link_discovery(agent_name, seed, config),
        "adaptive_exploration": adaptive_exploration(agent_name, seed, config),
        "discovered_relation_precision": discovered_relation_precision(agent_name, seed, config),
        "discovery_action_success": discovery_action_success(agent_name, seed, config),
    }
    metrics["discovery_r12_gated_score"] = discovery_r12_gated_score(metrics, config["gates"])
    return metrics
