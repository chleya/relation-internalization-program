from __future__ import annotations

from statistics import mean
from typing import Any

from .agents import BaseAgent, RelationAgent
from .env import TRUE_LINKS, ProcessWorld


def train_agent(agent: BaseAgent, seed: int, steps: int) -> None:
    env = ProcessWorld(seed=seed, mode="base")
    state = env.reset()
    for _ in range(steps):
        action = agent.act(state)
        transition = env.step(action)
        agent.observe(transition)
        state = env.reset()


def action_success(agent: BaseAgent, seed: int, steps: int, mode: str = "base") -> float:
    env = ProcessWorld(seed=seed + 1000, mode=mode)
    hits = []
    for _ in range(steps):
        state = env.reset()
        action = agent.act(state)
        hits.append(1.0 if action in env.optimal_actions(state) else 0.0)
    return mean(hits)


def relation_recovery(agent: BaseAgent) -> float:
    if not isinstance(agent, RelationAgent):
        return 0.0
    learned = agent.learned_links()
    return len(TRUE_LINKS & learned) / len(TRUE_LINKS)


def counterfactual_accuracy(agent: BaseAgent) -> float:
    cases = [
        (
            {
                "rainfall": "high",
                "load": "high",
                "drainage": "poor",
                "support": "absent",
                "pore_pressure": "high",
                "displacement": "high",
                "risk": "high",
                "warning": "warning",
            },
            "improve_drainage",
            {"pore_pressure": "normal"},
        ),
        (
            {
                "rainfall": "high",
                "load": "high",
                "drainage": "poor",
                "support": "absent",
                "pore_pressure": "high",
                "displacement": "high",
                "risk": "high",
                "warning": "warning",
            },
            "add_support",
            {"displacement": "normal", "risk": "low"},
        ),
        (
            {
                "rainfall": "low",
                "load": "high",
                "drainage": "poor",
                "support": "absent",
                "pore_pressure": "normal",
                "displacement": "normal",
                "risk": "low",
                "warning": "normal",
            },
            "reduce_load",
            {"load": "low"},
        ),
    ]
    hits = []
    for state, action, expected in cases:
        predicted = agent.counterfactual(state, action)
        hits.append(1.0 if all(predicted.get(k) == v for k, v in expected.items()) else 0.0)
    return mean(hits)


def edit_success(agent: BaseAgent) -> float:
    if not isinstance(agent, RelationAgent):
        return 0.0
    state = {
        "rainfall": "high",
        "load": "low",
        "drainage": "poor",
        "support": "absent",
        "pore_pressure": "high",
        "displacement": "normal",
        "risk": "low",
        "warning": "warning",
    }
    before = agent.counterfactual(state, "noop").get("pore_pressure")
    edited = agent.edit_relation("rainfall=high", "pore_pressure=normal")
    after = agent.counterfactual(state, "noop").get("pore_pressure")
    return 1.0 if edited and before == "high" and after == "normal" else 0.0


def active_exploration(agent: BaseAgent) -> float:
    if hasattr(agent, "active_exploration_score"):
        return agent.active_exploration_score()
    return 0.0


def gated_r1_score(metrics: dict[str, float], gates: dict[str, float]) -> float:
    required = [
        metrics["action_success"] >= gates["action_success"],
        metrics["ood_action_success"] >= gates["ood_action_success"],
        metrics["relation_recovery"] >= gates["relation_recovery"],
        metrics["counterfactual_accuracy"] >= gates["counterfactual_accuracy"],
        metrics["edit_success"] >= gates["edit_success"],
        metrics["active_exploration"] >= gates["active_exploration"],
    ]
    if not all(required):
        return 0.0
    return mean(metrics[key] for key in gates)


def evaluate_agent(agent: BaseAgent, seed: int, config: dict[str, Any]) -> dict[str, float]:
    train_agent(agent, seed, int(config["train_steps"]))
    metrics = {
        "action_success": action_success(agent, seed, int(config["test_steps"]), mode="base"),
        "ood_action_success": action_success(agent, seed, int(config["test_steps"]), mode="ood_warning"),
        "relation_recovery": relation_recovery(agent),
        "counterfactual_accuracy": counterfactual_accuracy(agent),
        "edit_success": edit_success(agent),
        "active_exploration": active_exploration(agent),
    }
    metrics["gated_r1_score"] = gated_r1_score(metrics, config["gates"])
    return metrics
