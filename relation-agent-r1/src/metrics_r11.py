from __future__ import annotations

import random
from statistics import mean
from typing import Any

from .agents import BaseAgent, RelationAgent, make_agent
from .env import ACTIONS, PHYSICAL_ACTIONS, REVERSAL_LINKS, TRUE_LINKS, ProcessWorld, Transition
from .metrics import active_exploration


FALSE_CANDIDATE_LINKS = {
    ("season=storm", "risk=high"),
    ("season=calm", "risk=low"),
    ("contractor=alarm", "risk=high"),
    ("contractor=clear", "risk=low"),
    ("support=absent", "risk=high"),
    ("drainage=poor", "risk=high"),
}

EXPANDED_CANDIDATE_LINKS = TRUE_LINKS | REVERSAL_LINKS | FALSE_CANDIDATE_LINKS


def make_r11_agent(name: str, seed: int, config: dict[str, Any]) -> BaseAgent:
    if name == "relation_agent":
        relation_config = config["relation_agent"]
        physical_cost = float(relation_config["physical_action_cost"])
        return RelationAgent(
            min_support=int(relation_config["min_support"]),
            min_confidence=float(relation_config["min_confidence"]),
            candidate_links=set(EXPANDED_CANDIDATE_LINKS),
            ignored_features=set(relation_config["ignored_features"]),
            decay=float(relation_config["decay"]),
            action_costs={
                "improve_drainage": physical_cost,
                "add_support": physical_cost,
                "reduce_load": physical_cost,
                "inspect": float(relation_config["inspect_cost"]),
                "noop": 0.0,
            },
        )
    if name == "relation_no_explore":
        relation_config = config["relation_agent"]
        agent = RelationAgent(
            min_support=int(relation_config["min_support"]),
            min_confidence=float(relation_config["min_confidence"]),
            candidate_links=set(EXPANDED_CANDIDATE_LINKS),
            ignored_features=set(relation_config["ignored_features"]),
            explore=False,
            decay=float(relation_config["decay"]),
        )
        agent.name = "relation_no_explore"
        return agent
    return make_agent(name, seed)


def add_confounders(
    state: dict[str, str],
    rng: random.Random,
    correlation: float,
) -> dict[str, str]:
    out = dict(state)
    risk_high = out["risk"] == "high"
    correlated = rng.random() < correlation
    storm = risk_high if correlated else not risk_high
    out["season"] = "storm" if storm else "calm"
    out["contractor"] = "alarm" if storm else "clear"
    return out


def train_agent_mode(
    agent: BaseAgent,
    seed: int,
    steps: int,
    mode: str = "base",
    confounder_correlation: float | None = None,
) -> None:
    env = ProcessWorld(seed=seed, mode=mode)
    rng = random.Random(seed + 100_000)
    for _ in range(steps):
        state = env.reset()
        if confounder_correlation is not None:
            state = add_confounders(state, rng, confounder_correlation)
            env.state = dict(state)
        action = agent.act(state)
        transition = env.step(action)
        if confounder_correlation is not None:
            after = add_confounders(transition.after, rng, confounder_correlation)
            transition = Transition(before=state, action=transition.action, after=after, reward=transition.reward)
            env.state = dict(after)
        agent.observe(transition)


def simulate_action(state: dict[str, str], action: str, mode: str = "base") -> dict[str, str]:
    env = ProcessWorld(seed=0, mode=mode)
    candidate = dict(state)
    env._apply_action(candidate, action)
    return env._evaluate_process(candidate)


def optimal_actions_for_mode(state: dict[str, str], mode: str = "base") -> set[str]:
    risk_reducing = set()
    for action in PHYSICAL_ACTIONS:
        after = simulate_action(state, action, mode)
        if state["risk"] == "high" and after["risk"] == "low":
            risk_reducing.add(action)
    if risk_reducing:
        return risk_reducing
    if state["risk"] == "low" and state["displacement"] == "normal":
        return {"noop", "inspect"}
    return {"noop", "inspect"}


def action_success_for_mode(
    agent: BaseAgent,
    seed: int,
    steps: int,
    mode: str = "base",
    confounder_correlation: float | None = None,
) -> float:
    env = ProcessWorld(seed=seed + 30_000, mode=mode)
    rng = random.Random(seed + 40_000)
    hits = []
    for _ in range(steps):
        state = env.reset()
        if confounder_correlation is not None:
            state = add_confounders(state, rng, confounder_correlation)
        action = agent.act(state)
        hits.append(1.0 if action in optimal_actions_for_mode(state, mode) else 0.0)
    return mean(hits)


def hidden_confounder_rejection(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_r11_agent(agent_name, seed, config)
    train_agent_mode(
        agent,
        seed,
        int(config["train_steps"]),
        mode="base",
        confounder_correlation=float(config["confounder"]["train_correlation"]),
    )
    return action_success_for_mode(
        agent,
        seed,
        int(config["test_steps"]),
        mode="base",
        confounder_correlation=float(config["confounder"]["test_correlation"]),
    )


def reversal_adaptation(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_r11_agent(agent_name, seed, config)
    train_agent_mode(agent, seed, int(config["train_steps"]), mode="base")
    train_agent_mode(agent, seed + 10_000, int(config["reversal_adapt_steps"]), mode="rule_reversal")
    return action_success_for_mode(agent, seed, int(config["test_steps"]), mode="rule_reversal")


def cost_tradeoff_success(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_r11_agent(agent_name, seed, config)
    train_agent_mode(agent, seed, int(config["train_steps"]), mode="base")
    cases = [
        (
            {
                "rainfall": "high",
                "load": "low",
                "drainage": "poor",
                "support": "absent",
                "pore_pressure": "high",
                "displacement": "normal",
                "risk": "low",
                "warning": "warning",
            },
            {"noop", "inspect"},
        ),
        (
            {
                "rainfall": "low",
                "load": "low",
                "drainage": "poor",
                "support": "absent",
                "pore_pressure": "normal",
                "displacement": "normal",
                "risk": "low",
                "warning": "warning",
            },
            {"noop", "inspect"},
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
                "warning": "normal",
            },
            set(PHYSICAL_ACTIONS),
        ),
    ]
    hits = []
    for state, expected in cases:
        hits.append(1.0 if agent.act(dict(state)) in expected else 0.0)
    return mean(hits)


def candidate_expansion_precision(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_r11_agent(agent_name, seed, config)
    if not isinstance(agent, RelationAgent):
        return 0.0
    train_agent_mode(
        agent,
        seed,
        int(config["train_steps"]),
        mode="base",
        confounder_correlation=float(config["confounder"]["train_correlation"]),
    )
    learned = agent.learned_links()
    if not learned:
        return 0.0
    false_learned = learned & FALSE_CANDIDATE_LINKS
    return 1.0 - (len(false_learned) / len(learned))


def active_discovery_score(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_r11_agent(agent_name, seed, config)
    train_agent_mode(agent, seed, int(config["train_steps"]), mode="base")
    return active_exploration(agent)


def hardening_gated_score(metrics: dict[str, float], gates: dict[str, float]) -> float:
    required = [
        metrics["hidden_confounder_rejection"] >= gates["hidden_confounder_rejection"],
        metrics["reversal_adaptation"] >= gates["reversal_adaptation"],
        metrics["cost_tradeoff_success"] >= gates["cost_tradeoff_success"],
        metrics["candidate_expansion_precision"] >= gates["candidate_expansion_precision"],
        metrics["active_discovery_score"] >= gates["active_discovery_score"],
    ]
    if not all(required):
        return 0.0
    return mean(metrics[key] for key in gates)


def evaluate_hardening_agent(agent_name: str, seed: int, config: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "hidden_confounder_rejection": hidden_confounder_rejection(agent_name, seed, config),
        "reversal_adaptation": reversal_adaptation(agent_name, seed, config),
        "cost_tradeoff_success": cost_tradeoff_success(agent_name, seed, config),
        "candidate_expansion_precision": candidate_expansion_precision(agent_name, seed, config),
        "active_discovery_score": active_discovery_score(agent_name, seed, config),
    }
    metrics["hardening_r11_gated_score"] = hardening_gated_score(metrics, config["gates"])
    return metrics
