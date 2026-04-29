from __future__ import annotations

import random
from statistics import mean
from typing import Any

from .agents import BaseAgent, DiscoveryRelationAgent, UncertaintyDiscoveryAgent, make_agent
from .env import ProcessWorld
from .metrics import action_success, train_agent
from .metrics_r12 import train_in_mode


PARTIAL_FIELDS = ["rainfall", "load", "drainage", "support", "pore_pressure", "displacement", "risk"]


def make_r2_agent(name: str, seed: int, config: dict[str, Any]) -> BaseAgent:
    if name == "uncertainty_discovery_agent":
        agent_config = config["uncertainty_agent"]
        return UncertaintyDiscoveryAgent(
            min_support=int(agent_config["min_support"]),
            min_confidence=float(agent_config["min_confidence"]),
            min_action_trials=int(agent_config["min_action_trials"]),
        )
    if name == "discovery_relation_agent":
        agent_config = config["uncertainty_agent"]
        return DiscoveryRelationAgent(
            min_support=int(agent_config["min_support"]),
            min_confidence=float(agent_config["min_confidence"]),
            min_action_trials=int(agent_config["min_action_trials"]),
        )
    return make_agent(name, seed)


def train_r2_agent(agent: BaseAgent, seed: int, steps: int) -> None:
    if isinstance(agent, (DiscoveryRelationAgent, UncertaintyDiscoveryAgent)):
        train_in_mode(agent, seed, steps, mode="base")
    else:
        train_agent(agent, seed, steps)


def mask_state(state: dict[str, str], rng: random.Random, missing_probability: float) -> dict[str, str]:
    out = dict(state)
    for key in PARTIAL_FIELDS:
        if rng.random() < missing_probability:
            out[key] = "unknown"
    return out


def noisy_state(state: dict[str, str], rng: random.Random, noise_probability: float) -> dict[str, str]:
    out = dict(state)
    binary_values = {
        "rainfall": ("high", "low"),
        "load": ("high", "low"),
        "drainage": ("poor", "good"),
        "support": ("absent", "present"),
        "pore_pressure": ("high", "normal"),
        "displacement": ("high", "normal"),
        "risk": ("high", "low"),
    }
    for key, values in binary_values.items():
        if rng.random() < noise_probability:
            out[key] = values[1] if out[key] == values[0] else values[0]
    return out


def needs_inspection(state: dict[str, str]) -> bool:
    if any(state.get(key) == "unknown" for key in ["pore_pressure", "displacement", "risk"]):
        return True
    if state.get("rainfall") == "high" and state.get("drainage") == "poor" and state.get("pore_pressure") == "normal":
        return True
    if state.get("displacement") == "normal" and state.get("risk") == "high":
        return True
    if state.get("displacement") == "high" and state.get("risk") == "low":
        return True
    return False


def two_step_action_success(agent: BaseAgent, true_state: dict[str, str], observed_state: dict[str, str], env: ProcessWorld) -> float:
    first_action = agent.act(observed_state)
    if first_action == "inspect":
        second_action = agent.act(true_state)
        return 1.0 if second_action in env.optimal_actions(true_state) else 0.0
    return 1.0 if first_action in env.optimal_actions(true_state) else 0.0


def partial_observation_success(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_r2_agent(agent_name, seed, config)
    train_r2_agent(agent, seed, int(config["train_steps"]))
    env = ProcessWorld(seed=seed + 70_000)
    rng = random.Random(seed + 71_000)
    hits = []
    for _ in range(int(config["test_steps"])):
        true_state = env.reset()
        observed = mask_state(true_state, rng, float(config["partial_observation"]["missing_probability"]))
        hits.append(two_step_action_success(agent, true_state, observed, env))
    return mean(hits)


def inspection_recall(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_r2_agent(agent_name, seed, config)
    train_r2_agent(agent, seed, int(config["train_steps"]))
    env = ProcessWorld(seed=seed + 72_000)
    rng = random.Random(seed + 73_000)
    hits = []
    for _ in range(int(config["partial_observation"]["critical_missing_cases"])):
        true_state = env.reset()
        observed = dict(true_state)
        observed[rng.choice(["pore_pressure", "displacement", "risk"])] = "unknown"
        hits.append(1.0 if agent.act(observed) == "inspect" else 0.0)
    return mean(hits)


def unsafe_action_rate(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_r2_agent(agent_name, seed, config)
    train_r2_agent(agent, seed, int(config["train_steps"]))
    env = ProcessWorld(seed=seed + 74_000)
    rng = random.Random(seed + 75_000)
    unsafe = []
    for _ in range(int(config["partial_observation"]["critical_missing_cases"])):
        true_state = env.reset()
        observed = dict(true_state)
        observed[rng.choice(["pore_pressure", "displacement", "risk"])] = "unknown"
        action = agent.act(observed)
        unsafe.append(1.0 if needs_inspection(observed) and action in {"improve_drainage", "add_support", "reduce_load", "noop"} else 0.0)
    return mean(unsafe)


def uncertainty_audit_score(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_r2_agent(agent_name, seed, config)
    train_r2_agent(agent, seed, int(config["train_steps"]))
    cases = [
        {"pore_pressure": "unknown"},
        {"displacement": "unknown"},
        {"risk": "unknown"},
        {"rainfall": "high", "drainage": "poor", "pore_pressure": "normal"},
        {"displacement": "normal", "risk": "high"},
    ]
    hits = []
    for patch in cases:
        state = {
            "rainfall": "high",
            "load": "high",
            "drainage": "poor",
            "support": "absent",
            "pore_pressure": "high",
            "displacement": "high",
            "risk": "high",
            "warning": "warning",
        }
        state.update(patch)
        audit = agent.uncertainty_audit(state) if hasattr(agent, "uncertainty_audit") else []
        text = " ".join(str(item) for item in audit)
        hits.append(1.0 if audit and "inspect" in text and "link" in text else 0.0)
    return mean(hits)


def noisy_observation_robustness(agent_name: str, seed: int, config: dict[str, Any]) -> float:
    agent = make_r2_agent(agent_name, seed, config)
    train_r2_agent(agent, seed, int(config["train_steps"]))
    env = ProcessWorld(seed=seed + 76_000)
    rng = random.Random(seed + 77_000)
    hits = []
    for _ in range(int(config["test_steps"])):
        true_state = env.reset()
        observed = noisy_state(true_state, rng, float(config["partial_observation"]["noise_probability"]))
        if needs_inspection(observed):
            hits.append(1.0 if agent.act(observed) == "inspect" else 0.0)
        else:
            hits.append(two_step_action_success(agent, true_state, observed, env))
    return mean(hits)


def partial_r2_gated_score(metrics: dict[str, float], gates: dict[str, float]) -> float:
    required = [
        metrics["partial_observation_success"] >= gates["partial_observation_success"],
        metrics["inspection_recall"] >= gates["inspection_recall"],
        metrics["unsafe_action_rate"] <= gates["unsafe_action_rate"],
        metrics["uncertainty_audit_score"] >= gates["uncertainty_audit_score"],
        metrics["noisy_observation_robustness"] >= gates["noisy_observation_robustness"],
    ]
    if not all(required):
        return 0.0
    safety_score = 1.0 - metrics["unsafe_action_rate"]
    return mean(
        [
            metrics["partial_observation_success"],
            metrics["inspection_recall"],
            safety_score,
            metrics["uncertainty_audit_score"],
            metrics["noisy_observation_robustness"],
        ]
    )


def evaluate_partial_observability_agent(agent_name: str, seed: int, config: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "partial_observation_success": partial_observation_success(agent_name, seed, config),
        "inspection_recall": inspection_recall(agent_name, seed, config),
        "unsafe_action_rate": unsafe_action_rate(agent_name, seed, config),
        "uncertainty_audit_score": uncertainty_audit_score(agent_name, seed, config),
        "noisy_observation_robustness": noisy_observation_robustness(agent_name, seed, config),
    }
    metrics["partial_r2_gated_score"] = partial_r2_gated_score(metrics, config["gates"])
    return metrics
