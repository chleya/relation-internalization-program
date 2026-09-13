from __future__ import annotations

import math
import random
from typing import Any
import numpy as np

WORLD_SIZE = 4.0
DT = 0.05
STEPS_PER_EPISODE = 20
DAMPING = 0.3
COLLISION_DIST = 1.2
COLLISION_FORCE = 3.0
COOLING = 0.4
AMBIENT_T = 0.3
HEAT_TRANSFER = 0.8
NOISE_FRAC = 0.05
RISK_THRESHOLD = 0.65

ALL_FEATURES_CONT = [
    "pos_x", "pos_y", "speed", "temperature", "risk",
    "prediction_error", "compression_surprise",
]
N_FEATS_CONT = len(ALL_FEATURES_CONT)
F_RISK = 4
F_PE = 5
F_CS = 6


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _init_objects(rng: random.Random, n_objects: int) -> np.ndarray:
    state = np.zeros((n_objects, 5), dtype=np.float32)
    for i in range(n_objects):
        angle = 2 * math.pi * i / n_objects
        r = WORLD_SIZE * 0.3 * rng.random()
        state[i, 0] = WORLD_SIZE / 2 + r * math.cos(angle)
        state[i, 1] = WORLD_SIZE / 2 + r * math.sin(angle)
        state[i, 2] = (rng.random() - 0.5) * 1.5
        state[i, 3] = (rng.random() - 0.5) * 1.5
        state[i, 4] = AMBIENT_T + rng.random() * 0.8
    return state


def _compute_forces_heat(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = state.shape[0]
    forces = np.zeros((n, 2), dtype=np.float32)
    heat_flux = np.zeros(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dx = state[i, 0] - state[j, 0]
            dy = state[i, 1] - state[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < COLLISION_DIST and dist > 0.01:
                force = COLLISION_FORCE * (COLLISION_DIST - dist) / dist
                fx = force * dx / dist
                fy = force * dy / dist
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
                hf = HEAT_TRANSFER * (state[j, 4] - state[i, 4]) / (1.0 + dist)
                heat_flux[i] += hf
                heat_flux[j] -= hf
    return forces, heat_flux


def _step_euler(state: np.ndarray) -> np.ndarray:
    n = state.shape[0]
    forces, heat = _compute_forces_heat(state)
    new_state = state.copy()
    for i in range(n):
        new_state[i, 0] += state[i, 2] * DT
        new_state[i, 1] += state[i, 3] * DT
        new_state[i, 2] += (forces[i, 0] - DAMPING * state[i, 2]) * DT
        new_state[i, 3] += (forces[i, 1] - DAMPING * state[i, 3]) * DT
        new_state[i, 4] += (-COOLING * (state[i, 4] - AMBIENT_T) + heat[i]) * DT
    new_state[:, 0] = np.clip(new_state[:, 0], 0.2, WORLD_SIZE - 0.2)
    new_state[:, 1] = np.clip(new_state[:, 1], 0.2, WORLD_SIZE - 0.2)
    new_state[:, 4] = np.clip(new_state[:, 4], AMBIENT_T - 0.3, AMBIENT_T + 1.2)
    return new_state


def _state_to_features(state: np.ndarray, noise_scale: float = NOISE_FRAC) -> np.ndarray:
    n = state.shape[0]
    feats = np.zeros((n, N_FEATS_CONT), dtype=np.float32)
    for i in range(n):
        feats[i, 0] = state[i, 0] / WORLD_SIZE + noise_scale * np.random.randn()
        feats[i, 1] = state[i, 1] / WORLD_SIZE + noise_scale * np.random.randn()
        speed = math.sqrt(state[i, 2] ** 2 + state[i, 3] ** 2)
        feats[i, 2] = _clamp(speed / 2.0 + noise_scale * np.random.randn())
        feats[i, 3] = _clamp(state[i, 4] + noise_scale * np.random.randn())
        feats[i, 4] = _clamp(
            max(0, feats[i, 3] - AMBIENT_T) * 0.8
            + feats[i, 2] * 0.4
            + noise_scale * np.random.randn(),
        )
        feats[i, 5] = _clamp(0.15 + noise_scale * np.random.randn())
        feats[i, 6] = _clamp(0.10 + noise_scale * np.random.randn())
    return np.clip(feats, 0.0, 1.0)


def _compute_actionability(feats: np.ndarray) -> dict:
    n = feats.shape[0]
    direct = np.zeros(n, dtype=bool)
    indirect = np.zeros(n, dtype=bool)
    inspectable = np.zeros(n, dtype=bool)

    high_risk = feats[:, F_RISK] > RISK_THRESHOLD
    direct[high_risk] = True
    inspectable[feats[:, F_RISK] > 0.3] = True

    for i in range(n):
        for j in range(n):
            if i != j and feats[j, F_RISK] > 0.8 * RISK_THRESHOLD and feats[i, F_RISK] > 0.4:
                indirect[i] = True
                break

    return {
        "direct": direct,
        "indirect": indirect,
        "inspect": inspectable,
        "risk": feats[:, F_RISK],
    }


def make_continuous_episode(
    rng: random.Random, n_objects: int, condition: str, episode_id: int,
) -> dict[str, Any]:
    state = _init_objects(rng, n_objects)
    interactions = []
    for step in range(STEPS_PER_EPISODE):
        state = _step_euler(state)
        feats = _state_to_features(state)
        act = _compute_actionability(feats)
        for oi in range(n_objects):
            interactions.append({
                "step": step,
                "object_id": oi,
                "region_id": f"{oi}_{step}",
                "pos_x": float(feats[oi, 0]),
                "pos_y": float(feats[oi, 1]),
                "speed": float(feats[oi, 2]),
                "temperature": float(feats[oi, 3]),
                "risk": float(feats[oi, 4]),
                "prediction_error": float(feats[oi, 5]),
                "compression_surprise": float(feats[oi, 6]),
                "observed_direct_reward": 1.0 if act["direct"][oi] else 0.0,
                "observed_heat_transfer": float(
                    _clamp(abs(feats[oi, 3] - AMBIENT_T) * 0.5)
                ) if act["indirect"][oi] else 0.0,
            })

    truth = []
    unique_regions = [(oi, step) for step in range(STEPS_PER_EPISODE) for oi in range(n_objects)]
    for oi, step in unique_regions:
        idx = oi + step * n_objects
        feats_i = _state_to_features(state)[oi]
        act_i = _compute_actionability(np.array([feats_i]))
        truth.append({
            "object_id": oi,
            "region_id": f"{oi}_{step}",
            "direct_actionable": bool(act_i["direct"][0]),
            "indirect_actionable": bool(act_i["indirect"][0]),
            "inspectable": bool(act_i["inspect"][0]),
            "latent_risk": float(act_i["risk"][0]),
        })

    return {
        "episode_id": episode_id,
        "condition": condition,
        "n_objects": n_objects,
        "dt": DT,
        "steps": STEPS_PER_EPISODE,
        "model_input": {
            "condition": condition,
            "interaction_history": interactions,
            "n_regions": len(interactions),
        },
        "evaluator_ground_truth": {"regions": truth},
    }


def make_continuous_datasets(
    n_objects: int = 3, n_train: int = 24, n_test: int = 8, n_ood: int = 8, seed: int = 0,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    datasets: dict[str, list[dict]] = {}
    ei = 0

    for _ in range(n_train):
        ep = make_continuous_episode(rng, n_objects, "train", ei)
        datasets.setdefault("train", []).append(ep)
        ei += 1
    for _ in range(n_test):
        ep = make_continuous_episode(rng, n_objects, "test", ei)
        datasets.setdefault("test", []).append(ep)
        ei += 1
    for _ in range(n_ood):
        ep = make_continuous_episode(rng, n_objects + 1, "ood_remap", ei)
        datasets.setdefault("ood_remap", []).append(ep)
        ei += 1

    return datasets
