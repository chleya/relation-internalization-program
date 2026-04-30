from __future__ import annotations

from typing import Any

import numpy as np

from .b21_trace_metrics import b21_trace_hardening_score
from .models.base import BasePLOSModel, logits_from_region


class RandomTraceBaseline(BasePLOSModel):
    name = "random_trace_baseline"
    structural_family = "random_trace"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        past = np.asarray(batch["past_frames"], dtype=np.float32)
        region = int(self.rng.integers(0, int(batch["grid_size"]) ** 2))
        return {
            "future_frames": np.repeat(past[-1][None, ...], int(batch["future_horizon"]), axis=0),
            "identity_logits": None,
            "event_logits": np.zeros((int(batch["frame_size"]), int(batch["frame_size"])), dtype=np.float32),
            "inspection_logits": logits_from_region(region, int(batch["grid_size"]) ** 2, strength=4.0),
            "structure": {"applicable": True, "trace_family": "random", "random_trace_region": region},
        }

    def get_structure(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.forward(batch)["structure"]

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        return {"applicable": False, "reason": "random_baseline_no_causal_trace"}


class OracleTraceBaseline(BasePLOSModel):
    name = "oracle_trace_baseline"
    structural_family = "oracle_trace"

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        past = np.asarray(batch["past_frames"], dtype=np.float32)
        region = int(batch.get("oracle_trace_region", 0))
        return {
            "future_frames": np.repeat(past[-1][None, ...], int(batch["future_horizon"]), axis=0),
            "identity_logits": None,
            "event_logits": np.zeros((int(batch["frame_size"]), int(batch["frame_size"])), dtype=np.float32),
            "inspection_logits": logits_from_region(region, int(batch["grid_size"]) ** 2, strength=8.0),
            "structure": {"applicable": True, "trace_family": "oracle", "oracle_trace_region": region},
        }

    def get_structure(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.forward(batch)["structure"]

    def intervene_structure(self, batch: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
        base = self.forward(batch)
        return {
            "applicable": True,
            "future_frames": base["future_frames"] * 0.0,
            "base_future_frames": base["future_frames"],
            "target": "oracle_trace",
            "target_region": int(batch.get("oracle_trace_region", 0)),
        }


class HeuristicSaliencyTraceBaseline(BasePLOSModel):
    name = "heuristic_saliency_trace_baseline"
    structural_family = "heuristic_saliency_trace"

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        past = np.asarray(batch["past_frames"], dtype=np.float32)
        intensity = past.mean(axis=(0, 3))
        grid_size = int(batch["grid_size"])
        frame_size = int(batch["frame_size"])
        cell = frame_size // grid_size
        scores = []
        for gy in range(grid_size):
            for gx in range(grid_size):
                patch = intensity[gy * cell : (gy + 1) * cell, gx * cell : (gx + 1) * cell]
                scores.append(float(patch.max()))
        region = int(np.argmax(scores))
        return {
            "future_frames": np.repeat(past[-1][None, ...], int(batch["future_horizon"]), axis=0),
            "identity_logits": None,
            "event_logits": np.zeros((frame_size, frame_size), dtype=np.float32),
            "inspection_logits": logits_from_region(region, grid_size**2, strength=4.0),
            "structure": {"applicable": True, "trace_family": "heuristic_saliency", "saliency_region": region},
        }


def evaluate_random_trace_baseline(episodes: list[dict[str, Any]], config: dict[str, Any], seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_regions = int(config.get("env", {}).get("grid_size", 8)) ** 2
    rows = []
    for episode in episodes:
        pred = int(rng.integers(0, n_regions))
        true_region = int(episode["ground_truth"].get("true_trace_region", episode["ground_truth"].get("true_delayed_checkpoint_region", -1)))
        rows.append(1.0 if pred == true_region else 0.0)
    accuracy = float(np.mean(rows)) if rows else 0.0
    metrics = {
        "false_trace_rejection": accuracy,
        "trace_swap_sensitivity": 0.0,
        "trace_deletion_specificity_ratio": 0.0,
        "multi_source_conflict_resolution": accuracy,
        "noisy_trace_robustness": accuracy,
        "trace_length_extrapolation": accuracy,
        "trace_compression_survival": accuracy,
        "true_trace_intervention_drop": 0.0,
        "non_trace_stability": 1.0,
    }
    return {
        "random_b21_score": b21_trace_hardening_score(metrics, {}),
        "random_false_trace_rejection": accuracy,
        "random_trace_swap_sensitivity": 0.0,
        "random_deletion_specificity": 0.0,
        "random_length_extrapolation": accuracy,
    }


def evaluate_oracle_trace_baseline(episodes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "false_trace_rejection": 1.0,
        "trace_swap_sensitivity": 1.0,
        "trace_deletion_specificity_ratio": 999.0,
        "multi_source_conflict_resolution": 1.0,
        "noisy_trace_robustness": 1.0,
        "trace_length_extrapolation": 1.0,
        "trace_compression_survival": 1.0,
        "true_trace_intervention_drop": 1.0,
        "non_trace_stability": 1.0,
    }
    return {"oracle_b21_score": b21_trace_hardening_score(metrics, {}), "oracle_trace_accuracy": 1.0}
