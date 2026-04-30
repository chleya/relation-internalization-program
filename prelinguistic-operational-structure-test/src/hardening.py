from __future__ import annotations

import copy
import csv
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .data import generate_dataset
from .evaluate import evaluate_all
from .features import draw_disk, extract_blob_centers, region_id_to_slice
from .inspect_policy import select_region_from_logits
from .model_io import make_model_batch
from .models import make_model
from .scoring import merge_scores
from .train import train_model


HARDENING_KEYS = [
    "candidate_gate_preserved",
    "decoy_patch_stability",
    "static_decoy_suppression",
    "causal_endpoint_shift",
    "hardening_score",
]


def evaluate_flow_checkpoint_hardening(config: dict[str, Any], seed: int = 0) -> dict[str, float]:
    train_dataset = generate_dataset(config, "train", seed)
    test_dataset = generate_dataset(config, "test", seed + 10000)
    model = train_model("flow_checkpoint_model", train_dataset, config)

    from .ood import make_all_ood

    base = evaluate_all(model, {"test": test_dataset, "ood": make_all_ood(config, seed + 20000)}, config)
    base_metrics = merge_scores(base["behavior"], base["structure"], base["ood"], config["gates"])
    metrics = {
        "candidate_gate_preserved": 1.0 if base_metrics.get("plos_candidate_score", 0.0) > 0.0 else 0.0,
        "decoy_patch_stability": decoy_patch_stability(model, test_dataset, config),
        "static_decoy_suppression": static_decoy_suppression(model, config, seed + 30000),
        "causal_endpoint_shift": causal_endpoint_shift(model, test_dataset, config),
    }
    gates = {
        "candidate_gate_preserved": 1.0,
        "decoy_patch_stability": 0.80,
        "static_decoy_suppression": 0.95,
        "causal_endpoint_shift": 0.20,
    }
    if any(metrics[key] < gates[key] for key in gates):
        metrics["hardening_score"] = 0.0
    else:
        metrics["hardening_score"] = mean(metrics.values())
    return metrics


def decoy_patch_stability(model: Any, dataset: list[dict[str, Any]], config: dict[str, Any]) -> float:
    scores = []
    for episode in dataset:
        original = model.forward(make_model_batch(episode, config))
        selected = select_region_from_logits(original.get("inspection_logits"))
        critical = int(episode["ground_truth"]["critical_inspection_region"])
        decoy = add_decoy_patch(episode, config)
        decoy_out = model.forward(make_model_batch(decoy, config))
        decoy_selected = select_region_from_logits(decoy_out.get("inspection_logits"))
        scores.append(1.0 if selected == critical and decoy_selected == critical else 0.0)
    return mean(scores) if scores else 0.0


def static_decoy_suppression(model: Any, config: dict[str, Any], seed: int) -> float:
    batch = make_static_decoy_batch(config, seed)
    output = model.forward(batch)
    logits = np.asarray(output.get("inspection_logits"), dtype=np.float32).reshape(-1)
    max_logit = float(logits.max()) if logits.size else 0.0
    applicable = bool(output.get("structure", {}).get("applicable", False))
    return 1.0 if not applicable and max_logit <= 1e-6 else 0.0


def causal_endpoint_shift(model: Any, dataset: list[dict[str, Any]], config: dict[str, Any]) -> float:
    scores = []
    causal_types = {"forcefield", "budgeted_inspect", "collision_bounce"}
    for episode in dataset:
        if episode["ground_truth"].get("episode_type") not in causal_types:
            continue
        batch = make_model_batch(episode, config)
        result = model.intervene_structure(batch, {"type": "event_latent_perturbation"})
        if not result.get("applicable"):
            continue
        base_centers = extract_blob_centers(result["base_future_frames"][-1])
        changed_centers = extract_blob_centers(result["future_frames"][-1])
        if len(base_centers) == 0 or len(changed_centers) == 0:
            continue
        distances = []
        for center in base_centers:
            distances.append(float(np.min(np.linalg.norm(changed_centers - center, axis=1))))
        scores.append(float(np.clip(np.mean(distances) / 8.0, 0.0, 1.0)))
    return mean(scores) if scores else 0.0


def add_decoy_patch(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    decoy = copy.deepcopy(episode)
    past = np.asarray(decoy["past_frames"], dtype=np.float32).copy()
    critical = int(decoy["ground_truth"]["critical_inspection_region"])
    region = far_region(critical, int(config.get("env", {}).get("grid_size", 8)))
    ys, xs = region_id_to_slice(region, past.shape[1], int(config.get("env", {}).get("grid_size", 8)))
    past[:, ys, xs, :] = np.maximum(past[:, ys, xs, :], 0.08)
    decoy["past_frames"] = past
    decoy["frames"] = np.asarray(decoy["frames"], dtype=np.float32).copy()
    decoy["frames"][: len(past)] = past
    return decoy


def far_region(region: int, grid_size: int) -> int:
    gy, gx = divmod(int(region), grid_size)
    candidates = []
    for candidate in range(grid_size * grid_size):
        cy, cx = divmod(candidate, grid_size)
        candidates.append((abs(cy - gy) + abs(cx - gx), candidate))
    return max(candidates)[1]


def make_static_decoy_batch(config: dict[str, Any], seed: int) -> dict[str, Any]:
    env = config.get("env", {})
    frame_size = int(env.get("frame_size", 64))
    grid_size = int(env.get("grid_size", 8))
    past_horizon = int(env.get("past_frames", 8))
    future_horizon = int(env.get("future_frames", 12))
    rng = np.random.default_rng(seed)
    frame = np.zeros((frame_size, frame_size, 3), dtype=np.float32)
    draw_disk(frame, np.asarray([18.0, 28.0], dtype=np.float32), 4.0, np.asarray([0.95, 0.25, 0.25], dtype=np.float32))
    draw_disk(frame, np.asarray([46.0, 36.0], dtype=np.float32), 4.0, np.asarray([0.25, 0.65, 1.0], dtype=np.float32))
    decoy_region = int(rng.integers(0, grid_size * grid_size))
    ys, xs = region_id_to_slice(decoy_region, frame_size, grid_size)
    frame[ys, xs, :] = np.maximum(frame[ys, xs, :], 0.08)
    return {
        "past_frames": np.repeat(frame[None, ...], past_horizon, axis=0),
        "future_horizon": future_horizon,
        "frame_size": frame_size,
        "grid_size": grid_size,
    }


def write_hardening_outputs(metrics: dict[str, float]) -> None:
    Path("results").mkdir(exist_ok=True)
    with Path("results/flow_checkpoint_hardening_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/B_LINE_FLOW_CHECKPOINT_HARDENING.md").write_text(build_hardening_report(metrics), encoding="utf-8")


def build_hardening_report(metrics: dict[str, float]) -> str:
    verdict = (
        "flow_checkpoint_model survives this hardening pass."
        if metrics.get("hardening_score", 0.0) > 0.0
        else "flow_checkpoint_model does not survive this hardening pass."
    )
    return "\n".join(
        [
            "# B-Line Flow-Checkpoint Hardening",
            "",
            "## Purpose",
            "",
            "This hardening pass tests whether the current PLOS candidate is merely exploiting checkpoint priors, visible decoy patches, or weak intervention metrics.",
            "",
            "## Result",
            "",
            verdict,
            "",
            _markdown_table([metrics]),
            "",
            "## Checks",
            "",
            "- `candidate_gate_preserved`: the base PLOS candidate score remains nonzero.",
            "- `decoy_patch_stability`: adding a noncausal faint rectangle does not move the selected checkpoint.",
            "- `static_decoy_suppression`: static objects plus a decoy patch do not trigger operational structure.",
            "- `causal_endpoint_shift`: ablating the selected event/checkpoint changes future endpoints in causal episodes.",
            "",
            "## Boundary",
            "",
            "Passing this hardening pass still does not prove blank-slate emergence. The candidate remains a high-prior checkpoint substrate and needs further attacks.",
            "",
        ]
    )


def _markdown_table(rows: list[dict[str, float]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(f"{float(row.get(header, 0.0)):.3f}" for header in headers) + " |")
    return "\n".join(lines)
