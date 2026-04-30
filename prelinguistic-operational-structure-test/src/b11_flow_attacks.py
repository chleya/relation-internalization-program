from __future__ import annotations

import copy
import csv
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import yaml

from .data import generate_dataset, generate_episode
from .evaluate import evaluate_all
from .features import draw_disk, extract_blob_centers, point_to_region_id, rect_to_region_id, region_id_to_slice
from .inspect_policy import select_region_from_logits
from .model_io import make_model_batch
from .ood import make_all_ood
from .scoring import merge_scores
from .train import train_model


B11_SUMMARY_KEYS = [
    "candidate_gate_preserved",
    "dynamic_decoy_rejection",
    "decoy_selected_rate",
    "true_checkpoint_preservation",
    "delayed_checkpoint_accuracy",
    "immediate_saliency_error_rate",
    "delayed_endpoint_shift",
    "competing_checkpoint_choice",
    "fixed_priority_error_rate",
    "oracle_value_rank",
    "relocation_ood_stability",
    "corner_checkpoint_accuracy",
    "edge_checkpoint_accuracy",
    "rare_position_accuracy",
    "causal_deletion_shift",
    "visual_deletion_shift",
    "causal_over_visual_deletion_ratio",
    "anti_prior_survival",
    "prior_trap_selection_rate",
    "anti_prior_critical_accuracy",
    "causal_endpoint_shift",
    "b11_hardening_score",
]

B11_GATES = {
    "dynamic_decoy_rejection": 0.80,
    "delayed_checkpoint_accuracy": 0.75,
    "competing_checkpoint_choice": 0.75,
    "relocation_ood_stability": 0.75,
    "causal_over_visual_deletion_ratio": 1.50,
    "anti_prior_survival": 0.70,
    "causal_endpoint_shift": 0.25,
}

B11_WEIGHTS = {
    "dynamic_decoy_rejection": 0.15,
    "delayed_checkpoint_accuracy": 0.15,
    "competing_checkpoint_choice": 0.20,
    "relocation_ood_stability": 0.15,
    "causal_over_visual_deletion_ratio": 0.20,
    "anti_prior_survival": 0.15,
}

ANTI_PRIOR_TYPES = [
    "visible_occlusion_noncritical",
    "predicted_collision_prevented",
    "motion_midpoint_irrelevant",
    "force_looking_decoy",
]


def run_b11_flow_hardening(config: dict[str, Any], seed: int = 0) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """
    Run the B1.1 reviewer-hardening attacks against flow_checkpoint_model.

    This does not change PLOS v1 scoring. It first checks that the base
    candidate gate remains nonzero, then runs six separate attacks that look
    for checkpoint-prior, saliency, relocation, and weak-causality shortcuts.
    """

    base_config = resolve_base_config(config)
    b11 = b11_config(config)
    target_model = str(config.get("target_model", "flow_checkpoint_model"))
    if target_model != "flow_checkpoint_model":
        raise ValueError("B1.1 targets flow_checkpoint_model only")

    train_dataset = generate_dataset(base_config, "train", seed)
    test_dataset = generate_dataset(base_config, "test", seed + 10000)
    model = train_model(target_model, train_dataset, base_config)

    base_eval = evaluate_all(model, {"test": test_dataset, "ood": make_all_ood(base_config, seed + 20000)}, base_config)
    base_metrics = merge_scores(base_eval["behavior"], base_eval["structure"], base_eval["ood"], base_config["gates"])

    metrics: dict[str, float] = {
        "candidate_gate_preserved": 1.0 if base_metrics.get("plos_candidate_score", 0.0) > 0.0 else 0.0
    }
    records: list[dict[str, Any]] = []

    n = attack_episode_count(config)
    attack_flags = b11.get("attacks", {})

    if attack_flags.get("dynamic_decoy_checkpoint", True):
        episodes = [
            make_dynamic_decoy_episode(generate_episode(base_config, seed + 30000 + idx * 13), config, seed + 31000 + idx)
            for idx in range(n)
        ]
        attack_metrics, attack_records = _evaluate_dynamic_decoy_rejection(model, episodes, base_config, return_records=True)
        metrics.update(attack_metrics)
        records.extend(attack_records)

    if attack_flags.get("delayed_checkpoint", True):
        episodes = [make_delayed_checkpoint_episode(config, seed + 40000 + idx * 17) for idx in range(n)]
        attack_metrics, attack_records = _evaluate_delayed_checkpoint_accuracy(model, episodes, base_config, return_records=True)
        metrics.update(attack_metrics)
        records.extend(attack_records)

    if attack_flags.get("competing_checkpoints", True):
        episodes = [make_competing_checkpoints_episode(config, seed + 50000 + idx * 19) for idx in range(n)]
        attack_metrics, attack_records = _evaluate_competing_checkpoint_choice(model, episodes, base_config, return_records=True)
        metrics.update(attack_metrics)
        records.extend(attack_records)

    if attack_flags.get("checkpoint_relocation_ood", True):
        episodes = make_checkpoint_relocation_ood_dataset(config, seed + 60000)
        attack_metrics, attack_records = _evaluate_checkpoint_relocation_ood(model, episodes, base_config, return_records=True)
        metrics.update(attack_metrics)
        records.extend(attack_records)

    if attack_flags.get("causal_deletion_vs_visual_deletion", True):
        episodes = [
            generate_episode(base_config, seed + 70000 + idx * 23, ["forcefield", "budgeted_inspect", "collision_bounce"][idx % 3])
            for idx in range(n)
        ]
        attack_metrics, attack_records = _evaluate_causal_vs_visual_deletion(model, episodes, base_config, return_records=True)
        metrics.update(attack_metrics)
        records.extend(attack_records)

    if attack_flags.get("anti_prior_world", True):
        episodes = [
            make_anti_prior_episode(config, seed + 80000 + idx * 29, ANTI_PRIOR_TYPES[idx % len(ANTI_PRIOR_TYPES)])
            for idx in range(n)
        ]
        attack_metrics, attack_records = _evaluate_anti_prior_survival(model, episodes, base_config, return_records=True)
        metrics.update(attack_metrics)
        records.extend(attack_records)

    for key in B11_SUMMARY_KEYS:
        metrics.setdefault(key, 0.0)

    metrics["b11_hardening_score"] = compute_b11_hardening_score(metrics, b11.get("gates", B11_GATES))
    return metrics, records


def make_dynamic_decoy_episode(base_episode: dict[str, Any], config: dict[str, Any], seed: int) -> dict[str, Any]:
    """
    Add a moving noncausal decoy patch to past frames.

    The decoy changes only visual input. Ground-truth critical region and future
    frames remain unchanged.
    """

    episode = copy.deepcopy(base_episode)
    base_config = resolve_base_config(config)
    env = base_config.get("env", {})
    frame_size = int(env.get("frame_size", 64))
    grid_size = int(env.get("grid_size", 8))
    critical = int(episode["ground_truth"]["critical_inspection_region"])
    decoy_region = choose_far_region(critical, grid_size, seed)
    add_moving_decoy(episode, decoy_region, frame_size, grid_size, seed)
    episode["ground_truth"]["b11_attack"] = "dynamic_decoy_checkpoint"
    episode["ground_truth"]["b11_decoy_region"] = int(decoy_region)
    episode["ground_truth"]["b11_true_critical_region"] = int(critical)
    return episode


def make_delayed_checkpoint_episode(config: dict[str, Any], seed: int, delay: int = 4) -> dict[str, Any]:
    """
    Generate an episode whose B1.1 causal checkpoint is only visible by rollout.

    The evaluator marks a future trajectory checkpoint as critical and inserts a
    visually salient immediate decoy into the observed prefix.
    """

    base_config = resolve_base_config(config)
    env = base_config.get("env", {})
    frame_size = int(env.get("frame_size", 64))
    grid_size = int(env.get("grid_size", 8))
    past_frames = int(env.get("past_frames", 8))
    episode = generate_episode(base_config, seed, "forcefield")
    positions = np.asarray(episode["ground_truth"]["true_positions"], dtype=np.float32)
    causal_time = min(past_frames + int(delay), len(positions) - 1)
    critical = int(point_to_region_id(positions[causal_time, 0], frame_size, grid_size))
    old_critical = int(episode["ground_truth"]["critical_inspection_region"])
    immediate_region = choose_far_region(critical, grid_size, seed + 1)
    add_moving_decoy(episode, immediate_region, frame_size, grid_size, seed + 2, radius=3.0, intensity=0.26)
    episode["ground_truth"]["critical_inspection_region"] = critical
    episode["ground_truth"]["b11_attack"] = "delayed_checkpoint"
    episode["ground_truth"]["b11_delayed_causal_time"] = int(causal_time)
    episode["ground_truth"]["b11_delayed_delay"] = int(delay)
    episode["ground_truth"]["b11_immediate_saliency_region"] = int(immediate_region)
    episode["ground_truth"]["b11_original_critical_region"] = int(old_critical)
    episode["ground_truth"]["b11_true_critical_region"] = int(critical)
    return episode


def make_competing_checkpoints_episode(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """
    Generate an episode with multiple plausible checkpoint regions.
    """

    base_config = resolve_base_config(config)
    env = base_config.get("env", {})
    frame_size = int(env.get("frame_size", 64))
    grid_size = int(env.get("grid_size", 8))
    episode = generate_episode(base_config, seed, "budgeted_inspect")
    gt = episode["ground_truth"]
    critical = int(gt["critical_inspection_region"])
    positions = np.asarray(gt["true_positions"], dtype=np.float32)
    midpoint_region = int(point_to_region_id(positions[min(len(positions) - 1, int(env.get("past_frames", 8)) - 1)].mean(axis=0), frame_size, grid_size))
    occluder_region = int(rect_to_region_id(gt["occluder"], frame_size, grid_size))
    event_points = gt.get("event_points", [])
    collision_like = int(point_to_region_id(event_points[0], frame_size, grid_size)) if event_points else midpoint_region
    decoy_region = choose_far_region(critical, grid_size, seed + 1)
    add_moving_decoy(episode, decoy_region, frame_size, grid_size, seed + 2, intensity=0.24)
    candidates = _unique_regions([critical, occluder_region, collision_like, midpoint_region, decoy_region])
    values = {region: 0.10 for region in candidates}
    values[critical] = 1.00
    for region, value in [(collision_like, 0.62), (occluder_region, 0.50), (midpoint_region, 0.35), (decoy_region, 0.20)]:
        if region != critical:
            values[region] = max(values.get(region, 0.0), value)
    fixed_priority_region = next((region for region in [occluder_region, collision_like, critical, midpoint_region, decoy_region] if region in candidates), critical)
    gt["b11_attack"] = "competing_checkpoints"
    gt["b11_candidate_regions"] = candidates
    gt["b11_candidate_values"] = {str(region): float(value) for region, value in values.items()}
    gt["b11_oracle_best_region"] = int(max(values, key=values.get))
    gt["b11_fixed_priority_region"] = int(fixed_priority_region)
    gt["b11_decoy_region"] = int(decoy_region)
    gt["b11_true_critical_region"] = int(gt["b11_oracle_best_region"])
    gt["critical_inspection_region"] = int(gt["b11_oracle_best_region"])
    return episode


def compute_checkpoint_information_values(episode: dict[str, Any], config: dict[str, Any]) -> dict[int, float]:
    gt = episode["ground_truth"]
    if "b11_candidate_values" in gt:
        return {int(key): float(value) for key, value in gt["b11_candidate_values"].items()}
    critical = int(gt["critical_inspection_region"])
    decoy = int(gt.get("b11_decoy_region", choose_far_region(critical, int(resolve_base_config(config).get("env", {}).get("grid_size", 8)), 0)))
    return {critical: 1.0, decoy: 0.1}


def make_checkpoint_relocation_ood_dataset(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """
    Generate OOD episodes with causal checkpoints in edge/corner-like regions.
    """

    base_config = resolve_base_config(config)
    n = attack_episode_count(config)
    episodes: list[dict[str, Any]] = []
    env = dict(base_config.get("env", {}))
    env["forcefield_ood"] = True
    ood_config = {**base_config, "env": env}
    grid_size = int(env.get("grid_size", 8))
    for idx in range(n):
        episode = generate_episode(ood_config, seed + idx * 31, "forcefield")
        critical = int(episode["ground_truth"]["critical_inspection_region"])
        kind = rare_position_kind(critical, grid_size)
        episode["ground_truth"]["b11_attack"] = "checkpoint_relocation_ood"
        episode["ground_truth"]["b11_relocation_kind"] = kind
        episode["ground_truth"]["b11_true_critical_region"] = critical
        episodes.append(episode)
    return episodes


def apply_causal_checkpoint_deletion(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """
    Mask the true causal checkpoint region from the observed prefix.
    """

    altered = copy.deepcopy(episode)
    critical = int(altered["ground_truth"].get("b11_true_critical_region", altered["ground_truth"]["critical_inspection_region"]))
    mask_region_in_past(altered, critical, config, fill=0.0)
    altered["ground_truth"]["b11_deletion_type"] = "causal"
    altered["ground_truth"]["b11_deleted_region"] = int(critical)
    return altered


def apply_visual_decoy_deletion(episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """
    Mask a visually salient but noncausal region from the observed prefix.
    """

    altered = copy.deepcopy(episode)
    gt = altered["ground_truth"]
    base_config = resolve_base_config(config)
    grid_size = int(base_config.get("env", {}).get("grid_size", 8))
    critical = int(gt.get("b11_true_critical_region", gt["critical_inspection_region"]))
    region = int(gt.get("b11_decoy_region", gt.get("b11_trap_region", choose_far_region(critical, grid_size, 7))))
    mask_region_in_past(altered, region, config, fill=0.0)
    gt["b11_deletion_type"] = "visual_decoy"
    gt["b11_deleted_region"] = int(region)
    return altered


def endpoint_shift(base_future: Any, altered_future: Any) -> float:
    """
    Compute normalized final-endpoint shift between two predicted rollouts.
    """

    base = np.asarray(base_future, dtype=np.float32)
    altered = np.asarray(altered_future, dtype=np.float32)
    if base.size == 0 or altered.size == 0:
        return 0.0
    base_centers = extract_blob_centers(base[-1])
    altered_centers = extract_blob_centers(altered[-1])
    if len(base_centers) and len(altered_centers):
        distances = [float(np.min(np.linalg.norm(altered_centers - center, axis=1))) for center in base_centers]
        return float(np.clip(np.mean(distances) / 8.0, 0.0, 1.0))
    return float(np.clip(np.mean(np.abs(base[-1] - altered[-1])) * 4.0, 0.0, 1.0))


def make_anti_prior_episode(config: dict[str, Any], seed: int, anti_type: str) -> dict[str, Any]:
    """
    Generate episodes where the default checkpoint prior is misleading.
    """

    base_config = resolve_base_config(config)
    env = base_config.get("env", {})
    frame_size = int(env.get("frame_size", 64))
    grid_size = int(env.get("grid_size", 8))
    if anti_type == "force_looking_decoy":
        episode = generate_episode(base_config, seed, "occlusion_crossing")
    else:
        episode = generate_episode(base_config, seed, "forcefield")

    gt = episode["ground_truth"]
    critical = int(gt["critical_inspection_region"])
    if anti_type == "visible_occlusion_noncritical":
        trap = choose_far_region(critical, grid_size, seed + 1)
        add_occluder_decoy(episode, trap, frame_size, grid_size)
    elif anti_type == "predicted_collision_prevented":
        positions = np.asarray(gt["true_positions"], dtype=np.float32)
        trap = int(point_to_region_id(positions[min(len(positions) - 1, int(env.get("past_frames", 8)) - 1)].mean(axis=0), frame_size, grid_size))
        if trap == critical:
            trap = choose_far_region(critical, grid_size, seed + 2)
        add_moving_decoy(episode, trap, frame_size, grid_size, seed + 3, radius=3.0, intensity=0.25)
    elif anti_type == "motion_midpoint_irrelevant":
        positions = np.asarray(gt["true_positions"], dtype=np.float32)
        trap = int(point_to_region_id(positions[int(env.get("past_frames", 8)) - 1].mean(axis=0), frame_size, grid_size))
        if trap == critical:
            trap = choose_far_region(critical, grid_size, seed + 4)
    else:
        trap = choose_far_region(critical, grid_size, seed + 5)
        add_moving_decoy(episode, trap, frame_size, grid_size, seed + 6, radius=3.0, intensity=0.28)
        critical = int(rect_to_region_id(gt.get("occluder", (24.0, 20.0, 40.0, 44.0)), frame_size, grid_size))
        gt["critical_inspection_region"] = critical

    gt["b11_attack"] = "anti_prior_world"
    gt["b11_anti_type"] = str(anti_type)
    gt["b11_trap_region"] = int(trap)
    gt["b11_true_critical_region"] = int(critical)
    return episode


def evaluate_dynamic_decoy_rejection(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    return _evaluate_dynamic_decoy_rejection(model, episodes, config, return_records=False)[0]


def evaluate_delayed_checkpoint_accuracy(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    return _evaluate_delayed_checkpoint_accuracy(model, episodes, config, return_records=False)[0]


def evaluate_competing_checkpoint_choice(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    return _evaluate_competing_checkpoint_choice(model, episodes, config, return_records=False)[0]


def evaluate_checkpoint_relocation_ood(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    return _evaluate_checkpoint_relocation_ood(model, episodes, config, return_records=False)[0]


def evaluate_causal_vs_visual_deletion(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    return _evaluate_causal_vs_visual_deletion(model, episodes, config, return_records=False)[0]


def evaluate_anti_prior_survival(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    return _evaluate_anti_prior_survival(model, episodes, config, return_records=False)[0]


def compute_b11_hardening_score(metrics: dict[str, float], gates: dict[str, float]) -> float:
    gates = {**B11_GATES, **(gates or {})}
    if float(metrics.get("candidate_gate_preserved", 0.0)) < 1.0:
        return 0.0
    required = [
        "dynamic_decoy_rejection",
        "delayed_checkpoint_accuracy",
        "competing_checkpoint_choice",
        "relocation_ood_stability",
        "causal_over_visual_deletion_ratio",
        "anti_prior_survival",
        "causal_endpoint_shift",
    ]
    if any(float(metrics.get(key, 0.0)) < float(gates.get(key, 0.0)) for key in required):
        return 0.0

    total = 0.0
    for key, weight in B11_WEIGHTS.items():
        value = float(metrics.get(key, 0.0))
        if key == "causal_over_visual_deletion_ratio":
            value = min(value / float(gates["causal_over_visual_deletion_ratio"]), 1.0)
        total += weight * min(max(value, 0.0), 1.0)
    return float(total)


def write_b11_outputs(metrics: dict[str, float], records: list[dict[str, Any]]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    summary_path = Path("results/b11_flow_checkpoint_hardening_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=B11_SUMMARY_KEYS)
        writer.writeheader()
        writer.writerow({key: float(metrics.get(key, 0.0)) for key in B11_SUMMARY_KEYS})

    records_path = Path("results/b11_flow_checkpoint_records.csv")
    fieldnames = sorted({key for record in records for key in record}) if records else ["attack"]
    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    Path("reports/B1_1_FLOW_CHECKPOINT_HARDENING.md").write_text(build_b11_hardening_report(metrics), encoding="utf-8")
    Path("reports/B1_1_FLOW_CHECKPOINT_SELF_AUDIT.md").write_text(build_b11_self_audit_report(), encoding="utf-8")


def resolve_base_config(config: dict[str, Any]) -> dict[str, Any]:
    if "env" in config and "gates" in config:
        return config
    base_path = Path(str(config.get("base_config", "configs/sweep.yaml")))
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    with base_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def b11_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get(
        "b11",
        {
            "n_attack_episodes": 32,
            "attacks": {
                "dynamic_decoy_checkpoint": True,
                "delayed_checkpoint": True,
                "competing_checkpoints": True,
                "checkpoint_relocation_ood": True,
                "causal_deletion_vs_visual_deletion": True,
                "anti_prior_world": True,
            },
            "gates": dict(B11_GATES),
        },
    )


def attack_episode_count(config: dict[str, Any]) -> int:
    b11 = b11_config(config)
    declared = int(b11.get("n_attack_episodes", 32))
    runtime_cap = b11.get("max_attack_episodes")
    if runtime_cap is not None:
        declared = min(declared, int(runtime_cap))
    return max(1, declared)


def add_moving_decoy(
    episode: dict[str, Any],
    region: int,
    frame_size: int,
    grid_size: int,
    seed: int,
    radius: float = 2.4,
    intensity: float = 0.24,
) -> None:
    rng = np.random.default_rng(seed)
    past = np.asarray(episode["past_frames"], dtype=np.float32).copy()
    center = region_center(region, frame_size, grid_size)
    direction = rng.normal(0.0, 1.0, size=2).astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / norm
    color = np.asarray([intensity, intensity, intensity], dtype=np.float32)
    for t in range(len(past)):
        offset = direction * (float(t) - (len(past) - 1) / 2.0) * 0.65
        draw_disk(past[t], center + offset, radius, color)
    episode["past_frames"] = past
    frames = np.asarray(episode["frames"], dtype=np.float32).copy()
    frames[: len(past)] = past
    episode["frames"] = frames
    episode["ground_truth"]["b11_decoy_region"] = int(region)


def add_occluder_decoy(episode: dict[str, Any], region: int, frame_size: int, grid_size: int) -> None:
    past = np.asarray(episode["past_frames"], dtype=np.float32).copy()
    ys, xs = region_id_to_slice(region, frame_size, grid_size)
    past[:, ys, xs, :] = np.maximum(past[:, ys, xs, :], 0.08)
    episode["past_frames"] = past
    frames = np.asarray(episode["frames"], dtype=np.float32).copy()
    frames[: len(past)] = past
    episode["frames"] = frames


def mask_region_in_past(episode: dict[str, Any], region: int, config: dict[str, Any], fill: float = 0.0) -> None:
    base_config = resolve_base_config(config)
    env = base_config.get("env", {})
    frame_size = int(env.get("frame_size", 64))
    grid_size = int(env.get("grid_size", 8))
    past = np.asarray(episode["past_frames"], dtype=np.float32).copy()
    ys, xs = region_id_to_slice(region, frame_size, grid_size)
    past[:, ys, xs, :] = fill
    episode["past_frames"] = past
    frames = np.asarray(episode["frames"], dtype=np.float32).copy()
    frames[: len(past)] = past
    episode["frames"] = frames


def choose_far_region(region: int, grid_size: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    gy, gx = divmod(int(region), grid_size)
    candidates: list[tuple[int, int]] = []
    for candidate in range(grid_size * grid_size):
        cy, cx = divmod(candidate, grid_size)
        dist = abs(cy - gy) + abs(cx - gx)
        candidates.append((dist, candidate))
    max_dist = max(dist for dist, _ in candidates)
    far = [candidate for dist, candidate in candidates if dist == max_dist]
    return int(far[int(rng.integers(0, len(far)))])


def region_center(region: int, frame_size: int, grid_size: int) -> np.ndarray:
    ys, xs = region_id_to_slice(region, frame_size, grid_size)
    return np.asarray([(xs.start + xs.stop - 1) / 2.0, (ys.start + ys.stop - 1) / 2.0], dtype=np.float32)


def rare_position_kind(region: int, grid_size: int) -> str:
    gy, gx = divmod(int(region), grid_size)
    if gy in {0, grid_size - 1} and gx in {0, grid_size - 1}:
        return "corner"
    if gy in {0, grid_size - 1} or gx in {0, grid_size - 1}:
        return "edge"
    return "rare_interior"


def _evaluate_dynamic_decoy_rejection(
    model: Any, episodes: list[dict[str, Any]], config: dict[str, Any], return_records: bool
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[float] = []
    decoy_hits: list[float] = []
    true_preserved: list[float] = []
    records: list[dict[str, Any]] = []
    for idx, episode in enumerate(episodes):
        if "b11_decoy_region" not in episode.get("ground_truth", {}):
            episode = make_dynamic_decoy_episode(episode, config, idx)
        base_episode = apply_visual_decoy_deletion(episode, config)
        base_selected = selected_checkpoint(model, base_episode, config)
        selected = selected_checkpoint(model, episode, config)
        true_region = int(episode["ground_truth"]["b11_true_critical_region"])
        decoy_region = int(episode["ground_truth"]["b11_decoy_region"])
        is_true = int(base_selected == true_region and selected == true_region)
        is_decoy = int(selected == decoy_region)
        rows.append(float(is_true))
        decoy_hits.append(float(is_decoy))
        true_preserved.append(float(is_true and not is_decoy))
        records.append(
            {
                "attack": "dynamic_decoy_checkpoint",
                "index": idx,
                "base_selected_region": base_selected,
                "selected_region": selected,
                "true_region": true_region,
                "decoy_region": decoy_region,
                "pass": is_true,
                "decoy_selected": is_decoy,
            }
        )
    metrics = {
        "dynamic_decoy_rejection": _mean(rows),
        "decoy_selected_rate": _mean(decoy_hits),
        "true_checkpoint_preservation": _mean(true_preserved),
    }
    return metrics, records if return_records else []


def _evaluate_delayed_checkpoint_accuracy(
    model: Any, episodes: list[dict[str, Any]], config: dict[str, Any], return_records: bool
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[float] = []
    saliency_errors: list[float] = []
    shifts: list[float] = []
    records: list[dict[str, Any]] = []
    for idx, episode in enumerate(episodes):
        selected = selected_checkpoint(model, episode, config)
        true_region = int(episode["ground_truth"]["b11_true_critical_region"])
        saliency_region = int(episode["ground_truth"]["b11_immediate_saliency_region"])
        causal_deleted = apply_causal_checkpoint_deletion(episode, config)
        base_out = model.forward(make_model_batch(episode, config))
        causal_out = model.forward(make_model_batch(causal_deleted, config))
        shift = endpoint_shift(base_out["future_frames"], causal_out["future_frames"])
        hit = int(selected == true_region)
        saliency_error = int(selected == saliency_region)
        rows.append(float(hit))
        saliency_errors.append(float(saliency_error))
        shifts.append(float(shift))
        records.append(
            {
                "attack": "delayed_checkpoint",
                "index": idx,
                "selected_region": selected,
                "true_region": true_region,
                "immediate_saliency_region": saliency_region,
                "pass": hit,
                "immediate_saliency_error": saliency_error,
                "endpoint_shift": shift,
            }
        )
    metrics = {
        "delayed_checkpoint_accuracy": _mean(rows),
        "immediate_saliency_error_rate": _mean(saliency_errors),
        "delayed_endpoint_shift": _mean(shifts),
    }
    return metrics, records if return_records else []


def _evaluate_competing_checkpoint_choice(
    model: Any, episodes: list[dict[str, Any]], config: dict[str, Any], return_records: bool
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    hits: list[float] = []
    fixed_errors: list[float] = []
    ranks: list[float] = []
    records: list[dict[str, Any]] = []
    for idx, episode in enumerate(episodes):
        selected = selected_checkpoint(model, episode, config)
        values = compute_checkpoint_information_values(episode, config)
        oracle = int(max(values, key=values.get))
        fixed_region = int(episode["ground_truth"].get("b11_fixed_priority_region", oracle))
        ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
        rank = next((rank_idx + 1 for rank_idx, (region, _) in enumerate(ordered) if int(region) == selected), len(ordered) + 1)
        hit = int(selected == oracle)
        fixed_error = int(selected == fixed_region and fixed_region != oracle)
        hits.append(float(hit))
        fixed_errors.append(float(fixed_error))
        ranks.append(float(rank))
        records.append(
            {
                "attack": "competing_checkpoints",
                "index": idx,
                "selected_region": selected,
                "oracle_region": oracle,
                "fixed_priority_region": fixed_region,
                "pass": hit,
                "fixed_priority_error": fixed_error,
                "oracle_value_rank": rank,
            }
        )
    metrics = {
        "competing_checkpoint_choice": _mean(hits),
        "fixed_priority_error_rate": _mean(fixed_errors),
        "oracle_value_rank": _mean(ranks),
    }
    return metrics, records if return_records else []


def _evaluate_checkpoint_relocation_ood(
    model: Any, episodes: list[dict[str, Any]], config: dict[str, Any], return_records: bool
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    hits: list[float] = []
    corner_hits: list[float] = []
    edge_hits: list[float] = []
    rare_hits: list[float] = []
    records: list[dict[str, Any]] = []
    for idx, episode in enumerate(episodes):
        selected = selected_checkpoint(model, episode, config)
        true_region = int(episode["ground_truth"]["b11_true_critical_region"])
        kind = str(episode["ground_truth"].get("b11_relocation_kind", "rare_interior"))
        hit = float(selected == true_region)
        hits.append(hit)
        if kind == "corner":
            corner_hits.append(hit)
        elif kind == "edge":
            edge_hits.append(hit)
        else:
            rare_hits.append(hit)
        records.append(
            {
                "attack": "checkpoint_relocation_ood",
                "index": idx,
                "selected_region": selected,
                "true_region": true_region,
                "relocation_kind": kind,
                "pass": int(hit),
            }
        )
    metrics = {
        "relocation_ood_stability": _mean(hits),
        "corner_checkpoint_accuracy": _mean(corner_hits),
        "edge_checkpoint_accuracy": _mean(edge_hits),
        "rare_position_accuracy": _mean(rare_hits),
    }
    return metrics, records if return_records else []


def _evaluate_causal_vs_visual_deletion(
    model: Any, episodes: list[dict[str, Any]], config: dict[str, Any], return_records: bool
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    causal_shifts: list[float] = []
    visual_shifts: list[float] = []
    records: list[dict[str, Any]] = []
    for idx, episode in enumerate(episodes):
        true_region = int(episode["ground_truth"]["critical_inspection_region"])
        episode["ground_truth"]["b11_true_critical_region"] = true_region
        visual_base = make_dynamic_decoy_episode(episode, config, seed=90000 + idx)
        causal_deleted = apply_causal_checkpoint_deletion(episode, config)
        visual_deleted = apply_visual_decoy_deletion(visual_base, config)

        base_out = model.forward(make_model_batch(episode, config))
        causal_out = model.forward(make_model_batch(causal_deleted, config))
        visual_base_out = model.forward(make_model_batch(visual_base, config))
        visual_deleted_out = model.forward(make_model_batch(visual_deleted, config))
        causal_shift = endpoint_shift(base_out["future_frames"], causal_out["future_frames"])
        visual_shift = endpoint_shift(visual_base_out["future_frames"], visual_deleted_out["future_frames"])
        causal_shifts.append(causal_shift)
        visual_shifts.append(visual_shift)
        records.append(
            {
                "attack": "causal_deletion_vs_visual_deletion",
                "index": idx,
                "causal_region": true_region,
                "visual_region": int(visual_base["ground_truth"]["b11_decoy_region"]),
                "causal_shift": causal_shift,
                "visual_shift": visual_shift,
                "ratio": causal_shift / (visual_shift + 1e-6),
            }
        )
    causal = _mean(causal_shifts)
    visual = _mean(visual_shifts)
    ratio = causal / (visual + 1e-6)
    metrics = {
        "causal_deletion_shift": causal,
        "visual_deletion_shift": visual,
        "causal_over_visual_deletion_ratio": float(ratio),
        "causal_endpoint_shift": causal,
    }
    return metrics, records if return_records else []


def _evaluate_anti_prior_survival(
    model: Any, episodes: list[dict[str, Any]], config: dict[str, Any], return_records: bool
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    hits: list[float] = []
    trap_hits: list[float] = []
    records: list[dict[str, Any]] = []
    for idx, episode in enumerate(episodes):
        selected = selected_checkpoint(model, episode, config)
        true_region = int(episode["ground_truth"]["b11_true_critical_region"])
        trap_region = int(episode["ground_truth"]["b11_trap_region"])
        hit = int(selected == true_region)
        trap_hit = int(selected == trap_region)
        hits.append(float(hit))
        trap_hits.append(float(trap_hit))
        records.append(
            {
                "attack": "anti_prior_world",
                "anti_type": episode["ground_truth"].get("b11_anti_type", ""),
                "index": idx,
                "selected_region": selected,
                "true_region": true_region,
                "trap_region": trap_region,
                "pass": hit,
                "trap_selected": trap_hit,
            }
        )
    metrics = {
        "anti_prior_survival": _mean(hits),
        "prior_trap_selection_rate": _mean(trap_hits),
        "anti_prior_critical_accuracy": _mean(hits),
    }
    return metrics, records if return_records else []


def selected_checkpoint(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> int:
    output = model.forward(make_model_batch(episode, config))
    return int(select_region_from_logits(output.get("inspection_logits")))


def _unique_regions(regions: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for region in regions:
        value = int(region)
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def build_b11_hardening_report(metrics: dict[str, float]) -> str:
    verdict = (
        "Under B1.1 reviewer hardening, the high-prior flow_checkpoint_model remains a PLOS candidate."
        if metrics.get("b11_hardening_score", 0.0) > 0.0
        else "The current flow_checkpoint_model PLOS v1 pass is not robust under B1.1 attacks."
    )
    return "\n".join(
        [
            "# B1.1 Flow-Checkpoint Reviewer Hardening",
            "",
            "## 1. Purpose",
            "",
            "Attack the current only PLOS candidate: flow_checkpoint_model.",
            "",
            "## 2. Why This Is Needed",
            "",
            "flow_checkpoint_model passed PLOS v1 but has a high checkpoint-selection prior.",
            "",
            "## 3. Attack Set",
            "",
            "- dynamic decoy checkpoint",
            "- delayed checkpoint",
            "- competing checkpoints",
            "- checkpoint relocation OOD",
            "- causal deletion vs visual deletion",
            "- anti-prior world",
            "",
            "## 4. Results Table",
            "",
            _markdown_table(metrics),
            "",
            "## 5. Interpretation",
            "",
            verdict,
            "",
            "If passed, the flow-checkpoint substrate survives stronger reviewer attacks, but remains high-prior.",
            "If failed, the prior PLOS pass was likely checkpoint-prior or saliency dependent.",
            "",
            "## 6. Claim Boundary",
            "",
            "Do not claim blank-slate emergence.",
            "Do not claim general physical reasoning.",
            "Do not claim real-world cognition.",
            "Do not claim language-free intelligence solved.",
            "",
        ]
    )


def build_b11_self_audit_report() -> str:
    return "\n".join(
        [
            "# B1.1 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Attacks the only PLOS candidate.",
            "- Tests dynamic decoy robustness.",
            "- Tests delayed causality.",
            "- Tests competing checkpoint selection.",
            "- Tests relocation OOD.",
            "- Compares causal deletion vs visual deletion.",
            "- Tests anti-prior worlds.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Still a toy 64x64 world.",
            "- flow_checkpoint_model still contains strong priors.",
            "- Hardening attacks are hand-designed.",
            "- Passing does not imply blank-slate emergence.",
            "- Causal deletion may create out-of-distribution inputs.",
            "- Endpoint shift may under-measure relational changes.",
            "- Competing checkpoint oracle may encode evaluator assumptions.",
            "",
            "## False Positive Risks",
            "",
            "- Model may exploit attack generator regularities.",
            "- Delayed checkpoint may still be visually inferable.",
            "- Decoys may be too weak.",
            "- Anti-prior episodes may be too narrow.",
            "- Causal deletion may affect visual continuity rather than causal structure.",
            "",
            "## Required Failure Checks",
            "",
            "1. dynamic decoy moves selection",
            "2. delayed checkpoint fails",
            "3. fixed priority wins over information value",
            "4. relocation OOD collapses",
            "5. causal deletion not stronger than visual deletion",
            "6. anti-prior trap selected",
            "7. hardening pass with tiny causal endpoint shift",
            "",
        ]
    )


def _markdown_table(metrics: dict[str, float]) -> str:
    lines = ["| metric | value | gate | pass |", "| --- | ---: | ---: | --- |"]
    for key in B11_SUMMARY_KEYS:
        if key not in metrics:
            continue
        gate = B11_GATES.get(key)
        value = float(metrics.get(key, 0.0))
        if gate is None:
            lines.append(f"| `{key}` | {value:.3f} |  |  |")
        else:
            passed = "pass" if value >= gate else "fail"
            lines.append(f"| `{key}` | {value:.3f} | {gate:.3f} | {passed} |")
    return "\n".join(lines)
