from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .b2_delayed_env import choose_far_region, draw_trace, make_delayed_checkpoint_episode
from .b2_delayed_metrics import delayed_endpoint_shift
from .b21_trace_interventions import (
    add_false_trace_to_episode,
    apply_matched_non_trace_deletion,
    apply_trace_compression,
    apply_true_trace_deletion,
    swap_trace_regions_between_episodes,
    trace_family,
)
from .b21_trace_metrics import (
    B21_GATES,
    B21_SUMMARY_KEYS,
    b21_trace_hardening_score,
    false_trace_rejection,
    mean_or_zero,
    multi_source_conflict_resolution,
    noisy_trace_robustness,
    trace_compression_survival,
    trace_deletion_specificity_ratio,
    trace_length_extrapolation,
    trace_swap_sensitivity,
)
from .features import region_id_to_slice
from .inspect_policy import select_region_from_logits
from .model_io import make_model_batch
from .train import train_model


RECORD_KEYS = [
    "model",
    "seed",
    "attack",
    "episode_id",
    "delay",
    "true_trace_region",
    "false_trace_region",
    "predicted_region",
    "selected_trace_source",
    "intervention_type",
    "base_prediction",
    "intervened_prediction",
    "endpoint_shift",
    "correct",
    "gate_pass",
]


def run_b21_trace_hardening(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime_config = b21_runtime_config(config)
    b21 = b21_config(config)
    n = attack_episode_count(config)
    train_delays = [int(value) for value in b21.get("train_delays", [2, 4, 6])]
    extrapolation_delays = [int(value) for value in b21.get("extrapolation_delays", [8, 10])]
    target_models = config.get(
        "target_models",
        ["recurrent_flow_checkpoint_model", "field_memory_model", "schema_memory_model"],
    )

    false_episodes = [
        make_false_delayed_trace_episode(runtime_config, seed + 10000 + idx * 13, train_delays[idx % len(train_delays)])
        for idx in range(n)
    ]
    swap_pairs = [
        make_trace_swap_pair(runtime_config, seed + 20000 + idx * 17, train_delays[idx % len(train_delays)])
        for idx in range(n)
    ]
    deletion_episodes = [
        make_delayed_checkpoint_episode(runtime_config, seed + 30000 + idx * 19, train_delays[idx % len(train_delays)])
        for idx in range(n)
    ]
    conflict_episodes = [
        make_multi_source_trace_conflict_episode(runtime_config, seed + 40000 + idx * 23, "trace_conflict")
        for idx in range(n)
    ]
    noise_levels = [0.10, 0.25, 0.45]
    noisy_episodes = [
        make_noisy_delayed_trace_episode(
            runtime_config,
            seed + 50000 + idx * 29,
            train_delays[idx % len(train_delays)],
            noise_levels[idx % len(noise_levels)],
        )
        for idx in range(n)
    ]
    extrapolation_episodes = make_trace_length_extrapolation_dataset(runtime_config, seed + 60000, extrapolation_delays, n)
    compression_episodes = [
        make_delayed_checkpoint_episode(runtime_config, seed + 70000 + idx * 31, train_delays[idx % len(train_delays)])
        for idx in range(n)
    ]

    summary: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for model_name in target_models:
        model = train_model(str(model_name), [], runtime_config)
        metrics: dict[str, float] = {}
        model_records: list[dict[str, Any]] = []
        for evaluator, episodes in [
            (evaluate_false_trace_rejection, false_episodes),
            (evaluate_trace_deletion_specificity, deletion_episodes),
            (evaluate_multi_source_trace_conflict, conflict_episodes),
            (evaluate_noisy_trace_robustness, noisy_episodes),
            (evaluate_trace_length_extrapolation, extrapolation_episodes),
            (evaluate_trace_compression_pressure, compression_episodes),
        ]:
            attack_metrics, attack_records = evaluator(model, episodes, runtime_config)
            metrics.update(attack_metrics)
            model_records.extend(attack_records)
        swap_metrics, swap_records = evaluate_trace_swap_sensitivity(model, swap_pairs, runtime_config)
        metrics.update(swap_metrics)
        model_records.extend(swap_records)
        metrics["b21_trace_hardening_score"] = b21_trace_hardening_score(metrics, b21.get("gates", B21_GATES))

        row = {"model": str(model_name), "seed": int(seed)}
        for key in B21_SUMMARY_KEYS:
            if key not in {"model", "seed"}:
                row[key] = float(metrics.get(key, 0.0))
        summary.append(row)

        for record in model_records:
            enriched = {"model": str(model_name), "seed": int(seed)}
            enriched.update(record)
            records.append(enriched)
    return summary, records


def make_false_delayed_trace_episode(config: dict[str, Any], seed: int, delay: int) -> dict[str, Any]:
    episode = make_delayed_checkpoint_episode(config, seed, delay, saliency_decoy=True)
    env = config.get("env", {})
    grid_size = int(env.get("grid_size", 8))
    true_region = int(episode["ground_truth"]["true_delayed_checkpoint_region"])
    false_region = choose_far_region(true_region, grid_size, seed + 1)
    episode = add_false_trace_to_episode(episode, false_region, config, strength=0.72)
    episode["ground_truth"].update(
        {
            "episode_type": "b21_false_delayed_trace",
            "delay": int(delay),
            "true_trace_region": int(true_region),
            "false_trace_region": int(false_region),
            "critical_inspection_region": int(true_region),
            "false_trace_has_future_effect": False,
        }
    )
    return episode


def make_trace_swap_pair(config: dict[str, Any], seed: int, delay: int) -> tuple[dict[str, Any], dict[str, Any]]:
    episode_a = make_delayed_checkpoint_episode(config, seed, delay, saliency_decoy=True)
    episode_b = make_delayed_checkpoint_episode(config, seed + 991, delay, saliency_decoy=True)
    attempts = 0
    while (
        int(episode_a["ground_truth"]["true_delayed_checkpoint_region"])
        == int(episode_b["ground_truth"]["true_delayed_checkpoint_region"])
        and attempts < 8
    ):
        attempts += 1
        episode_b = make_delayed_checkpoint_episode(config, seed + 991 + attempts * 37, delay, saliency_decoy=True)
    for name, episode in {"a": episode_a, "b": episode_b}.items():
        episode["ground_truth"]["episode_type"] = "b21_trace_swap"
        episode["ground_truth"]["trace_swap_member"] = name
        episode["ground_truth"]["true_trace_region"] = int(episode["ground_truth"]["true_delayed_checkpoint_region"])
    return episode_a, episode_b


def apply_trace_swap(model: Any, batch_a: dict[str, Any], batch_b: dict[str, Any], trace_family_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if trace_family_name not in {"recurrent_flow_checkpoint", "field_memory", "schema_memory"}:
        return {"applicable": False}, {"applicable": False}
    return {
        "applicable": True,
        "base_output": model.forward(batch_a),
        "swapped_output": model.forward(batch_b),
    }, {
        "applicable": True,
        "base_output": model.forward(batch_b),
        "swapped_output": model.forward(batch_a),
    }


def make_multi_source_trace_conflict_episode(config: dict[str, Any], seed: int, conflict_type: str) -> dict[str, Any]:
    episode = make_false_delayed_trace_episode(config, seed, delay=4)
    true_region = int(episode["ground_truth"]["true_trace_region"])
    wrong_region = int(episode["ground_truth"]["false_trace_region"])
    episode["ground_truth"].update(
        {
            "episode_type": "b21_multi_source_trace_conflict",
            "conflict_type": conflict_type,
            "causal_trace_source": "true_trace",
            "wrong_trace_region": int(wrong_region),
            "true_trace_region": int(true_region),
        }
    )
    return episode


def inject_trace_conflict(model: Any, batch: dict[str, Any], conflict_spec: dict[str, Any]) -> dict[str, Any]:
    family = trace_family(model, batch)
    if family not in {"recurrent_flow_checkpoint", "field_memory", "schema_memory"}:
        return {"applicable": False, "reason": "unsupported_trace_family"}
    return {"applicable": True, "conflict_spec": dict(conflict_spec), "output": model.forward(batch)}


def make_noisy_delayed_trace_episode(config: dict[str, Any], seed: int, delay: int, noise_level: float) -> dict[str, Any]:
    episode = make_delayed_checkpoint_episode(config, seed, delay, saliency_decoy=True)
    env = config.get("env", {})
    frame_size = int(env.get("frame_size", 64))
    grid_size = int(env.get("grid_size", 8))
    rng = np.random.default_rng(seed + 5)
    true_region = int(episode["ground_truth"]["true_delayed_checkpoint_region"])
    control_region = choose_far_region(true_region, grid_size, seed + 6)
    for region in (true_region, control_region):
        ys, xs = region_id_to_slice(region, frame_size, grid_size)
        noise = rng.normal(0.0, float(noise_level) * 0.035, size=episode["past_frames"][:, ys, xs, :].shape).astype(np.float32)
        episode["past_frames"][:, ys, xs, :] = np.clip(episode["past_frames"][:, ys, xs, :] + noise, 0.0, 1.0)
    episode["frames"][: len(episode["past_frames"])] = episode["past_frames"]
    episode["ground_truth"].update(
        {
            "episode_type": "b21_noisy_delayed_trace",
            "delay": int(delay),
            "noise_level": float(noise_level),
            "true_trace_region": int(true_region),
            "noise_control_region": int(control_region),
        }
    )
    return episode


def make_trace_length_extrapolation_dataset(config: dict[str, Any], seed: int, delays: list[int], n: int | None = None) -> list[dict[str, Any]]:
    count = int(n or len(delays))
    episodes = []
    for idx in range(count):
        delay = int(delays[idx % len(delays)])
        episode = make_delayed_checkpoint_episode(config, seed + idx * 41, delay, saliency_decoy=True)
        episode["ground_truth"].update(
            {
                "episode_type": "b21_trace_length_extrapolation",
                "delay": delay,
                "true_trace_region": int(episode["ground_truth"]["true_delayed_checkpoint_region"]),
                "extrapolation_delay": delay,
            }
        )
        episodes.append(episode)
    return episodes


def evaluate_false_trace_rejection(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    false_hits = []
    true_hits = []
    records = []
    for idx, episode in enumerate(episodes):
        pred = predict_region(model, episode, config)
        true_region = int(episode["ground_truth"]["true_trace_region"])
        false_region = int(episode["ground_truth"]["false_trace_region"])
        correct = false_trace_rejection(pred, true_region, false_region)
        rows.append(correct)
        false_hits.append(1.0 if pred == false_region else 0.0)
        true_hits.append(1.0 if pred == true_region else 0.0)
        records.append(record_row("false_delayed_trace", idx, episode, pred, "false_trace" if pred == false_region else "true_or_other", correct))
    return {
        "false_trace_rejection": mean_or_zero(rows),
        "false_trace_selected_rate": mean_or_zero(false_hits),
        "true_trace_selected_rate": mean_or_zero(true_hits),
    }, records


def evaluate_trace_swap_sensitivity(
    model: Any, episode_pairs: list[tuple[dict[str, Any], dict[str, Any]]], config: dict[str, Any]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    shifts = []
    records = []
    family = trace_family(model)
    for idx, (episode_a, episode_b) in enumerate(episode_pairs):
        swapped_a, swapped_b = swap_trace_regions_between_episodes(episode_a, episode_b, config)
        base_a = model.forward(make_model_batch(episode_a, config))
        base_b = model.forward(make_model_batch(episode_b, config))
        out_a = model.forward(make_model_batch(swapped_a, config))
        out_b = model.forward(make_model_batch(swapped_b, config))
        base_region_a = int(select_region_from_logits(base_a["inspection_logits"]))
        base_region_b = int(select_region_from_logits(base_b["inspection_logits"]))
        pred_a = int(select_region_from_logits(out_a["inspection_logits"]))
        pred_b = int(select_region_from_logits(out_b["inspection_logits"]))
        expected_a = int(episode_b["ground_truth"]["true_trace_region"])
        expected_b = int(episode_a["ground_truth"]["true_trace_region"])
        score_a = trace_swap_sensitivity(base_region_a, pred_a, expected_a)
        score_b = trace_swap_sensitivity(base_region_b, pred_b, expected_b)
        shift_a = delayed_endpoint_shift(base_a["future_frames"], out_a["future_frames"])
        shift_b = delayed_endpoint_shift(base_b["future_frames"], out_b["future_frames"])
        rows.extend([score_a, score_b])
        shifts.extend([shift_a, shift_b])
        records.append(record_row("trace_swap", idx * 2, swapped_a, pred_a, family, score_a, base_region_a, pred_a, shift_a))
        records.append(record_row("trace_swap", idx * 2 + 1, swapped_b, pred_b, family, score_b, base_region_b, pred_b, shift_b))
    return {
        "trace_swap_sensitivity": mean_or_zero(rows),
        "trace_swap_endpoint_shift": mean_or_zero(shifts),
    }, records


def evaluate_trace_deletion_specificity(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    true_drops = []
    control_drops = []
    records = []
    family = trace_family(model)
    for idx, episode in enumerate(episodes):
        batch = make_model_batch(episode, config)
        true_result = apply_true_trace_deletion(model, batch, family)
        control_result = apply_matched_non_trace_deletion(model, batch, family)
        if not true_result.get("applicable") or not control_result.get("applicable"):
            records.append(record_row("trace_deletion_specificity", idx, episode, -1, family, 0.0, "", "", 0.0, "unsupported"))
            continue
        true_drop = delayed_endpoint_shift(true_result["base_future_frames"], true_result["future_frames"])
        control_drop = delayed_endpoint_shift(control_result["base_future_frames"], control_result["future_frames"])
        true_drops.append(true_drop)
        control_drops.append(control_drop)
        records.append(
            record_row(
                "trace_deletion_specificity",
                idx,
                episode,
                int(true_result.get("target_region", -1)),
                family,
                1.0,
                "",
                "",
                true_drop,
                str(true_result.get("intervention_type", "true_trace_deletion")),
            )
        )
    true_drop = mean_or_zero(true_drops)
    control_drop = mean_or_zero(control_drops)
    return {
        "true_trace_intervention_drop": true_drop,
        "matched_non_trace_drop": control_drop,
        "trace_deletion_specificity_ratio": trace_deletion_specificity_ratio(true_drop, control_drop),
        "non_trace_stability": float(np.clip(1.0 - control_drop, 0.0, 1.0)),
    }, records


def evaluate_multi_source_trace_conflict(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    wrong_hits = []
    records = []
    for idx, episode in enumerate(episodes):
        pred = predict_region(model, episode, config)
        true_region = int(episode["ground_truth"]["true_trace_region"])
        wrong_region = int(episode["ground_truth"]["wrong_trace_region"])
        score = multi_source_conflict_resolution(pred, true_region, wrong_region)
        rows.append(score)
        wrong_hits.append(1.0 if pred == wrong_region else 0.0)
        records.append(record_row("multi_source_trace_conflict", idx, episode, pred, "wrong_trace" if pred == wrong_region else "causal_trace", score))
    return {
        "multi_source_conflict_resolution": mean_or_zero(rows),
        "wrong_trace_follow_rate": mean_or_zero(wrong_hits),
        "causal_trace_priority": mean_or_zero(rows),
    }, records


def evaluate_noisy_trace_robustness(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    by_level: dict[str, list[float]] = {"mild": [], "medium": [], "strong": []}
    rows = []
    records = []
    for idx, episode in enumerate(episodes):
        pred = predict_region(model, episode, config)
        true_region = int(episode["ground_truth"]["true_trace_region"])
        correct = 1.0 if pred == true_region else 0.0
        level = noise_bucket(float(episode["ground_truth"]["noise_level"]))
        by_level[level].append(correct)
        rows.append(correct)
        records.append(record_row("noisy_delayed_trace", idx, episode, pred, level, correct))
    mild = mean_or_zero(by_level["mild"])
    medium = mean_or_zero(by_level["medium"])
    strong = mean_or_zero(by_level["strong"])
    return {
        "noisy_trace_robustness": noisy_trace_robustness(rows),
        "accuracy_under_mild_noise": mild,
        "accuracy_under_medium_noise": medium,
        "accuracy_under_strong_noise": strong,
        "noise_degradation_slope": float(mild - strong),
    }, records


def evaluate_trace_length_extrapolation(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    by_delay: dict[int, list[float]] = {}
    rows = []
    records = []
    for idx, episode in enumerate(episodes):
        pred = predict_region(model, episode, config)
        true_region = int(episode["ground_truth"]["true_trace_region"])
        delay = int(episode["ground_truth"]["delay"])
        correct = 1.0 if pred == true_region else 0.0
        rows.append(correct)
        by_delay.setdefault(delay, []).append(correct)
        records.append(record_row("trace_length_extrapolation", idx, episode, pred, f"delay_{delay}", correct))
    delay_8 = mean_or_zero(by_delay.get(8, []))
    delay_10 = mean_or_zero(by_delay.get(10, []))
    return {
        "trace_length_extrapolation": trace_length_extrapolation(rows),
        "delay_8_accuracy": delay_8,
        "delay_10_accuracy": delay_10,
        "long_delay_degradation": float(max(0.0, delay_8 - delay_10)),
    }, records


def evaluate_trace_compression_pressure(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    levels = [0.75, 0.50, 0.25]
    by_level: dict[float, list[float]] = {level: [] for level in levels}
    saliency_hits = []
    records = []
    family = trace_family(model)
    for idx, episode in enumerate(episodes):
        batch = make_model_batch(episode, config)
        true_region = int(episode["ground_truth"]["true_delayed_checkpoint_region"])
        saliency_region = int(episode["ground_truth"]["early_saliency_region"])
        for level in levels:
            result = apply_trace_compression(model, batch, level, family)
            if not result.get("applicable"):
                correct = 0.0
                pred = -1
            else:
                pred = int(select_region_from_logits(result["compressed_output"]["inspection_logits"]))
                correct = 1.0 if pred == true_region else 0.0
            by_level[level].append(correct)
            saliency_hits.append(1.0 if pred == saliency_region else 0.0)
            records.append(record_row("trace_compression_pressure", idx, episode, pred, f"compression_{level:.2f}", correct))
    acc075 = mean_or_zero(by_level[0.75])
    acc050 = mean_or_zero(by_level[0.50])
    acc025 = mean_or_zero(by_level[0.25])
    values = [acc075, acc050, acc025]
    return {
        "trace_compression_survival": trace_compression_survival(values),
        "accuracy_at_075": acc075,
        "accuracy_at_050": acc050,
        "accuracy_at_025": acc025,
        "causal_trace_retention_under_compression": trace_compression_survival(values),
        "saliency_retention_bias": mean_or_zero(saliency_hits),
    }, records


def predict_region(model: Any, episode: dict[str, Any], config: dict[str, Any]) -> int:
    output = model.forward(make_model_batch(episode, config))
    return int(select_region_from_logits(output.get("inspection_logits")))


def noise_bucket(level: float) -> str:
    if level <= 0.12:
        return "mild"
    if level <= 0.30:
        return "medium"
    return "strong"


def record_row(
    attack: str,
    idx: int,
    episode: dict[str, Any],
    pred: int,
    source: str,
    correct: float,
    base_prediction: Any = "",
    intervened_prediction: Any = "",
    endpoint_shift: float | str = "",
    intervention_type: str = "",
) -> dict[str, Any]:
    gt = episode["ground_truth"]
    true_region = gt.get("true_trace_region", gt.get("true_delayed_checkpoint_region", ""))
    false_region = gt.get("false_trace_region", gt.get("wrong_trace_region", ""))
    return {
        "attack": attack,
        "episode_id": idx,
        "delay": gt.get("delay", gt.get("extrapolation_delay", "")),
        "true_trace_region": true_region,
        "false_trace_region": false_region,
        "predicted_region": pred,
        "selected_trace_source": source,
        "intervention_type": intervention_type,
        "base_prediction": base_prediction,
        "intervened_prediction": intervened_prediction,
        "endpoint_shift": endpoint_shift,
        "correct": float(correct),
        "gate_pass": int(float(correct) > 0.0),
    }


def b21_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    b2 = resolve_b2_config(config)
    plos = resolve_plos_config(b2)
    b21 = b21_config(config)
    env = dict(plos.get("env", {}))
    for key in ("frame_size", "grid_size", "past_frames", "future_frames"):
        if key in b21:
            env[key] = b21[key]
    runtime = {**plos, "env": env}
    runtime["b2"] = {
        "n_train": int(b21.get("n_attack_episodes", 32)),
        "n_test": int(b21.get("n_attack_episodes", 32)),
        "n_ood": int(b21.get("n_attack_episodes", 32)),
        "frame_size": int(env.get("frame_size", 64)),
        "grid_size": int(env.get("grid_size", 8)),
        "past_frames": int(env.get("past_frames", 8)),
        "future_frames": int(env.get("future_frames", 12)),
        "delays": [int(value) for value in b21.get("train_delays", [2, 4, 6])],
        "heldout_delays": [int(value) for value in b21.get("heldout_delays", [3, 5, 7])],
    }
    return runtime


def resolve_b2_config(config: dict[str, Any]) -> dict[str, Any]:
    if "b2" in config:
        return config
    path = Path(str(config.get("base_config", "configs/b2_delayed_checkpoint.yaml")))
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_plos_config(b2_config: dict[str, Any]) -> dict[str, Any]:
    if "env" in b2_config and "gates" in b2_config:
        return b2_config
    path = Path(str(b2_config.get("base_config", "configs/sweep.yaml")))
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def b21_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get(
        "b21",
        {
            "n_attack_episodes": 32,
            "train_delays": [2, 4, 6],
            "heldout_delays": [3, 5, 7],
            "extrapolation_delays": [8, 10],
            "compression_levels": [1.0, 0.75, 0.5, 0.25],
            "gates": dict(B21_GATES),
        },
    )


def attack_episode_count(config: dict[str, Any]) -> int:
    declared = int(b21_config(config).get("n_attack_episodes", 32))
    cap = b21_config(config).get("max_attack_episodes")
    if cap is not None:
        declared = min(declared, int(cap))
    return max(1, declared)


def write_b21_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    with Path("results/b21_trace_hardening_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=B21_SUMMARY_KEYS)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row.get(key, "") for key in B21_SUMMARY_KEYS})
    with Path("results/b21_trace_hardening_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_KEYS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in RECORD_KEYS})
    Path("reports/B2_1_TRACE_HARDENING_REPORT.md").write_text(build_b21_report(summary), encoding="utf-8")
    Path("reports/B2_1_TRACE_HARDENING_SELF_AUDIT.md").write_text(build_b21_self_audit(), encoding="utf-8")


def build_b21_report(summary: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# B2.1 Trace-Bearing Substrate Hardening",
            "",
            "## 1. Purpose",
            "",
            "B2 introduced trace-bearing substrates for delayed operational checkpoints. B2.1 tests whether these models truly use delayed causal trace.",
            "",
            "## 2. Background",
            "",
            "PLOS v1 found flow_checkpoint_model. B1.1 showed it fails delayed checkpoint. B2 introduced temporal/field/schema memory substrates. B2.1 attacks the trace itself.",
            "",
            "## 3. Target Models",
            "",
            "- recurrent_flow_checkpoint_model",
            "- field_memory_model",
            "- schema_memory_model",
            "",
            "Optional reference:",
            "- flow_checkpoint_model",
            "",
            "## 4. Attack Set",
            "",
            "- false delayed trace",
            "- trace swap",
            "- trace deletion specificity",
            "- multi-source trace conflict",
            "- noisy delayed trace",
            "- trace length extrapolation",
            "- trace compression pressure",
            "",
            "## 5. Results Table",
            "",
            markdown_table(summary),
            "",
            "## 6. Interpretation",
            "",
            interpretation(summary),
            "",
            "If a model passes, it remains a hardened trace-bearing substrate candidate.",
            "If it fails, its B2 success may rely on trace prior, clean-generator cues, or shallow delay shortcuts.",
            "",
            "## 7. Claim Boundary",
            "",
            "Do not claim blank-slate emergence.",
            "Do not claim general physical reasoning.",
            "Do not claim human-like trace cognition.",
            "Do not claim real-world deployment.",
            "",
        ]
    )


def build_b21_self_audit() -> str:
    return "\n".join(
        [
            "# B2.1 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Tests whether trace-bearing models actually use causal trace.",
            "- Adds false delayed trace attacks.",
            "- Adds trace swap.",
            "- Adds trace deletion specificity.",
            "- Adds multi-source trace conflict.",
            "- Adds noisy trace robustness.",
            "- Adds trace length extrapolation.",
            "- Adds trace compression pressure.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Still a 64x64 toy world.",
            "- Trace attacks are hand-designed.",
            "- Ground truth trace comes from simulator.",
            "- Models may exploit generator artifacts.",
            "- Trace interventions may create OOD hidden states.",
            "- Compression pressure may not match natural resource limits.",
            "- Passing does not prove blank-slate emergence.",
            "",
            "## False Positive Risks",
            "",
            "- Model rejects false trace using superficial difference.",
            "- Trace swap creates unnatural hidden state.",
            "- Deletion affects visual continuity rather than trace.",
            "- Conflict oracle encodes evaluator bias.",
            "- Noisy trace remains too clean.",
            "- Length extrapolation may still be interpolation in disguise.",
            "- Compression keeps causal trace only because architecture privileges it.",
            "",
            "## Required Failure Checks",
            "",
            "1. false trace selected",
            "2. trace swap has no behavioral effect",
            "3. true trace deletion no stronger than non-trace deletion",
            "4. wrong trace source wins under conflict",
            "5. noisy trace causes collapse",
            "6. delay=8/10 extrapolation fails",
            "7. compression preserves saliency over causal trace",
            "",
        ]
    )


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(B21_SUMMARY_KEYS) + " |", "| " + " | ".join("---" for _ in B21_SUMMARY_KEYS) + " |"]
    for row in rows:
        values = []
        for key in B21_SUMMARY_KEYS:
            value = row.get(key, "")
            values.append(f"{float(value):.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def interpretation(summary: list[dict[str, Any]]) -> str:
    passed = [str(row["model"]) for row in summary if float(row.get("b21_trace_hardening_score", 0.0)) > 0.0]
    if not passed:
        return "B2 trace-bearing success is not robust under B2.1 trace-specific hardening. The next hard problem is robust causal trace use."
    notes = []
    if "recurrent_flow_checkpoint_model" in passed:
        notes.append("Temporal memory trace may be sufficient under B2.1, discounted by checkpoint and recurrent priors.")
    if "field_memory_model" in passed:
        notes.append("Delayed operational trace may be field-form O, not object-form O.")
    if "schema_memory_model" in passed:
        notes.append("Sparse schema memory may be a strong delayed trace carrier, discounted by schema-slot priors.")
    return " ".join(notes)
