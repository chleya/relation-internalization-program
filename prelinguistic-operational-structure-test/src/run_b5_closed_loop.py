from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .b23_private_selectors import enforce_private_selector
from .b4_intervention_policy import MODEL_TO_TRACE_FAMILY
from .b5_closed_loop_baselines import evaluate_b5_baselines
from .b5_closed_loop_env import EPISODE_TYPES, b5_config, b5_runtime_config, make_b5_closed_loop_episode
from .b5_closed_loop_metrics import B5_GATES, B5_SUMMARY_KEYS, b5_closed_loop_score, closed_loop_gain_over_baseline
from .b5_closed_loop_policy import evaluate_closed_loop_policy
from .b5_epistemic_pragmatic_values import compute_wrong_inspect_penalty, compute_wrong_intervention_penalty
from .b5_outputs import write_b5_outputs
from .models import make_model


def run_b5_closed_loop(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime = b5_runtime_config(config)
    b5 = b5_config(config)
    n_test = effective_count(int(b5.get("n_test", 32)), b5)
    model_names = [str(name) for name in config.get("target_models", MODEL_TO_TRACE_FAMILY)]
    models = {name: make_model(name) for name in model_names}
    for name, model in models.items():
        enforce_private_selector(model, name)

    summary = []
    records: list[dict[str, Any]] = []
    for model_name, model in models.items():
        family = MODEL_TO_TRACE_FAMILY.get(model_name, "recurrent")
        episodes = [make_b5_closed_loop_episode(runtime, seed + idx * 17, EPISODE_TYPES[idx % len(EPISODE_TYPES)], family) for idx in range(n_test)]
        policy_metrics, policy_records = evaluate_closed_loop_policy(model, episodes, runtime, seed, model_name)
        baseline_metrics, baseline_records = evaluate_b5_baselines(episodes, runtime, seed)
        model_score = float(policy_metrics.get("closed_loop_model_score", 0.0))
        wrong_metrics = {
            "wrong_inspect_penalty_sensitivity": mean_or_zero([compute_wrong_inspect_penalty(episode, runtime) for episode in episodes]),
            "wrong_intervention_penalty_sensitivity": mean_or_zero([compute_wrong_intervention_penalty(episode, runtime) for episode in episodes]),
        }
        metrics = {
            **policy_metrics,
            **baseline_metrics,
            **wrong_metrics,
            "closed_loop_gain_over_inspect_always": closed_loop_gain_over_baseline(model_score, baseline_metrics["inspect_always_score"]),
            "closed_loop_gain_over_intervene_immediately": closed_loop_gain_over_baseline(model_score, baseline_metrics["intervene_immediately_score"]),
            "closed_loop_gain_over_random": closed_loop_gain_over_baseline(model_score, baseline_metrics["random_closed_loop_score"]),
            "closed_loop_gain_over_saliency": closed_loop_gain_over_baseline(model_score, baseline_metrics["saliency_closed_loop_score"]),
            "closed_loop_gain_over_short_horizon": closed_loop_gain_over_baseline(model_score, baseline_metrics["short_horizon_closed_loop_score"]),
        }
        metrics["b5_closed_loop_score"] = b5_closed_loop_score(metrics, b5.get("gates", B5_GATES))
        row = {"model": model_name, "seed": int(seed)}
        for key in B5_SUMMARY_KEYS:
            if key not in {"model", "seed"}:
                row[key] = float(metrics.get(key, 0.0))
        summary.append(row)
        records.extend(policy_records)
        for record in baseline_records:
            records.append({"model": model_name, "seed": seed, **record})
    return summary, records


def effective_count(value: int, b5: dict[str, Any]) -> int:
    cap = b5.get("max_eval_episodes")
    if cap is not None:
        value = min(value, int(cap))
    return max(1, int(value))


def mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b5_closed_loop.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records = run_b5_closed_loop(config, seed=args.seed)
    write_b5_outputs(summary, records)
    best = max(float(row.get("b5_closed_loop_score", 0.0)) for row in summary) if summary else 0.0
    print(f"best_b5_closed_loop_score={best:.3f}")


if __name__ == "__main__":
    main()
