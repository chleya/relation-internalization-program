from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .g1_baselines import BASELINE_NAMES, baseline_action
from .g1_env import CONDITIONS, make_g1_datasets
from .g1_generator import choose_action, fit_generator
from .g1_metrics import RECORD_FIELDS, SUMMARY_FIELDS, score_episode, summarize


def run_g1(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    datasets = make_g1_datasets(config, seed)
    train_inputs = [episode["model_input"] for episode in datasets["train"]]
    artifact = fit_generator(train_inputs, config)
    records: list[dict[str, Any]] = []
    raw: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for condition, episodes in datasets.items():
        for policy_name in BASELINE_NAMES:
            rows = []
            for episode in episodes:
                if policy_name == "g1_generator":
                    output = choose_action(episode["model_input"], artifact)
                else:
                    output = baseline_action(episode["model_input"], policy_name, seed=int(episode["episode_id"]), truth=episode["evaluator_ground_truth"])
                row = score_episode(episode, output, policy_name)
                rows.append(row)
                records.append(row)
            raw[(condition, policy_name)] = rows
    summary = []
    for condition in CONDITIONS:
        baseline_scores = {policy_name: mean(raw.get((condition, policy_name), []), "generator_score") for policy_name in BASELINE_NAMES}
        for policy_name in BASELINE_NAMES:
            summary.append(summarize(raw.get((condition, policy_name), []), baseline_scores))
    metrics = build_metrics(summary, records, artifact)
    return summary, records, metrics, artifact


def build_metrics(summary: list[dict[str, Any]], records: list[dict[str, Any]], artifact: dict[str, Any]) -> dict[str, Any]:
    gen_rows = [row for row in summary if row["policy_name"] == "g1_generator"]
    ood = next((row for row in gen_rows if row["condition"] == "ood_remap"), {})
    test = next((row for row in gen_rows if row["condition"] == "test"), {})
    return {
        "summary_rows": len(summary),
        "record_rows": len(records),
        "conditions": sorted({row["condition"] for row in summary}),
        "generated_components": artifact["generated_components"],
        "search_size": artifact["search_size"],
        "train_objective": artifact["train_objective"],
        "g1_generator_mean_score": mean(gen_rows, "generator_score"),
        "g1_test_score": float(test.get("generator_score", 0.0)),
        "g1_ood_score": float(ood.get("generator_score", 0.0)),
        "g1_ood_gain_over_random": float(ood.get("gain_over_random", 0.0)),
        "g1_ood_gain_over_hand_designed": float(ood.get("gain_over_hand_designed", 0.0)),
        "g1_oracle_gap": mean(gen_rows, "oracle_gap"),
        "g1_mask_f1": mean(gen_rows, "mask_f1"),
        "g1_compression_cost": mean(gen_rows, "compression_cost"),
        "invalid_metric_count_total": sum(int(row["invalid_metric_count"]) for row in summary),
        "generator_rule": artifact["rule"].__dict__,
        "claim": "minimal generated operational-structure diagnostic",
    }


def write_g1_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    out_dir = Path("results/g1_minimal")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "summary.csv", summary, SUMMARY_FIELDS)
    write_csv(out_dir / "records.csv", records, RECORD_FIELDS)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/G1_MINIMAL_GENERATOR_RESULTS.md").write_text(build_results_report(metrics), encoding="utf-8")


def build_results_report(metrics: dict[str, Any]) -> str:
    better_random = float(metrics["g1_ood_gain_over_random"]) > 0
    better_hand = float(metrics["g1_ood_gain_over_hand_designed"]) > 0
    return "\n".join(
        [
            "# G1 Minimal Generator Results",
            "",
            "## Summary",
            "G1 runs a compact rule-search generator that induces an actionability/update mask from interaction history rather than receiving the B-line mask directly.",
            "",
            "## Metrics",
            f"- g1_generator_mean_score = {float(metrics['g1_generator_mean_score']):.3f}",
            f"- g1_test_score = {float(metrics['g1_test_score']):.3f}",
            f"- g1_ood_score = {float(metrics['g1_ood_score']):.3f}",
            f"- g1_ood_gain_over_random = {float(metrics['g1_ood_gain_over_random']):.3f}",
            f"- g1_ood_gain_over_hand_designed = {float(metrics['g1_ood_gain_over_hand_designed']):.3f}",
            f"- g1_oracle_gap = {float(metrics['g1_oracle_gap']):.3f}",
            f"- g1_mask_f1 = {float(metrics['g1_mask_f1']):.3f}",
            f"- g1_compression_cost = {float(metrics['g1_compression_cost']):.3f}",
            "",
            "## Interpretation",
            f"- beats_random_on_ood: {str(better_random).lower()}",
            f"- beats_hand_designed_on_ood: {str(better_hand).lower()}",
            "G1 is useful only if the generated rule improves over weak baselines while preserving an explicit oracle gap and failure report.",
            "",
            "## Claim Boundary",
            "G1 is a minimal toy generator experiment. It does not prove autonomous cognition, real-world risk intelligence, robotics capability, safety certification, construction-site autonomy, or deployable engineering control.",
            "",
            "## Next Step",
            "G2 should test whether the generated rule family can become less hand-scaffolded, for example by inducing feature combinations or update primitives rather than selecting thresholds over a fixed vocabulary.",
            "",
        ]
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
