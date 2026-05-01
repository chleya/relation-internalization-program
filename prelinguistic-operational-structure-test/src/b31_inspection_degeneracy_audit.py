from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .b31_baseline_sanity import run_b31_baseline_sanity_check
from .b31_disagreement_inspection_env import evaluate_disagreement_inspection_divergence, make_disagreement_inspection_episode
from .b31_inspection_provenance import collect_inspection_provenance, summarize_inspection_provenance
from .b31_inspection_scorers import evaluate_inspection_scorer_specificity
from .b31_shared_policy_ablation import evaluate_shared_inspection_policy_ablation
from .b31_trace_ablation_specificity import evaluate_trace_ablation_specificity
from .b3_active_inspection_env import b3_runtime_config, make_b3_active_inspection_episode
from .b3_inspection_policy import trace_guided_inspection_policy
from .b23_private_selectors import enforce_private_selector
from .models import make_model


B31_GATES = {
    "max_cross_model_inspect_region_match_rate": 0.80,
    "max_shared_inspection_policy_usage_rate": 0.05,
    "min_private_inspection_score_usage_rate": 0.95,
    "fallback_usage_rate": 0.05,
    "min_inspection_scorer_specificity": 0.70,
    "max_inspection_scorer_correlation": 0.90,
    "max_shared_policy_ablation_drop": 0.05,
    "min_private_trace_ablation_drop": 0.20,
    "min_disagreement_inspection_divergence": 0.50,
    "min_family_aligned_inspection_rate": 0.60,
    "max_random_inspection_score": 0.25,
    "min_oracle_inspection_score": 0.95,
    "min_trace_over_saliency_gain_margin": 0.15,
    "min_trace_over_short_horizon_gain_margin": 0.15,
    "private_trace_over_non_trace_ratio": 1.50,
}

B31_SUMMARY_KEYS = [
    "model",
    "seed",
    "cross_model_inspect_region_match_rate",
    "exact_all_model_same_region_rate",
    "pairwise_region_match_rate",
    "gt_region_match_rate",
    "saliency_region_match_rate",
    "shared_inspection_policy_usage_rate",
    "private_trace_inspection_score_usage_rate",
    "fallback_usage_rate",
    "unknown_policy_source_rate",
    "mean_inspection_scorer_correlation",
    "inspection_scorer_specificity",
    "shared_policy_ablation_drop",
    "private_inspection_retention_after_shared_ablation",
    "shared_policy_usage_rate_after_ablation",
    "disagreement_inspection_divergence",
    "family_aligned_inspection_rate",
    "cross_model_same_inspect_rate_on_disagreement",
    "causal_family_inspection_accuracy",
    "random_inspection_score",
    "saliency_inspection_score",
    "short_horizon_inspection_score",
    "oracle_inspection_score",
    "trace_over_saliency_gain_margin",
    "trace_over_short_horizon_gain_margin",
    "private_trace_ablation_drop",
    "matched_non_trace_ablation_drop",
    "saliency_ablation_drop",
    "private_trace_over_non_trace_ratio",
    "private_trace_over_saliency_ratio",
    "non_trace_inspection_stability",
    "b31_inspection_audit_score",
]


def compute_inspect_region_overlap(b3_records: list[dict[str, Any]], config: dict[str, Any] | None = None) -> dict[str, float]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in b3_records:
        if "model" not in row or "predicted_inspect_region" not in row:
            continue
        grouped.setdefault(int(row["episode_id"]), []).append(row)
    exact_all = []
    pairwise = []
    gt_hits = []
    saliency_hits = []
    for rows in grouped.values():
        regions = [int(row["predicted_inspect_region"]) for row in rows]
        if not regions:
            continue
        exact_all.append(1.0 if len(set(regions)) == 1 and len(regions) > 1 else 0.0)
        pairwise.extend(pairwise_matches(regions))
        for row in rows:
            pred = int(row["predicted_inspect_region"])
            gt_hits.append(1.0 if pred == int(row["oracle_best_inspect_region"]) else 0.0)
            saliency_hits.append(1.0 if pred == int(row["saliency_region"]) else 0.0)
    exact_rate = mean_or_zero(exact_all)
    return {
        "cross_model_inspect_region_match_rate": exact_rate,
        "exact_all_model_same_region_rate": exact_rate,
        "pairwise_region_match_rate": mean_or_zero(pairwise),
        "gt_region_match_rate": mean_or_zero(gt_hits),
        "saliency_region_match_rate": mean_or_zero(saliency_hits),
    }


def run_b31_inspection_audit(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime = b31_runtime_config(config)
    b31 = b31_config(config)
    n = effective_audit_count(b31)
    target_names = [str(name) for name in config.get("target_models", ["recurrent_flow_checkpoint_model", "field_memory_model", "schema_memory_model"])]
    models = {name: make_model(name) for name in target_names}
    for name, model in models.items():
        enforce_private_selector(model, name)

    episodes = make_b31_episodes(runtime, n, seed)
    disagreement = make_b31_disagreement_episodes(runtime, n, seed)
    b3_records = collect_b3_like_records(models, episodes, runtime, seed)
    overlap_metrics = compute_inspect_region_overlap(b3_records, config)

    provenance_records = []
    for model in models.values():
        provenance_records.extend(collect_inspection_provenance(model, episodes, runtime))
    provenance_metrics = summarize_inspection_provenance(provenance_records, config)

    scorer_metrics, scorer_records = evaluate_inspection_scorer_specificity(models, disagreement, runtime, seed)
    shared_metrics, shared_records = evaluate_shared_inspection_policy_ablation(models, episodes, runtime, seed)
    disagreement_metrics, disagreement_records = evaluate_disagreement_inspection_divergence(models, disagreement, runtime, seed)
    baseline_metrics, baseline_records = run_b31_baseline_sanity_check(models, episodes, runtime, seed)
    ablation_metrics, ablation_records = evaluate_trace_ablation_specificity(models, episodes, runtime, seed)

    metrics: dict[str, Any] = {
        "model": "cross_model_audit",
        "seed": int(seed),
        **overlap_metrics,
        **provenance_metrics,
        **scorer_metrics,
        **shared_metrics,
        **disagreement_metrics,
        **baseline_metrics,
        **ablation_metrics,
    }
    metrics["b31_inspection_audit_score"] = b31_inspection_audit_score(metrics, b31.get("gates", B31_GATES))
    records = []
    records.extend({"audit_type": "inspect_region_overlap", **row} for row in b3_records)
    records.extend({"audit_type": "inspection_provenance", **row} for row in provenance_records)
    records.extend({"audit_type": "inspection_scorer", **row} for row in scorer_records)
    records.extend({"audit_type": "shared_policy_ablation", **row} for row in shared_records)
    records.extend({"audit_type": "disagreement_inspection", **row} for row in disagreement_records)
    records.extend({"audit_type": "baseline_sanity", **row} for row in baseline_records)
    records.extend({"audit_type": "trace_ablation_specificity", **row} for row in ablation_records)
    return [metrics], records


def collect_b3_like_records(models: dict[str, Any], episodes: list[dict[str, Any]], config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    records = []
    for episode_id, episode in enumerate(episodes):
        gt = episode["ground_truth"]
        for model_name, model in models.items():
            policy = trace_guided_inspection_policy(model, episode, config)
            records.append(
                {
                    "seed": seed,
                    "episode_id": episode_id,
                    "delay": int(gt.get("delay", 0)),
                    "model": model_name,
                    "predicted_inspect_region": int(policy["inspect_region"]),
                    "true_trace_region": int(gt["true_trace_region"]),
                    "saliency_region": int(gt["saliency_region"]),
                    "oracle_best_inspect_region": int(gt["oracle_best_inspect_region"]),
                    "policy_source": policy.get("policy_source", ""),
                    "trace_family": policy.get("trace_family", ""),
                }
            )
    return records


def b31_inspection_audit_score(metrics: dict[str, float], gates: dict[str, float] | None = None) -> float:
    gates = {**B31_GATES, **(gates or {})}
    max_checks = {
        "cross_model_inspect_region_match_rate": "max_cross_model_inspect_region_match_rate",
        "shared_inspection_policy_usage_rate": "max_shared_inspection_policy_usage_rate",
        "fallback_usage_rate": "fallback_usage_rate",
        "mean_inspection_scorer_correlation": "max_inspection_scorer_correlation",
        "shared_policy_ablation_drop": "max_shared_policy_ablation_drop",
        "random_inspection_score": "max_random_inspection_score",
    }
    min_checks = {
        "private_trace_inspection_score_usage_rate": "min_private_inspection_score_usage_rate",
        "inspection_scorer_specificity": "min_inspection_scorer_specificity",
        "private_trace_ablation_drop": "min_private_trace_ablation_drop",
        "private_trace_over_non_trace_ratio": "private_trace_over_non_trace_ratio",
        "disagreement_inspection_divergence": "min_disagreement_inspection_divergence",
        "family_aligned_inspection_rate": "min_family_aligned_inspection_rate",
        "oracle_inspection_score": "min_oracle_inspection_score",
        "trace_over_saliency_gain_margin": "min_trace_over_saliency_gain_margin",
        "trace_over_short_horizon_gain_margin": "min_trace_over_short_horizon_gain_margin",
    }
    for metric_key, gate_key in max_checks.items():
        if float(metrics.get(metric_key, 1.0)) > float(gates[gate_key]):
            return 0.0
    ratio_gate = float(gates.get("private_trace_over_non_trace_ratio", gates.get("min_private_trace_over_non_trace_ratio", 1.50)))
    for metric_key, gate_key in min_checks.items():
        threshold = ratio_gate if metric_key == "private_trace_over_non_trace_ratio" else float(gates[gate_key])
        if float(metrics.get(metric_key, 0.0)) < threshold:
            return 0.0
    weights = {
        "private_trace_inspection_score_usage_rate": 0.15,
        "inspection_scorer_specificity": 0.20,
        "disagreement_inspection_divergence": 0.20,
        "family_aligned_inspection_rate": 0.15,
        "private_trace_ablation_drop": 0.15,
        "trace_over_saliency_gain_margin": 0.15,
    }
    return float(sum(weight * min(max(float(metrics.get(key, 0.0)), 0.0), 1.0) for key, weight in weights.items()))


def write_b31_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b31_inspection_audit_summary.csv"), summary, B31_SUMMARY_KEYS)
    write_csv(Path("results/b31_inspection_audit_records.csv"), records)
    write_csv(Path("results/b31_inspect_region_overlap.csv"), [row for row in records if row.get("audit_type") == "inspect_region_overlap"])
    write_csv(Path("results/b31_inspection_provenance.csv"), [row for row in records if row.get("audit_type") == "inspection_provenance"])
    write_csv(Path("results/b31_inspection_scorer_correlation.csv"), [row for row in records if row.get("audit_type") == "inspection_scorer"])
    write_csv(Path("results/b31_disagreement_inspection_results.csv"), [row for row in records if row.get("audit_type") == "disagreement_inspection"])
    write_csv(Path("results/b31_shared_policy_ablation.csv"), [row for row in records if row.get("audit_type") == "shared_policy_ablation"])
    write_csv(Path("results/b31_baseline_sanity.csv"), [row for row in records if row.get("audit_type") == "baseline_sanity"])
    write_csv(Path("results/b31_trace_ablation_specificity.csv"), [row for row in records if row.get("audit_type") == "trace_ablation_specificity"])
    Path("reports/B3_1_ACTIVE_INSPECTION_DEGENERACY_AUDIT.md").write_text(build_b31_report(summary), encoding="utf-8")
    Path("reports/B3_1_ACTIVE_INSPECTION_DEGENERACY_SELF_AUDIT.md").write_text(build_b31_self_audit(), encoding="utf-8")


def b31_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    base = resolve_b31_base_config(config)
    runtime = b3_runtime_config(base)
    b31 = b31_config(config)
    env = dict(runtime.get("env", {}))
    for key in ("frame_size", "grid_size", "past_frames", "future_frames"):
        if key in b31:
            env[key] = int(b31[key])
    runtime["env"] = env
    return runtime


def resolve_b31_base_config(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(config.get("base_config", "configs/b3_active_inspection.yaml")))
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def b31_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("b31", {"n_audit_episodes": 32, "gates": dict(B31_GATES)})


def effective_audit_count(b31: dict[str, Any]) -> int:
    count = int(b31.get("n_audit_episodes", 32))
    if "max_audit_episodes" in b31:
        count = min(count, int(b31["max_audit_episodes"]))
    return max(1, count)


def make_b31_episodes(config: dict[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    delays = [2, 4, 6]
    return [
        make_b3_active_inspection_episode(config, seed + idx * 17, "b31_inspection_overlap_audit", delays[idx % len(delays)])
        for idx in range(count)
    ]


def make_b31_disagreement_episodes(config: dict[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    families = ["recurrent_memory", "field_trace", "schema_memory"]
    return [make_disagreement_inspection_episode(config, seed + 70000 + idx * 29, families[idx % len(families)]) for idx in range(count)]


def pairwise_matches(regions: list[int]) -> list[float]:
    matches = []
    for left in range(len(regions)):
        for right in range(left + 1, len(regions)):
            matches.append(1.0 if int(regions[left]) == int(regions[right]) else 0.0)
    return matches


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_b31_report(summary: list[dict[str, Any]]) -> str:
    row = summary[0] if summary else {}
    score = float(row.get("b31_inspection_audit_score", 0.0))
    interpretation = (
        "B3.1 reduces shared-inspection-policy risk under the current diagnostics."
        if score > 0.0
        else "B3.1 does not clear the degeneracy gates; B3 remains evidence for trace-guided inspection behavior, not independent active-inspection mechanisms."
    )
    return "\n".join(
        [
            "# B3.1 Active Inspection Degeneracy Audit",
            "",
            "## 1. Purpose",
            "",
            "B3.1 audits whether identical B3 scores come from model-specific private trace-guided inspection or from shared inspection policy / evaluator degeneracy.",
            "",
            "## 2. Background",
            "",
            "PLOS v1 found a short-horizon checkpoint candidate. B1.1 showed delayed checkpoint failure. B2 introduced trace-bearing delayed checkpoints. B2.1 hardened trace use. B2.1a found score degeneracy. B2.2 exposed shared selector dependence. B2.3 reconstructed private selectors. B3 showed delayed trace-guided active inspection. B3.1 audits whether B3 active inspection is independently mechanism-separated or still degenerate.",
            "",
            "## 3. Audits",
            "",
            "- per-episode inspect region overlap",
            "- inspection policy provenance",
            "- trace-family-specific inspection scorer",
            "- shared inspection policy ablation",
            "- disagreement-inspection episodes",
            "- baseline sanity check",
            "- trace-ablation specificity",
            "",
            "## 4. Results",
            "",
            markdown_summary(row),
            "",
            "## 5. Interpretation",
            "",
            interpretation,
            "",
            "## 6. Claim Boundary",
            "",
            "B3.1 does not add a new capability claim.",
            "B3.1 does not prove independent active-inspection mechanisms.",
            "B3.1 only validates or weakens the reliability of B3 active-inspection scoring.",
            "",
        ]
    )


def build_b31_self_audit() -> str:
    return "\n".join(
        [
            "# B3.1 Active Inspection Degeneracy Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Checks per-episode inspect-region overlap.",
            "- Adds inspection policy provenance.",
            "- Compares trace-family-specific inspection scorers.",
            "- Disables any shared inspection policy path.",
            "- Adds disagreement-inspection episodes.",
            "- Rechecks random, saliency, short-horizon, and oracle baselines.",
            "- Tests trace-ablation specificity.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Still a 64x64 toy world.",
            "- Audit episodes are synthetic.",
            "- Provenance can miss hidden shared low-level features.",
            "- High overlap can be caused by a simple task rather than literal shared code.",
            "- Passing does not prove independent mechanisms.",
            "",
            "## Required Failure Checks",
            "",
            "1. cross-model inspect regions match too often",
            "2. shared inspection policy is used",
            "3. private inspection scorer is not used",
            "4. inspection scorer correlation is too high",
            "5. disagreement-inspection episodes do not diverge",
            "6. random/saliency/short-horizon baselines are too strong",
            "7. private trace ablation is not specific",
            "",
        ]
    )


def markdown_summary(summary: dict[str, Any]) -> str:
    keys = B31_SUMMARY_KEYS
    lines = ["| metric | value |", "| --- | --- |"]
    for key in keys:
        value = summary.get(key, "")
        if isinstance(value, float):
            value = f"{value:.3f}"
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
