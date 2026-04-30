from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .b2_delayed_env import make_delayed_checkpoint_episode
from .b21_trace_attacks import b21_runtime_config
from .b22_disagreement_env import evaluate_disagreement_episode_divergence, make_trace_disagreement_episode
from .b22_provenance import collect_trace_provenance, cross_model_prediction_match_rate, summarize_trace_provenance
from .b22_selector_ablation import evaluate_shared_selector_ablation, evaluate_source_specific_ablation
from .b22_selector_free_models import evaluate_selector_free_retention, make_selector_free_model
from .b22_trace_scorers import compute_trace_scorer_correlation, private_trace_scores
from .models import make_model


B22_GATES = {
    "shared_selector_usage_rate": 0.05,
    "max_cross_model_exact_prediction_match_rate": 0.80,
    "min_trace_family_specificity": 0.70,
    "min_model_private_trace_drop": 0.20,
    "max_shared_selector_ablation_advantage": 0.00,
    "min_disagreement_episode_divergence": 0.50,
    "max_trace_scorer_correlation": 0.90,
    "min_selector_free_retention": 0.60,
}

B22_SUMMARY_KEYS = [
    "model",
    "seed",
    "shared_selector_usage_rate",
    "model_private_score_usage_rate",
    "fallback_usage_rate",
    "cross_model_exact_prediction_match_rate",
    "selector_free_b21_score",
    "selector_free_retention",
    "disagreement_episode_divergence",
    "family_aligned_selection_rate",
    "trace_family_specificity",
    "mean_trace_scorer_correlation",
    "model_private_trace_drop",
    "shared_selector_ablation_drop",
    "shared_selector_ablation_advantage",
    "private_trace_retention_after_shared_ablation",
    "b22_disentanglement_score",
]

B22_RECORD_KEYS = [
    "model",
    "seed",
    "episode_id",
    "audit_type",
    "selected_region",
    "source_module",
    "source_trace_family",
    "source_score",
    "fallback_used",
    "shared_selector_used",
    "model_private_score_used",
    "causal_family",
    "temporal_trace_region",
    "field_trace_region",
    "schema_trace_region",
    "predicted_region",
    "gate_pass",
    "note",
]


def run_b22_selector_disentanglement(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    b22 = b22_config(config)
    runtime_config = b21_runtime_config(resolve_b21_config_for_b22(config))
    n = int(b22.get("n_episodes", 32))
    delays = [2, 4, 6]
    target_model_names = [str(name) for name in config.get("target_models", ["recurrent_flow_checkpoint_model", "field_memory_model", "schema_memory_model"])]
    episodes = [make_delayed_checkpoint_episode(runtime_config, seed + idx * 31, delays[idx % len(delays)]) for idx in range(n)]
    causal_families = ["recurrent", "field", "schema"]
    disagreement_episodes = [
        make_trace_disagreement_episode(runtime_config, seed + 10000 + idx * 37, causal_families[idx % len(causal_families)])
        for idx in range(n)
    ]
    models = {name: make_model(name) for name in target_model_names}

    all_provenance = []
    by_model_provenance: dict[str, list[dict[str, Any]]] = {}
    for name, model in models.items():
        rows = collect_trace_provenance(model, episodes, runtime_config)
        by_model_provenance[name] = rows
        all_provenance.extend(rows)
    exact_match_rate = cross_model_prediction_match_rate(all_provenance)
    disagreement_metrics = evaluate_disagreement_episode_divergence(models, disagreement_episodes, runtime_config)

    summary: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    selector_free_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    source_ablation_rows: list[dict[str, Any]] = []
    scorer_rows: list[dict[str, Any]] = []
    shared_ablation_rows: list[dict[str, Any]] = []

    for name, model in models.items():
        provenance_metrics = summarize_trace_provenance(by_model_provenance[name], runtime_config)
        selector_free = make_selector_free_model(name, runtime_config)
        selector_free_metrics = evaluate_selector_free_retention(model, selector_free, episodes, runtime_config)
        source_metrics = evaluate_source_specific_ablation(model, episodes[: min(n, 96)], runtime_config)
        shared_metrics = evaluate_shared_selector_ablation(model, episodes[: min(n, 96)], runtime_config)
        scorer_metrics = evaluate_trace_scorer_correlation(episodes[: min(n, 96)], runtime_config)
        metrics = {
            **provenance_metrics,
            **selector_free_metrics,
            **disagreement_metrics,
            **source_metrics,
            **shared_metrics,
            **scorer_metrics,
            "cross_model_exact_prediction_match_rate": exact_match_rate,
        }
        metrics["b22_disentanglement_score"] = b22_disentanglement_score(metrics, b22.get("gates", B22_GATES))
        row = {"model": name, "seed": int(seed)}
        for key in B22_SUMMARY_KEYS:
            if key not in {"model", "seed"}:
                row[key] = float(metrics.get(key, 0.0))
        summary.append(row)

        for provenance in by_model_provenance[name]:
            records.append(record_from_provenance(provenance, seed))
        selector_free_rows.append({"model": name, "seed": seed, **selector_free_metrics})
        source_ablation_rows.append({"model": name, "seed": seed, **source_metrics})
        scorer_rows.append({"model": name, "seed": seed, **scorer_metrics})
        shared_ablation_rows.append({"model": name, "seed": seed, **shared_metrics})

    for idx, episode in enumerate(disagreement_episodes):
        gt = episode["ground_truth"]
        disagreement_rows.append(
            {
                "episode_id": idx,
                "causal_family": gt.get("causal_family", ""),
                "temporal_trace_region": gt.get("temporal_trace_region", ""),
                "field_trace_region": gt.get("field_trace_region", ""),
                "schema_trace_region": gt.get("schema_trace_region", ""),
                "causal_region": gt.get("causal_region", ""),
            }
        )
    write_auxiliary_outputs(
        all_provenance,
        selector_free_rows,
        disagreement_rows,
        source_ablation_rows,
        scorer_rows,
        shared_ablation_rows,
    )
    records.extend(metric_records("selector_free_variants", selector_free_rows, seed))
    records.extend(metric_records("trace_disagreement_episodes", disagreement_rows, seed))
    records.extend(metric_records("source_specific_ablation", source_ablation_rows, seed))
    records.extend(metric_records("independent_trace_scorers", scorer_rows, seed))
    records.extend(metric_records("shared_selector_ablation", shared_ablation_rows, seed))
    return summary, records


def b22_disentanglement_score(metrics: dict[str, float], gates: dict[str, float] | None = None) -> float:
    gates = {**B22_GATES, **(gates or {})}
    if float(metrics.get("shared_selector_usage_rate", 1.0)) > float(gates["shared_selector_usage_rate"]):
        return 0.0
    if float(metrics.get("cross_model_exact_prediction_match_rate", 1.0)) > float(gates["max_cross_model_exact_prediction_match_rate"]):
        return 0.0
    if float(metrics.get("trace_family_specificity", 0.0)) < float(gates["min_trace_family_specificity"]):
        return 0.0
    if float(metrics.get("model_private_trace_drop", 0.0)) < float(gates["min_model_private_trace_drop"]):
        return 0.0
    if float(metrics.get("shared_selector_ablation_advantage", 1.0)) > float(gates["max_shared_selector_ablation_advantage"]):
        return 0.0
    if float(metrics.get("disagreement_episode_divergence", 0.0)) < float(gates["min_disagreement_episode_divergence"]):
        return 0.0
    if float(metrics.get("selector_free_retention", 0.0)) < float(gates["min_selector_free_retention"]):
        return 0.0
    weights = {
        "selector_free_retention": 0.20,
        "disagreement_episode_divergence": 0.20,
        "trace_family_specificity": 0.20,
        "model_private_trace_drop": 0.20,
        "private_trace_retention_after_shared_ablation": 0.20,
    }
    return float(
        sum(weight * min(max(float(metrics.get(key, 0.0)), 0.0), 1.0) for key, weight in weights.items())
    )


def evaluate_trace_scorer_correlation(episodes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    rows = []
    from .model_io import make_model_batch

    for episode in episodes:
        batch = make_model_batch(episode, config)
        rows.append(
            compute_trace_scorer_correlation(
                {
                    "recurrent": private_trace_scores(batch, "recurrent_flow_checkpoint"),
                    "field": private_trace_scores(batch, "field_memory"),
                    "schema": private_trace_scores(batch, "schema_memory"),
                }
            )
        )
    if not rows:
        return {
            "recurrent_field_score_correlation": 1.0,
            "recurrent_schema_score_correlation": 1.0,
            "field_schema_score_correlation": 1.0,
            "mean_trace_scorer_correlation": 1.0,
            "trace_family_specificity": 0.0,
        }
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def b22_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("b22", {"n_episodes": 32, "gates": dict(B22_GATES)})


def resolve_b21_config_for_b22(config: dict[str, Any]) -> dict[str, Any]:
    current = dict(config)
    visited: set[Path] = set()
    while "b21" not in current:
        base_config = current.get("base_config")
        if not base_config:
            break
        path = Path(str(base_config))
        if not path.is_absolute():
            path = Path.cwd() / path
        if path in visited:
            break
        visited.add(path)
        with path.open("r", encoding="utf-8") as handle:
            current = yaml.safe_load(handle)
    return current


def write_b22_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    with Path("results/b22_selector_disentanglement_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=B22_SUMMARY_KEYS)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row.get(key, "") for key in B22_SUMMARY_KEYS})
    with Path("results/b22_selector_disentanglement_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=B22_RECORD_KEYS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in B22_RECORD_KEYS})
    Path("reports/B2_2_TRACE_SELECTOR_DISENTANGLEMENT.md").write_text(build_b22_report(summary), encoding="utf-8")
    Path("reports/B2_2_TRACE_SELECTOR_SELF_AUDIT.md").write_text(build_b22_self_audit(), encoding="utf-8")


def write_auxiliary_outputs(
    provenance: list[dict[str, Any]],
    selector_free_rows: list[dict[str, Any]],
    disagreement_rows: list[dict[str, Any]],
    source_ablation_rows: list[dict[str, Any]],
    scorer_rows: list[dict[str, Any]],
    shared_ablation_rows: list[dict[str, Any]],
) -> None:
    write_csv(Path("results/b22_trace_provenance.csv"), provenance)
    write_csv(Path("results/b22_selector_free_comparison.csv"), selector_free_rows)
    write_csv(Path("results/b22_disagreement_episodes.csv"), disagreement_rows)
    write_csv(Path("results/b22_source_specific_ablation.csv"), source_ablation_rows)
    write_csv(Path("results/b22_trace_scorer_correlation.csv"), scorer_rows)
    write_csv(Path("results/b22_shared_selector_ablation.csv"), shared_ablation_rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def record_from_provenance(row: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "model": row.get("model", ""),
        "seed": seed,
        "episode_id": row.get("episode_id", ""),
        "audit_type": "trace_provenance",
        "selected_region": row.get("selected_region", ""),
        "source_module": row.get("source_module", ""),
        "source_trace_family": row.get("source_trace_family", ""),
        "source_score": row.get("source_score", ""),
        "fallback_used": row.get("fallback_used", ""),
        "shared_selector_used": row.get("shared_selector_used", ""),
        "model_private_score_used": row.get("model_private_score_used", ""),
        "predicted_region": row.get("selected_region", ""),
        "gate_pass": int(not bool(row.get("shared_selector_used"))),
        "note": "shared selector provenance" if bool(row.get("shared_selector_used")) else "private selector provenance",
    }


def metric_records(audit_type: str, rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        for key, value in row.items():
            if key in {"model", "seed", "episode_id"}:
                continue
            records.append(
                {
                    "model": row.get("model", ""),
                    "seed": row.get("seed", seed),
                    "episode_id": row.get("episode_id", idx),
                    "audit_type": audit_type,
                    "predicted_region": row.get("selected_region", ""),
                    "causal_family": row.get("causal_family", ""),
                    "temporal_trace_region": row.get("temporal_trace_region", ""),
                    "field_trace_region": row.get("field_trace_region", ""),
                    "schema_trace_region": row.get("schema_trace_region", ""),
                    "gate_pass": "",
                    "note": f"{key}={value}",
                }
            )
    return records


def build_b22_report(summary: list[dict[str, Any]]) -> str:
    passed = [row["model"] for row in summary if float(row.get("b22_disentanglement_score", 0.0)) > 0.0]
    interpretation = (
        "The trace-bearing models show evidence of separated trace mechanisms under current diagnostics."
        if passed
        else "B2/B2.1 should currently be interpreted as shared trace-selector success, not independent recurrent/field/schema mechanism validation."
    )
    return "\n".join(
        [
            "# B2.2 Trace Selector Disentanglement",
            "",
            "## 1. Purpose",
            "",
            "B2.1a found that three trace-bearing models produce identical per-episode predictions. B2.2 tests whether B2/B2.1 success comes from independent trace mechanisms or a shared trace selector.",
            "",
            "## 2. Background",
            "",
            "PLOS v1: flow_checkpoint_model became the first checkpoint candidate. B1.1: flow_checkpoint_model failed delayed checkpoint. B2: trace-bearing substrates solved delayed checkpoint. B2.1: trace-bearing models passed trace hardening. B2.1a: identical predictions revealed selector degeneracy. B2.2: disentangles trace selector provenance.",
            "",
            "## 3. Methods",
            "",
            "- trace provenance audit",
            "- selector-free variants",
            "- trace disagreement episodes",
            "- source-specific ablation",
            "- independent trace scorers",
            "- shared-selector ablation",
            "",
            "## 4. Results",
            "",
            markdown_table(summary),
            "",
            "## 5. Interpretation",
            "",
            interpretation,
            "",
            "The current base trace-bearing models still report shared selector provenance through delayed_common.select_delayed_region. This is the intended reviewer-facing failure mode: B2.2 separates useful trace-bearing behavior from evidence for independent mechanism families.",
            "",
            "## 6. Claim Boundary",
            "",
            "Do not claim blank-slate emergence.",
            "Do not claim general delayed causality.",
            "Do not claim human-like trace cognition.",
            "Do not claim real-world deployment.",
            "",
        ]
    )


def build_b22_self_audit() -> str:
    return "\n".join(
        [
            "# B2.2 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Addresses B2.1a identical prediction degeneracy.",
            "- Tracks trace provenance.",
            "- Adds selector-free variants.",
            "- Adds trace disagreement episodes.",
            "- Adds source-specific ablations.",
            "- Adds independent trace scorer comparison.",
            "- Adds shared-selector ablation.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Still a 64x64 toy world.",
            "- Selector-free variants may be weaker by construction.",
            "- Provenance reporting can be incomplete or misleading.",
            "- Disagreement episodes are hand-designed.",
            "- Trace scorer correlation may be high because task is simple.",
            "- Source-specific ablation may create OOD hidden states.",
            "- Passing does not prove natural emergence.",
            "",
            "## False Positive Risks",
            "",
            "- Model-private scorer may still reuse shared features.",
            "- Selector-free variants may retain hidden shared heuristic.",
            "- Disagreement episodes may encode evaluator assumptions.",
            "- Shared selector usage may be underreported.",
            "- Ablation may harm unrelated capacity.",
            "- Low prediction overlap may come from noise, not mechanism separation.",
            "",
            "## Required Failure Checks",
            "",
            "1. shared selector usage remains high.",
            "2. selector-free variants collapse.",
            "3. trace disagreement episodes do not produce divergence.",
            "4. model-private trace ablation has small effect.",
            "5. shared selector ablation has larger effect than private trace ablation.",
            "6. trace scorers are nearly identical.",
            "7. provenance fields are missing or unknown.",
            "",
        ]
    )


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(B22_SUMMARY_KEYS) + " |", "| " + " | ".join("---" for _ in B22_SUMMARY_KEYS) + " |"]
    for row in rows:
        values = []
        for key in B22_SUMMARY_KEYS:
            value = row.get(key, "")
            values.append(f"{float(value):.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
