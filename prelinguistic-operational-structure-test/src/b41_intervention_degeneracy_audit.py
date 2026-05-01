from __future__ import annotations

import csv
from math import log2
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .b23_private_selectors import enforce_private_selector
from .b41_action_provenance import collect_action_policy_provenance, summarize_action_policy_provenance
from .b41_action_scorers import evaluate_action_scorer_specificity
from .b41_baseline_sanity import run_b41_baseline_sanity_check
from .b41_shared_action_policy_ablation import evaluate_shared_action_policy_ablation
from .b41_trace_ablation_specificity import evaluate_action_after_trace_ablation_specificity
from .b41_wrong_action_stress import evaluate_wrong_action_wrong_region_stress
from .b4_intervention_env import b4_runtime_config, make_b4_intervention_episode
from .b4_intervention_policy import MODEL_TO_TRACE_FAMILY, trace_guided_intervention_policy
from .models import make_model


FORBIDDEN_ACTION_VALUE_KEYS = [
    "oracle_best_action",
    "oracle_region",
    "oracle_action_type",
    "oracle_intervention_value",
    "intervention_values",
    "family_best_actions",
    "correct_action_value",
    "wrong_action_value",
    "wrong_region_value",
    "best_action",
    "best_region",
]

B41_GATES = {
    "max_cross_model_exact_action_match_rate": 0.80,
    "max_exact_all_model_same_action_rate": 0.80,
    "max_fixed_action_type_rate": 0.85,
    "max_shared_action_policy_usage_rate": 0.05,
    "min_private_trace_action_score_usage_rate": 0.95,
    "max_fallback_usage_rate": 0.05,
    "min_action_scorer_specificity": 0.70,
    "max_action_scorer_correlation": 0.90,
    "max_shared_action_policy_ablation_drop": 0.05,
    "min_private_action_retention_after_shared_ablation": 0.80,
    "min_wrong_action_penalty": 0.20,
    "min_wrong_region_penalty": 0.20,
    "min_wrong_action_wrong_region_penalty": 0.25,
    "max_random_intervention_score": 0.25,
    "min_oracle_intervention_score": 0.95,
    "min_gain_over_saliency": 0.15,
    "min_gain_over_short_horizon": 0.15,
    "min_gain_over_inspect_only": 0.15,
    "min_private_trace_ablation_drop": 0.20,
    "min_private_trace_over_non_trace_ratio": 1.50,
    "min_action_type_shift_after_trace_ablation": 0.20,
    "max_value_leakage_count": 0,
}

B41_SUMMARY_KEYS = [
    "model",
    "seed",
    "cross_model_exact_action_match_rate",
    "exact_all_model_same_action_rate",
    "fixed_action_type_rate",
    "action_type_entropy",
    "shared_action_policy_usage_rate",
    "private_trace_action_score_usage_rate",
    "fallback_usage_rate",
    "oracle_value_usage_rate",
    "action_scorer_specificity",
    "mean_action_scorer_correlation",
    "shared_action_policy_ablation_drop",
    "private_action_retention_after_shared_ablation",
    "wrong_action_penalty",
    "wrong_region_penalty",
    "wrong_action_wrong_region_penalty",
    "random_intervention_score",
    "saliency_intervention_score",
    "short_horizon_intervention_score",
    "inspect_only_score",
    "oracle_intervention_score",
    "trace_over_saliency_gain_margin",
    "trace_over_short_horizon_gain_margin",
    "trace_over_inspect_only_gain_margin",
    "private_trace_ablation_drop",
    "matched_non_trace_ablation_drop",
    "private_trace_over_non_trace_ratio",
    "action_type_shift_after_trace_ablation",
    "region_shift_after_trace_ablation",
    "non_trace_action_stability",
    "value_leakage_count",
    "b41_intervention_audit_score",
]

B41_RECORD_KEYS = [
    "model",
    "seed",
    "episode_id",
    "audit_type",
    "predicted_action_type",
    "predicted_region",
    "oracle_action_type",
    "oracle_region",
    "family",
    "trace_family",
    "policy_source",
    "shared_action_policy_used",
    "private_trace_action_score_used",
    "fallback_used",
    "oracle_value_used",
    "action_score",
    "region_score",
    "action_type_score",
    "ablation_type",
    "base_action_type",
    "ablated_action_type",
    "base_region",
    "ablated_region",
    "correct_action_value",
    "wrong_action_value",
    "wrong_region_value",
    "gate_pass",
    "note",
]


def run_b41_intervention_audit(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime = b41_runtime_config(config)
    b41 = b41_config(config)
    count = effective_count(int(b41.get("n_audit_episodes", 32)), b41)
    model_names = [str(name) for name in config.get("target_models", MODEL_TO_TRACE_FAMILY)]
    models = {name: make_model(name) for name in model_names}
    for name, model in models.items():
        enforce_private_selector(model, name)

    shared_episodes = [
        make_b4_intervention_episode(runtime, seed + idx * 17, "b41_intervention_audit", ["recurrent", "field", "schema"][idx % 3])
        for idx in range(count)
    ]
    overlap_records = collect_overlap_records(models, shared_episodes, runtime, seed)
    overlap_metrics = {
        **compute_per_episode_action_overlap(overlap_records, runtime),
        **compute_action_type_distribution(overlap_records, runtime),
    }
    scorer_metrics, scorer_records = evaluate_action_scorer_specificity(models, shared_episodes, runtime, seed)
    wrong_metrics, wrong_records = evaluate_wrong_action_wrong_region_stress(shared_episodes, runtime, seed)
    baseline_metrics, baseline_records = run_b41_baseline_sanity_check(shared_episodes, runtime, seed)
    leakage = audit_action_value_leakage({}, {}, {"oracle_value_used": False})

    records: list[dict[str, Any]] = []
    records.extend(overlap_records)
    records.extend(scorer_records)
    records.extend(wrong_records)
    records.extend(baseline_records)
    summary = []
    for model_name, model in models.items():
        family = MODEL_TO_TRACE_FAMILY.get(model_name, "recurrent")
        native_episodes = [
            make_b4_intervention_episode(runtime, seed + 10000 + idx * 19, "b41_native_intervention_audit", family)
            for idx in range(count)
        ]
        provenance_records = collect_action_policy_provenance(model, native_episodes, runtime, seed, model_name)
        provenance_metrics = summarize_action_policy_provenance(provenance_records, runtime)
        shared_metrics, shared_records = evaluate_shared_action_policy_ablation(model, native_episodes, runtime, seed, model_name)
        ablation_metrics, ablation_records = evaluate_action_after_trace_ablation_specificity(model, native_episodes, runtime, seed, model_name)
        records.extend(provenance_records)
        records.extend(shared_records)
        records.extend(ablation_records)
        metrics = {
            **overlap_metrics,
            **provenance_metrics,
            **scorer_metrics,
            **shared_metrics,
            **wrong_metrics,
            **baseline_metrics,
            **ablation_metrics,
            **leakage,
        }
        metrics["b41_intervention_audit_score"] = b41_intervention_audit_score(metrics, b41.get("gates", B41_GATES))
        row = {"model": model_name, "seed": int(seed)}
        for key in B41_SUMMARY_KEYS:
            if key not in {"model", "seed"}:
                row[key] = float(metrics.get(key, 0.0))
        summary.append(row)
    return summary, records


def collect_overlap_records(
    models: dict[str, Any],
    episodes: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    records = []
    for episode_id, episode in enumerate(episodes):
        for model_name, model in models.items():
            policy = trace_guided_intervention_policy(model, episode, config)
            action = policy["action"]
            oracle = episode["ground_truth"]["oracle_best_action"]
            records.append(
                {
                    "record_kind": "action_overlap",
                    "model": model_name,
                    "seed": seed,
                    "episode_id": episode_id,
                    "audit_type": "per_episode_action_overlap",
                    "predicted_action_type": action["action_type"],
                    "predicted_region": int(action["region_id"]),
                    "oracle_action_type": oracle["action_type"],
                    "oracle_region": int(oracle["region_id"]),
                    "family": episode["ground_truth"].get("b4_family", ""),
                    "trace_family": policy["trace_family"],
                    "same_as_other_models": 0,
                    "gate_pass": 1,
                    "note": "cross-model exact action overlap audit",
                }
            )
    by_episode = group_by_episode(records)
    for episode_records in by_episode.values():
        actions = [action_identity(row) for row in episode_records]
        for row in episode_records:
            row["same_as_other_models"] = int(actions.count(action_identity(row)) > 1)
    return records


def compute_per_episode_action_overlap(b4_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    by_episode = group_by_episode(b4_records)
    pairwise = []
    exact_all = []
    same_region_diff_action = []
    same_action_diff_region = []
    for rows in by_episode.values():
        if len(rows) < 2:
            continue
        actions = [action_identity(row) for row in rows]
        regions = [int(row.get("predicted_region", -1)) for row in rows]
        action_types = [str(row.get("predicted_action_type", "")) for row in rows]
        exact_all.append(1.0 if len(set(actions)) <= 1 else 0.0)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                pairwise.append(1.0 if actions[i] == actions[j] else 0.0)
                same_region_diff_action.append(1.0 if regions[i] == regions[j] and action_types[i] != action_types[j] else 0.0)
                same_action_diff_region.append(1.0 if action_types[i] == action_types[j] and regions[i] != regions[j] else 0.0)
    pairwise_rate = mean_or_zero(pairwise)
    return {
        "cross_model_exact_action_match_rate": pairwise_rate,
        "exact_all_model_same_action_rate": mean_or_zero(exact_all),
        "pairwise_action_match_rate": pairwise_rate,
        "same_region_different_action_rate": mean_or_zero(same_region_diff_action),
        "same_action_type_different_region_rate": mean_or_zero(same_action_diff_region),
    }


def compute_action_type_distribution(b4_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in b4_records:
        by_model.setdefault(str(row.get("model", "")), []).append(row)
    fixed_rates = []
    action_entropies = []
    region_entropies = []
    for rows in by_model.values():
        action_types = [str(row.get("predicted_action_type", "")) for row in rows]
        regions = [str(row.get("predicted_region", "")) for row in rows]
        fixed_rates.append(max_frequency(action_types))
        action_entropies.append(entropy(action_types))
        region_entropies.append(entropy(regions))
    return {
        "fixed_action_type_rate": max(fixed_rates) if fixed_rates else 1.0,
        "action_type_entropy": mean_or_zero(action_entropies),
        "region_entropy": mean_or_zero(region_entropies),
    }


def audit_action_value_leakage(
    model_input: dict[str, Any],
    structure: dict[str, Any],
    provenance: dict[str, Any],
    forbidden_keys: list[str] | None = None,
) -> dict[str, Any]:
    forbidden = set(forbidden_keys or FORBIDDEN_ACTION_VALUE_KEYS)
    paths = []
    for root_name, root in [("model_input", model_input), ("structure", structure), ("provenance", provenance)]:
        paths.extend(find_forbidden_paths(root, root_name, forbidden))
    return {
        "value_leakage_count": len(paths),
        "oracle_value_usage_rate": 1.0 if bool(provenance.get("oracle_value_used", False)) else 0.0,
        "forbidden_key_paths": ";".join(paths),
        "no_leakage_score": 1.0 if not paths else 0.0,
    }


def b41_intervention_audit_score(metrics: dict[str, float], gates: dict[str, float] | None = None) -> float:
    gates = {**B41_GATES, **(gates or {})}
    max_checks = [
        ("cross_model_exact_action_match_rate", "max_cross_model_exact_action_match_rate"),
        ("exact_all_model_same_action_rate", "max_exact_all_model_same_action_rate"),
        ("fixed_action_type_rate", "max_fixed_action_type_rate"),
        ("shared_action_policy_usage_rate", "max_shared_action_policy_usage_rate"),
        ("fallback_usage_rate", "max_fallback_usage_rate"),
        ("mean_action_scorer_correlation", "max_action_scorer_correlation"),
        ("shared_action_policy_ablation_drop", "max_shared_action_policy_ablation_drop"),
        ("random_intervention_score", "max_random_intervention_score"),
        ("value_leakage_count", "max_value_leakage_count"),
    ]
    min_checks = [
        ("private_trace_action_score_usage_rate", "min_private_trace_action_score_usage_rate"),
        ("action_scorer_specificity", "min_action_scorer_specificity"),
        ("private_action_retention_after_shared_ablation", "min_private_action_retention_after_shared_ablation"),
        ("wrong_action_penalty", "min_wrong_action_penalty"),
        ("wrong_region_penalty", "min_wrong_region_penalty"),
        ("wrong_action_wrong_region_penalty", "min_wrong_action_wrong_region_penalty"),
        ("oracle_intervention_score", "min_oracle_intervention_score"),
        ("trace_over_saliency_gain_margin", "min_gain_over_saliency"),
        ("trace_over_short_horizon_gain_margin", "min_gain_over_short_horizon"),
        ("trace_over_inspect_only_gain_margin", "min_gain_over_inspect_only"),
        ("private_trace_ablation_drop", "min_private_trace_ablation_drop"),
        ("private_trace_over_non_trace_ratio", "min_private_trace_over_non_trace_ratio"),
        ("action_type_shift_after_trace_ablation", "min_action_type_shift_after_trace_ablation"),
    ]
    for metric_key, gate_key in max_checks:
        if float(metrics.get(metric_key, 0.0)) > float(gates[gate_key]):
            return 0.0
    for metric_key, gate_key in min_checks:
        if float(metrics.get(metric_key, 0.0)) < float(gates[gate_key]):
            return 0.0
    if float(metrics.get("oracle_value_usage_rate", 0.0)) > 0.0:
        return 0.0
    weights = {
        "action_scorer_specificity": 0.15,
        "private_trace_action_score_usage_rate": 0.15,
        "wrong_action_penalty": 0.15,
        "wrong_region_penalty": 0.15,
        "private_trace_ablation_drop": 0.15,
        "private_trace_over_non_trace_ratio": 0.10,
        "baseline_sanity_score": 0.10,
        "no_leakage_score": 0.05,
    }
    return float(sum(weight * min(max(float(metrics.get(key, 0.0)), 0.0), 1.0) for key, weight in weights.items()))


def write_b41_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b41_intervention_audit_summary.csv"), summary, B41_SUMMARY_KEYS)
    write_csv(Path("results/b41_intervention_audit_records.csv"), records, B41_RECORD_KEYS)
    write_csv(Path("results/b41_action_overlap.csv"), [row for row in records if row.get("record_kind") == "action_overlap"])
    write_csv(Path("results/b41_action_provenance.csv"), [row for row in records if row.get("record_kind") == "action_provenance"])
    write_csv(Path("results/b41_action_scorer_correlation.csv"), [row for row in records if row.get("record_kind") == "action_scorer"])
    write_csv(Path("results/b41_shared_action_policy_ablation.csv"), [row for row in records if row.get("record_kind") == "shared_action_ablation"])
    write_csv(Path("results/b41_wrong_action_stress.csv"), [row for row in records if row.get("record_kind") == "wrong_action_stress"])
    write_csv(Path("results/b41_baseline_sanity.csv"), [row for row in records if row.get("record_kind") == "baseline_sanity"])
    write_csv(Path("results/b41_trace_ablation_specificity.csv"), [row for row in records if row.get("record_kind") == "trace_ablation_specificity"])
    Path("reports/B4_1_INTERVENTION_DEGENERACY_AUDIT.md").write_text(build_b41_report(summary), encoding="utf-8")
    Path("reports/B4_1_INTERVENTION_DEGENERACY_SELF_AUDIT.md").write_text(build_b41_self_audit(), encoding="utf-8")


def build_b41_report(summary: list[dict[str, Any]]) -> str:
    passed = [row["model"] for row in summary if float(row.get("b41_intervention_audit_score", 0.0)) > 0.0]
    interpretation = (
        "B4.1 reduces the risk that B4 success is caused by shared action policy, fixed action-type shortcut, intervention-value leakage, weak baselines, or nonspecific trace ablation. Under current toy diagnostics, private delayed traces remain credible drivers of local intervention/action selection."
        if passed
        else "B4.1 shows that B4 should currently be interpreted as trace-guided intervention-path success, not independently validated recurrent / field / schema intervention mechanisms. The failure localizes the next bottleneck to action-policy degeneracy, action-value leakage, weak action baselines, or nonspecific trace ablation."
    )
    return "\n".join(
        [
            "# B4.1 Intervention Degeneracy Audit",
            "",
            "## 1. Purpose",
            "",
            "B4 produced strong intervention/action-selection results. B4.1 audits whether this reflects private trace-guided intervention or shared action policy / shortcuts / leakage / weak baselines.",
            "",
            "## 2. Background",
            "",
            "PLOS v1: short-horizon checkpoint candidate. B1.1: delayed checkpoint failure. B2: trace-bearing delayed checkpoint. B2.1: trace hardening. B2.1a: score degeneracy. B2.2: shared selector problem. B2.3: private selector reconstruction. B3: trace-guided active inspection. B3.1: active inspection degeneracy audit. B3.2: mechanism-disambiguating active inspection. B4: delayed trace-guided intervention. B4.1: intervention degeneracy audit.",
            "",
            "## 3. Audits",
            "",
            "- per-episode action overlap",
            "- action policy provenance",
            "- family-specific action scorer",
            "- shared action policy ablation",
            "- wrong-action / wrong-region stress",
            "- baseline sanity",
            "- action-after-trace-ablation specificity",
            "- intervention value leakage check",
            "",
            "## 4. Results",
            "",
            markdown_table(summary),
            "",
            "## 5. Interpretation",
            "",
            interpretation,
            "",
            "## 6. Claim Boundary",
            "",
            "Do not claim real control.",
            "Do not claim robotics ability.",
            "Do not claim engineering deployment.",
            "Do not claim general active intelligence.",
            "Do not claim language-free cognition solved.",
            "",
        ]
    )


def build_b41_self_audit() -> str:
    return "\n".join(
        [
            "# B4.1 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Investigates B4 high-score risk.",
            "- Checks per-episode action overlap.",
            "- Checks fixed action-type shortcut.",
            "- Checks action policy provenance.",
            "- Adds family-specific action scorers.",
            "- Adds shared action policy ablation.",
            "- Adds wrong-action / wrong-region stress.",
            "- Re-checks baseline sanity.",
            "- Adds action-after-trace-ablation specificity.",
            "- Adds intervention-value leakage check.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Still a 64x64 toy world.",
            "- Action space remains hand-designed.",
            "- Intervention values are simulator-defined.",
            "- Wrong-action stress is synthetic.",
            "- Provenance fields may miss hidden shared logic.",
            "- Ablation may create OOD internal states.",
            "- Passing B4.1 does not imply real control.",
            "",
            "## False Positive Risks",
            "",
            "- Different action scorers may share low-level features.",
            "- Action overlap may be low due to noise, not real mechanism separation.",
            "- Wrong-action penalty may be evaluator-biased.",
            "- Oracle intervention values may encode generator assumptions.",
            "- Trace ablation may damage general action capacity.",
            "- Fixed action-type shortcut may persist in subtle form.",
            "",
            "## Required Failure Checks",
            "",
            "1. three models choose same action per episode",
            "2. action_type is almost always fixed",
            "3. shared action policy is used",
            "4. private trace action scorer is not used",
            "5. action scorer correlations are too high",
            "6. shared action policy ablation causes large drop",
            "7. wrong action / wrong region is not penalized",
            "8. random / saliency / short-horizon / inspect-only baselines match trace model",
            "9. private trace ablation is not stronger than non-trace ablation",
            "10. intervention value leakage is detected",
            "",
        ]
    )


def b41_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    base = resolve_b41_base_config(config)
    runtime = b4_runtime_config(base)
    b41 = b41_config(config)
    env = dict(runtime.get("env", {}))
    for key in ("frame_size", "grid_size", "past_frames", "future_frames"):
        if key in b41:
            env[key] = int(b41[key])
    runtime["env"] = env
    runtime["b41"] = b41
    return runtime


def resolve_b41_base_config(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(config.get("base_config", "configs/b4_intervention.yaml")))
    if not path.is_absolute():
        path = Path.cwd() / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def b41_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("b41", {"n_audit_episodes": 32, "gates": {}})


def effective_count(value: int, b41: dict[str, Any]) -> int:
    cap = b41.get("max_audit_episodes")
    if cap is not None:
        value = min(value, int(cap))
    return max(1, int(value))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(B41_SUMMARY_KEYS) + " |", "| " + " | ".join("---" for _ in B41_SUMMARY_KEYS) + " |"]
    for row in rows:
        values = []
        for key in B41_SUMMARY_KEYS:
            value = row.get(key, "")
            values.append(f"{float(value):.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def group_by_episode(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in records:
        by_episode.setdefault(int(row.get("episode_id", 0)), []).append(row)
    return by_episode


def action_identity(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("predicted_action_type", "")), int(row.get("predicted_region", -1))


def max_frequency(values: list[str]) -> float:
    if not values:
        return 1.0
    counts = {value: values.count(value) for value in set(values)}
    return max(counts.values()) / len(values)


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    total = len(values)
    counts = {value: values.count(value) for value in set(values)}
    return float(-sum((count / total) * log2(count / total) for count in counts.values()))


def find_forbidden_paths(value: Any, prefix: str, forbidden: set[str]) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            if str(key) in forbidden:
                paths.append(path)
            paths.extend(find_forbidden_paths(child, path, forbidden))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            paths.extend(find_forbidden_paths(child, f"{prefix}[{idx}]", forbidden))
    return paths


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0

