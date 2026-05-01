from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .b23_private_selectors import enforce_private_selector
from .b4_intervention_policy import MODEL_TO_TRACE_FAMILY
from .b5_closed_loop_env import EPISODE_TYPES, b5_runtime_config, make_b5_closed_loop_episode
from .b5_closed_loop_policy import closed_loop_policy, evaluate_closed_loop_policy
from .b51_baseline_sanity import run_b51_baseline_sanity_check
from .b51_decision_diversity import compute_decision_diversity, compute_shortcut_rates, evaluate_episode_type_conditioned_decisions
from .b51_feedback_revision_specificity import evaluate_feedback_revision_specificity
from .b51_plan_overlap import compute_fixed_closed_loop_plan_rate, compute_per_episode_plan_overlap
from .b51_planning_budget_stress import run_planning_budget_stress
from .b51_policy_provenance import collect_closed_loop_policy_provenance, summarize_closed_loop_policy_provenance
from .b51_trace_update_specificity import evaluate_trace_update_specificity
from .b51_value_leakage_audit import audit_b5_value_leakage, summarize_b5_value_leakage
from .models import make_model


B51_SUMMARY_KEYS = [
    "model",
    "seed",
    "cross_model_exact_plan_match_rate",
    "exact_all_model_same_plan_rate",
    "fixed_closed_loop_plan_rate",
    "inspect_always_rate",
    "intervene_immediately_rate",
    "always_inspect_then_intervene_rate",
    "skip_inspect_when_not_needed_rate",
    "skip_intervention_when_not_needed_rate",
    "decision_diversity_score",
    "shared_closed_loop_policy_usage_rate",
    "private_trace_closed_loop_usage_rate",
    "fallback_usage_rate",
    "oracle_plan_usage_rate",
    "oracle_trace_update_usage_rate",
    "oracle_feedback_revision_usage_rate",
    "trace_update_specificity",
    "trace_update_ablation_drop",
    "trace_update_over_non_trace_ratio",
    "wrong_inspection_update_drop",
    "shuffled_inspection_update_drop",
    "scripted_update_score",
    "model_gain_over_scripted_update",
    "feedback_revision_specificity",
    "feedback_revision_ablation_drop",
    "feedback_revision_over_scripted_ratio",
    "contradictory_feedback_sensitivity",
    "random_closed_loop_score",
    "saliency_closed_loop_score",
    "short_horizon_closed_loop_score",
    "inspect_always_score",
    "intervene_immediately_score",
    "oracle_closed_loop_score",
    "model_gain_over_inspect_always",
    "model_gain_over_intervene_immediately",
    "value_leakage_count",
    "planning_budget_stress_retention",
    "strict_budget_compliance",
    "b51_closed_loop_audit_score",
]

B51_RECORD_KEYS = [
    "model",
    "seed",
    "episode_id",
    "audit_type",
    "episode_type",
    "inspect_decision",
    "inspect_region",
    "trace_before_region",
    "trace_after_inspection_region",
    "intervention_decision",
    "intervention_action_type",
    "intervention_region",
    "trace_after_feedback_region",
    "policy_source",
    "trace_update_source",
    "feedback_revision_source",
    "shared_closed_loop_policy_used",
    "private_trace_used",
    "oracle_plan_used",
    "oracle_trace_update_used",
    "oracle_feedback_revision_used",
    "baseline_name",
    "baseline_score",
    "budget_used",
    "gate_pass",
    "note",
]


def run_b51_closed_loop_audit(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime = b51_runtime_config(config)
    b51 = b51_config(config)
    n_audit = effective_count(int(b51.get("n_audit_episodes", 32)), b51)
    model_names = [str(name) for name in config.get("target_models", MODEL_TO_TRACE_FAMILY)]
    models = {name: make_model(name) for name in model_names}
    for name, model in models.items():
        enforce_private_selector(model, name)

    families = ["recurrent", "field", "schema"]
    episodes = [make_b5_closed_loop_episode(runtime, seed + idx * 17, EPISODE_TYPES[idx % len(EPISODE_TYPES)], families[idx % 3]) for idx in range(n_audit)]
    all_records: list[dict[str, Any]] = []
    policy_records: list[dict[str, Any]] = []
    policy_metrics_by_model: dict[str, dict[str, float]] = {}
    for model_name, model in models.items():
        policy_metrics, records = evaluate_closed_loop_policy(model, episodes, runtime, seed, model_name)
        policy_metrics_by_model[model_name] = policy_metrics
        policy_records.extend(records)
    overlap_metrics = {
        **compute_per_episode_plan_overlap(policy_records, runtime),
        **compute_fixed_closed_loop_plan_rate(policy_records, runtime),
    }
    diversity_metrics = {
        **compute_shortcut_rates(policy_records, runtime),
        **compute_decision_diversity(policy_records, runtime),
        **evaluate_episode_type_conditioned_decisions(policy_records, runtime),
    }
    all_records.extend(normalize_policy_records(policy_records, seed))

    summary = []
    for model_name, model in models.items():
        model_score = float(policy_metrics_by_model[model_name].get("closed_loop_model_score", 0.0))
        provenance_records = collect_closed_loop_policy_provenance(model, episodes, runtime, model_name)
        provenance_metrics = summarize_closed_loop_policy_provenance(provenance_records, runtime)
        trace_metrics, trace_records = evaluate_trace_update_specificity(model, episodes, runtime, seed, model_name)
        feedback_metrics, feedback_records = evaluate_feedback_revision_specificity(model, episodes, runtime, seed, model_name)
        baseline_metrics, baseline_records = run_b51_baseline_sanity_check(episodes, runtime, model_score, seed)
        leakage_records = collect_leakage_records(model, episodes, runtime, model_name)
        leakage_metrics = summarize_b5_value_leakage(leakage_records, runtime)
        budget_metrics, budget_records = run_planning_budget_stress(model, episodes, runtime, model_name)
        metrics = {
            **overlap_metrics,
            **diversity_metrics,
            **provenance_metrics,
            **trace_metrics,
            **feedback_metrics,
            **baseline_metrics,
            **leakage_metrics,
            **budget_metrics,
        }
        metrics["b51_closed_loop_audit_score"] = b51_closed_loop_audit_score(metrics, b51.get("gates", {}))
        row = {"model": model_name, "seed": int(seed)}
        for key in B51_SUMMARY_KEYS:
            if key not in {"model", "seed"}:
                row[key] = float(metrics.get(key, 0.0))
        summary.append(row)
        all_records.extend(normalize_provenance_records(provenance_records, seed))
        all_records.extend(normalize_generic_records(trace_records, seed, "trace_update_specificity"))
        all_records.extend(normalize_generic_records(feedback_records, seed, "feedback_revision_specificity"))
        all_records.extend(normalize_baseline_records(baseline_records, seed, model_name))
        all_records.extend(normalize_leakage_records(leakage_records, seed, model_name))
        all_records.extend(normalize_generic_records(budget_records, seed, "planning_budget_stress"))
    return summary, all_records


def b51_closed_loop_audit_score(metrics: dict[str, float], gates: dict[str, float] | None = None) -> float:
    gates = gates or {}
    max_gates = {
        "cross_model_exact_plan_match_rate": gates.get("max_cross_model_exact_plan_match_rate", 0.80),
        "exact_all_model_same_plan_rate": gates.get("max_exact_all_model_same_plan_rate", 0.80),
        "fixed_closed_loop_plan_rate": gates.get("max_fixed_closed_loop_plan_rate", 0.85),
        "inspect_always_rate": gates.get("max_inspect_always_rate", 0.85),
        "intervene_immediately_rate": gates.get("max_intervene_immediately_rate", 0.85),
        "shared_closed_loop_policy_usage_rate": gates.get("max_shared_closed_loop_policy_usage_rate", 0.05),
        "fallback_usage_rate": gates.get("max_fallback_usage_rate", 0.05),
        "random_closed_loop_score": gates.get("max_random_closed_loop_score", 0.25),
        "value_leakage_count": gates.get("max_value_leakage_count", 0),
        "oracle_plan_usage_rate": gates.get("max_oracle_plan_usage_rate", 0.0),
        "oracle_trace_update_usage_rate": gates.get("max_oracle_trace_update_usage_rate", 0.0),
        "oracle_feedback_revision_usage_rate": gates.get("max_oracle_feedback_revision_usage_rate", 0.0),
    }
    for key, threshold in max_gates.items():
        if float(metrics.get(key, 0.0)) > float(threshold):
            return 0.0
    min_gates = {
        "skip_inspect_when_not_needed_rate": gates.get("min_skip_inspect_when_not_needed_rate", 0.70),
        "skip_intervention_when_not_needed_rate": gates.get("min_skip_intervention_when_not_needed_rate", 0.70),
        "decision_diversity_score": gates.get("min_decision_diversity_score", 0.50),
        "private_trace_closed_loop_usage_rate": gates.get("min_private_trace_closed_loop_usage_rate", 0.95),
        "trace_update_specificity": gates.get("min_trace_update_specificity", 0.70),
        "trace_update_ablation_drop": gates.get("min_trace_update_ablation_drop", 0.20),
        "trace_update_over_non_trace_ratio": gates.get("min_trace_update_over_non_trace_ratio", 1.50),
        "feedback_revision_specificity": gates.get("min_feedback_revision_specificity", 0.65),
        "feedback_revision_ablation_drop": gates.get("min_feedback_revision_ablation_drop", 0.20),
        "feedback_revision_over_scripted_ratio": gates.get("min_feedback_revision_over_scripted_ratio", 1.50),
        "oracle_closed_loop_score": gates.get("min_oracle_closed_loop_score", 0.95),
        "model_gain_over_inspect_always": gates.get("min_model_gain_over_inspect_always", 0.15),
        "model_gain_over_intervene_immediately": gates.get("min_model_gain_over_intervene_immediately", 0.15),
        "model_gain_over_scripted_update": gates.get("min_model_gain_over_scripted_update", 0.10),
        "planning_budget_stress_retention": gates.get("min_planning_budget_stress_retention", 0.70),
        "strict_budget_compliance": gates.get("min_strict_budget_compliance", 1.0),
    }
    for key, threshold in min_gates.items():
        if float(metrics.get(key, 0.0)) < float(threshold):
            return 0.0
    weights = {
        "decision_diversity_score": 0.15,
        "trace_update_specificity": 0.15,
        "feedback_revision_specificity": 0.15,
        "private_trace_closed_loop_usage_rate": 0.15,
        "model_gain_over_inspect_always": 0.10,
        "model_gain_over_intervene_immediately": 0.10,
        "planning_budget_stress_retention": 0.10,
    }
    no_leakage_score = 1.0 if float(metrics.get("value_leakage_count", 1.0)) == 0.0 else 0.0
    return float(sum(weight * min(max(float(metrics.get(key, 0.0)), 0.0), 1.0) for key, weight in weights.items()) + 0.10 * no_leakage_score)


def collect_leakage_records(model: Any, episodes: list[dict[str, Any]], config: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    records = []
    for episode_id, episode in enumerate(episodes):
        output = closed_loop_policy(model, episode, config)
        result = audit_b5_value_leakage(episode, output, output.get("provenance", {}))
        records.append({"record_kind": "value_leakage_audit", "episode_id": episode_id, "model": model_name, **result, "gate_pass": int(result["value_leakage_count"] == 0), "note": "B5 value leakage audit"})
    return records


def normalize_policy_records(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        action_type = str(record.get("predicted_intervention_action_type", "do_nothing"))
        normalized.append(
            {
                "model": record.get("model", ""),
                "seed": seed,
                "episode_id": int(record.get("episode_id", 0)),
                "audit_type": "plan_overlap",
                "episode_type": record.get("episode_type", ""),
                "inspect_decision": "skip" if int(record.get("inspect_skipped", 0)) else "inspect",
                "inspect_region": int(record.get("predicted_inspect_region", -1)),
                "trace_before_region": int(record.get("trace_before_region", -1)),
                "trace_after_inspection_region": int(record.get("trace_after_inspection_region", -1)),
                "intervention_decision": "skip" if action_type == "do_nothing" else "intervene",
                "intervention_action_type": action_type,
                "intervention_region": int(record.get("predicted_intervention_region", -1)),
                "trace_after_feedback_region": int(record.get("trace_after_feedback_region", -1)),
                "budget_used": int(record.get("planning_budget_used", 0)),
                "gate_pass": int(record.get("gate_pass", 0)),
                "note": "closed-loop plan record",
            }
        )
    return normalized


def normalize_provenance_records(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        normalized.append(
            {
                "model": record.get("model", ""),
                "seed": seed,
                "episode_id": int(record.get("episode_id", 0)),
                "audit_type": "policy_provenance",
                "policy_source": record.get("inspect_policy_source", ""),
                "trace_update_source": record.get("trace_update_source", ""),
                "feedback_revision_source": record.get("feedback_revision_source", ""),
                "shared_closed_loop_policy_used": int(record.get("shared_closed_loop_policy_used", False)),
                "private_trace_used": int(record.get("private_trace_used_for_inspect", False)),
                "oracle_plan_used": int(record.get("oracle_plan_used", False)),
                "oracle_trace_update_used": int(record.get("oracle_trace_update_used", False)),
                "oracle_feedback_revision_used": int(record.get("oracle_feedback_revision_used", False)),
                "gate_pass": int(record.get("gate_pass", 0)),
                "note": record.get("note", ""),
            }
        )
    return normalized


def normalize_generic_records(records: list[dict[str, Any]], seed: int, audit_type: str) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        normalized.append(
            {
                "model": record.get("model", ""),
                "seed": seed,
                "episode_id": int(record.get("episode_id", 0)),
                "audit_type": audit_type,
                "trace_before_region": int(record.get("trace_before_region", -1)),
                "trace_after_inspection_region": int(record.get("trace_after_inspection_region", -1)),
                "trace_after_feedback_region": int(record.get("trace_after_feedback_region", -1)),
                "trace_update_source": record.get("trace_update_source", ""),
                "feedback_revision_source": record.get("feedback_revision_source", ""),
                "gate_pass": int(record.get("gate_pass", 0)),
                "note": record.get("note", ""),
            }
        )
    return normalized


def normalize_baseline_records(records: list[dict[str, Any]], seed: int, model_name: str) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        normalized.append(
            {
                "model": model_name,
                "seed": seed,
                "episode_id": int(record.get("episode_id", 0)),
                "audit_type": "baseline_sanity",
                "episode_type": record.get("episode_type", ""),
                "baseline_name": record.get("baseline_name", ""),
                "baseline_score": float(record.get("baseline_value", 0.0)),
                "intervention_action_type": record.get("predicted_intervention_action_type", ""),
                "intervention_region": int(record.get("predicted_intervention_region", -1)),
                "gate_pass": int(record.get("gate_pass", 0)),
                "note": record.get("note", ""),
            }
        )
    return normalized


def normalize_leakage_records(records: list[dict[str, Any]], seed: int, model_name: str) -> list[dict[str, Any]]:
    return [
        {
            "model": model_name,
            "seed": seed,
            "episode_id": int(record.get("episode_id", 0)),
            "audit_type": "value_leakage_audit",
            "gate_pass": int(record.get("gate_pass", 0)),
            "note": record.get("note", ""),
        }
        for record in records
    ]


def write_b51_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b51_closed_loop_audit_summary.csv"), summary, B51_SUMMARY_KEYS)
    write_csv(Path("results/b51_closed_loop_audit_records.csv"), records, B51_RECORD_KEYS)
    for audit_type, path in [
        ("plan_overlap", "results/b51_plan_overlap.csv"),
        ("policy_provenance", "results/b51_policy_provenance.csv"),
        ("decision_diversity", "results/b51_decision_diversity.csv"),
        ("trace_update_specificity", "results/b51_trace_update_specificity.csv"),
        ("feedback_revision_specificity", "results/b51_feedback_revision_specificity.csv"),
        ("baseline_sanity", "results/b51_baseline_sanity.csv"),
        ("value_leakage_audit", "results/b51_value_leakage_audit.csv"),
        ("planning_budget_stress", "results/b51_planning_budget_stress.csv"),
    ]:
        write_csv(Path(path), [row for row in records if row.get("audit_type") == audit_type])
    Path("reports/B5_1_CLOSED_LOOP_DEGENERACY_AUDIT.md").write_text(build_b51_report(summary), encoding="utf-8")
    Path("reports/B5_1_CLOSED_LOOP_DEGENERACY_SELF_AUDIT.md").write_text(build_b51_self_audit(), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_b51_report(summary: list[dict[str, Any]]) -> str:
    passed = [row["model"] for row in summary if float(row.get("b51_closed_loop_audit_score", 0.0)) > 0.0]
    interpretation = (
        "B5.1 reduces the risk that B5 success is caused by fixed scripts, shortcut policies, value leakage, loose budgets, or nonspecific ablations."
        if passed
        else "B5.1 shows that B5 should currently be interpreted as closed-loop path success under toy diagnostics, not genuine adaptive closed-loop operational structure."
    )
    return "\n".join(
        [
            "# B5.1 Closed-Loop Degeneracy Audit",
            "",
            "## 1. Purpose",
            "",
            "B5 produced strong closed-loop results. B5.1 audits whether this reflects genuine private delayed operational trace closed-loop operation or fixed scripts, shortcut policies, oracle-like updates, value leakage, weak baselines, loose planning budget, or nonspecific ablations.",
            "",
            "## 2. Background",
            "",
            "PLOS v1: short-horizon checkpoint candidate. B1.1: delayed checkpoint failure. B2: trace-bearing delayed checkpoint. B2.1: trace hardening. B2.1a: score degeneracy. B2.2: shared selector problem. B2.3: private selector reconstruction. B3: trace-guided active inspection. B3.1: inspection degeneracy audit. B3.2: mechanism-disambiguating active inspection. B4: trace-guided intervention. B4.1: fixed action-type shortcut discovered. B4.2: action-type disambiguation. B5: epistemic-pragmatic closed-loop operation. B5.1: closed-loop degeneracy audit.",
            "",
            "## 3. Audits",
            "",
            "- per-episode plan overlap",
            "- closed-loop policy provenance",
            "- inspect/skip/intervene/skip diversity",
            "- trace-update specificity",
            "- feedback-revision specificity",
            "- baseline sanity",
            "- value leakage audit",
            "- planning-budget stress",
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
            "Do not claim human-like active inference.",
            "Do not claim language-free cognition solved.",
            "",
        ]
    )


def build_b51_self_audit() -> str:
    return "\n".join(
        [
            "# B5.1 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Investigates B5 high-score risk.",
            "- Checks fixed closed-loop script.",
            "- Checks inspect-always shortcut.",
            "- Checks intervene-immediately shortcut.",
            "- Checks closed-loop policy provenance.",
            "- Checks trace-update specificity.",
            "- Checks feedback-revision specificity.",
            "- Checks epistemic/pragmatic value leakage.",
            "- Checks planning-budget robustness.",
            "- Checks baseline sanity.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Still a 64x64 toy world.",
            "- Still only a two-step closed loop.",
            "- Trace update and feedback revision may remain partly engineered.",
            "- Budget stress may not reflect real compute constraints.",
            "- Passing B5.1 does not prove real active inference.",
            "- Passing B5.1 does not prove natural emergence.",
            "",
            "## Required Failure Checks",
            "",
            "1. three models choose same closed-loop plan per episode",
            "2. model always inspects",
            "3. model always intervenes immediately",
            "4. update does not depend on inspection content",
            "5. feedback revision does not depend on consequence",
            "6. oracle/value leakage detected",
            "7. planning budget stress collapses performance",
            "8. inspect-always baseline matches model",
            "9. intervene-immediately baseline matches model",
            "10. trace ablation is nonspecific",
            "",
        ]
    )


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(B51_SUMMARY_KEYS) + " |", "| " + " | ".join("---" for _ in B51_SUMMARY_KEYS) + " |"]
    for row in rows:
        values = []
        for key in B51_SUMMARY_KEYS:
            value = row.get(key, "")
            values.append(f"{float(value):.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def b51_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    if "base_config" not in config:
        runtime = b5_runtime_config(config)
    else:
        path = Path(str(config.get("base_config", "configs/b5_closed_loop.yaml")))
        if not path.is_absolute():
            path = Path.cwd() / path
        with path.open("r", encoding="utf-8") as handle:
            runtime = b5_runtime_config(yaml.safe_load(handle))
    runtime["b51"] = b51_config(config)
    return runtime


def b51_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("b51", {"n_audit_episodes": 32, "gates": {}})


def effective_count(value: int, b51: dict[str, Any]) -> int:
    cap = b51.get("max_eval_episodes")
    if cap is not None:
        value = min(value, int(cap))
    return max(1, int(value))
