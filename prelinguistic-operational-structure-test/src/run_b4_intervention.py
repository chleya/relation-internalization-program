from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from .b23_private_selectors import enforce_private_selector
from .b4_intervention_baselines import evaluate_b4_baselines
from .b4_intervention_env import b4_config, b4_runtime_config, make_b4_intervention_episode
from .b4_intervention_metrics import B4_GATES, B4_SUMMARY_KEYS, b4_intervention_score
from .b4_intervention_policy import MODEL_TO_TRACE_FAMILY, evaluate_trace_guided_intervention_policy
from .b4_trace_ablation_eval import evaluate_action_after_trace_ablation
from .models import make_model


B4_RECORD_KEYS = [
    "model",
    "seed",
    "episode_id",
    "episode_type",
    "family",
    "delay",
    "true_trace_region",
    "saliency_region",
    "short_horizon_region",
    "predicted_action_type",
    "predicted_region",
    "oracle_action_type",
    "oracle_region",
    "family_expected_action_type",
    "family_expected_region",
    "baseline_outcome_error",
    "intervened_outcome_error",
    "outcome_improvement",
    "wrong_region_value",
    "wrong_region_penalty",
    "trace_ablated_action_type",
    "trace_ablated_region",
    "trace_ablation_drop",
    "gate_pass",
    "note",
]


def run_b4_intervention(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime = b4_runtime_config(config)
    b4 = b4_config(config)
    n_test = effective_count(int(b4.get("n_test", 32)), b4)
    n_ood = effective_count(int(b4.get("n_ood", 32)), b4)
    model_names = [str(name) for name in config.get("target_models", MODEL_TO_TRACE_FAMILY)]
    models = {name: make_model(name) for name in model_names}
    for name, model in models.items():
        enforce_private_selector(model, name)

    payloads: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for model_name, model in models.items():
        family = MODEL_TO_TRACE_FAMILY.get(model_name, "recurrent")
        episodes = make_family_episodes(runtime, seed, n_test, family, "b4_trace_guided_intervention", offset=0)
        ood_episodes = make_family_episodes(runtime, seed, n_ood, family, "b4_delay_ood_intervention", offset=20000)
        policy_metrics, policy_records = evaluate_trace_guided_intervention_policy(model, episodes, runtime, seed, model_name)
        baseline_metrics, baseline_records = evaluate_b4_baselines(episodes, runtime, seed)
        ablation_metrics, ablation_records = evaluate_action_after_trace_ablation(model, episodes, runtime, seed, model_name)
        ood_metrics, ood_records = evaluate_trace_guided_intervention_policy(model, ood_episodes, runtime, seed, model_name)
        records.extend(policy_records)
        records.extend({"model": model_name, **row} for row in baseline_records)
        records.extend(ablation_records)
        records.extend({"record_kind": "ood", **row} for row in ood_records)
        payloads[model_name] = {
            "family": family,
            "metrics": {
                **policy_metrics,
                **baseline_metrics,
                **ablation_metrics,
                "delay_ood_intervention_accuracy": ood_metrics["trace_guided_intervention_accuracy"],
            },
        }

    family_accs = {
        payload["family"]: float(payload["metrics"].get("trace_guided_intervention_accuracy", 0.0))
        for payload in payloads.values()
    }
    family_specific = mean_or_zero(list(family_accs.values()))
    summary = []
    for model_name, payload in payloads.items():
        metrics = dict(payload["metrics"])
        metrics.update(
            {
                "family_specific_intervention_accuracy": family_specific,
                "recurrent_intervention_accuracy": family_accs.get("recurrent", 0.0),
                "field_intervention_accuracy": family_accs.get("field", 0.0),
                "schema_intervention_accuracy": family_accs.get("schema", 0.0),
            }
        )
        metrics["b4_intervention_score"] = b4_intervention_score(metrics, b4.get("gates", B4_GATES))
        row = {"model": model_name, "seed": int(seed)}
        for key in B4_SUMMARY_KEYS:
            if key not in {"model", "seed"}:
                row[key] = float(metrics.get(key, 0.0))
        summary.append(row)
        records.append(
            {
                "record_kind": "family",
                "model": model_name,
                "seed": seed,
                "family": payload["family"],
                "family_specific_intervention_accuracy": family_specific,
                "recurrent_intervention_accuracy": family_accs.get("recurrent", 0.0),
                "field_intervention_accuracy": family_accs.get("field", 0.0),
                "schema_intervention_accuracy": family_accs.get("schema", 0.0),
            }
        )
    return summary, records


def make_family_episodes(
    config: dict[str, Any],
    seed: int,
    count: int,
    family: str,
    episode_type: str,
    offset: int,
) -> list[dict[str, Any]]:
    return [
        make_b4_intervention_episode(config, seed + offset + idx * 17, episode_type, family)
        for idx in range(count)
    ]


def write_b4_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b4_intervention_summary.csv"), summary, B4_SUMMARY_KEYS)
    write_csv(Path("results/b4_intervention_records.csv"), records, B4_RECORD_KEYS)
    write_csv(Path("results/b4_baseline_comparison.csv"), [row for row in records if row.get("record_kind") == "baseline"])
    write_csv(Path("results/b4_trace_ablation_results.csv"), [row for row in records if row.get("record_kind") == "trace_ablation"])
    write_csv(Path("results/b4_wrong_region_penalty.csv"), [row for row in records if row.get("record_kind") == "policy"])
    write_csv(Path("results/b4_family_intervention_results.csv"), [row for row in records if row.get("record_kind") == "family"])
    Path("reports/B4_DELAYED_TRACE_GUIDED_INTERVENTION.md").write_text(build_b4_report(summary), encoding="utf-8")
    Path("reports/B4_INTERVENTION_SELF_AUDIT.md").write_text(build_b4_self_audit(), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_b4_report(summary: list[dict[str, Any]]) -> str:
    passed = [row["model"] for row in summary if float(row.get("b4_intervention_score", 0.0)) > 0.0]
    interpretation = (
        "B4 supports that, in the toy PLOS environment, B3.2 private delayed traces can guide local intervention/action selection under a constrained action budget. The models choose both intervention region and action type, outperform random/saliency/short-horizon/inspect-only baselines, show outcome improvement, and lose action performance after trace ablation."
        if passed
        else "B4 shows that private delayed traces can guide inspection, but not yet local intervention/action selection under the current minimal action space."
    )
    return "\n".join(
        [
            "# B4 Delayed Trace-Guided Intervention / Action Selection",
            "",
            "## 1. Purpose",
            "",
            "B4 tests whether B3.2 private delayed traces can guide local intervention/action selection under a constrained action budget.",
            "",
            "## 2. Background",
            "",
            "PLOS v1: short-horizon checkpoint candidate. B1.1: delayed checkpoint failure. B2: trace-bearing delayed checkpoint. B2.1: trace hardening. B2.1a: score degeneracy. B2.2: shared selector problem. B2.3: private selector reconstruction. B3: trace-guided active inspection. B3.1: active inspection degeneracy audit. B3.2: mechanism-disambiguating active inspection. B4: trace-guided intervention/action selection.",
            "",
            "## 3. Task",
            "",
            "The model must select one local intervention action: action type and region id under budget = 1.",
            "",
            "## 4. Minimal Action Space",
            "",
            "- do_nothing",
            "- inspect_only",
            "- apply_local_damping",
            "- apply_local_push",
            "- block_force_region",
            "- stabilize_trace_region",
            "",
            "## 5. Models",
            "",
            "- recurrent_flow_checkpoint_model",
            "- field_memory_model",
            "- schema_memory_model",
            "",
            "## 6. Baselines",
            "",
            "- random intervention",
            "- saliency intervention",
            "- short-horizon intervention",
            "- inspect-only",
            "- oracle intervention",
            "",
            "## 7. Metrics",
            "",
            "- trace-guided intervention accuracy",
            "- region accuracy",
            "- action type accuracy",
            "- outcome improvement",
            "- intervention-vs-inspection gain",
            "- wrong-region penalty sensitivity",
            "- family-specific intervention accuracy",
            "- trace ablation intervention drop",
            "- baseline gains",
            "- delay OOD intervention",
            "",
            "## 8. Results",
            "",
            markdown_table(summary),
            "",
            "## 9. Interpretation",
            "",
            interpretation,
            "",
            "## 10. Claim Boundary",
            "",
            "Do not claim real control.",
            "Do not claim robotics deployment.",
            "Do not claim real engineering intervention.",
            "Do not claim general active intelligence.",
            "Do not claim language-free cognition solved.",
            "",
        ]
    )


def build_b4_self_audit() -> str:
    return "\n".join(
        [
            "# B4 Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Moves from inspection to local intervention/action selection.",
            "- Keeps action space minimal.",
            "- Tests intervention vs inspection distinction.",
            "- Adds wrong-region intervention penalty.",
            "- Adds family-specific intervention targets.",
            "- Adds trace-ablation action test.",
            "- Adds random/saliency/short-horizon/inspect-only/oracle baselines.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Still a 64x64 toy world.",
            "- Actions are simplified.",
            "- Intervention values are simulator-defined.",
            "- No continuous control.",
            "- No real robot or real engineering environment.",
            "- Family-specific intervention targets are synthetic.",
            "- Passing B4 does not imply deployable intervention intelligence.",
            "",
            "## False Positive Risks",
            "",
            "- Action type may be inferred from episode type shortcut or model-family priors.",
            "- Region may be selected correctly but action type may be trivial.",
            "- Inspect-only baseline may be too weak.",
            "- Oracle intervention values may encode evaluator bias.",
            "- Wrong-region penalty may be too easy.",
            "- Trace ablation may damage general capacity, not action selection specifically.",
            "- Family-specific targets may leak through synthetic generator regularities.",
            "",
            "## Required Failure Checks",
            "",
            "1. random baseline passes",
            "2. saliency baseline matches trace model",
            "3. short-horizon baseline matches trace model",
            "4. inspect-only matches intervention",
            "5. wrong-region intervention is not penalized",
            "6. trace ablation does not reduce action performance",
            "7. oracle intervention score is low",
            "8. action type accuracy is low despite region accuracy",
            "",
        ]
    )


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(B4_SUMMARY_KEYS) + " |", "| " + " | ".join("---" for _ in B4_SUMMARY_KEYS) + " |"]
    for row in rows:
        values = []
        for key in B4_SUMMARY_KEYS:
            value = row.get(key, "")
            values.append(f"{float(value):.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def effective_count(value: int, b4: dict[str, Any]) -> int:
    cap = b4.get("max_eval_episodes")
    if cap is not None:
        value = min(value, int(cap))
    return max(1, int(value))


def mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b4_intervention.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, records = run_b4_intervention(config, seed=args.seed)
    write_b4_outputs(summary, records)
    best = max(float(row.get("b4_intervention_score", 0.0)) for row in summary) if summary else 0.0
    print(f"best_b4_intervention_score={best:.3f}")


if __name__ == "__main__":
    main()
