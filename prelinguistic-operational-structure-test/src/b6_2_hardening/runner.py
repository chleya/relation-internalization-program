from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .audit import audit_for_episode, write_audit
from .baselines import BASELINES, get_baseline
from .env import make_b62_datasets
from .metrics import RECORD_FIELDS, SUMMARY_FIELDS, score_output, summarize
from .policy import b62_policy


POLICIES = {"b62_policy": b62_policy, **{name: get_baseline(name) for name in BASELINES}}


def run_b6_2_hardening(config: dict[str, Any], seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    datasets = make_b62_datasets(config, seed)
    summary: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for condition, episodes in datasets.items():
        if not episodes:
            context = {"condition": condition, "seed": seed, "policy_name": "b62_policy", "mask_visibility": "", "hide_indirect_target": 0, "trace_mode": "", "delay_steps": 0}
            summary.append(summarize([], context, {}, {"forbidden_reference_count": 0, "poisoned_ground_truth_invariance_pass": 0, "policy_uses_model_input_only": 1}))
            continue
        audit = audit_for_episode(episodes[0])
        scored_by_policy: dict[str, list[dict[str, Any]]] = {}
        for policy_name, policy in POLICIES.items():
            scored_rows = []
            for episode in episodes:
                output = policy(episode, config)
                scored = {
                    **base_context(episode, seed, policy_name),
                    **score_output(episode, output),
                    "note": "B6.2 fallback-risk/delayed-credit record",
                }
                scored_rows.append(scored)
                records.append(scored)
            scored_by_policy[policy_name] = scored_rows
        baseline_scores = {name: mean(rows, "risk_constrained_score") for name, rows in scored_by_policy.items()}
        baseline_scores["always_abstain_utility"] = mean(scored_by_policy.get("always_abstain", []), "utility_score")
        for policy_name, rows in scored_by_policy.items():
            context = base_summary_context(episodes[0], seed, policy_name)
            summary.append(summarize(rows, context, baseline_scores, audit))
    metrics = build_metrics(summary, records)
    return summary, records, metrics


def write_b6_2_outputs(summary: list[dict[str, Any]], records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    Path("results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    write_csv(Path("results/b6_2_hardening_summary.csv"), summary, SUMMARY_FIELDS)
    write_csv(Path("results/b6_2_hardening_records.csv"), records, RECORD_FIELDS)
    Path("results/b6_2_hardening_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_audit(Path("results/b6_2_audit_summary.json"), summary)
    Path("reports/B6_2_FALLBACK_RISK_AND_DELAYED_CREDIT.md").write_text(build_report(summary, metrics), encoding="utf-8")


def base_context(episode: dict[str, Any], seed: int, policy_name: str) -> dict[str, Any]:
    evaluator = episode["evaluator_ground_truth"]
    return {
        "condition": evaluator["condition"],
        "seed": int(seed),
        "policy_name": policy_name,
        "mask_visibility": evaluator["mask_visibility"],
        "hide_indirect_target": int(evaluator["hide_indirect_target"]),
        "trace_mode": evaluator["trace_mode"],
        "delay_steps": int(evaluator["delay_steps"]),
    }


def base_summary_context(episode: dict[str, Any], seed: int, policy_name: str) -> dict[str, Any]:
    return base_context(episode, seed, policy_name)


def build_metrics(summary: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "summary_rows": len(summary),
        "record_rows": len(records),
        "conditions": sorted({row["condition"] for row in summary}),
        "b62_policy_mean_score": mean([row for row in summary if row["policy_name"] == "b62_policy"], "risk_constrained_score"),
        "oracle_mean_score": mean([row for row in summary if row["policy_name"] == "oracle"], "risk_constrained_score"),
        "random_mean_score": mean([row for row in summary if row["policy_name"] == "random"], "risk_constrained_score"),
    }


def build_report(summary: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# B6.2 Fallback Risk Inference / Delayed Credit Hardening",
            "",
            "## Purpose",
            "",
            "B6.2 addresses audit-derived structural issues before any higher-stage claim: missing masks, delayed indirect credit, wrong trace regions, reduced mask visibility, poisoned evaluator invariance, and stronger baselines.",
            "",
            "## Validation",
            "",
            "Local validation commands:",
            "",
            "```bash",
            "cd prelinguistic-operational-structure-test",
            "pytest -q",
            "python -m src.run_b6_2_hardening --config configs/b6_2_hardening.yaml --seed 0",
            "python -m src.visualize_b6_2 --summary results/b6_2_hardening_summary.csv",
            "git diff --check",
            "```",
            "",
            "## Key Results",
            "",
            f"- b62_policy_mean_score = {metrics.get('b62_policy_mean_score', 0.0):.3f}",
            f"- oracle_mean_score = {metrics.get('oracle_mean_score', 0.0):.3f}",
            f"- random_mean_score = {metrics.get('random_mean_score', 0.0):.3f}",
            "",
            "## Second-Pass Caveats",
            "",
            "B6.2 is a diagnostic layer, not a completed robustness claim. The aggregate mean can hide split-level failures, so B6.2 must be interpreted together with `reports/B6_2_RESULT_REVIEW.md`.",
            "",
            "- wrong_trace is the key unresolved weakness: if b62_policy does not outperform mask_only or state_only, B6.2 does not prove autonomous target correction or robust trace repair.",
            "- hide_indirect_target tests candidate-search fallback. It does not prove hidden indirect causal path discovery unless the policy succeeds without public indirect cues and without oracle target leakage.",
            "- missing_mask improvements are fallback diagnostics. If they depend mostly on public state estimates rather than private trace, the claim remains narrow.",
            "- delayed_indirect improvements are toy outcome-history diagnostics. A high aggregate score can still coexist with weak delayed indirect success if safety or abstention receives partial credit.",
            "- mask_only and state_only baselines must remain visible in the interpretation. If they explain most performance, B6.2 remains partly a mask/state diagnostic benchmark.",
            "",
            "## Audit-Derived Limitations",
            "",
            "1. B6/B6.1 clean mask exposed strong operational cues.",
            "2. previous_trace_state[\"region\"] remains a strong target prior, now explicitly stressed through wrong/missing/ambiguous trace conditions.",
            "3. B6 clean config had n_ood but no active OOD split; B6.2 uses explicit condition-level stress splits instead.",
            "4. Some trace-uncertainty modes may be represented inside a wrong_trace split rather than as standalone missing_trace / ambiguous_trace / low_confidence_trace splits; they must not be reported as independently passed unless generated separately.",
            "5. Hidden indirect causal path discovery is not proven.",
            "6. Delayed credit assignment improvements, if any, are only in toy delayed outcome history.",
            "7. If mask_only remains strong, B6.2 remains partly a mask-diagnostic task.",
            "8. If random/always_abstain remain non-trivial, conservative default behavior still earns partial score.",
            "9. B6.2 does not support real-world risk intelligence, safety certification, robotics ability, or engineering deployment.",
            "",
        ]
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
