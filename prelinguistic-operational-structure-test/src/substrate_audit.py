from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


SUBSTRATE_PROFILES: dict[str, dict[str, Any]] = {
    "pixel_predictor": {
        "role": "first_order_baseline",
        "observation_only_interface": True,
        "language_or_relation_table_input": False,
        "explicit_object_prior": False,
        "explicit_field_prior": False,
        "schema_head_prior": False,
        "intervenable_structure_claim": False,
        "prior_level": "low",
        "claim_discount": "Prediction behavior only; cannot count as operational structure without intervention evidence.",
    },
    "predictive_coding_model": {
        "role": "prediction_error_substrate",
        "observation_only_interface": True,
        "language_or_relation_table_input": False,
        "explicit_object_prior": False,
        "explicit_field_prior": False,
        "schema_head_prior": False,
        "intervenable_structure_claim": True,
        "prior_level": "medium_prediction_error_prior",
        "claim_discount": "Prediction-error structure is a temporal-compression prior; it must pass local intervention and OOD gates before counting as O.",
    },
    "trajectory_memory": {
        "role": "memory_baseline",
        "observation_only_interface": True,
        "language_or_relation_table_input": False,
        "explicit_object_prior": False,
        "explicit_field_prior": False,
        "schema_head_prior": False,
        "intervenable_structure_claim": False,
        "prior_level": "low_to_medium",
        "claim_discount": "Memory can fit fragments; OOD failure should be interpreted as non-structure.",
    },
    "patch_graph_model": {
        "role": "local_patch_graph_substrate",
        "observation_only_interface": True,
        "language_or_relation_table_input": False,
        "explicit_object_prior": False,
        "explicit_field_prior": False,
        "schema_head_prior": False,
        "intervenable_structure_claim": True,
        "prior_level": "medium_grid_prior",
        "claim_discount": "Patch graph success would support local operational structure, but grid locality is still supplied as a substrate prior.",
    },
    "koopman_model": {
        "role": "low_rank_dynamics_substrate",
        "observation_only_interface": True,
        "language_or_relation_table_input": False,
        "explicit_object_prior": False,
        "explicit_field_prior": False,
        "schema_head_prior": False,
        "intervenable_structure_claim": True,
        "prior_level": "medium_linear_dynamics_prior",
        "claim_discount": "Koopman success would support low-rank dynamic O only if mode interventions are local and OOD-stable.",
    },
    "world_model": {
        "role": "latent_sequence_baseline",
        "observation_only_interface": True,
        "language_or_relation_table_input": False,
        "explicit_object_prior": False,
        "explicit_field_prior": False,
        "schema_head_prior": False,
        "intervenable_structure_claim": False,
        "prior_level": "medium",
        "claim_discount": "Latent dynamics may help behavior, but structure is not localized by default.",
    },
    "slot_model": {
        "role": "object_biased_candidate",
        "observation_only_interface": True,
        "language_or_relation_table_input": False,
        "explicit_object_prior": True,
        "explicit_field_prior": False,
        "schema_head_prior": False,
        "intervenable_structure_claim": True,
        "prior_level": "high_object_prior",
        "claim_discount": "Slot success cannot prove object structure emerged; it tests whether object-form O is usable under an object-biased substrate.",
    },
    "field_model": {
        "role": "field_form_candidate",
        "observation_only_interface": True,
        "language_or_relation_table_input": False,
        "explicit_object_prior": False,
        "explicit_field_prior": True,
        "schema_head_prior": False,
        "intervenable_structure_claim": True,
        "prior_level": "medium_field_prior",
        "claim_discount": "Field success supports field-form O only if local field intervention and OOD gates pass.",
    },
    "flow_checkpoint_model": {
        "role": "flow_checkpoint_substrate",
        "observation_only_interface": True,
        "language_or_relation_table_input": False,
        "explicit_object_prior": False,
        "explicit_field_prior": False,
        "schema_head_prior": True,
        "intervenable_structure_claim": True,
        "prior_level": "high_checkpoint_prior",
        "claim_discount": "Flow-checkpoint success would show that continuity, anomaly, occlusion, and collision checkpoints are usable; it is discounted because checkpoint selection is an architectural prior.",
    },
    "schema_model": {
        "role": "field_first_hybrid_schema_candidate",
        "observation_only_interface": True,
        "language_or_relation_table_input": False,
        "explicit_object_prior": False,
        "explicit_field_prior": True,
        "schema_head_prior": True,
        "intervenable_structure_claim": True,
        "prior_level": "high_field_schema_prior",
        "claim_discount": "Schema success is strongest only when readouts are locally causal, task-aligned, and robust OOD.",
    },
}


def substrate_profiles(model_names: list[str] | None = None) -> list[dict[str, Any]]:
    names = model_names or list(SUBSTRATE_PROFILES)
    return [{"model": name, **SUBSTRATE_PROFILES[name]} for name in names]


def write_substrate_audit(
    model_names: list[str] | None = None,
    csv_path: str | Path = "results/substrate_audit.csv",
    report_path: str | Path = "reports/B_LINE_SUBSTRATE_AUDIT.md",
) -> None:
    rows = substrate_profiles(model_names)
    write_csv(csv_path, rows)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(build_substrate_audit(rows), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_substrate_audit(rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# B-Line Substrate Audit",
            "",
            "## Purpose",
            "",
            "This audit addresses whether a branch starts with operational structure already injected. It cannot prove a substrate is free of all priors. It records the designed priors and how much each result must be discounted.",
            "",
            "## Input Firewall",
            "",
            "All model `forward` and `intervene_structure` calls use an observation-only batch: `past_frames`, `future_horizon`, `frame_size`, and `grid_size`. Forbidden fields include `future_frames`, `ground_truth`, object IDs, event labels, relation labels, language, rules, and relation tables.",
            "",
            "## Branch Audit",
            "",
            _markdown_table(rows),
            "",
            "## Interpretation Rule",
            "",
            "- Low-prior baselines can reject prediction and memory false positives, but they do not expose enough structure to qualify by behavior alone.",
            "- Slot success is discounted because object structure is supplied as an architectural prior.",
            "- Field success is not object-biased, but it is still a field prior rather than a blank substrate.",
            "- Schema success would be strongest only if behavior, local structural intervention, and OOD gates pass together.",
            "- A PLOS claim requires passing gates plus this audit; the audit itself is not evidence of operational structure.",
            "- `results/null_control_summary.csv` is an additional blank/static-frame check for structure emitted without dynamics.",
            "",
        ]
    )


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)
