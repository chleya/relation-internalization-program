from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .null_controls import NULL_CONTROL_COLS
from .run_experiment import run_one
from .substrate_audit import write_substrate_audit


BEHAVIOR_COLS = [
    "identity_after_occlusion",
    "identity_after_crossing",
    "event_boundary_alignment",
    "intervention_sensitivity",
    "relation_locality",
    "noncausal_region_invariance",
    "critical_region_selection_accuracy",
    "inspection_value_gain",
    "behavior_score",
]

STRUCTURE_COLS = [
    "slot_causal_drop",
    "slot_swap_consistency",
    "event_latent_causal_drop",
    "relation_edge_causal_drop",
    "inspection_map_causal_drop",
    "field_causal_drop",
    "critical_field_locality",
    "noncritical_field_invariance",
    "structure_intervention_score",
]

OOD_COLS = ["ood_trajectory_generalization", "ood_score"]


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], cols: list[str]) -> list[dict[str, Any]]:
    out = []
    models = sorted({row["model"] for row in rows})
    for model in models:
        subset = [row for row in rows if row["model"] == model]
        item = {"model": model, "n": len(subset)}
        for col in cols:
            item[col] = mean(float(row.get(col, 0.0)) for row in subset)
        out.append(item)
    return out


def build_report(
    behavior: list[dict[str, Any]],
    structure: list[dict[str, Any]],
    ood: list[dict[str, Any]],
    overall: list[dict[str, Any]],
) -> str:
    qualified = [row["model"] for row in overall if float(row.get("plos_candidate_score", 0.0)) > 0.0]
    qualification_text = (
        "Qualified PLOS candidates: " + ", ".join(qualified) + "."
        if qualified
        else "No model qualifies as a PLOS candidate in this v1 sweep."
    )
    lines = [
        "# B-Line PLOS Report",
        "",
        "## 1. Research Question",
        "",
        "Can a model form operational structure from continuous 2D dynamics without language labels, verified by behavior, structural intervention, and OOD gates?",
        "",
        "## 2. W/O1/O2/L Framework",
        "",
        "`W` is world process, `O1` is trajectory schema, `O2` is operational structure, and `L` is language overlay. This v1 tests `W -> O1 -> O2`, not `L -> L`.",
        "",
        "## 3. Why Field-First Hybrid Schema Was Chosen",
        "",
        "The main candidate starts from continuous fields and only then reads out sparse schemas. This avoids assuming objects first while retaining intervenable structure.",
        "",
        "## 4. Why Behavior Alone Is Insufficient",
        "",
        "Prediction, tracking, or inspection success can be produced by smoothness, memory, or saliency. PLOS requires behavior plus local structural intervention plus OOD generalization.",
        "",
        "## 5. Environment Description",
        "",
        "A 64x64 world with two moving balls, occlusion/crossing, collision/bounce, hidden local force field, and one budgeted 8x8 inspect action.",
        "",
        "## 6. Model Branches And Roles",
        "",
        "- `pixel_predictor`: first-order frame prediction baseline.",
        "- `predictive_coding_model`: prediction-error substrate candidate.",
        "- `trajectory_memory`: nearest-neighbor trajectory fragment baseline.",
        "- `patch_graph_model`: local patch dynamic graph substrate candidate.",
        "- `koopman_model`: low-rank dynamics substrate candidate.",
        "- `world_model`: latent sequence predictor baseline.",
        "- `slot_model`: object-biased second-order candidate.",
        "- `field_model`: non-object-centric field-form candidate.",
        "- `flow_checkpoint_model`: trajectory-flow checkpoint substrate candidate.",
        "- `schema_model`: field-first hybrid schema main candidate.",
        "",
        "## 7. Behavior Gate Results",
        "",
        _markdown_table(behavior),
        "",
        "## 8. Structural Intervention Results",
        "",
        _markdown_table(structure),
        "",
        "## 9. OOD Results",
        "",
        _markdown_table(ood),
        "",
        "## 10. Interpretation Of Each Branch",
        "",
        *branch_interpretations(overall),
        "",
        "## 11. Whether Any Model Qualifies As PLOS Candidate",
        "",
        qualification_text,
        "",
        "Overall diagnostic summary:",
        "",
        _markdown_table(overall),
        "",
        "A model qualifies only when behavior, at least one relevant structural intervention gate, and OOD gates all pass. A zero score is not failure of the project; it means the current substrate did not satisfy the minimal evidence rule.",
        "",
        "## 12. Substrate Audit Summary",
        "",
        "Model inputs are observation-only: `past_frames`, `future_horizon`, `frame_size`, and `grid_size`. The substrate audit records architectural priors separately. Slot branches are discounted for object priors; field branches are discounted for field priors; schema branches are discounted for field plus schema-head priors. This audit does not prove blank-slate emergence, but it prevents treating injected structure as discovered structure.",
        "",
        "The sweep also writes `results/null_control_summary.csv`. These blank/static controls flag branches that emit strong event or inspection structure when no dynamics are present.",
        "",
        "Read `reports/B_LINE_SUBSTRATE_AUDIT.md` for the full branch-by-branch audit.",
        "",
        "Important caveat: a qualifying score is still subject to structural audit. In this v1 diagnostic, some branches can score high on locality and invariance while causal-drop magnitudes remain small. Treat such a pass as a PLOS candidate, not as settled evidence of strong causal structure.",
        "",
        "## 13. Claim Boundary",
        "",
        "This is a toy diagnostic. It does not claim understanding, world-model intelligence, general physical reasoning, or language-free cognition.",
        "",
    ]
    return "\n".join(lines)


def branch_interpretations(overall: list[dict[str, Any]]) -> list[str]:
    lookup = {row["model"]: row for row in overall}
    templates = {
        "pixel_predictor": "If this branch has behavior without intervention evidence, interpret it as smoothness/prediction behavior, not operational structure.",
        "predictive_coding_model": "Success would suggest prediction-error dynamics can serve as an O1-to-O2 substrate, but only if the residual structures are local, causal, and OOD-stable.",
        "trajectory_memory": "If this branch works in distribution but weakens on OOD, interpret it as trajectory memory rather than schema formation.",
        "patch_graph_model": "Success would suggest local patch dynamics can carry operational structure without object slots, discounted by the built-in grid prior.",
        "koopman_model": "Success would suggest low-rank dynamic modes can support operational structure, discounted by the linear-dynamics prior.",
        "world_model": "Good behavior from this branch is first-order latent prediction unless structure can be localized and intervened on.",
        "slot_model": "Slot success would support an object-form O candidate only if slot interventions are local and task-aligned.",
        "field_model": "Field success would support field-form O if field perturbations produce local, predictable, task-aligned drops.",
        "flow_checkpoint_model": "Success would support trajectory-checkpoint O, but it is discounted because checkpoint selection is an architectural prior.",
        "schema_model": "The schema model is the strongest candidate only if it passes behavior, intervention, and OOD gates together.",
    }
    lines = []
    for model in sorted(lookup):
        score = float(lookup[model].get("plos_candidate_score", 0.0))
        status = "qualifies" if score > 0.0 else "does not qualify"
        lines.append(f"- `{model}` {status}. {templates.get(model, 'Interpret only through the stated gates.')}")
    return lines


def build_self_audit() -> str:
    return "\n".join(
        [
            "# B-Line PLOS Self-Audit",
            "",
            "## What This Improves",
            "",
            "- Tests `W -> O1 -> O2` rather than `L -> L`.",
            "- Separates behavior success from structure evidence.",
            "- Adds structural intervention gates.",
            "- Adds field-based branches to avoid object-centric bias.",
            "- Keeps language out of v1.",
            "- Adds an observation-only model input firewall.",
            "- Adds a substrate audit for injected object, field, and schema priors.",
            "",
            "## Remaining Weaknesses",
            "",
            "- Toy world is simple.",
            "- Ground truth comes from simulator.",
            "- Auxiliary heads may bias structure.",
            "- Structural interventions may miss distributed representations.",
            "- Field model may secretly encode object-like structure.",
            "- Slot model has built-in object bias.",
            "- Field and schema branches still contain architectural priors, not blank-slate emergence.",
            "- Flow-checkpoint success is high-prior checkpoint structure, not blank-slate emergence.",
            "- Current intervention metrics can under-measure sparse local causal effects and over-credit locality/invariance.",
            "- Passing does not imply real-world physical intelligence.",
            "",
            "## False Positive Risks",
            "",
            "- Pixel predictor may exploit smoothness.",
            "- Trajectory memory may overfit common fragments.",
            "- Slot model may pass because objects are built in.",
            "- Field model may pass because force field is too simple.",
            "- Inspection policy may learn saliency rather than information value.",
            "- Intervention may create out-of-distribution hidden states.",
            "",
            "## Required Failure Checks",
            "",
            "1. Behavior high but structure intervention low.",
            "2. Structure intervention high but behavior low.",
            "3. OOD collapse.",
            "4. Pixel predictor passes all gates.",
            "5. Slot-only success with field failure.",
            "6. Field-only success with slot failure.",
            "7. Schema model succeeds only through one brittle head.",
            "",
        ]
    )


def build_substrate_search_report(overall: list[dict[str, Any]], null_controls: list[dict[str, Any]]) -> str:
    new_models = {"predictive_coding_model", "patch_graph_model", "koopman_model", "flow_checkpoint_model"}
    new_rows = [row for row in overall if row["model"] in new_models]
    null_rows = [row for row in null_controls if row["model"] in new_models]
    qualified = [row["model"] for row in new_rows if float(row.get("plos_candidate_score", 0.0)) > 0.0]
    verdict = (
        "At least one new substrate qualifies in this sweep: " + ", ".join(qualified) + "."
        if qualified
        else "No new substrate qualifies as a PLOS candidate in this sweep."
    )
    return "\n".join(
        [
            "# B-Line Substrate Search",
            "",
            "## Question",
            "",
            "Can a less object-biased substrate reach the PLOS evidence rule: behavior + structural intervention + OOD?",
            "",
            "## New Substrates",
            "",
            "- `predictive_coding_model`: uses temporal prediction error and surprise fields as an O1 substrate.",
            "- `patch_graph_model`: learns local patch-to-patch dynamics without object slots.",
            "- `koopman_model`: learns a low-rank linear dynamic basis from observation-only frame pairs.",
            "- `flow_checkpoint_model`: extracts continuity, anomaly, occlusion, and predicted-contact checkpoints from past frames.",
            "",
            "These are not blank-slate substrates. They are candidate bases with explicit prior discounts recorded in `B_LINE_SUBSTRATE_AUDIT.md`.",
            "",
            "## Result",
            "",
            verdict,
            "",
            "Caveat: `flow_checkpoint_model` is a PLOS candidate under the current v1 gates, but it has a high checkpoint-selection prior. Its causal-drop metrics remain small, so the result should be treated as a candidate foothold requiring hardening, not as a final internalization claim.",
            "",
            _markdown_table(new_rows),
            "",
            "## Null Controls",
            "",
            "Blank/static-frame controls test whether a substrate emits structure without dynamics.",
            "",
            _markdown_table(null_rows),
            "",
            "## Interpretation",
            "",
            "A zero PLOS score means the candidate did not pass the full gate. It does not mean the substrate is useless; it means current evidence is insufficient to call it pre-linguistic operational structure.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = []
    for seed in config.get("seeds", [config.get("seed", 0)]):
        for model in config["models"]:
            rows.append(run_one(args.config, model, int(seed)))

    overall_cols = sorted({key for row in rows for key in row if key not in {"seed"}})
    ordered_rows = [{key: row.get(key, "") for key in ["model", "seed", *[c for c in overall_cols if c != "model"]]} for row in rows]
    behavior = summarize(rows, BEHAVIOR_COLS)
    structure = summarize(rows, STRUCTURE_COLS)
    ood = summarize(rows, OOD_COLS)
    overall = summarize(rows, ["behavior_score", "structure_intervention_score", "ood_score", "plos_candidate_score"])
    write_csv("results/records.csv", ordered_rows)
    write_csv("results/behavior_summary.csv", behavior)
    write_csv("results/structure_intervention_summary.csv", structure)
    write_csv("results/ood_summary.csv", ood)
    null_controls = summarize(rows, NULL_CONTROL_COLS)
    write_csv("results/null_control_summary.csv", null_controls)
    write_csv("results/overall_summary.csv", overall)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/B_LINE_PLOS_REPORT.md").write_text(build_report(behavior, structure, ood, overall), encoding="utf-8")
    Path("reports/B_LINE_SELF_AUDIT.md").write_text(build_self_audit(), encoding="utf-8")
    Path("reports/B_LINE_SUBSTRATE_SEARCH.md").write_text(build_substrate_search_report(overall, null_controls), encoding="utf-8")
    write_substrate_audit(list(config["models"]))
    for row in overall:
        print(f"{row['model']}: plos_candidate_score={row['plos_candidate_score']:.3f}")


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        vals = []
        for header in headers:
            value = row.get(header, "")
            vals.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
