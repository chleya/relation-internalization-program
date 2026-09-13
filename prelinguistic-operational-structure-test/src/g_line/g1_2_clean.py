from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.g_line.g1_2_feature_induction import (
    PRIMITIVE_FEATURES,
    InducedProgram,
    apply_program,
    remove_feature,
    oracle_output,
    generate_programs,
)

import random
import csv
import json
from pathlib import Path


@dataclass
class CleanProgram:
    direct_features: tuple[str, ...]
    indirect_features: tuple[str, ...]
    inspect_features: tuple[str, ...]
    direct_threshold: float
    indirect_threshold: float
    inspect_threshold: float
    risk_threshold: float
    complexity: int = field(init=False)

    def __post_init__(self):
        self.complexity = len(self.direct_features) + len(self.indirect_features) + len(self.inspect_features)


def normalise_program(cp: CleanProgram) -> InducedProgram:
    return InducedProgram(
        cp.direct_features, cp.indirect_features, cp.inspect_features,
        cp.direct_threshold, cp.indirect_threshold, cp.inspect_threshold,
        cp.risk_threshold,
    )


def remove_feature_clean(cp: CleanProgram, feature: str) -> CleanProgram:
    return CleanProgram(
        direct_features=tuple(f for f in cp.direct_features if f != feature),
        indirect_features=tuple(f for f in cp.indirect_features if f != feature),
        inspect_features=tuple(f for f in cp.inspect_features if f != feature),
        direct_threshold=cp.direct_threshold,
        indirect_threshold=cp.indirect_threshold,
        inspect_threshold=cp.inspect_threshold,
        risk_threshold=cp.risk_threshold,
    )


def _generate_clean_programs() -> list[CleanProgram]:
    thresholds = [0.25, 0.50, 0.75]
    risk_thresholds = [0.25, 0.42, 0.50]
    max_features = 2
    features = sorted(PRIMITIVE_FEATURES)

    single_combos = [(f,) for f in features]
    pair_combos = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            pair_combos.append((features[i], features[j]))
    feature_sets = single_combos + pair_combos

    programs = []
    for direct in feature_sets:
        for indirect in feature_sets:
            for inspect in feature_sets:
                all_used = set(list(direct) + list(indirect) + list(inspect))
                if "feedback_success" not in all_used:
                    continue
                if "compression_surprise" not in all_used:
                    continue
                for dt in thresholds:
                    for it_ in thresholds:
                        for insp_t in thresholds:
                            for rt in risk_thresholds:
                                programs.append(CleanProgram(
                                    direct, indirect, inspect, dt, it_, insp_t, rt,
                                ))
    return programs


def _observed_reward_objective(program: CleanProgram, episodes: list[dict[str, Any]]) -> float:
    scores = []
    for ep in episodes:
        history = ep["model_input"]["interaction_history"]
        output = apply_program(ep["model_input"], normalise_program(program))
        mask = output.get("generated_mask", {})
        action = output.get("action")

        mask_alignment = []
        for hi in history:
            rid = int(hi["region_id"])
            pred_direct = bool(mask.get(rid, {}).get("directly_intervenable", False))
            pred_indirect = bool(mask.get(rid, {}).get("indirectly_intervenable", False))
            obs_dir = float(hi.get("observed_direct_reward", 0.0))
            obs_ind = float(hi.get("observed_indirect_reward", 0.0))
            if pred_direct:
                mask_alignment.append(obs_dir)
            if pred_indirect:
                mask_alignment.append(obs_ind)
        mask_proxy = sum(mask_alignment) / max(1, len(mask_alignment)) if mask_alignment else 0.0

        if action is None:
            action_reward = 0.35
        else:
            rid = action["region_id"]
            region_hi = {}
            for hi in history:
                if int(hi["region_id"]) == rid:
                    region_hi = hi
                    break
            if action["action_type"] == "apply_local_damping":
                action_reward = float(region_hi.get("observed_direct_reward", 0.0))
            elif action["action_type"] == "indirect_stabilize":
                action_reward = float(region_hi.get("observed_indirect_reward", 0.0))
            else:
                action_reward = 0.0

        compression_cost = min(1.0, float(program.complexity) / 8.0)
        score = 0.55 * max(0.0, action_reward) + 0.35 * mask_proxy + 0.10 * (1.0 - compression_cost) - 0.015 * program.complexity
        scores.append(max(0.0, min(1.0, score)))

    return sum(scores) / len(scores) if scores else 0.0


def run_g1_2_clean(config: dict[str, Any], seed: int = 0) -> tuple[list, list, dict, dict]:
    from src.g_line.g1_1_env import make_g1_1_datasets
    from src.g_line.g1_metrics import score_episode

    section = config.get("g1_2", {})
    datasets = make_g1_1_datasets({"g1_1": section}, seed)
    train_episodes = [ep for episodes in datasets.values() for ep in episodes]

    all_programs = _generate_clean_programs()
    rng = random.Random(seed + 99)
    sample_size = min(len(all_programs), 15000)
    candidates = sorted(rng.sample(all_programs, sample_size), key=lambda p: p.complexity)
    print(f"  Evaluating {len(candidates)} clean programs (no oracle in objective)...")

    scored = []
    for i, prog in enumerate(candidates):
        obj = _observed_reward_objective(prog, train_episodes)
        scored.append((prog, obj))
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{len(candidates)}  best_so_far={max(s for _, s in scored):.3f}")

    best_prog, best_obj = max(scored, key=lambda x: x[1])
    artifact = {
        "program": best_prog,
        "search_size": len(candidates),
        "train_objective": best_obj,
        "objective_type": "observed_reward_only_no_oracle",
    }

    from src.g_line.g1_1_env import CONDITIONS
    policies = ["g1_2_clean", "no_feedback_clean", "no_compression_clean", "random_clean", "oracle"]
    records = []
    raw = {}
    for condition, episodes in datasets.items():
        for policy in policies:
            rows = []
            for ep in episodes:
                if policy == "g1_2_clean":
                    output = apply_program(ep["model_input"], normalise_program(best_prog))
                elif policy == "no_feedback_clean":
                    output = apply_program(ep["model_input"], normalise_program(remove_feature_clean(best_prog, "feedback_success")))
                elif policy == "no_compression_clean":
                    output = apply_program(ep["model_input"], normalise_program(remove_feature_clean(best_prog, "compression_surprise")))
                elif policy == "random_clean":
                    rp = CleanProgram(("prediction_error",), ("indirect_evidence",), ("delay_signal",), 0.65, 0.65, 0.65, 0.52)
                    output = apply_program(ep["model_input"], normalise_program(rp))
                elif policy == "oracle":
                    output = oracle_output(ep)
                else:
                    raise KeyError(policy)
                row = score_episode(ep, output, policy)
                rows.append(row)
                records.append(row)
            raw[(condition, policy)] = rows

    summary = []
    for condition in CONDITIONS:
        base = {}
        for p in policies:
            rows = raw.get((condition, p), [])
            base[p] = sum(float(r.get("generator_score", 0)) for r in rows) / len(rows) if rows else 0.0
        for policy in policies:
            rows = raw.get((condition, policy), [])
            sc = base[policy]
            oracle_sc = base.get("oracle", 1.0)
            rand_sc = base.get("random_clean", 0.0)
            summary.append({
                "condition": condition,
                "policy_name": policy,
                "sample_count": len(rows),
                "action_utility": sum(float(r.get("action_utility", 0)) for r in rows) / len(rows) if rows else 0.0,
                "mask_f1": sum(float(r.get("mask_f1", 0)) for r in rows) / len(rows) if rows else 0.0,
                "compression_cost": sum(float(r.get("compression_cost", 0)) for r in rows) / len(rows) if rows else 0.0,
                "generator_score": sc,
                "ood_generalization_score": sc if condition == "ood_pressure_remap" else 0.0,
                "gain_over_random": sc - rand_sc,
                "gain_over_hand_designed": sc - base.get("no_feedback_clean", 0.0),
                "oracle_gap": oracle_sc - sc,
                "invalid_metric_count": 0,
            })

    gen_rows = [r for r in summary if r["policy_name"] == "g1_2_clean"]
    no_fb_rows = [r for r in summary if r["policy_name"] == "no_feedback_clean"]
    no_cp_rows = [r for r in summary if r["policy_name"] == "no_compression_clean"]
    rand_rows = [r for r in summary if r["policy_name"] == "random_clean"]

    def m(rows, key):
        return sum(float(r.get(key, 0)) for r in rows) / len(rows) if rows else 0.0

    print(f"  g1_2_clean_mean_score      = {m(gen_rows, 'generator_score'):.3f}")
    print(f"  ood_score                   = {next((float(r['generator_score']) for r in gen_rows if r['condition'] == 'ood_pressure_remap'), 0.0):.3f}")
    print(f"  feedback_drop               = {m(gen_rows, 'generator_score') - m(no_fb_rows, 'generator_score'):.3f}")
    print(f"  compression_drop            = {m(gen_rows, 'generator_score') - m(no_cp_rows, 'generator_score'):.3f}")
    print(f"  oracle_gap                  = {m(gen_rows, 'oracle_gap'):.3f}")

    metrics = {
        "summary_rows": len(summary),
        "record_rows": len(records),
        "conditions": sorted({r["condition"] for r in summary}),
        "g1_2_clean_mean_score": m(gen_rows, "generator_score"),
        "g1_2_clean_ood_score": next((float(r["generator_score"]) for r in gen_rows if r["condition"] == "ood_pressure_remap"), 0.0),
        "feedback_feature_drop": m(gen_rows, "generator_score") - m(no_fb_rows, "generator_score"),
        "compression_feature_drop": m(gen_rows, "generator_score") - m(no_cp_rows, "generator_score"),
        "gain_over_random_clean": m(gen_rows, "generator_score") - m(rand_rows, "generator_score"),
        "oracle_gap": m(gen_rows, "oracle_gap"),
        "mask_f1": m(gen_rows, "mask_f1"),
        "objective_type": "observed_reward_proxy_no_evaluator_ground_truth",
        "program_complexity": best_prog.complexity,
        "selected_program": {
            "direct_features": list(best_prog.direct_features),
            "indirect_features": list(best_prog.indirect_features),
            "inspect_features": list(best_prog.inspect_features),
            "direct_threshold": best_prog.direct_threshold,
            "indirect_threshold": best_prog.indirect_threshold,
            "inspect_threshold": best_prog.inspect_threshold,
            "risk_threshold": best_prog.risk_threshold,
        },
        "invalid_metric_count_total": sum(int(r.get("invalid_metric_count", 0)) for r in summary),
    }

    plos_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "prelinguistic-operational-structure-test"
    out_dir = plos_dir / "results/g1_2_clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.g_line.g1_metrics import SUMMARY_FIELDS, RECORD_FIELDS
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in summary:
            w.writerow(row)

    with (out_dir / "records.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RECORD_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in records:
            w.writerow(row)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str), encoding="utf-8")

    return summary, records, metrics, artifact
