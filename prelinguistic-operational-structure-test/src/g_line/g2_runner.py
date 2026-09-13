from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from .g1_metrics import SUMMARY_FIELDS, RECORD_FIELDS
from .g2_env import CONDITIONS, make_g2_datasets
from .g2_generator import (
    CrossObjectRule,
    _generate_cross_object_rules,
    apply_cross_object_rule,
    training_objective_g2,
)
from .g2_metrics import score_episode_g2


def run_g2(config: dict[str, Any], seed: int = 0, hardening: bool = False) -> tuple[list, list, dict, dict]:
    datasets = make_g2_datasets(config, seed)
    train_episodes = datasets["train"]

    all_rules = _generate_cross_object_rules(config)
    if hardening:
        all_rules = [r for r in all_rules if r.use_indirect]
    rng = random.Random(seed + 99)
    sample_n = min(len(all_rules), 5000)
    candidates = sorted(rng.sample(all_rules, sample_n), key=lambda r: r.complexity)
    mode = "hardening (indirect_required)" if hardening else "baseline"
    print(f"  Searching {len(candidates)} cross-object rules [{mode}]...")

    scored = []
    for i, rule in enumerate(candidates):
        obj = training_objective_g2(rule, train_episodes, config)
        scored.append((rule, obj))
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{len(candidates)}  best={max(s for _, s in scored):.3f}")

    best_rule, best_obj = max(scored, key=lambda x: x[1])
    print(f"  Best rule: n_clusters={best_rule.n_clusters} sim={best_rule.similarity_threshold:.2f} "
          f"risk={best_rule.risk_threshold:.2f} priority={best_rule.action_priority} indirect={best_rule.use_indirect}")

    artifact = {
        "rule": best_rule,
        "search_size": len(candidates),
        "train_objective": best_obj,
        "objective_type": "observed_reward_no_oracle",
    }

    policies = ["g2_cross_object", "g2_random_group", "g2_no_indirect", "g2_kill_indirect", "g2_oracle"]
    records = []
    raw = {}
    for condition, episodes in datasets.items():
        for policy in policies:
            rows = []
            for ep in episodes:
                if policy == "g2_cross_object":
                    output = apply_cross_object_rule(ep["model_input"], best_rule, seed=ep["episode_id"])
                elif policy == "g2_random_group":
                    rand_rule = CrossObjectRule(5, "random", 0.50, 0.50, "closest", False)
                    output = apply_cross_object_rule(ep["model_input"], rand_rule, seed=ep["episode_id"] + 5000)
                elif policy == "g2_no_indirect":
                    noind = CrossObjectRule(
                        best_rule.n_clusters, best_rule.grouping_method,
                        best_rule.similarity_threshold, best_rule.risk_threshold,
                        best_rule.action_priority, False,
                    )
                    output = apply_cross_object_rule(ep["model_input"], noind, seed=ep["episode_id"])
                elif policy == "g2_kill_indirect":
                    output = apply_cross_object_rule(ep["model_input"], best_rule, seed=ep["episode_id"])
                    for rid_key in output.get("generated_mask", {}):
                        output["generated_mask"][rid_key]["similar_groups"] = []
                        output["generated_mask"][rid_key]["indirectly_intervenable"] = False
                elif policy == "g2_oracle":
                    output = {"action": None, "generated_mask": {}, "discovered_groups": [], "n_discovered_groups": 0}
                    truth = ep["evaluator_ground_truth"]["regions"]
                    best_reg = None
                    for t in truth:
                        if t["direct_actionable"]:
                            best_reg = t["region_id"]
                            break
                        if t["indirect_actionable"] and best_reg is None:
                            best_reg = t["region_id"]
                    if best_reg is not None:
                        brt = {}
                        for t in truth:
                            if t["region_id"] == best_reg:
                                brt = t
                                break
                        if brt.get("direct_actionable"):
                            output["action"] = {"region_id": best_reg, "action_type": "apply_local_damping"}
                        elif brt.get("indirect_actionable"):
                            output["action"] = {"region_id": best_reg, "action_type": "indirect_stabilize"}
                    output["discovered_groups"] = [
                        [t["region_id"] for t in truth if t["object_id"] == oid]
                        for oid in sorted(set(t["object_id"] for t in truth))
                    ]
                    output["n_discovered_groups"] = len(output["discovered_groups"])
                else:
                    raise KeyError(policy)

                row = score_episode_g2(ep, output, policy)
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
            oracle_sc = base.get("g2_oracle", 1.0)
            rand_sc = base.get("g2_random_group", 0.0)
            srow = {
                "condition": condition,
                "policy_name": policy,
                "sample_count": len(rows),
                "action_utility": sum(float(r.get("action_utility", 0)) for r in rows) / len(rows) if rows else 0.0,
                "mask_f1": sum(float(r.get("mask_f1", 0)) for r in rows) / len(rows) if rows else 0.0,
                "compression_cost": sum(float(r.get("compression_cost", 0)) for r in rows) / len(rows) if rows else 0.0,
                "generator_score": sc,
                "ood_generalization_score": sc if "ood" in condition else 0.0,
                "gain_over_random": sc - rand_sc,
                "gain_over_hand_designed": sc - base.get("g2_no_indirect", 0.0),
                "oracle_gap": oracle_sc - sc,
                "invalid_metric_count": 0,
                "object_discovery_ari": sum(float(r.get("object_discovery_ari", 0)) for r in rows) / len(rows) if rows else 0.0,
                "cross_group_transfer": sum(float(r.get("cross_group_transfer", 0)) for r in rows) / len(rows) if rows else 0.0,
                "edge_discovery_f1": sum(float(r.get("edge_discovery_f1", 0)) for r in rows) / len(rows) if rows else 0.0,
                "group_compression_efficiency": sum(float(r.get("group_compression_efficiency", 0)) for r in rows) / len(rows) if rows else 0.0,
            }
            summary.append(srow)

    gen_rows = [r for r in summary if r["policy_name"] == "g2_cross_object"]
    noind_rows = [r for r in summary if r["policy_name"] == "g2_no_indirect"]
    kill_rows = [r for r in summary if r["policy_name"] == "g2_kill_indirect"]
    rand_rows = [r for r in summary if r["policy_name"] == "g2_random_group"]

    def m(rows, k):
        return sum(float(r.get(k, 0)) for r in rows) / len(rows) if rows else 0.0

    metrics = {
        "g2_mean_score": m(gen_rows, "generator_score"),
        "g2_ood_score": next((float(r["generator_score"]) for r in gen_rows if r["condition"] == "ood_remap"), 0.0),
        "g2_ood_novel_score": next((float(r["generator_score"]) for r in gen_rows if r["condition"] == "ood_novel_object"), 0.0),
        "g2_ood_topology_score": next((float(r["generator_score"]) for r in gen_rows if r["condition"] == "ood_topology_shift"), 0.0),
        "object_discovery_ari": m(gen_rows, "object_discovery_ari"),
        "edge_discovery_f1": m(gen_rows, "edge_discovery_f1"),
        "indirect_ablation_drop": m(gen_rows, "generator_score") - m(noind_rows, "generator_score"),
        "indirect_kill_drop": m(gen_rows, "generator_score") - m(kill_rows, "generator_score"),
        "gain_over_random": m(gen_rows, "generator_score") - m(rand_rows, "generator_score"),
        "oracle_gap": m(gen_rows, "oracle_gap"),
        "mask_f1": m(gen_rows, "mask_f1"),
        "rule": {
            "n_clusters": best_rule.n_clusters,
            "similarity_threshold": best_rule.similarity_threshold,
            "risk_threshold": best_rule.risk_threshold,
            "action_priority": best_rule.action_priority,
            "use_indirect": best_rule.use_indirect,
        },
    }

    print(f"  g2_mean_score       = {metrics['g2_mean_score']:.3f}")
    print(f"  g2_ood_score        = {metrics['g2_ood_score']:.3f}")
    print(f"  g2_ood_novel        = {metrics['g2_ood_novel_score']:.3f}")
    print(f"  g2_ood_topology     = {metrics['g2_ood_topology_score']:.3f}")
    print(f"  object_discovery_ari= {metrics['object_discovery_ari']:.3f}")
    print(f"  edge_discovery_f1   = {metrics['edge_discovery_f1']:.3f}")
    print(f"  indirect_drop       = {metrics['indirect_ablation_drop']:.3f}")
    print(f"  indirect_kill_drop  = {metrics['indirect_kill_drop']:.3f}")
    print(f"  oracle_gap          = {metrics['oracle_gap']:.3f}")

    plos_dir = Path(__file__).resolve().parent.parent.parent
    out_dir = plos_dir / "results/g2_compositional"
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_fields = SUMMARY_FIELDS + ["object_discovery_ari", "cross_group_transfer", "edge_discovery_f1", "group_compression_efficiency"]

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=extra_fields, extrasaction="ignore")
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
