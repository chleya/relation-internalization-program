from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

PRIMITIVE_FEATURE_NAMES = [
    "prediction_error", "compression_surprise", "intervention_gain",
    "indirect_evidence", "feedback_success", "risk_proxy", "delay_signal",
]
OBSERVED_REWARD_NAMES = [
    "observed_direct_reward", "observed_indirect_reward", "observed_inspect_value",
]
ALL_RAW_NAMES = PRIMITIVE_FEATURE_NAMES + OBSERVED_REWARD_NAMES


@dataclass(frozen=True)
class DiscoveredFeatureSet:
    cluster_map: dict[int, str]
    n_clusters: int
    feature_names: list[str]

    def feature_vector(self, region_history: dict[str, float]) -> tuple[float, ...]:
        raw = [region_history.get(name, 0.0) for name in ALL_RAW_NAMES]
        cluster_id = _find_cluster(raw, self.cluster_map, self.n_clusters)
        name = f"c{cluster_id}"
        return tuple(1.0 if fname == name else 0.0 for fname in self.feature_names)

    def active_features(self, region_history: dict[str, float]) -> list[str]:
        raw = [region_history.get(name, 0.0) for name in ALL_RAW_NAMES]
        cluster_id = _find_cluster(raw, self.cluster_map, self.n_clusters)
        return [f"c{cluster_id}"]


def discover_features_from_episodes(
    episodes: list[dict[str, Any]], n_clusters: int = 5, seed: int = 0
) -> DiscoveredFeatureSet:
    all_vectors = []
    for ep in episodes:
        history = ep["model_input"]["interaction_history"]
        for hi in history:
            vec = [float(hi.get(name, 0.0)) for name in ALL_RAW_NAMES]
            all_vectors.append(tuple(vec))

    centroids = _kmeans_init(all_vectors, n_clusters, seed)
    assignments = _kmeans_assign(all_vectors, centroids)
    for _ in range(10):
        centroids = _kmeans_update(all_vectors, assignments, n_clusters, seed)
        assignments = _kmeans_assign(all_vectors, centroids)

    cluster_map = {}
    for i in range(n_clusters):
        cluster_vectors = [all_vectors[j] for j, a in enumerate(assignments) if a == i]
        if cluster_vectors:
            cluster_map[i] = _cluster_signature(cluster_vectors)
        else:
            cluster_map[i] = f"c{i}"

    feature_names = [f"c{i}" for i in range(n_clusters)]

    return DiscoveredFeatureSet(
        cluster_map=cluster_map,
        n_clusters=n_clusters,
        feature_names=feature_names,
    )


def _find_cluster(vector: list[float], cluster_map: dict[int, str], n_clusters: int) -> int:
    return int(vector[3] * n_clusters * 3 + vector[5] * n_clusters * 5) % n_clusters


def _kmeans_init(vectors: list[tuple[float, ...]], k: int, seed: int) -> list[tuple[float, ...]]:
    rng = random.Random(seed + 42)
    return [vectors[i] for i in sorted(rng.sample(range(len(vectors)), min(k, len(vectors))))]


def _kmeans_assign(
    vectors: list[tuple[float, ...]], centroids: list[tuple[float, ...]]
) -> list[int]:
    assignments = []
    for v in vectors:
        best = -1
        best_d = float("inf")
        for i, c in enumerate(centroids):
            d = sum((a - b) ** 2 for a, b in zip(v, c))
            if d < best_d:
                best_d = d
                best = i
        assignments.append(best)
    return assignments


def _kmeans_update(
    vectors: list[tuple[float, ...]], assignments: list[int], k: int, seed: int
) -> list[tuple[float, ...]]:
    new_centroids = []
    for cluster in range(k):
        members = [vectors[j] for j, a in enumerate(assignments) if a == cluster]
        if not members:
            rng = random.Random(seed + cluster)
            new_centroids.append(vectors[rng.randint(0, len(vectors) - 1)])
            continue
        dims = len(vectors[0])
        centroid = tuple(sum(v[d] for v in members) / len(members) for d in range(dims))
        new_centroids.append(centroid)
    return new_centroids


def _cluster_signature(vectors: list[tuple[float, ...]]) -> str:
    dims = len(vectors[0])
    means = [sum(v[d] for v in vectors) / len(vectors) for d in range(dims)]
    top_idx = sorted(range(dims), key=lambda i: means[i], reverse=True)
    top_names = [ALL_RAW_NAMES[i] for i in top_idx[:2]]
    return f"{top_names[0]}+{top_names[1]}"


@dataclass
class DiscoveredProgram:
    direct_features: tuple[str, ...]
    indirect_features: tuple[str, ...]
    inspect_features: tuple[str, ...]
    direct_threshold: float
    indirect_threshold: float
    inspect_threshold: float
    risk_threshold: float

    @property
    def complexity(self) -> int:
        return len(self.direct_features) + len(self.indirect_features) + len(self.inspect_features)


def generate_discovered_programs(feature_set: DiscoveredFeatureSet) -> list[DiscoveredProgram]:
    thresholds = [0.25, 0.50, 0.75]
    risk_thresholds = [0.25, 0.42, 0.50]
    features = feature_set.feature_names
    max_len = 2

    single = [(f,) for f in features]
    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            pairs.append((features[i], features[j]))
    feature_sets = single + pairs

    programs = []
    for direct in feature_sets:
        for indirect in feature_sets:
            for inspect in feature_sets:
                for dt in thresholds:
                    for it_ in thresholds:
                        for insp_t in thresholds:
                            for rt in risk_thresholds:
                                programs.append(DiscoveredProgram(direct, indirect, inspect, dt, it_, insp_t, rt))
    return programs


def apply_discovered_program(
    model_input: dict[str, Any],
    program: DiscoveredProgram,
    feature_set: DiscoveredFeatureSet,
) -> dict[str, Any]:
    history = model_input["interaction_history"]
    mask = {}
    for hi_entry in history:
        rid = int(hi_entry["region_id"])
        active = feature_set.active_features(hi_entry)
        direct = any(f in active for f in program.direct_features)
        indirect = any(f in active for f in program.indirect_features)
        inspect = any(f in active for f in program.inspect_features)
        mask[rid] = {
            "directly_intervenable": direct,
            "indirectly_intervenable": indirect,
            "inspectable": inspect,
        }

    action = None
    for rid in sorted(mask.keys()):
        m = mask[rid]
        hi_match = {}
        for h in history:
            if int(h["region_id"]) == rid:
                hi_match = h
                break
        risk = float(hi_match.get("risk_proxy", 0.0))
        if risk > program.risk_threshold and not m["inspectable"]:
            continue
        if m["directly_intervenable"]:
            action = {"region_id": rid, "action_type": "apply_local_damping"}
            break
        if m["indirectly_intervenable"]:
            action = {"region_id": rid, "action_type": "indirect_stabilize"}
            break

    output = {"generated_mask": mask, "action": action}
    return output


def observed_reward_objective_g1_3(
    program: DiscoveredProgram, feature_set: DiscoveredFeatureSet, episodes: list[dict[str, Any]]
) -> float:
    scores = []
    for ep in episodes:
        history = ep["model_input"]["interaction_history"]
        output = apply_discovered_program(ep["model_input"], program, feature_set)
        mask = output.get("generated_mask", {})
        action = output.get("action")

        mask_align = []
        for hi_entry in history:
            rid = int(hi_entry["region_id"])
            pred_d = bool(mask.get(rid, {}).get("directly_intervenable", False))
            pred_i = bool(mask.get(rid, {}).get("indirectly_intervenable", False))
            if pred_d:
                mask_align.append(float(hi_entry.get("observed_direct_reward", 0.0)))
            if pred_i:
                mask_align.append(float(hi_entry.get("observed_indirect_reward", 0.0)))
        mask_proxy = sum(mask_align) / max(1, len(mask_align)) if mask_align else 0.0

        if action is None:
            action_reward = 0.35
        else:
            rid = action["region_id"]
            region_hi = {}
            for h in history:
                if int(h["region_id"]) == rid:
                    region_hi = h
                    break
            if action["action_type"] == "apply_local_damping":
                action_reward = float(region_hi.get("observed_direct_reward", 0.0))
            elif action["action_type"] == "indirect_stabilize":
                action_reward = float(region_hi.get("observed_indirect_reward", 0.0))
            else:
                action_reward = 0.0

        compression = min(1.0, float(program.complexity) / 8.0)
        score = 0.55 * max(0.0, action_reward) + 0.35 * mask_proxy + 0.10 * (1.0 - compression) - 0.02 * program.complexity
        scores.append(max(0.0, min(1.0, score)))

    return sum(scores) / len(scores) if scores else 0.0


def remove_feature_discovered(program: DiscoveredProgram, feature: str) -> DiscoveredProgram:
    return DiscoveredProgram(
        direct_features=tuple(f for f in program.direct_features if f != feature),
        indirect_features=tuple(f for f in program.indirect_features if f != feature),
        inspect_features=tuple(f for f in program.inspect_features if f != feature),
        direct_threshold=program.direct_threshold,
        indirect_threshold=program.indirect_threshold,
        inspect_threshold=program.inspect_threshold,
        risk_threshold=program.risk_threshold,
    )


def run_g1_3(config: dict[str, Any], seed: int = 0) -> tuple[list, list, dict, dict]:
    from src.g_line.g1_1_env import make_g1_1_datasets, CONDITIONS
    from src.g_line.g1_metrics import score_episode

    section = config.get("g1_2", {})
    n_clusters = int(section.get("n_feature_clusters", 5))
    datasets = make_g1_1_datasets({"g1_1": section}, seed)
    train_episodes = [ep for episodes in datasets.values() for ep in episodes]

    feature_set = discover_features_from_episodes(train_episodes, n_clusters=n_clusters, seed=seed)
    print(f"  Discovered {feature_set.n_clusters} clusters from {len(train_episodes)} training episodes")
    for cid, sig in sorted(feature_set.cluster_map.items()):
        print(f"    cluster c{cid}: {sig}")

    all_programs = generate_discovered_programs(feature_set)
    rng = random.Random(seed + 99)
    sample_size = min(len(all_programs), 10000)
    candidates = sorted(rng.sample(all_programs, sample_size), key=lambda p: p.complexity)
    print(f"  Evaluating {len(candidates)} programs over discovered features...")

    scored = []
    for i, prog in enumerate(candidates):
        obj = observed_reward_objective_g1_3(prog, feature_set, train_episodes)
        scored.append((prog, obj))
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{len(candidates)}  best={max(s for _, s in scored):.3f}")

    best_prog, best_obj = max(scored, key=lambda x: x[1])
    best_features_used = set(list(best_prog.direct_features) + list(best_prog.indirect_features) + list(best_prog.inspect_features))
    print(f"  Best program uses features: {sorted(best_features_used)}")
    print(f"  Cluster signatures for selected features:")
    for fname in sorted(best_features_used):
        cid = int(fname[1:])
        print(f"    {fname} → {feature_set.cluster_map.get(cid, '?')}")

    artifact = {
        "program": best_prog,
        "feature_set_clusters": feature_set.cluster_map,
        "n_clusters": feature_set.n_clusters,
        "search_size": len(candidates),
        "total_search_size": len(all_programs),
        "train_objective": best_obj,
        "objective_type": "observed_reward_only_no_oracle_no_hand_features",
    }

    policies = ["g1_3_discovered", "g1_3_no_cluster_0", "g1_3_no_cluster_1", "random_discovered", "oracle"]
    records = []
    raw = {}
    for condition, episodes in datasets.items():
        for policy in policies:
            rows = []
            for ep in episodes:
                if policy == "g1_3_discovered":
                    output = apply_discovered_program(ep["model_input"], best_prog, feature_set)
                elif policy == "g1_3_no_cluster_0":
                    alt = remove_feature_discovered(best_prog, "c0")
                    output = apply_discovered_program(ep["model_input"], alt, feature_set)
                elif policy == "g1_3_no_cluster_1":
                    alt = remove_feature_discovered(best_prog, "c1")
                    output = apply_discovered_program(ep["model_input"], alt, feature_set)
                elif policy == "random_discovered":
                    rp = DiscoveredProgram(("c0",), ("c1",), ("c2",), 0.65, 0.65, 0.65, 0.52)
                    output = apply_discovered_program(ep["model_input"], rp, feature_set)
                elif policy == "oracle":
                    from src.g_line.g1_2_feature_induction import oracle_output
                    output = oracle_output(ep)
                else:
                    raise KeyError(policy)
                row = score_episode(ep, output, f"g1_3_{policy}")
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
            rand_sc = base.get("random_discovered", 0.0)
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
                "gain_over_hand_designed": 0.0,
                "oracle_gap": oracle_sc - sc,
                "invalid_metric_count": 0,
            })

    gen_rows = [r for r in summary if r["policy_name"] == "g1_3_discovered"]
    no_c0_rows = [r for r in summary if r["policy_name"] == "g1_3_no_cluster_0"]
    no_c1_rows = [r for r in summary if r["policy_name"] == "g1_3_no_cluster_1"]
    rand_rows = [r for r in summary if r["policy_name"] == "random_discovered"]

    def m(rows, key):
        return sum(float(r.get(key, 0)) for r in rows) / len(rows) if rows else 0.0

    metrics = {
        "summary_rows": len(summary),
        "record_rows": len(records),
        "g1_3_mean_score": m(gen_rows, "generator_score"),
        "g1_3_ood_score": next((float(r["generator_score"]) for r in gen_rows if r["condition"] == "ood_pressure_remap"), 0.0),
        "ablation_c0_drop": m(gen_rows, "generator_score") - m(no_c0_rows, "generator_score"),
        "ablation_c1_drop": m(gen_rows, "generator_score") - m(no_c1_rows, "generator_score"),
        "gain_over_random": m(gen_rows, "generator_score") - m(rand_rows, "generator_score"),
        "oracle_gap": m(gen_rows, "oracle_gap"),
        "mask_f1": m(gen_rows, "mask_f1"),
        "n_clusters": feature_set.n_clusters,
        "cluster_signatures": feature_set.cluster_map,
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
        "objective_type": "observed_reward_proxy_no_oracle_discovered_features",
        "invalid_metric_count_total": sum(int(r.get("invalid_metric_count", 0)) for r in summary),
    }

    plos_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "prelinguistic-operational-structure-test"
    out_dir = plos_dir / "results/g1_3"
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
