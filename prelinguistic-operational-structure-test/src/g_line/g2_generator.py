from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .g2_env import _OBJECT_LATENT_PROFILES

PRIMITIVE_FEATURE_NAMES = [
    "prediction_error", "compression_surprise", "intervention_gain",
    "indirect_evidence", "feedback_success", "risk_proxy", "delay_signal",
]
OBSERVED_REWARD_NAMES = [
    "observed_direct_reward", "observed_indirect_reward", "observed_inspect_value",
]
ALL_OBSERVABLE = PRIMITIVE_FEATURE_NAMES + OBSERVED_REWARD_NAMES
N_OBJECT_TYPES = len(_OBJECT_LATENT_PROFILES)


def _agg(values: list[float], method: str) -> float:
    if not values:
        return 0.0
    if method == "mean":
        return sum(values) / len(values)
    if method == "max":
        return max(values)
    if method == "min":
        return min(values)
    return sum(values) / len(values)


def _group_regions_by_feature_profile(
    history: list[dict[str, Any]], n_clusters: int, seed: int
) -> tuple[list[list[int]], dict[int, tuple[float, ...]]]:
    vectors = []
    rids = []
    for hi in history:
        v = tuple(float(hi.get(name, 0.0)) for name in ALL_OBSERVABLE)
        vectors.append(v)
        rids.append(int(hi["region_id"]))

    dim = len(vectors[0]) if vectors else 10
    rng = __import__("random").Random(seed + 77)
    centroids = [vectors[i] for i in sorted(rng.sample(range(len(vectors)), min(n_clusters, len(vectors))))]

    for _ in range(15):
        assignments = []
        for v in vectors:
            best_d, best_c = float("inf"), 0
            for ci, c in enumerate(centroids):
                d = sum((a - b) ** 2 for a, b in zip(v, c))
                if d < best_d:
                    best_d, best_c = d, ci
            assignments.append(best_c)
        new_c = []
        for ci in range(n_clusters):
            members = [vectors[j] for j, a in enumerate(assignments) if a == ci]
            if not members:
                new_c.append(centroids[ci])
            else:
                new_c.append(tuple(sum(v[d] for v in members) / len(members) for d in range(dim)))
        centroids = new_c

    final_assignments = []
    for v in vectors:
        best_d, best_c = float("inf"), 0
        for ci, c in enumerate(centroids):
            d = sum((a - b) ** 2 for a, b in zip(v, c))
            if d < best_d:
                best_d, best_c = d, ci
        final_assignments.append(best_c)

    groups = [[] for _ in range(n_clusters)]
    for rid, cluster in zip(rids, final_assignments):
        groups[cluster].append(rid)

    cluster_signatures = {}
    for ci in range(n_clusters):
        members = [vectors[j] for j, a in enumerate(final_assignments) if a == ci]
        if members:
            cluster_signatures[ci] = tuple(sum(v[d] for v in members) / len(members) for d in range(dim))
        else:
            cluster_signatures[ci] = tuple(0.0 for _ in range(dim))

    return groups, cluster_signatures


def _compute_similarity_matrix(
    groups: list[list[int]], history: list[dict[str, Any]]
) -> tuple[list[list[float]], int]:
    n = len(groups)
    sim = [[0.0] * n for _ in range(n)]
    group_features = {}
    for gi, gids in enumerate(groups):
        inter = [h for h in history if int(h["region_id"]) in gids]
        if inter:
            group_features[gi] = tuple(
                sum(float(h.get(name, 0.0)) for h in inter) / len(inter)
                for name in ALL_OBSERVABLE
            )
        else:
            group_features[gi] = tuple(0.0 for _ in ALL_OBSERVABLE)

    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = group_features[i], group_features[j]
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(vi, vj)))
            sim_i = 1.0 / (1.0 + d)
            sim[i][j] = sim_i
            sim[j][i] = sim_i
        sim[i][i] = 1.0

    active = sum(1 for g in groups if g)
    return sim, active


@dataclass
class CrossObjectRule:
    n_clusters: int
    grouping_method: str
    similarity_threshold: float
    risk_threshold: float
    action_priority: str
    use_indirect: bool

    @property
    def complexity(self) -> int:
        return 4 + int(self.use_indirect)


def _generate_cross_object_rules(config: dict[str, Any]) -> list[CrossObjectRule]:
    section = config.get("g2", {})
    n_clusters_options = [4, 5, 6]
    sim_thresholds = [0.25, 0.50, 0.75]
    risk_thresholds = [0.25, 0.42, 0.50]
    action_priorities = ["closest", "best_similarity", "lowest_risk"]
    use_indirect_options = [True, False]

    rules = []
    for nc in n_clusters_options:
        for st in sim_thresholds:
            for rt in risk_thresholds:
                for ap in action_priorities:
                    for ui in use_indirect_options:
                        rules.append(CrossObjectRule(nc, "kmeans_adaptive", st, rt, ap, ui))
    return rules


def apply_cross_object_rule(
    model_input: dict[str, Any],
    rule: CrossObjectRule,
    seed: int,
) -> dict[str, Any]:
    history = model_input["interaction_history"]
    groups, cluster_sigs = _group_regions_by_feature_profile(history, rule.n_clusters, seed)
    sim_matrix, n_active = _compute_similarity_matrix(groups, history)

    mask = {}
    for gi, gids in enumerate(groups):
        for rid in gids:
            similar_groups = [gj for gj in range(len(groups)) if sim_matrix[gi][gj] >= rule.similarity_threshold and gj != gi]
            mask[rid] = {
                "discovered_group": gi,
                "similar_groups": similar_groups,
                "directly_intervenable": True,
                "indirectly_intervenable": rule.use_indirect and len(similar_groups) > 0,
                "inspectable": True,
            }

    action = None
    scored_candidates = []
    for rid in sorted(mask.keys()):
        m = mask[rid]
        ri_hist = {}
        for h in history:
            if int(h["region_id"]) == rid:
                ri_hist = h
                break
        risk_val = float(ri_hist.get("risk_proxy", 0.0))
        if risk_val > rule.risk_threshold and not m["inspectable"]:
            continue

        sim_groups = m.get("similar_groups", [])
        if rule.action_priority == "closest":
            priority = float(ri_hist.get("intervention_gain", 0.0))
        elif rule.action_priority == "best_similarity":
            priority = max((sim_matrix[m["discovered_group"]][sg] for sg in sim_groups), default=0.0)
        elif rule.action_priority == "lowest_risk":
            priority = 1.0 - risk_val
        else:
            priority = float(ri_hist.get("intervention_gain", 0.0))

        scored_candidates.append((priority, rid))

    if scored_candidates:
        scored_candidates.sort(reverse=True)
        _, best_rid = scored_candidates[0]
        m = mask[best_rid]
        if m["indirectly_intervenable"]:
            action = {"region_id": best_rid, "action_type": "indirect_stabilize"}
        elif m["directly_intervenable"]:
            action = {"region_id": best_rid, "action_type": "apply_local_damping"}

    return {
        "generated_mask": mask,
        "action": action,
        "discovered_groups": [g for g in groups if g],
        "n_discovered_groups": sum(1 for g in groups if g),
        "similarity_matrix": sim_matrix,
    }


def training_objective_g2(
    rule: CrossObjectRule, episodes: list[dict[str, Any]], config: dict[str, Any]
) -> float:
    scores = []
    for i, ep in enumerate(episodes):
        output = apply_cross_object_rule(ep["model_input"], rule, seed=1000 + i)
        history = ep["model_input"]["interaction_history"]
        mask = output.get("generated_mask", {})
        action = output.get("action")
        n_groups = output.get("n_discovered_groups", 0)

        mask_reward = 0.0
        for hi in history:
            rid = int(hi["region_id"])
            pred_d = bool(mask.get(rid, {}).get("directly_intervenable", False))
            pred_i = bool(mask.get(rid, {}).get("indirectly_intervenable", False))
            if pred_d:
                mask_reward += float(hi.get("observed_direct_reward", 0.0))
            if pred_i:
                mask_reward += float(hi.get("observed_indirect_reward", 0.0))
        mask_reward /= max(1, len(history))

        if action is None:
            action_reward = 0.35
        else:
            rid = action["region_id"]
            ri_h = {}
            for h in history:
                if int(h["region_id"]) == rid:
                    ri_h = h
                    break
            if action["action_type"] == "apply_local_damping":
                action_reward = float(ri_h.get("observed_direct_reward", 0.0))
            else:
                action_reward = float(ri_h.get("observed_indirect_reward", 0.0))

        compression = min(1.0, float(rule.complexity + (n_groups - 3)) / 10.0)
        score = 0.50 * max(0.0, action_reward) + 0.30 * mask_reward + 0.10 * (1.0 - compression)
        score += 0.10 * min(1.0, n_groups / max(1, rule.n_clusters))
        scores.append(max(0.0, min(1.0, score)))

    return sum(scores) / len(scores) if scores else 0.0
