from __future__ import annotations

from typing import Any

from .g1_metrics import SUMMARY_FIELDS, RECORD_FIELDS, score_episode as _score_g1


G2_EXTRA_SUMMARY = SUMMARY_FIELDS + [
    "object_discovery_ari",
    "cross_group_transfer",
    "edge_discovery_f1",
    "group_compression_efficiency",
]


def score_episode_g2(episode: dict[str, Any], output: dict[str, Any], policy_name: str) -> dict[str, Any]:
    base = _score_g1(episode, output, policy_name)
    truth = episode["evaluator_ground_truth"]["regions"]
    truth_groups = {}
    for row in truth:
        oid = row["object_id"]
        truth_groups.setdefault(oid, []).append(int(row["region_id"]))

    mask = output.get("generated_mask", {})
    discovered_groups = output.get("discovered_groups", [])
    if discovered_groups:
        ri = _adjusted_rand_index(truth_groups, discovered_groups)
        base["object_discovery_ari"] = ri
        base["cross_group_transfer"] = min(1.0, ri)
    else:
        base["object_discovery_ari"] = 0.0
        base["cross_group_transfer"] = 0.0

    edges = episode["evaluator_ground_truth"].get("edges", [])
    if edges:
        sim_matrix = output.get("similarity_matrix", [])
        n_groups = len(sim_matrix)
        edge_hits = 0
        total_edges = len(edges)
        for edge in edges:
            src_obj = edge["src_object"]
            dst_obj = edge["dst_object"]
            src_rids = truth_groups.get(src_obj, [])
            dst_rids = truth_groups.get(dst_obj, [])
            if src_rids and dst_rids:
                src_mask_group = None
                dst_mask_group = None
                for rid in src_rids:
                    if rid in mask:
                        src_mask_group = mask[rid].get("discovered_group")
                        break
                for rid in dst_rids:
                    if rid in mask:
                        dst_mask_group = mask[rid].get("discovered_group")
                        break
                if src_mask_group is not None and dst_mask_group is not None and src_mask_group < n_groups and dst_mask_group < n_groups:
                    if sim_matrix[src_mask_group][dst_mask_group] >= 0.15:
                        edge_hits += 1
        base["edge_discovery_f1"] = edge_hits / total_edges if total_edges else 0.0
    else:
        base["edge_discovery_f1"] = 0.0

    base["group_compression_efficiency"] = output.get("n_discovered_groups", 0) / max(1, len(truth_groups))

    return base


def _adjusted_rand_index(
    true_groups: dict[int, list[int]], pred_groups: list[list[int]]
) -> float:
    all_regions = []
    for g in true_groups.values():
        all_regions.extend(g)
    all_pred = []
    for g in pred_groups:
        all_pred.extend(g)
    if not all_regions or not all_pred:
        return 0.0

    region_set = set(all_regions) & set(all_pred)
    if len(region_set) < 2:
        return 0.0

    pairs_tp = 0
    pairs_fp = 0
    pairs_fn = 0
    pairs_tn = 0
    region_list = sorted(region_set)
    for i in range(len(region_list)):
        for j in range(i + 1, len(region_list)):
            ri, rj = region_list[i], region_list[j]
            same_true = False
            for g in true_groups.values():
                if ri in g and rj in g:
                    same_true = True
                    break
            same_pred = False
            for g in pred_groups:
                if ri in g and rj in g:
                    same_pred = True
                    break
            if same_true and same_pred:
                pairs_tp += 1
            elif not same_true and same_pred:
                pairs_fp += 1
            elif same_true and not same_pred:
                pairs_fn += 1
            else:
                pairs_tn += 1

    total = pairs_tp + pairs_fp + pairs_fn + pairs_tn
    if total == 0:
        return 0.0
    return (pairs_tp + pairs_tn) / total
