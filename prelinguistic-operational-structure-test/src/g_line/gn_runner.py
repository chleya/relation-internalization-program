from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any

from . import gn_model
from .g2_env import CONDITIONS, make_g2_datasets
from .g2_metrics import score_episode_g2


def _make_action_output(feats: np.ndarray, output: np.ndarray, obj_ids: list[int]) -> dict:
    mask = gn_model.decode_output(output, feats)
    action = gn_model.choose_action_from_output(output, feats)
    discovered_group_keys = sorted(mask.keys())
    discovered_groups = [[int(obj_ids[g])] if g < len(obj_ids) else [g] for g in discovered_group_keys]
    return {
        "action": action,
        "generated_mask": mask,
        "discovered_groups": discovered_groups,
        "n_discovered_groups": len(discovered_groups),
    }


def run_gn(config: dict[str, Any], seed: int = 0) -> dict:
    datasets = make_g2_datasets(config, seed)
    train_eps = datasets["train"]

    train_encoded = []
    for ep in train_eps:
        feats, targets, obj_ids = gn_model.encode_episode(ep)
        if feats.shape[0] >= 2:
            train_encoded.append((ep, feats, targets, obj_ids))
    if len(train_encoded) < 2:
        print("  Not enough training episodes with >=2 objects")
        return {"gn_mean_score": 0.0, "gn_cross_ablation_drop": 0.0, "gn_ood_score": 0.0}

    model = gn_model.NeuralGenerator(seed=seed)
    train_eps_raw = [e for e, _, _, _ in train_encoded]
    losses = gn_model.train(model, train_eps_raw, lr=0.02, epochs=300, mask_train=True)
    final_loss = float(losses[-1]) if losses else 1.0
    print(f"  Final train loss (masked): {final_loss:.4f}")

    records = []
    plos_dir = Path(__file__).resolve().parent.parent.parent

    for policy_name, use_cross in [("gn_neural", True), ("gn_no_cross", False)]:
        for condition, eps in datasets.items():
            for ep in eps:
                feats, _, obj_ids = gn_model.encode_episode(ep)
                output_arr = model.forward(feats, use_cross=use_cross)
                g2_output = _make_action_output(feats, output_arr, obj_ids)
                row = score_episode_g2(ep, g2_output, policy_name)
                records.append(row)

    summary = []
    for condition in CONDITIONS:
        base = {}
        for p in ["gn_neural", "gn_no_cross"]:
            rows = [r for r in records if r["condition"] == condition and r["policy_name"] == p]
            base[p] = sum(float(r["generator_score"]) for r in rows) / len(rows) if rows else 0.0
        for policy in ["gn_neural", "gn_no_cross"]:
            rows = [r for r in records if r["condition"] == condition and r["policy_name"] == policy]
            sc = base[policy]
            rand_sc = base.get("gn_no_cross", 0.0)
            srow = {
                "condition": condition,
                "policy_name": policy,
                "sample_count": len(rows),
                "generator_score": sc,
                "ood_generalization_score": sc if "ood" in condition else 0.0,
                "gain_over_random": sc - rand_sc,
                "oracle_gap": 0.0,
            }
            summary.append(srow)

    gn_rows = [r for r in summary if r["policy_name"] == "gn_neural"]
    no_rows = [r for r in summary if r["policy_name"] == "gn_no_cross"]

    def m(rows, k):
        return sum(float(r.get(k, 0)) for r in rows) / len(rows) if rows else 0.0

    gn_mean = m(gn_rows, "generator_score")
    gn_ood = next((float(r["generator_score"]) for r in gn_rows if r["condition"] == "ood_remap"), 0.0)
    no_mean = m(no_rows, "generator_score")
    cross_drop = gn_mean - no_mean

    metrics = {
        "gn_mean_score": gn_mean,
        "gn_ood_score": gn_ood,
        "gn_no_cross_score": no_mean,
        "gn_cross_ablation_drop": cross_drop,
        "gn_params": len(model.params()),
        "gn_hidden": gn_model.HIDDEN,
        "train_final_loss": final_loss,
    }

    print(f"\n  === GN RESULT ===")
    print(f"  gn_mean_score          = {gn_mean:.3f}")
    print(f"  gn_no_cross_score      = {no_mean:.3f}")
    print(f"  gn_cross_ablation_drop  = {cross_drop:.3f}")
    print(f"  gn_ood_score           = {gn_ood:.3f}")

    return metrics
