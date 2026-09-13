from __future__ import annotations

import json
import random
import numpy as np
from pathlib import Path
from typing import Any

from . import gn_model
from .g2_env import CONDITIONS, make_g2_datasets
from .g2_metrics import score_episode_g2


def _make_action_output(feats: np.ndarray, output: np.ndarray, obj_ids: list[int]) -> dict:
    mask = gn_model.decode_output(output, feats)
    action = gn_model.choose_action_from_output(output, feats)
    if action is not None:
        idx = action["region_id"]
        action["region_id"] = obj_ids[idx] if idx < len(obj_ids) else idx
    discovered_group_keys = sorted(mask.keys())
    discovered_groups = [[int(obj_ids[g])] if g < len(obj_ids) else [g] for g in discovered_group_keys]
    return {
        "action": action,
        "generated_mask": mask,
        "discovered_groups": discovered_groups,
        "n_discovered_groups": len(discovered_groups),
    }


def _eval_policy(gen_output_fn, datasets: dict, policy_name: str) -> tuple[list, dict]:
    records = []
    for condition, eps in datasets.items():
        for ep in eps:
            output = gen_output_fn(ep)
            row = score_episode_g2(ep, output, policy_name)
            records.append(row)
    scores_by_cond = {}
    for condition in CONDITIONS:
        rows = [r for r in records if r["condition"] == condition]
        sc = sum(float(r["generator_score"]) for r in rows) / len(rows) if rows else 0.0
        scores_by_cond[condition] = sc
    mean_score = sum(scores_by_cond.values()) / len(scores_by_cond) if scores_by_cond else 0.0
    return records, {"mean_score": mean_score, "by_condition": scores_by_cond}


def run_gn_contrastive(config: dict[str, Any], seed: int = 0) -> dict:
    datasets = make_g2_datasets(config, seed)
    train_eps = datasets["train"]

    print("  Training contrastive scorer...")
    c_model = gn_model.ContrastiveScorer(seed=seed)
    gn_model.train_contrastive(c_model, train_eps, lr=0.02, epochs=300)

    def gen_full(ep):
        feats, _, obj_ids = gn_model.encode_episode(ep)
        n = feats.shape[0]
        compat_matrix = np.eye(n, dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    compat_matrix[i, j] = c_model.score_pair(feats[i], feats[j], use_cross=True)
        group_map = {}
        for i in range(n):
            group_map[i] = i
        for i in range(n):
            for j in range(i + 1, n):
                if compat_matrix[i, j] > 0.5 and compat_matrix[j, i] > 0.5:
                    g = min(group_map[i], group_map[j])
                    group_map[i] = group_map[j] = g
        mask = {}
        for i in range(n):
            mask[i] = {
                "directly_intervenable": bool(float(feats[i, 2]) > float(feats[i, 5])),
                "indirectly_intervenable": bool(float(feats[i, 3]) > 0.5),
                "inspectable": bool(float(feats[i, 9]) > 0.5),
                "discovered_group": group_map[i],
                "similar_groups": [group_map[j] for j in range(n) if j != i and group_map[j] != group_map[i]],
            }
        action = None
        best_score = -1.0
        for i in range(n):
            if mask[i]["directly_intervenable"]:
                s = float(feats[i, 2]) - float(feats[i, 5])
                if s > best_score:
                    best_score = s
                    action = {"region_id": obj_ids[i], "action_type": "apply_local_damping"}
        discovered_groups = [[int(obj_ids[g])] for g in sorted(set(group_map.values()))]
        return {
            "action": action,
            "generated_mask": mask,
            "discovered_groups": discovered_groups,
            "n_discovered_groups": len(discovered_groups),
        }

    def gen_no_cross(ep):
        feats, _, obj_ids = gn_model.encode_episode(ep)
        n = feats.shape[0]
        compat_matrix = np.eye(n, dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    compat_matrix[i, j] = c_model.score_pair(feats[i], feats[j], use_cross=False)
        mask = {}
        for i in range(n):
            mask[i] = {
                "directly_intervenable": bool(float(feats[i, 2]) > float(feats[i, 5])),
                "indirectly_intervenable": False,
                "inspectable": bool(float(feats[i, 9]) > 0.5),
                "discovered_group": i,
                "similar_groups": [],
            }
        action = None
        best_score = -1.0
        for i in range(n):
            if mask[i]["directly_intervenable"]:
                s = float(feats[i, 2]) - float(feats[i, 5])
                if s > best_score:
                    best_score = s
                    action = {"region_id": obj_ids[i], "action_type": "apply_local_damping"}
        return {
            "action": action,
            "generated_mask": mask,
            "discovered_groups": [[int(obj_ids[i])] for i in range(n)],
            "n_discovered_groups": n,
        }

    print("  Evaluating contrastive...")
    _, full_metrics = _eval_policy(gen_full, datasets, "gn_contrast")
    _, no_metrics = _eval_policy(gen_no_cross, datasets, "gn_contrast_nocross")

    full_mean = full_metrics["mean_score"]
    no_mean = no_metrics["mean_score"]
    cross_drop = full_mean - no_mean

    print(f"  contrast_mean_score          = {full_mean:.3f}")
    print(f"  contrast_no_cross_score      = {no_mean:.3f}")
    print(f"  contrast_cross_ablation_drop  = {cross_drop:.3f}")

    return {
        "gn_contrast_mean_score": full_mean,
        "gn_contrast_no_cross_score": no_mean,
        "gn_contrast_ablation_drop": cross_drop,
    }


def run_gn_bottleneck(config: dict[str, Any], seed: int = 0) -> dict:
    datasets = make_g2_datasets(config, seed)
    train_eps = datasets["train"]

    print("  Training autoencoder...")
    ae = gn_model.BottleneckAE(seed=seed)
    gn_model.train_autoencoder(ae, train_eps, lr=0.01, epochs=200)

    print("  Training generator on 2D latents...")
    gn = gn_model.NeuralGenerator(seed=seed + 1, input_dim=gn_model.BOTTLENECK)

    latent_train_feats = []
    latent_train_targets = []
    for ep in train_eps:
        feats, targets, obj_ids = gn_model.encode_episode(ep)
        latents = np.zeros((feats.shape[0], gn_model.BOTTLENECK), dtype=np.float32)
        for oi in range(feats.shape[0]):
            latents[oi] = ae.encode(feats[oi])
        latent_train_feats.append(latents)
        latent_train_targets.append(targets)

    for epoch in range(200):
        epoch_loss = 0.0
        n_b = 0
        accum = [np.zeros_like(p) for p in gn.params()]
        rng = random.Random(epoch + 777)
        indices = list(range(len(latent_train_feats)))
        rng.shuffle(indices)
        for idx in indices[:min(16, len(indices))]:
            feats = latent_train_feats[idx]
            tgts = latent_train_targets[idx]
            n_o = feats.shape[0]
            if n_o < 2:
                continue
            masked = feats.copy()
            mask_id = rng.randint(0, n_o - 1)
            masked[mask_id] = 0.0
            loss, grads = gn_model.compute_gradients(gn, masked, tgts, use_cross=True)
            base_out = gn.forward(masked, use_cross=True)
            mask_loss = float(np.mean((base_out[mask_id] - tgts[mask_id]) ** 2).item())
            epoch_loss += mask_loss
            n_b += 1
            gn_g = gn_model._mask_only_grads(gn, masked, tgts, mask_id)
            for gi in range(len(gn_g)):
                accum[gi] += gn_g[gi]
        if n_b > 0:
            epoch_loss /= n_b
            for gi in range(len(accum)):
                accum[gi] /= n_b
        new_p = [p - 0.02 * g for p, g in zip(gn.params(), accum)]
        gn.set_params(new_p)
        if epoch % 50 == 0:
            print(f"    epoch {epoch:3d}  loss={epoch_loss:.4f} [bottleneck]")

    def make_gn_output(ep, use_cross):
        feats, _, obj_ids = gn_model.encode_episode(ep)
        latents = np.zeros((feats.shape[0], gn_model.BOTTLENECK), dtype=np.float32)
        for oi in range(feats.shape[0]):
            latents[oi] = ae.encode(feats[oi])
        output_arr = gn.forward(latents, use_cross=use_cross)
        return _make_action_output(latents, output_arr, obj_ids)

    print("  Evaluating bottleneck...")
    _, full_metrics = _eval_policy(lambda ep: make_gn_output(ep, True), datasets, "gn_bottleneck")
    _, no_metrics = _eval_policy(lambda ep: make_gn_output(ep, False), datasets, "gn_bottleneck_nocross")

    full_mean = full_metrics["mean_score"]
    no_mean = no_metrics["mean_score"]
    cross_drop = full_mean - no_mean

    print(f"  bottleneck_mean_score          = {full_mean:.3f}")
    print(f"  bottleneck_no_cross_score      = {no_mean:.3f}")
    print(f"  bottleneck_cross_ablation_drop  = {cross_drop:.3f}")

    return {
        "gn_bottleneck_mean_score": full_mean,
        "gn_bottleneck_no_cross_score": no_mean,
        "gn_bottleneck_ablation_drop": cross_drop,
    }
