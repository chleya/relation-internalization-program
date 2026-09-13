from __future__ import annotations

import json
import random
import numpy as np
from pathlib import Path
from typing import Any

from . import gn_model
from . import gn_world
from .g2_metrics import score_episode_g2

CONDITIONS_CONT = ["train", "test", "ood_remap"]


def _encode_episode_cont(ep: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    history = ep["model_input"]["interaction_history"]
    obj_map: dict[int, list[dict]] = {}
    for hi in history:
        oid = int(hi["object_id"])
        obj_map.setdefault(oid, []).append(hi)

    obj_ids = sorted(obj_map.keys())
    n_objs = len(obj_ids)
    features = np.zeros((n_objs, gn_world.N_FEATS_CONT), dtype=np.float32)
    targets = np.zeros((n_objs, 3), dtype=np.float32)
    truth = {t["region_id"]: t for t in ep["evaluator_ground_truth"]["regions"]}

    for idx, oid in enumerate(obj_ids):
        items = obj_map[oid]
        for name_i, name in enumerate(gn_world.ALL_FEATURES_CONT):
            vals = [float(it.get(name, 0.0)) for it in items]
            features[idx, name_i] = np.mean(vals) if vals else 0.0
        dir_good = any(
            truth.get(it["region_id"], {}).get("direct_actionable", False)
            for it in items
        )
        ind_good = any(
            truth.get(it["region_id"], {}).get("indirect_actionable", False)
            for it in items
        )
        insp_good = any(
            truth.get(it["region_id"], {}).get("inspectable", False)
            for it in items
        )
        targets[idx] = [float(dir_good), float(ind_good), float(insp_good)]
    return features, targets, obj_ids


def _make_action_output_cont(feats: np.ndarray, output: np.ndarray, obj_ids: list[int]) -> dict:
    mask = gn_model.decode_output(output, feats)
    action = gn_model.choose_action_from_output(output, feats)
    if action is not None:
        idx = action["region_id"]
        action["region_id"] = obj_ids[idx] if idx < len(obj_ids) else idx
    discovered_groups = [[int(obj_ids[g])] if g < len(obj_ids) else [g]
                          for g in sorted(mask.keys())]
    return {
        "action": action,
        "generated_mask": mask,
        "discovered_groups": discovered_groups,
        "n_discovered_groups": len(discovered_groups),
    }


def _eval_policy_cont(
    gen_fn, datasets: dict, policy_name: str,
) -> tuple[list, dict]:
    records = []
    for condition, eps in datasets.items():
        for ep in eps:
            output = gen_fn(ep)
            row = score_episode_g2(ep, output, policy_name)
            records.append(row)
    by_cond = {}
    for condition in CONDITIONS_CONT:
        rows = [r for r in records if r.get("condition") == condition]
        sc = sum(float(r["generator_score"]) for r in rows) / len(rows) if rows else 0.0
        by_cond[condition] = sc
    mean_score = sum(by_cond.values()) / max(len(by_cond), 1)
    return records, {"mean_score": mean_score, "by_condition": by_cond}


def run_gn_continuous(n_objects: int = 3, n_train: int = 24,
                      n_test: int = 8, n_ood: int = 8, seed: int = 0) -> dict:
    datasets = gn_world.make_continuous_datasets(
        n_objects=n_objects, n_train=n_train, n_test=n_test, n_ood=n_ood, seed=seed,
    )
    train_eps = datasets["train"]

    train_feats = []
    train_targets = []
    for ep in train_eps:
        feats, targets, obj_ids = _encode_episode_cont(ep)
        if feats.shape[0] >= 2:
            train_feats.append(feats)
            train_targets.append(targets)

    if len(train_feats) < 2:
        print("  Not enough training data")
        return {"gnc_mean_score": 0.0, "gnc_cross_drop": 0.0}

    model = gn_model.NeuralGenerator(seed=seed, input_dim=gn_world.N_FEATS_CONT)
    rng = random.Random(seed + 777)
    lr = 0.02

    for epoch in range(300):
        epoch_loss = 0.0
        n_batch = 0
        accum = [np.zeros_like(p) for p in model.params()]
        indices = list(range(len(train_feats)))
        rng.shuffle(indices)
        for idx in indices[:min(16, len(indices))]:
            feats = train_feats[idx].copy()
            tgts = train_targets[idx]
            n_o = feats.shape[0]
            if n_o < 2:
                continue
            mask_id = rng.randint(0, n_o - 1)
            feats[mask_id] = 0.0
            loss, grads = gn_model.compute_gradients(model, feats, tgts, use_cross=True)
            base_out = model.forward(feats, use_cross=True)
            mask_loss = float(np.mean((base_out[mask_id] - tgts[mask_id]) ** 2).item())
            epoch_loss += mask_loss
            n_batch += 1
            gn_g = gn_model._mask_only_grads(model, feats, tgts, mask_id)
            for gi in range(len(gn_g)):
                accum[gi] += gn_g[gi]
        if n_batch > 0:
            epoch_loss /= n_batch
            for gi in range(len(accum)):
                accum[gi] /= n_batch
        new_p = [p - lr * g for p, g in zip(model.params(), accum)]
        model.set_params(new_p)
        if epoch % 50 == 0:
            print(f"    epoch {epoch:3d}  loss={epoch_loss:.4f} [continuous]")

    print(f"  Final loss: {epoch_loss:.4f}")

    def gen_output(ep, use_cross):
        feats, _, obj_ids = _encode_episode_cont(ep)
        output_arr = model.forward(feats, use_cross=use_cross)
        return _make_action_output_cont(feats, output_arr, obj_ids)

    print("  Evaluating...")
    _, full_m = _eval_policy_cont(lambda ep: gen_output(ep, True), datasets, "gnc_neural")
    _, no_m = _eval_policy_cont(lambda ep: gen_output(ep, False), datasets, "gnc_nocross")

    full_mean = full_m["mean_score"]
    no_mean = no_m["mean_score"]
    drop = full_mean - no_mean

    print(f"  gnc_mean_score          = {full_mean:.3f}")
    print(f"  gnc_no_cross_score      = {no_mean:.3f}")
    print(f"  gnc_cross_ablation_drop  = {drop:.3f}")

    return {
        "gnc_mean_score": full_mean,
        "gnc_no_cross_score": no_mean,
        "gnc_cross_drop": drop,
        "by_condition": full_m["by_condition"],
        "no_cross_by_condition": no_m["by_condition"],
    }
