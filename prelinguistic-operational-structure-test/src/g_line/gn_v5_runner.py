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


def _encode_temporal(ep: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    history = ep["model_input"]["interaction_history"]
    obj_ids = sorted(set(int(hi["object_id"]) for hi in history))
    n = len(obj_ids)
    t_steps = gn_world.STEPS_PER_EPISODE
    traj = np.zeros((n, t_steps, gn_world.N_FEATS_CONT), dtype=np.float32)
    for hi in history:
        oi = obj_ids.index(int(hi["object_id"]))
        step = int(hi["step"])
        for fi, name in enumerate(gn_world.ALL_FEATURES_CONT):
            traj[oi, step, fi] = float(hi.get(name, 0.0))

    truth = {t["region_id"]: t for t in ep["evaluator_ground_truth"]["regions"]}
    targets = np.zeros((n, 3), dtype=np.float32)
    for oi, oid in enumerate(obj_ids):
        obj_truths = [truth[f"{oid}_{s}"] for s in range(t_steps) if f"{oid}_{s}" in truth]
        dir_g = any(t.get("direct_actionable", False) for t in obj_truths)
        ind_g = any(t.get("indirect_actionable", False) for t in obj_truths)
        insp_g = any(t.get("inspectable", False) for t in obj_truths)
        targets[oi] = [float(dir_g), float(ind_g), float(insp_g)]
    return traj, targets, obj_ids


def _make_action_output_temp(traj: np.ndarray, output_traj: np.ndarray, obj_ids: list[int]) -> dict:
    n = traj.shape[0]
    last_feats = output_traj[:, -1, :]
    mask = gn_model.decode_output(last_feats, last_feats)
    action = gn_model.choose_action_from_output(last_feats, last_feats)
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


def _eval_policy_temp(gen_fn, datasets: dict, policy_name: str) -> tuple[list, dict]:
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


def run_gn_temporal(n_objects: int = 3, n_train: int = 24,
                    n_test: int = 8, n_ood: int = 8, seed: int = 0) -> dict:
    datasets = gn_world.make_continuous_datasets(
        n_objects=n_objects, n_train=n_train, n_test=n_test, n_ood=n_ood, seed=seed,
    )
    train_eps = datasets["train"]

    train_trajs = []
    train_targets = []
    for ep in train_eps:
        traj, tgts, oids = _encode_temporal(ep)
        if traj.shape[0] >= 2:
            train_trajs.append(traj)
            train_targets.append(tgts)

    if len(train_trajs) < 2:
        return {"gnt_mean_score": 0.0, "gnt_cross_drop": 0.0}

    t_steps = gn_world.STEPS_PER_EPISODE
    model = gn_model.TemporalGenerator(
        input_dim=gn_world.N_FEATS_CONT, n_steps=t_steps, seed=seed,
    )
    rng = random.Random(seed + 777)

    for epoch in range(200):
        epoch_loss = 0.0
        n_b = 0
        accum = [np.zeros_like(p) for p in model.params()]
        indices = list(range(len(train_trajs)))
        rng.shuffle(indices)
        for idx in indices[:min(16, len(indices))]:
            traj = train_trajs[idx]
            n_o = traj.shape[0]
            mask_obj = rng.randint(0, n_o - 1)
            masked = traj.copy()
            masked[mask_obj] = 0.0
            pred = model.forward(masked, use_cross=True)
            loss = float(np.mean((pred[mask_obj] - traj[mask_obj]) ** 2).item())
            epoch_loss += loss
            n_b += 1
            grads = gn_model._temp_grads(model, traj, mask_obj, loss)
            for gi in range(len(grads)):
                accum[gi] += grads[gi]
        if n_b > 0:
            epoch_loss /= n_b
            for gi in range(len(accum)):
                accum[gi] /= n_b
        new_p = [p - 0.02 * g for p, g in zip(model.params(), accum)]
        model.set_params(new_p)
        if epoch % 50 == 0:
            print(f"    epoch {epoch:3d}  loss={epoch_loss:.4f} [temporal]")

    print(f"  Final temporal loss: {epoch_loss:.4f}")

    def gen_output(ep, use_cross):
        traj, _, obj_ids = _encode_temporal(ep)
        pred = model.forward(traj, use_cross=use_cross)
        return _make_action_output_temp(traj, pred, obj_ids)

    print("  Evaluating...")
    _, full_m = _eval_policy_temp(lambda ep: gen_output(ep, True), datasets, "gnt_temporal")
    _, no_m = _eval_policy_temp(lambda ep: gen_output(ep, False), datasets, "gnt_nocross")

    full_mean = full_m["mean_score"]
    no_mean = no_m["mean_score"]
    drop = full_mean - no_mean

    print(f"  gnt_mean_score          = {full_mean:.3f}")
    print(f"  gnt_no_cross_score      = {no_mean:.3f}")
    print(f"  gnt_cross_ablation_drop  = {drop:.3f}")

    return {
        "gnt_mean_score": full_mean,
        "gnt_no_cross_score": no_mean,
        "gnt_cross_drop": drop,
        "by_condition": full_m["by_condition"],
        "no_cross_by_condition": no_m["by_condition"],
    }
