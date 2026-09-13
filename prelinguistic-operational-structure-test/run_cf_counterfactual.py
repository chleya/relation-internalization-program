from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from g_line import gn_world
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent


def _run_forward_from(state: np.ndarray, start_step: int, n_steps: int,
                      intervene_obj: int | None = None) -> np.ndarray:
    traj = np.zeros((n_steps, state.shape[0], state.shape[1]), dtype=np.float32)
    cur = state.copy()
    for s in range(n_steps):
        if intervene_obj is not None and s == 0:
            cur[intervene_obj, 2] *= 0.3
            cur[intervene_obj, 3] *= 0.3
            cur[intervene_obj, 4] = gn_world.AMBIENT_T + 0.1
        cur = gn_world._step_euler(cur)
        traj[s] = cur
    return traj


def _state_traj_to_features(traj: np.ndarray) -> np.ndarray:
    T, n_o, _ = traj.shape
    feats = np.zeros((T, n_o, gn_world.N_FEATS_CONT), dtype=np.float32)
    for s in range(T):
        fs = gn_world._state_to_features(traj[s])
        feats[s] = fs
    return feats


def generate_counterfactual_data(n_episodes: int = 32, seed: int = 0,
                                 n_objects: int = 3) -> tuple[list, list]:
    rng = random.Random(seed)
    train_x = []
    train_y = []

    for ei in range(n_episodes):
        state = gn_world._init_objects(rng, n_objects)
        for _ in range(10):
            state = gn_world._step_euler(state)

        pre_state = state.copy()
        pre_feats = _state_traj_to_features(pre_state[np.newaxis])[0]

        for intervene_obj in range(n_objects):
            real_traj = _run_forward_from(pre_state, 10, 10, intervene_obj=intervene_obj)
            counter_traj = _run_forward_from(pre_state, 10, 10, intervene_obj=None)

            real_feats = _state_traj_to_features(real_traj)
            counter_feats = _state_traj_to_features(counter_traj)
            delta = real_feats - counter_feats

            train_x.append({
                "pre_feats": pre_feats,
                "intervene_obj": intervene_obj,
            })
            train_y.append(delta.transpose(1, 0, 2))

    return train_x, train_y


CF_HIDDEN = 12
CF_N_PERTURB = 60


class CounterfactualModel:
    def __init__(self, n_objects: int = 3):
        self.n_objs = n_objects
        rng = np.random.Generator(np.random.PCG64(42))
        self.w_self = _init_weights((gn_world.N_FEATS_CONT, CF_HIDDEN), rng)
        self.b_self = np.zeros(CF_HIDDEN, dtype=np.float32)
        self.w_cross = _init_weights((gn_world.N_FEATS_CONT, CF_HIDDEN), rng)
        self.b_cross = np.zeros(CF_HIDDEN, dtype=np.float32)
        self.w_act = _init_weights((n_objects, CF_HIDDEN), rng)
        self.w_head = _init_weights((CF_HIDDEN * 3, 10 * gn_world.N_FEATS_CONT), rng)
        self.b_head = np.zeros(10 * gn_world.N_FEATS_CONT, dtype=np.float32)

    def _relu(self, x):
        return np.maximum(0, x)

    def forward(self, pre_feats: np.ndarray, intervene_obj: int,
                use_cross: bool = True) -> np.ndarray:
        n_o = pre_feats.shape[0]
        self_h = self._relu(pre_feats @ self.w_self + self.b_self)
        if use_cross:
            cross_h = self._relu(pre_feats @ self.w_cross + self.b_cross)
            cross_agg = np.zeros((n_o, CF_HIDDEN), dtype=np.float32)
            for i in range(n_o):
                m = np.ones(n_o, dtype=bool)
                m[i] = False
                cross_agg[i] = cross_h[m].mean(axis=0)
        else:
            cross_agg = np.zeros((n_o, CF_HIDDEN), dtype=np.float32)
        action_onehot = np.zeros(CF_HIDDEN, dtype=np.float32)
        if intervene_obj < self.n_objs:
            action_onehot = self.w_act[intervene_obj]

        preds = np.zeros((n_o, 10 * gn_world.N_FEATS_CONT), dtype=np.float32)
        for i in range(n_o):
            combined = np.concatenate([self_h[i], cross_agg[i], action_onehot])
            preds[i] = combined @ self.w_head + self.b_head
        return preds.reshape(n_o, 10, gn_world.N_FEATS_CONT)

    def params(self):
        return [self.w_self, self.b_self, self.w_cross, self.b_cross,
                self.w_act, self.w_head, self.b_head]

    def set_params(self, flat):
        self.w_self, self.b_self, self.w_cross, self.b_cross, \
        self.w_act, self.w_head, self.b_head = flat


def _cf_grads(model: CounterfactualModel, pre_feats: np.ndarray,
              intervene_obj: int, delta_true: np.ndarray,
              loss_val: float) -> list[np.ndarray]:
    eps = 1e-4
    all_p = model.params()
    grads = [np.zeros_like(p) for p in all_p]
    rng = random.Random(int(loss_val * 1e7) + intervene_obj * 97)

    for pi, p in enumerate(all_p):
        flat = p.ravel()
        g = grads[pi].ravel()
        idxs = list(range(len(flat)))
        rng.shuffle(idxs)
        for idx in idxs[:min(CF_N_PERTURB, len(idxs))]:
            old = flat[idx]
            flat[idx] = old + eps
            model.set_params(all_p)
            pred = model.forward(pre_feats, intervene_obj, use_cross=True)
            l2 = float(np.mean((pred - delta_true) ** 2).item())
            flat[idx] = old
            g[idx] = (l2 - loss_val) / eps
        grads[pi] = g.reshape(p.shape)
    model.set_params(all_p)
    return grads


print("=== Route A: Counterfactual Training ===")
print("  Generating counterfactual data...")

train_x, train_y = generate_counterfactual_data(n_episodes=24, seed=0, n_objects=3)
n_samples = len(train_x)
print(f"  Generated {n_samples} counterfactual pairs")

test_x, test_y = generate_counterfactual_data(n_episodes=8, seed=100, n_objects=3)
test_x_ood, test_y_ood = generate_counterfactual_data(n_episodes=8, seed=200, n_objects=4)

model = CounterfactualModel(n_objects=3)
lr = 0.01
rng_train = random.Random(777)

for epoch in range(80):
    epoch_loss = 0.0
    n_b = 0
    accum = [np.zeros_like(p) for p in model.params()]
    idxs = list(range(n_samples))
    rng_train.shuffle(idxs)
    for idx in idxs[:min(8, n_samples)]:
        x = train_x[idx]
        y = train_y[idx]
        delta_pred = model.forward(x["pre_feats"], x["intervene_obj"], use_cross=True)
        loss = float(np.mean((delta_pred - y) ** 2).item())
        if math.isnan(loss) or math.isinf(loss):
            continue
        epoch_loss += loss
        n_b += 1
        grads = _cf_grads(model, x["pre_feats"], x["intervene_obj"], y, loss)
        for gi in range(len(grads)):
            accum[gi] += grads[gi]
    if n_b > 0:
        epoch_loss /= n_b
        for gi in range(len(accum)):
            accum[gi] /= n_b
    new_p = [p - lr * g for p, g in zip(model.params(), accum)]
    model.set_params(new_p)

    if epoch % 30 == 0:
        print(f"    epoch {epoch:3d}  loss={epoch_loss:.6f}")


def eval_counterfactual(data_x, data_y, use_cross, label):
    total_loss = 0.0
    n = 0
    per_obj_loss = np.zeros(3)
    for idx in range(len(data_x)):
        x = data_x[idx]
        y = data_y[idx]
        pred = model.forward(x["pre_feats"], x["intervene_obj"], use_cross=use_cross)
        loss = float(np.mean((pred - y) ** 2))
        if math.isnan(loss):
            continue
        total_loss += loss
        n += 1
        for oi in range(min(3, y.shape[0])):
            per_obj_loss[oi] += float(np.mean((pred[oi] - y[oi]) ** 2))
    avg = total_loss / max(n, 1)
    per_obj_loss /= max(n, 1)
    non_int = [per_obj_loss[oi] for oi in range(3)]
    print(f"    {label}: loss={avg:.6f}  per_obj={[f'{l:.4f}' for l in non_int]}")
    return avg


print("\n  Evaluating...")
full_loss = eval_counterfactual(test_x, test_y, True, "with_cross ")
no_loss = eval_counterfactual(test_x, test_y, False, "no_cross  ")
ood_loss = eval_counterfactual(test_x_ood, test_y_ood, True, "ood_cross")
ood_no = eval_counterfactual(test_x_ood, test_y_ood, False, "ood_nocross")

cf_drop = no_loss - full_loss
cf_ood_drop = ood_no - ood_loss

print(f"\n  === COUNTERFACTUAL RESULT ===")
print(f"  cf_loss_with_cross    = {full_loss:.6f}")
print(f"  cf_loss_no_cross      = {no_loss:.6f}")
print(f"  cf_cross_benefit      = {cf_drop:.6f}  (+ = cross helps)")
print(f"  cf_ood_cross_benefit  = {cf_ood_drop:.6f}")

metrics = {
    "cf_loss_with_cross": full_loss,
    "cf_loss_no_cross": no_loss,
    "cf_cross_benefit": cf_drop,
    "cf_ood_loss_with_cross": ood_loss,
    "cf_ood_loss_no_cross": ood_no,
    "cf_ood_cross_benefit": cf_ood_drop,
}

out_dir = plos_dir / "results" / "cf_counterfactual"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"  Saved to {out_dir}")
print("=== DONE ===")
