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

CROSS_HIDDEN = 16
CROSS_N_PERTURB = 60
CROSS_N_STEPS = 10


def _run_forward_others(state: np.ndarray, n_steps: int,
                         intervene_obj: int) -> np.ndarray:
    n_o = state.shape[0]
    cur = state.copy()
    cur[intervene_obj, 2] *= 0.3
    cur[intervene_obj, 3] *= 0.3
    cur[intervene_obj, 4] = gn_world.AMBIENT_T + 0.05
    others = [o for o in range(n_o) if o != intervene_obj]
    traj = np.zeros((n_steps, len(others), gn_world.N_FEATS_CONT), dtype=np.float32)
    for s in range(n_steps):
        cur = gn_world._step_euler(cur)
        feats = gn_world._state_to_features(cur).astype(np.float32)
        for di, oi in enumerate(others):
            traj[s, di] = feats[oi]
    return traj


class CrossOnlyModel:
    def __init__(self, n_objects: int = 3):
        self.n_objs = n_objects
        rng = np.random.Generator(np.random.PCG64(42))
        self.w_self = _init_weights((gn_world.N_FEATS_CONT, CROSS_HIDDEN), rng)
        self.b_self = np.zeros(CROSS_HIDDEN, dtype=np.float32)
        self.w_cross = _init_weights((gn_world.N_FEATS_CONT, CROSS_HIDDEN), rng)
        self.b_cross = np.zeros(CROSS_HIDDEN, dtype=np.float32)
        self.w_act = _init_weights((n_objects, CROSS_HIDDEN), rng)
        flat_out = CROSS_N_STEPS * gn_world.N_FEATS_CONT
        self.w_head = _init_weights((CROSS_HIDDEN * 3, flat_out), rng)
        self.b_head = np.zeros(flat_out, dtype=np.float32)

    def _relu(self, x):
        return np.maximum(0, x)

    def forward_others(self, pre_feats: np.ndarray, intervene_obj: int,
                       use_cross: bool = True) -> np.ndarray:
        n_o = pre_feats.shape[0]
        others = [o for o in range(n_o) if o != intervene_obj]
        n_other = len(others)

        self_h = self._relu(pre_feats @ self.w_self + self.b_self)
        if use_cross:
            cross_h = self._relu(pre_feats @ self.w_cross + self.b_cross)
            cross_agg = np.zeros((n_o, CROSS_HIDDEN), dtype=np.float32)
            for i in range(n_o):
                m = np.ones(n_o, dtype=bool)
                m[i] = False
                cross_agg[i] = cross_h[m].mean(axis=0)
        else:
            cross_agg = np.zeros((n_o, CROSS_HIDDEN), dtype=np.float32)
        act_emb = np.zeros(CROSS_HIDDEN, dtype=np.float32)
        if 0 <= intervene_obj < self.n_objs:
            act_emb = self.w_act[intervene_obj]

        flat_out = CROSS_N_STEPS * gn_world.N_FEATS_CONT
        preds = np.zeros((n_other, flat_out), dtype=np.float32)
        for di, oi in enumerate(others):
            combined = np.concatenate([self_h[oi], cross_agg[oi], act_emb])
            preds[di] = combined @ self.w_head + self.b_head
        return preds.reshape(n_other, CROSS_N_STEPS, gn_world.N_FEATS_CONT)

    def params(self):
        return [self.w_self, self.b_self, self.w_cross, self.b_cross,
                self.w_act, self.w_head, self.b_head]

    def set_params(self, flat):
        self.w_self, self.b_self, self.w_cross, self.b_cross, \
        self.w_act, self.w_head, self.b_head = flat


def generate_crossonly_data(n_episodes: int = 40, seed: int = 0,
                            n_objects: int = 3) -> tuple[list, list]:
    rng = random.Random(seed)
    train_x = []
    train_y = []
    for ei in range(n_episodes):
        state = gn_world._init_objects(rng, n_objects)
        for _ in range(8):
            state = gn_world._step_euler(state)
        pre_feats = gn_world._state_to_features(state).astype(np.float32)
        for intervene_obj in range(n_objects):
            others_traj = _run_forward_others(state.copy(), CROSS_N_STEPS, intervene_obj)
            train_x.append({"pre_feats": pre_feats, "intervene_obj": intervene_obj})
            train_y.append(others_traj.transpose(1, 0, 2))
    return train_x, train_y


def _crossonly_grads(model: CrossOnlyModel, pre_feats: np.ndarray,
                     intervene_obj: int, target: np.ndarray,
                     loss_val: float) -> list[np.ndarray]:
    eps = 1e-4
    all_p = model.params()
    grads = [np.zeros_like(p) for p in all_p]
    rng = random.Random(int(loss_val * 1e7) + intervene_obj * 97 + 14009)

    for pi, p in enumerate(all_p):
        flat = p.ravel()
        g = grads[pi].ravel()
        idxs = list(range(len(flat)))
        rng.shuffle(idxs)
        for idx in idxs[:min(CROSS_N_PERTURB, len(idxs))]:
            old = flat[idx]
            flat[idx] = old + eps
            model.set_params(all_p)
            pred = model.forward_others(pre_feats, intervene_obj, use_cross=True)
            l2 = float(np.mean((pred - target) ** 2).item())
            flat[idx] = old
            g[idx] = (l2 - loss_val) / eps
        grads[pi] = g.reshape(p.shape)
    model.set_params(all_p)
    return grads


print("=== Line 2: Cross-Only Prediction (no self-target) ===")
print("  Generating cross-only data...")

train_x, train_y = generate_crossonly_data(n_episodes=40, seed=0, n_objects=3)
n_samples = len(train_x)
print(f"  Generated {n_samples} samples (each: predict OTHERS only)")

test_x, test_y = generate_crossonly_data(n_episodes=12, seed=100, n_objects=3)
ood_x, ood_y = generate_crossonly_data(n_episodes=12, seed=200, n_objects=4)

model = CrossOnlyModel(n_objects=3)
lr = 0.01
rng_train = random.Random(777)

for epoch in range(100):
    epoch_loss = 0.0
    n_b = 0
    accum = [np.zeros_like(p) for p in model.params()]
    idxs = list(range(n_samples))
    rng_train.shuffle(idxs)
    for idx in idxs[:min(16, n_samples)]:
        x = train_x[idx]
        y = train_y[idx]
        pred = model.forward_others(x["pre_feats"], x["intervene_obj"], use_cross=True)
        loss = float(np.mean((pred - y) ** 2).item())
        if math.isnan(loss) or math.isinf(loss):
            continue
        epoch_loss += loss
        n_b += 1
        grads = _crossonly_grads(model, x["pre_feats"], x["intervene_obj"], y, loss)
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


def eval_crossonly(data_x, data_y, use_cross, label):
    total = 0.0
    n = 0
    for idx in range(len(data_x)):
        x = data_x[idx]
        y = data_y[idx]
        pred = model.forward_others(x["pre_feats"], x["intervene_obj"], use_cross=use_cross)
        loss = float(np.mean((pred - y) ** 2).item())
        if math.isnan(loss):
            continue
        total += loss
        n += 1
    avg = total / max(n, 1)
    print(f"    {label}: loss={avg:.6f}")
    return avg


print("\n  Evaluating...")
with_loss = eval_crossonly(test_x, test_y, True, "with_cross ")
no_loss = eval_crossonly(test_x, test_y, False, "no_cross  ")
ood_with = eval_crossonly(ood_x, ood_y, True, "ood_cross ")
ood_no = eval_crossonly(ood_x, ood_y, False, "ood_nocross")

cross_benefit = no_loss - with_loss
ood_benefit = ood_no - ood_with

print(f"\n  === CROSS-ONLY RESULT ===")
print(f"  co_with_cross        = {with_loss:.6f}")
print(f"  co_no_cross          = {no_loss:.6f}")
print(f"  co_cross_benefit     = {cross_benefit:.6f}  (+ = cross helps)")
print(f"  co_ood_benefit       = {ood_benefit:.6f}")

metrics = {
    "co_with_cross": with_loss, "co_no_cross": no_loss,
    "co_cross_benefit": cross_benefit,
    "co_ood_with": ood_with, "co_ood_no": ood_no,
    "co_ood_benefit": ood_benefit,
}
out_dir = plos_dir / "results" / "cross_only"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"  Saved to {out_dir}")
print("=== DONE ===")
