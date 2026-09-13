from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from g_line import gn_world as small_world
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent

BIG_WORLD_SIZE = 2.8
BIG_COLLISION_FORCE = 6.0
BIG_HEAT_TRANSFER = 1.5
BIG_DAMPING = 0.15
BIG_COLLISION_DIST = 1.0
BIG_STEPS = 50
BIG_N_OBJECTS = 6
BIG_NOISE = 0.03


def _init_objects_big(rng: random.Random) -> np.ndarray:
    n = BIG_N_OBJECTS
    state = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        angle = 2 * math.pi * i / n
        r = BIG_WORLD_SIZE * 0.35 * rng.random()
        state[i, 0] = BIG_WORLD_SIZE / 2 + r * math.cos(angle)
        state[i, 1] = BIG_WORLD_SIZE / 2 + r * math.sin(angle)
        state[i, 2] = (rng.random() - 0.5) * 2.0
        state[i, 3] = (rng.random() - 0.5) * 2.0
        state[i, 4] = small_world.AMBIENT_T + rng.random() * 0.8
    return state


def _compute_forces_heat_big(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = state.shape[0]
    forces = np.zeros((n, 2), dtype=np.float32)
    heat_flux = np.zeros(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dx = state[i, 0] - state[j, 0]
            dy = state[i, 1] - state[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < BIG_COLLISION_DIST and dist > 0.01:
                force = BIG_COLLISION_FORCE * (BIG_COLLISION_DIST - dist) / dist
                fx = force * dx / dist
                fy = force * dy / dist
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
                hf = BIG_HEAT_TRANSFER * (state[j, 4] - state[i, 4]) / (1.0 + dist)
                heat_flux[i] += hf
                heat_flux[j] -= hf
    return forces, heat_flux


def _step_euler_big(state: np.ndarray) -> np.ndarray:
    n = state.shape[0]
    forces, heat = _compute_forces_heat_big(state)
    new_state = state.copy()
    for i in range(n):
        new_state[i, 0] += state[i, 2] * small_world.DT
        new_state[i, 1] += state[i, 3] * small_world.DT
        new_state[i, 2] += (forces[i, 0] - BIG_DAMPING * state[i, 2]) * small_world.DT
        new_state[i, 3] += (forces[i, 1] - BIG_DAMPING * state[i, 3]) * small_world.DT
        new_state[i, 4] += (-small_world.COOLING * (state[i, 4] - small_world.AMBIENT_T) + heat[i]) * small_world.DT
    new_state[:, 0] = np.clip(new_state[:, 0], 0.15, BIG_WORLD_SIZE - 0.15)
    new_state[:, 1] = np.clip(new_state[:, 1], 0.15, BIG_WORLD_SIZE - 0.15)
    new_state[:, 4] = np.clip(new_state[:, 4], small_world.AMBIENT_T - 0.3, small_world.AMBIENT_T + 1.2)
    return new_state


def _state_to_features_big(state: np.ndarray) -> np.ndarray:
    n = state.shape[0]
    feats = np.zeros((n, small_world.N_FEATS_CONT), dtype=np.float32)
    for i in range(n):
        feats[i, 0] = state[i, 0] / BIG_WORLD_SIZE + BIG_NOISE * np.random.randn()
        feats[i, 1] = state[i, 1] / BIG_WORLD_SIZE + BIG_NOISE * np.random.randn()
        speed = math.sqrt(state[i, 2] ** 2 + state[i, 3] ** 2)
        feats[i, 2] = small_world._clamp(speed / 2.5 + BIG_NOISE * np.random.randn())
        feats[i, 3] = small_world._clamp(state[i, 4] + BIG_NOISE * np.random.randn())
        feats[i, 4] = small_world._clamp(
            max(0, feats[i, 3] - small_world.AMBIENT_T) * 0.8
            + feats[i, 2] * 0.4
            + BIG_NOISE * np.random.randn(),
        )
        feats[i, 5] = small_world._clamp(0.15 + BIG_NOISE * np.random.randn())
        feats[i, 6] = small_world._clamp(0.10 + BIG_NOISE * np.random.randn())
    return np.clip(feats, 0.0, 1.0)


def collate_big_trajectory(n_steps: int = 600, seed: int = 0) -> np.ndarray:
    rng_st = random.Random(seed)
    state = _init_objects_big(rng_st)
    all_feats = []
    for _ in range(n_steps):
        all_feats.append(_state_to_features_big(state).astype(np.float32))
        state = _step_euler_big(state)
    return np.array(all_feats, dtype=np.float32)


def granger_te(traj: np.ndarray, src: int, tgt: int, n_lags: int = 2) -> tuple[float, float, float]:
    T, n_o, d = traj.shape
    pairs = T - n_lags - 1
    if pairs < 20:
        return 0.0, 0.0, 0.0

    self_dim = n_lags * d
    X_self = np.zeros((pairs, self_dim), dtype=np.float32)
    X_full = np.zeros((pairs, self_dim * 2), dtype=np.float32)
    y = np.zeros((pairs, d), dtype=np.float32)

    for t in range(pairs):
        for lag in range(n_lags):
            base_self = lag * d
            base_full = lag * d * 2
            X_self[t, base_self:base_self + d] = traj[t + lag, tgt]
            X_full[t, base_full:base_full + d] = traj[t + lag, tgt]
            X_full[t, base_full + d:base_full + 2 * d] = traj[t + lag, src]
        y[t] = traj[t + n_lags, tgt]

    X_self_b = np.concatenate([X_self, np.ones((pairs, 1), dtype=np.float32)], axis=1)
    X_full_b = np.concatenate([X_full, np.ones((pairs, 1), dtype=np.float32)], axis=1)

    try:
        beta_self = np.linalg.lstsq(X_self_b, y, rcond=None)[0]
        beta_full = np.linalg.lstsq(X_full_b, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0

    mse_self = float(np.mean((y - X_self_b @ beta_self) ** 2).item())
    mse_full = float(np.mean((y - X_full_b @ beta_full) ** 2).item())
    if mse_self < 1e-10:
        return 0.0, 0.0, 0.0
    te = max(0.0, (mse_self - mse_full) / mse_self)

    rng_sh = random.Random(src * 100 + tgt * 7)
    idxs = list(range(pairs))
    rng_sh.shuffle(idxs)
    X_shuf = X_self.copy()
    X_shuf[:] = X_full[:, self_dim:][idxs]
    X_shuf_b = np.concatenate([X_self, X_shuf, np.ones((pairs, 1), dtype=np.float32)], axis=1)
    try:
        beta_shuf = np.linalg.lstsq(X_shuf_b, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return te, 0.0, 0.0
    mse_shuf = float(np.mean((y - X_shuf_b @ beta_shuf) ** 2).item())
    te_shuf = max(0.0, (mse_self - mse_shuf) / mse_self)
    return te, te_shuf, mse_self


print("=\"*60")
print("  BIG WORLD EXPERIMENT")
print("  Objects:", BIG_N_OBJECTS, "| World:", BIG_WORLD_SIZE, "x", BIG_WORLD_SIZE)
print("  Force:", BIG_COLLISION_FORCE, "| Heat:", BIG_HEAT_TRANSFER, "| Steps:", BIG_STEPS)
print("=\"*60")

print("\n--- Part 1: Transfer Entropy ---")
print("  Generating trajectory...")
traj_big = collate_big_trajectory(n_steps=600, seed=0)
print(f"  Shape: {traj_big.shape}")

print("  Computing TE on subset of pairs...")
all_te = []
all_shuf = []
sample_pairs = [(i, j) for i in range(BIG_N_OBJECTS) for j in range(BIG_N_OBJECTS) if i != j]
rng_te = random.Random(42)
rng_te.shuffle(sample_pairs)
for src, tgt in sample_pairs[:20]:
    te, te_shuf, _ = granger_te(traj_big, src, tgt)
    all_te.append(te)
    all_shuf.append(te_shuf)

avg_te = float(np.mean(all_te))
avg_shuf = float(np.mean(all_shuf))
te_excess = avg_te - avg_shuf

print(f"  Big world TE avg      = {avg_te:.4f}")
print(f"  Big world TE shuf     = {avg_shuf:.4f}")
print(f"  Big world TE excess   = {te_excess:.4f}")
print(f"  (Small world TE excess was 0.0333)")

ratio = te_excess / 0.0333
print(f"  TE ratio big/small    = {ratio:.2f}x")

print("\n--- Part 2: Transformer Generator ---")

TF_D = 20
TF_NP = 60


def _softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class TFGen:
    def __init__(self):
        rng = np.random.Generator(np.random.PCG64(42))
        self.w_in = _init_weights((small_world.N_FEATS_CONT, TF_D), rng)
        self.b_in = np.zeros(TF_D, dtype=np.float32)
        self.w_q = _init_weights((TF_D, TF_D), rng)
        self.w_k = _init_weights((TF_D, TF_D), rng)
        self.w_v = _init_weights((TF_D, TF_D), rng)
        self.w_merge = _init_weights((TF_D, TF_D), rng)
        self.b_merge = np.zeros(TF_D, dtype=np.float32)
        self.w_out = _init_weights((TF_D, small_world.N_FEATS_CONT), rng)
        self.b_out = np.zeros(small_world.N_FEATS_CONT, dtype=np.float32)

    def _relu(self, x):
        return np.maximum(0, x)

    def forward(self, feats, use_attn=True):
        n_o = feats.shape[0]
        h = self._relu(feats @ self.w_in + self.b_in)
        if use_attn and n_o > 1:
            Q = h @ self.w_q
            K = h @ self.w_k
            V = h @ self.w_v
            scores = Q @ K.T / math.sqrt(TF_D)
            attn = _softmax(scores, axis=-1)
            h = h + self._relu(attn @ V @ self.w_merge + self.b_merge)
        return h @ self.w_out + self.b_out

    def params(self):
        return [self.w_in, self.b_in, self.w_q, self.w_k, self.w_v,
                self.w_merge, self.b_merge, self.w_out, self.b_out]

    def set_params(self, flat):
        (self.w_in, self.b_in, self.w_q, self.w_k, self.w_v,
         self.w_merge, self.b_merge, self.w_out, self.b_out) = flat


def gen_frame_pairs(n_steps, seed):
    rng_st = random.Random(seed)
    state = _init_objects_big(rng_st)
    xs, ys = [], []
    for _ in range(n_steps):
        xs.append(_state_to_features_big(state).astype(np.float32))
        ns = _step_euler_big(state)
        ys.append(_state_to_features_big(ns).astype(np.float32))
        state = ns
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def _tf_grads(model, feats, target, loss_val):
    eps = 1e-4
    all_p = model.params()
    grads = [np.zeros_like(p) for p in all_p]
    rng = random.Random(int(loss_val * 1e7) + 13007)
    for pi, p in enumerate(all_p):
        flat = p.ravel()
        g = grads[pi].ravel()
        idxs = list(range(len(flat)))
        rng.shuffle(idxs)
        for idx in idxs[:min(TF_NP, len(idxs))]:
            old = flat[idx]
            flat[idx] = old + eps
            model.set_params(all_p)
            pred = model.forward(feats, use_attn=True)
            l2 = float(np.mean((pred - target) ** 2).item())
            flat[idx] = old
            g[idx] = (l2 - loss_val) / eps
        grads[pi] = g.reshape(p.shape)
    model.set_params(all_p)
    return grads


print("  Generating frame pairs...")
train_x, train_y = gen_frame_pairs(350, 0)
test_x, test_y = gen_frame_pairs(100, 100)
n_train = train_x.shape[0]
print(f"  Train: {n_train} frames, {BIG_N_OBJECTS} objects each")

model = TFGen()
lr = 0.006
rng_tr = random.Random(777)

for epoch in range(100):
    el = 0.0
    nb = 0
    accum = [np.zeros_like(p) for p in model.params()]
    idxs = list(range(n_train))
    rng_tr.shuffle(idxs)
    for idx in idxs[:min(32, n_train)]:
        pred = model.forward(train_x[idx], use_attn=True)
        loss = float(np.mean((pred - train_y[idx]) ** 2).item())
        if math.isnan(loss) or math.isinf(loss):
            continue
        el += loss
        nb += 1
        grads = _tf_grads(model, train_x[idx], train_y[idx], loss)
        for gi in range(len(grads)):
            accum[gi] += grads[gi]
    if nb > 0:
        el /= nb
        for gi in range(len(accum)):
            accum[gi] /= nb
    new_p = [p - lr * g for p, g in zip(model.params(), accum)]
    model.set_params(new_p)
    if epoch % 30 == 0:
        print(f"    epoch {epoch:3d}  loss={el:.6f}")


def eval_tf(data_x, data_y, use_attn, label):
    total = 0.0
    n = 0
    for idx in range(data_x.shape[0]):
        pred = model.forward(data_x[idx], use_attn=use_attn)
        loss = float(np.mean((pred - data_y[idx]) ** 2).item())
        if math.isnan(loss):
            continue
        total += loss
        n += 1
    avg = total / max(n, 1)
    print(f"    {label}: loss={avg:.6f}")
    return avg


def eval_shuf(data_x, data_y, label):
    orig = [model.w_q.copy(), model.w_k.copy(), model.w_v.copy()]
    rng = np.random.Generator(np.random.PCG64(555))
    for w in [model.w_q, model.w_k, model.w_v]:
        flat = w.ravel()
        rng.shuffle(flat)
        w[:] = flat.reshape(w.shape)
    total = 0.0
    n = 0
    for idx in range(data_x.shape[0]):
        pred = model.forward(data_x[idx], use_attn=True)
        loss = float(np.mean((pred - data_y[idx]) ** 2).item())
        if math.isnan(loss):
            continue
        total += loss
        n += 1
    avg = total / max(n, 1)
    model.w_q[:], model.w_k[:], model.w_v[:] = orig
    print(f"    {label}: loss={avg:.6f}")
    return avg


print("\n  Evaluating...")
w_loss = eval_tf(test_x, test_y, True, "with_attn ")
n_loss = eval_tf(test_x, test_y, False, "no_attn   ")
s_loss = eval_shuf(test_x, test_y, "shuf_attn ")

benefit = n_loss - w_loss
sdrop = s_loss - w_loss

print(f"\n  =\"*40")
print(f"  BIG WORLD RESULT")
print(f"  big_te_excess         = {te_excess:.4f}  (small=0.0333, ratio={ratio:.2f}x)")
print(f"  big_tf_with_attn      = {w_loss:.6f}")
print(f"  big_tf_no_attn        = {n_loss:.6f}")
print(f"  big_attn_benefit      = {benefit:.6f}  (+ = attention helps)")
print(f"  big_shuf_drop         = {sdrop:.6f}  (+ = shuffled hurts)")
print(f"  (Small world attn benefit was 0.002637)")

metrics = {
    "big_n_objects": BIG_N_OBJECTS,
    "big_world_size": BIG_WORLD_SIZE,
    "big_te_avg": avg_te, "big_te_shuf": avg_shuf, "big_te_excess": te_excess,
    "te_ratio_vs_small": ratio,
    "big_tf_with_attn": w_loss, "big_tf_no_attn": n_loss, "big_tf_shuf_attn": s_loss,
    "big_attn_benefit": benefit, "big_shuf_drop": sdrop,
}

out_dir = plos_dir / "results" / "big_world"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"  Saved to {out_dir}")
print("=\"*40")
