from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from g_line import gn_world

plos_dir = Path(__file__).resolve().parent


def collect_trajectory(n_steps: int = 500, seed: int = 0, n_objects: int = 3) -> np.ndarray:
    rng_st = random.Random(seed)
    state = gn_world._init_objects(rng_st, n_objects)
    all_feats = []
    for _ in range(n_steps):
        all_feats.append(gn_world._state_to_features(state).astype(np.float32))
        state = gn_world._step_euler(state)
    return np.array(all_feats, dtype=np.float32)


def granger_transfer_entropy(traj: np.ndarray, src: int, tgt: int,
                              n_lags: int = 2) -> tuple[float, float, float]:
    T, n_o, d = traj.shape
    pairs = T - n_lags - 1
    if pairs < 20:
        return 0.0, 0.0, 0.0

    X = np.zeros((pairs, n_lags * d * 2), dtype=np.float32)
    y = np.zeros((pairs, d), dtype=np.float32)

    for t in range(pairs):
        row = []
        for lag in range(n_lags):
            row.extend(traj[t + lag, tgt].tolist())
        for lag in range(n_lags):
            row.extend(traj[t + lag, src].tolist())
        X[t] = np.array(row, dtype=np.float32)
        y[t] = traj[t + n_lags, tgt]

    X_self = X[:, :n_lags * d]
    X_full = X

    X_self_b = np.concatenate([X_self, np.ones((pairs, 1), dtype=np.float32)], axis=1)
    X_full_b = np.concatenate([X_full, np.ones((pairs, 1), dtype=np.float32)], axis=1)

    try:
        beta_self = np.linalg.lstsq(X_self_b, y, rcond=None)[0]
        beta_full = np.linalg.lstsq(X_full_b, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0

    resid_self = y - X_self_b @ beta_self
    resid_full = y - X_full_b @ beta_full

    mse_self = float(np.mean(resid_self ** 2).item())
    mse_full = float(np.mean(resid_full ** 2).item())

    if mse_self < 1e-10:
        return 0.0, 0.0, 0.0

    te = max(0.0, (mse_self - mse_full) / mse_self)

    rng = random.Random(src * 100 + tgt * 7)
    idxs = list(range(pairs))
    rng.shuffle(idxs)
    X_shuf = X_self.copy()
    X_shuf[:, :] = X_full[:, n_lags * d:][idxs]
    X_shuf_b = np.concatenate([X_self, X_shuf, np.ones((pairs, 1), dtype=np.float32)], axis=1)
    try:
        beta_shuf = np.linalg.lstsq(X_shuf_b, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return te, 0.0, 0.0
    resid_shuf = y - X_shuf_b @ beta_shuf
    mse_shuf = float(np.mean(resid_shuf ** 2).item())
    te_shuf = max(0.0, (mse_self - mse_shuf) / mse_self)

    return te, te_shuf, mse_self


print("=== Route C: Information-Theoretic Causal Detection ===")
print("  Generating trajectory data...")

traj_3 = collect_trajectory(n_steps=500, seed=0, n_objects=3)
traj_4 = collect_trajectory(n_steps=500, seed=0, n_objects=4)

print(f"  Trajectory shape: {traj_3.shape}")

print("\n  Computing transfer entropy (3-object world)...")
te_matrix = np.zeros((3, 3), dtype=np.float32)
te_shuf_matrix = np.zeros((3, 3), dtype=np.float32)
base_mse = np.zeros(3, dtype=np.float32)

all_te = []
all_te_shuf = []

for src in range(3):
    for tgt in range(3):
        if src == tgt:
            continue
        te, te_shuf, mse_self = granger_transfer_entropy(traj_3, src, tgt)
        te_matrix[src, tgt] = te
        te_shuf_matrix[src, tgt] = te_shuf
        base_mse[tgt] = mse_self
        all_te.append(te)
        all_te_shuf.append(te_shuf)
        print(f"    {src}->{tgt}: TE={te:.6f}  (shuffled={te_shuf:.6f}, base_mse={mse_self:.6f})")

avg_te = float(np.mean(all_te))
avg_te_shuf = float(np.mean(all_te_shuf))
te_excess = avg_te - avg_te_shuf

print(f"\n    Average TE         = {avg_te:.6f}")
print(f"    Average TE (shuf)  = {avg_te_shuf:.6f}")
print(f"    TE excess          = {te_excess:.6f}  (above shuffle baseline)")

print("\n  Computing transfer entropy (4-object OOD)...")
te_4 = []
te_4_shuf = []
for src in range(4):
    for tgt in range(4):
        if src == tgt:
            continue
        te, te_shuf, _ = granger_transfer_entropy(traj_4, src, tgt)
        te_4.append(te)
        te_4_shuf.append(te_shuf)

avg_te_4 = float(np.mean(te_4))
avg_te_4_shuf = float(np.mean(te_4_shuf))
te_excess_4 = avg_te_4 - avg_te_4_shuf

print(f"    Average TE (4-obj)        = {avg_te_4:.6f}")
print(f"    Average TE shuf (4-obj)   = {avg_te_4_shuf:.6f}")
print(f"    TE excess (4-obj)         = {te_excess_4:.6f}")

print(f"\n  === INFO-THEORETIC RESULT ===")
print(f"  te_avg                 = {avg_te:.6f}")
print(f"  te_avg_shuffled        = {avg_te_shuf:.6f}")
print(f"  te_excess              = {te_excess:.6f}  (+ = real causal flow exists)")
print(f"  te_4obj_excess         = {te_excess_4:.6f}")

metrics = {
    "te_avg": avg_te,
    "te_avg_shuffled": avg_te_shuf,
    "te_excess": te_excess,
    "te_4obj_avg": avg_te_4,
    "te_4obj_avg_shuffled": avg_te_4_shuf,
    "te_4obj_excess": te_excess_4,
    "te_matrix_3obj": te_matrix.tolist(),
    "te_shuf_matrix_3obj": te_shuf_matrix.tolist(),
    "base_mse": base_mse.tolist(),
}

out_dir = plos_dir / "results" / "info_theoretic"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"  Saved to {out_dir}")
print("=== DONE ===")
