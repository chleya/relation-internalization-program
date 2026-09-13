from __future__ import annotations

import json
import random
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from g_line import gn_world
from g_line import gn_model

plos_dir = Path(__file__).resolve().parent

print("=== W -> O1: Compression Phase Transition ===")

datasets = gn_world.make_continuous_datasets(n_objects=3, n_train=48, n_test=16, n_ood=0, seed=0)
eps = datasets["train"] + datasets["test"]

all_trajs = []
for ep in eps:
    history = ep["model_input"]["interaction_history"]
    obj_ids = sorted(set(int(hi["object_id"]) for hi in history))
    n = len(obj_ids)
    T = gn_world.STEPS_PER_EPISODE
    traj = np.zeros((n, T, gn_world.N_FEATS_CONT), dtype=np.float32)
    for hi in history:
        oi = obj_ids.index(int(hi["object_id"]))
        s = int(hi["step"])
        for fi, name in enumerate(gn_world.ALL_FEATURES_CONT):
            traj[oi, s, fi] = float(hi.get(name, 0.0))
    flat = traj.reshape(-1).copy()
    all_trajs.append(flat)

all_trajs = np.array(all_trajs, dtype=np.float32)
D = all_trajs.shape[1]
rng = random.Random(42)
n_data = len(all_trajs)
train_n = int(n_data * 0.75)
idxs = list(range(n_data))
rng.shuffle(idxs)
train_idx = idxs[:train_n]
test_idx = idxs[train_n:]
train_data = all_trajs[train_idx]
test_data = all_trajs[test_idx]

print(f"  Data: {n_data} trajectories, dim={D}")
print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

K_VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
results = []

for k in K_VALUES:
    if k >= D:
        results.append({"k": k, "train_loss": 0.0, "test_loss": 0.0})
        print(f"  k={k:3d} -> trivial")
        continue

    hidden_ae = 12
    w_enc = gn_model._init_weights((D, hidden_ae), np.random.Generator(np.random.PCG64(k)))
    b_enc = np.zeros(hidden_ae, dtype=np.float32)
    w_bn = gn_model._init_weights((hidden_ae, k), np.random.Generator(np.random.PCG64(k + 1)))
    b_bn = np.zeros(k, dtype=np.float32)
    w_dec0 = gn_model._init_weights((k, hidden_ae), np.random.Generator(np.random.PCG64(k + 2)))
    b_dec0 = np.zeros(hidden_ae, dtype=np.float32)
    w_dec1 = gn_model._init_weights((hidden_ae, D), np.random.Generator(np.random.PCG64(k + 3)))
    b_dec1 = np.zeros(D, dtype=np.float32)

    params = [w_enc, b_enc, w_bn, b_bn, w_dec0, b_dec0, w_dec1, b_dec1]

    def encode(x, p):
        h = np.maximum(0, x @ p[0] + p[1])
        return h @ p[2] + p[3]

    def decode(z, p):
        h = np.maximum(0, z @ p[4] + p[5])
        return h @ p[6] + p[7]

    lr = 0.008
    eps_grad = 1e-4
    n_perturb = 60

    for epoch in range(80):
        rng_batch = random.Random(epoch + k * 100)
        batch_idx = list(range(len(train_data)))
        rng_batch.shuffle(batch_idx)
        batch = train_data[batch_idx[:min(8, len(train_data))]]

        grad_accum = [np.zeros_like(p) for p in params]
        n_b = 0
        for x in batch:
            z = encode(x, params)
            r = decode(z, params)
            base_loss = float(np.mean((r - x) ** 2).item())

            for pi, p in enumerate(params):
                flat_p = p.ravel()
                g_p = np.zeros_like(flat_p)
                all_idxs = list(range(len(flat_p)))
                rng_batch.shuffle(all_idxs)
                n_sel = min(n_perturb, len(all_idxs))
                for idx in all_idxs[:n_sel]:
                    old = flat_p[idx]
                    flat_p[idx] = old + eps_grad
                    z2 = encode(x, params)
                    r2 = decode(z2, params)
                    l2 = float(np.mean((r2 - x) ** 2).item())
                    flat_p[idx] = old
                    g_p[idx] = (l2 - base_loss) / eps_grad
                grad_accum[pi] += g_p.reshape(p.shape)

            n_b += 1

        if n_b > 0:
            for gi in range(len(grad_accum)):
                grad_accum[gi] /= n_b

        new_p = [p - lr * g for p, g in zip(params, grad_accum)]
        params = new_p

        if epoch % 40 == 0:
            z = encode(batch[0], params)
            r = decode(z, params)
            print(f"    k={k:3d} epoch {epoch:3d} loss={float(np.mean((r-batch[0])**2)):.4f}")

    train_loss = 0.0
    for x in train_data:
        z = encode(x, params)
        r = decode(z, params)
        train_loss += float(np.mean((r - x) ** 2).item())
    train_loss /= len(train_data)

    test_loss = 0.0
    for x in test_data:
        z = encode(x, params)
        r = decode(z, params)
        test_loss += float(np.mean((r - x) ** 2).item())
    test_loss /= len(test_data)

    results.append({"k": k, "train_loss": round(train_loss, 6), "test_loss": round(test_loss, 6)})
    print(f"  k={k:3d}  train_loss={train_loss:.6f}  test_loss={test_loss:.6f}")

print("\n=== Phase Transition Analysis ===")
for i in range(1, len(results)):
    prev = results[i - 1]
    curr = results[i]
    if prev["k"] < D:
        delta = prev["test_loss"] - curr["test_loss"]
        rel_drop = delta / max(prev["test_loss"], 1e-8)
        marker = " <-- PHASE?" if rel_drop > 0.25 else ""
        print(f"  k {prev['k']:3d}->{curr['k']:3d}  dLoss={delta:.6f}  rel_drop={rel_drop:.3f}{marker}")

out_dir = plos_dir / "results" / "o1_compression"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "phase_transition.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\n  Saved to {out_dir}")
print("=== DONE ===")
