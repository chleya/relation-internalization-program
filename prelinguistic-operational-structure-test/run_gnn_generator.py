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

GNN_HIDDEN = 16
GNN_N_PERTURB = 60


def _softplus(x):
    return np.log(1 + np.exp(np.clip(x, -20, 20)))


class GNNModel:
    def __init__(self):
        rng = np.random.Generator(np.random.PCG64(42))
        self.w_self = _init_weights((gn_world.N_FEATS_CONT, GNN_HIDDEN), rng)
        self.b_self = np.zeros(GNN_HIDDEN, dtype=np.float32)
        self.w_edge_k = _init_weights((gn_world.N_FEATS_CONT * 2, GNN_HIDDEN), rng)
        self.w_edge_v = _init_weights((gn_world.N_FEATS_CONT, GNN_HIDDEN), rng)
        self.w_out = _init_weights((GNN_HIDDEN * 2, gn_world.N_FEATS_CONT), rng)
        self.b_out = np.zeros(gn_world.N_FEATS_CONT, dtype=np.float32)

    def _relu(self, x):
        return np.maximum(0, x)

    def forward(self, feats: np.ndarray, use_edges: bool = True) -> np.ndarray:
        n_o = feats.shape[0]
        self_h = self._relu(feats @ self.w_self + self.b_self)

        msgs = np.zeros((n_o, GNN_HIDDEN), dtype=np.float32)
        if use_edges:
            for i in range(n_o):
                for j in range(n_o):
                    if i == j:
                        continue
                    pair = np.concatenate([feats[i], feats[j]])
                    edge_raw = pair @ self.w_edge_k
                    edge_w = float(_softplus(float(edge_raw.sum())))
                    msg = self._relu(feats[j] @ self.w_edge_v)
                    msgs[i] += edge_w * msg

        combined = np.concatenate([self_h, msgs], axis=1)
        return combined @ self.w_out + self.b_out

    def params(self):
        return [self.w_self, self.b_self, self.w_edge_k, self.w_edge_v, self.w_out, self.b_out]

    def set_params(self, flat):
        self.w_self, self.b_self, self.w_edge_k, self.w_edge_v, self.w_out, self.b_out = flat


def generate_frame_pairs(n_steps: int = 250, seed: int = 0, n_objects: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng_st = random.Random(seed)
    state = gn_world._init_objects(rng_st, n_objects)
    feats_x = []
    feats_y = []
    for _ in range(n_steps):
        feats_t = gn_world._state_to_features(state)
        state_next = gn_world._step_euler(state)
        feats_t1 = gn_world._state_to_features(state_next)
        feats_x.append(feats_t.astype(np.float32))
        feats_y.append(feats_t1.astype(np.float32))
        state = state_next
    return np.array(feats_x, dtype=np.float32), np.array(feats_y, dtype=np.float32)


def _gnn_grads(model: GNNModel, feats: np.ndarray, target: np.ndarray,
               loss_val: float) -> list[np.ndarray]:
    eps = 1e-4
    all_p = model.params()
    grads = [np.zeros_like(p) for p in all_p]
    rng = random.Random(int(loss_val * 1e7) + 9973)

    for pi, p in enumerate(all_p):
        flat = p.ravel()
        g = grads[pi].ravel()
        idxs = list(range(len(flat)))
        rng.shuffle(idxs)
        for idx in idxs[:min(GNN_N_PERTURB, len(idxs))]:
            old = flat[idx]
            flat[idx] = old + eps
            model.set_params(all_p)
            pred = model.forward(feats, use_edges=True)
            l2 = float(np.mean((pred - target) ** 2).item())
            flat[idx] = old
            g[idx] = (l2 - loss_val) / eps
        grads[pi] = g.reshape(p.shape)
    model.set_params(all_p)
    return grads


print("=== Route B: GNN Generator ===")
print("  Generating frame-pair data...")

train_x, train_y = generate_frame_pairs(n_steps=250, seed=0, n_objects=3)
test_x, test_y = generate_frame_pairs(n_steps=80, seed=100, n_objects=3)
ood_x, ood_y = generate_frame_pairs(n_steps=80, seed=200, n_objects=4)

n_train = train_x.shape[0]
print(f"  Training samples: {n_train}")

model = GNNModel()
lr = 0.01
rng_train = random.Random(777)

for epoch in range(80):
    epoch_loss = 0.0
    n_b = 0
    accum = [np.zeros_like(p) for p in model.params()]
    idxs = list(range(n_train))
    rng_train.shuffle(idxs)
    for idx in idxs[:min(32, n_train)]:
        pred = model.forward(train_x[idx], use_edges=True)
        loss = float(np.mean((pred - train_y[idx]) ** 2).item())
        if math.isnan(loss) or math.isinf(loss):
            continue
        epoch_loss += loss
        n_b += 1
        grads = _gnn_grads(model, train_x[idx], train_y[idx], loss)
        for gi in range(len(grads)):
            accum[gi] += grads[gi]
    if n_b > 0:
        epoch_loss /= n_b
        for gi in range(len(accum)):
            accum[gi] /= n_b
    new_p = [p - lr * g for p, g in zip(model.params(), accum)]
    model.set_params(new_p)

    if epoch % 25 == 0:
        print(f"    epoch {epoch:3d}  loss={epoch_loss:.6f}")


def eval_gnn(data_x, data_y, use_edges, label):
    total_loss = 0.0
    n = 0
    for idx in range(data_x.shape[0]):
        pred = model.forward(data_x[idx], use_edges=use_edges)
        loss = float(np.mean((pred - data_y[idx]) ** 2).item())
        if math.isnan(loss):
            continue
        total_loss += loss
        n += 1
    avg = total_loss / max(n, 1)
    print(f"    {label}: loss={avg:.6f}")
    return avg


def eval_with_shuffled_edges(data_x, data_y, label):
    orig_wk = model.w_edge_k.copy()
    flat = model.w_edge_k.ravel()
    rng = np.random.Generator(np.random.PCG64(999))
    rng.shuffle(flat)
    model.w_edge_k = flat.reshape(model.w_edge_k.shape)

    total_loss = 0.0
    n = 0
    for idx in range(data_x.shape[0]):
        pred = model.forward(data_x[idx], use_edges=True)
        loss = float(np.mean((pred - data_y[idx]) ** 2).item())
        if math.isnan(loss):
            continue
        total_loss += loss
        n += 1
    avg = total_loss / max(n, 1)
    model.w_edge_k = orig_wk
    print(f"    {label}: loss={avg:.6f}")
    return avg


print("\n  Evaluating...")
with_loss = eval_gnn(test_x, test_y, True, "with_edges  ")
no_loss = eval_gnn(test_x, test_y, False, "no_edges    ")
shuf_loss = eval_with_shuffled_edges(test_x, test_y, "shuf_edges  ")

ood_with = eval_gnn(ood_x, ood_y, True, "ood_edges   ")
ood_no = eval_gnn(ood_x, ood_y, False, "ood_noedges ")
ood_shuf = eval_with_shuffled_edges(ood_x, ood_y, "ood_shufedge")

edge_benefit = no_loss - with_loss
shuf_drop = shuf_loss - with_loss
ood_benefit = ood_no - ood_with
ood_shuf_drop = ood_shuf - ood_with

print(f"\n  === GNN RESULT ===")
print(f"  gnn_with_edges      = {with_loss:.6f}")
print(f"  gnn_no_edges        = {no_loss:.6f}")
print(f"  gnn_shuf_edges      = {shuf_loss:.6f}")
print(f"  gnn_edge_benefit    = {edge_benefit:.6f}  (+ = edges help)")
print(f"  gnn_shuf_drop       = {shuf_drop:.6f}  (+ = shuffled edges hurt)")
print(f"  gnn_ood_benefit     = {ood_benefit:.6f}")
print(f"  gnn_ood_shuf_drop   = {ood_shuf_drop:.6f}")

metrics = {
    "gnn_with_edges": with_loss,
    "gnn_no_edges": no_loss,
    "gnn_shuf_edges": shuf_loss,
    "gnn_edge_benefit": edge_benefit,
    "gnn_shuf_drop": shuf_drop,
    "gnn_ood_with_edges": ood_with,
    "gnn_ood_no_edges": ood_no,
    "gnn_ood_shuf_edges": ood_shuf,
    "gnn_ood_benefit": ood_benefit,
    "gnn_ood_shuf_drop": ood_shuf_drop,
}

out_dir = plos_dir / "results" / "gnn_generator"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"  Saved to {out_dir}")
print("=== DONE ===")
