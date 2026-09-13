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

TF_D_MODEL = 20
TF_N_PERTURB = 60


def _softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class TransformerGenerator:
    def __init__(self):
        rng = np.random.Generator(np.random.PCG64(42))
        self.w_in = _init_weights((gn_world.N_FEATS_CONT, TF_D_MODEL), rng)
        self.b_in = np.zeros(TF_D_MODEL, dtype=np.float32)
        self.w_q = _init_weights((TF_D_MODEL, TF_D_MODEL), rng)
        self.w_k = _init_weights((TF_D_MODEL, TF_D_MODEL), rng)
        self.w_v = _init_weights((TF_D_MODEL, TF_D_MODEL), rng)
        self.w_merge = _init_weights((TF_D_MODEL, TF_D_MODEL), rng)
        self.b_merge = np.zeros(TF_D_MODEL, dtype=np.float32)
        self.w_out = _init_weights((TF_D_MODEL, gn_world.N_FEATS_CONT), rng)
        self.b_out = np.zeros(gn_world.N_FEATS_CONT, dtype=np.float32)

    def _relu(self, x):
        return np.maximum(0, x)

    def forward(self, feats: np.ndarray, use_attn: bool = True) -> np.ndarray:
        n_o = feats.shape[0]
        h = self._relu(feats @ self.w_in + self.b_in)

        if use_attn and n_o > 1:
            Q = h @ self.w_q
            K = h @ self.w_k
            V = h @ self.w_v
            scores = Q @ K.T / math.sqrt(TF_D_MODEL)
            attn = _softmax(scores, axis=-1)
            context = attn @ V
            h = h + self._relu(context @ self.w_merge + self.b_merge)

        return h @ self.w_out + self.b_out

    def params(self):
        return [self.w_in, self.b_in, self.w_q, self.w_k, self.w_v,
                self.w_merge, self.b_merge, self.w_out, self.b_out]

    def set_params(self, flat):
        (self.w_in, self.b_in, self.w_q, self.w_k, self.w_v,
         self.w_merge, self.b_merge, self.w_out, self.b_out) = flat


def generate_frame_pairs(n_steps: int = 300, seed: int = 0, n_objects: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng_st = random.Random(seed)
    state = gn_world._init_objects(rng_st, n_objects)
    feats_x = []
    feats_y = []
    for _ in range(n_steps):
        feats_t = gn_world._state_to_features(state).astype(np.float32)
        state_next = gn_world._step_euler(state)
        feats_t1 = gn_world._state_to_features(state_next).astype(np.float32)
        feats_x.append(feats_t)
        feats_y.append(feats_t1)
        state = state_next
    return np.array(feats_x, dtype=np.float32), np.array(feats_y, dtype=np.float32)


def _tf_grads(model: TransformerGenerator, feats: np.ndarray,
              target: np.ndarray, loss_val: float) -> list[np.ndarray]:
    eps = 1e-4
    all_p = model.params()
    grads = [np.zeros_like(p) for p in all_p]
    rng = random.Random(int(loss_val * 1e7) + 13007)

    for pi, p in enumerate(all_p):
        flat = p.ravel()
        g = grads[pi].ravel()
        idxs = list(range(len(flat)))
        rng.shuffle(idxs)
        for idx in idxs[:min(TF_N_PERTURB, len(idxs))]:
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


print("=== Line 1: Transformer Generator ===")
print("  Generating frame-pair data...")

train_x, train_y = generate_frame_pairs(n_steps=300, seed=0, n_objects=3)
test_x, test_y = generate_frame_pairs(n_steps=80, seed=100, n_objects=3)
ood_x, ood_y = generate_frame_pairs(n_steps=80, seed=200, n_objects=4)

n_train = train_x.shape[0]
print(f"  Training samples: {n_train}")

model = TransformerGenerator()
lr = 0.008
rng_train = random.Random(777)

for epoch in range(100):
    epoch_loss = 0.0
    n_b = 0
    accum = [np.zeros_like(p) for p in model.params()]
    idxs = list(range(n_train))
    rng_train.shuffle(idxs)
    for idx in idxs[:min(32, n_train)]:
        pred = model.forward(train_x[idx], use_attn=True)
        loss = float(np.mean((pred - train_y[idx]) ** 2).item())
        if math.isnan(loss) or math.isinf(loss):
            continue
        epoch_loss += loss
        n_b += 1
        grads = _tf_grads(model, train_x[idx], train_y[idx], loss)
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


def eval_shuffled_attn(data_x, data_y, label):
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
with_loss = eval_tf(test_x, test_y, True, "with_attn   ")
no_loss = eval_tf(test_x, test_y, False, "no_attn     ")
shuf_loss = eval_shuffled_attn(test_x, test_y, "shuf_attn   ")

ood_with = eval_tf(ood_x, ood_y, True, "ood_attn    ")
ood_no = eval_tf(ood_x, ood_y, False, "ood_noattn  ")
ood_shuf = eval_shuffled_attn(ood_x, ood_y, "ood_shufattn")

attn_benefit = no_loss - with_loss
shuf_drop = shuf_loss - with_loss
ood_benefit = ood_no - ood_with
ood_shuf_drop = ood_shuf - ood_with

print(f"\n  === TRANSFORMER RESULT ===")
print(f"  tf_with_attn          = {with_loss:.6f}")
print(f"  tf_no_attn            = {no_loss:.6f}")
print(f"  tf_shuf_attn          = {shuf_loss:.6f}")
print(f"  tf_attn_benefit       = {attn_benefit:.6f}  (+ = attention helps)")
print(f"  tf_shuf_drop          = {shuf_drop:.6f}  (+ = shuffled attn hurts)")
print(f"  tf_ood_benefit        = {ood_benefit:.6f}")
print(f"  tf_ood_shuf_drop      = {ood_shuf_drop:.6f}")

metrics = {
    "tf_with_attn": with_loss, "tf_no_attn": no_loss, "tf_shuf_attn": shuf_loss,
    "tf_attn_benefit": attn_benefit, "tf_shuf_drop": shuf_drop,
    "tf_ood_with": ood_with, "tf_ood_no": ood_no, "tf_ood_shuf": ood_shuf,
    "tf_ood_benefit": ood_benefit, "tf_ood_shuf_drop": ood_shuf_drop,
}

out_dir = plos_dir / "results" / "transformer_generator"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"  Saved to {out_dir}")
print("=== DONE ===")
