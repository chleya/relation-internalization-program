from __future__ import annotations

import random
from typing import Any
import numpy as np

ALL_FEATURES = [
    "prediction_error", "compression_surprise", "intervention_gain",
    "indirect_evidence", "feedback_success", "risk_proxy", "delay_signal",
    "observed_direct_reward", "observed_indirect_reward", "observed_inspect_value",
]
N_FEATS = len(ALL_FEATURES)
HIDDEN = 32


def _init_weights(shape: tuple, rng: np.random.Generator, scale: float = 0.1):
    return rng.normal(0, scale, shape).astype(np.float32)


class NeuralGenerator:
    def __init__(self, seed: int = 42, input_dim: int = N_FEATS):
        self.input_dim = input_dim
        rng = np.random.Generator(np.random.PCG64(seed))
        self.w_self = _init_weights((input_dim, HIDDEN), rng)
        self.b_self = np.zeros(HIDDEN, dtype=np.float32)
        self.w_cross = _init_weights((input_dim, HIDDEN), rng)
        self.b_cross = np.zeros(HIDDEN, dtype=np.float32)
        self.w_head = _init_weights((HIDDEN * 2, 3), rng)
        self.b_head = np.zeros(3, dtype=np.float32)
        self.rng = rng

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def forward(self, obj_features: np.ndarray, use_cross: bool = True) -> np.ndarray:
        n_objs, n_feats = obj_features.shape
        self_hidden = self._relu(obj_features @ self.w_self + self.b_self)

        if use_cross and n_objs > 1:
            cross_hidden = self._relu(obj_features @ self.w_cross + self.b_cross)
            cross_agg = np.zeros((n_objs, HIDDEN), dtype=np.float32)
            for i in range(n_objs):
                mask = np.ones(n_objs, dtype=bool)
                mask[i] = False
                cross_agg[i] = cross_hidden[mask].mean(axis=0)
        else:
            cross_agg = np.zeros((n_objs, HIDDEN), dtype=np.float32)

        combined = np.concatenate([self_hidden, cross_agg], axis=1)
        return self._relu(combined @ self.w_head + self.b_head)

    def params(self) -> list[np.ndarray]:
        return [self.w_self, self.b_self, self.w_cross, self.b_cross, self.w_head, self.b_head]

    def set_params(self, flat: list[np.ndarray]):
        self.w_self, self.b_self, self.w_cross, self.b_cross, self.w_head, self.b_head = flat

    def copy(self) -> NeuralGenerator:
        new = NeuralGenerator(0)
        new.set_params([p.copy() for p in self.params()])
        return new


def encode_episode(ep: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    history = ep["model_input"]["interaction_history"]
    obj_map: dict[int, list[dict]] = {}
    for hi in history:
        oid = int(hi["object_id"])
        obj_map.setdefault(oid, []).append(hi)

    obj_ids = sorted(obj_map.keys())
    n_objs = len(obj_ids)
    features = np.zeros((n_objs, N_FEATS), dtype=np.float32)
    targets = np.zeros((n_objs, 3), dtype=np.float32)
    truth = {t["region_id"]: t for t in ep["evaluator_ground_truth"]["regions"]}

    for idx, oid in enumerate(obj_ids):
        items = obj_map[oid]
        for name_i, name in enumerate(ALL_FEATURES):
            features[idx, name_i] = np.mean([float(it.get(name, 0.0)) for it in items])
        dir_good = any(truth[it["region_id"]]["direct_actionable"] for it in items)
        ind_good = any(truth[it["region_id"]]["indirect_actionable"] for it in items)
        insp_good = any(truth[it["region_id"]]["inspectable"] for it in items)
        targets[idx] = [float(dir_good), float(ind_good), float(insp_good)]
    return features, targets, obj_ids


def mean_pool_obj(targets: np.ndarray) -> float:
    return float(targets.mean())


def decode_output(output: np.ndarray, obj_features: np.ndarray, risk_threshold: float = 0.5) -> dict[int, dict]:
    n_feats = obj_features.shape[1]
    risk_idx = 5 if n_feats >= 10 else -1
    mask = {}
    for i in range(output.shape[0]):
        risk = float(obj_features[i, risk_idx]) if risk_idx >= 0 else 0.5
        mask[i] = {
            "directly_intervenable": bool(output[i, 0] > 0.5 and risk <= risk_threshold),
            "indirectly_intervenable": bool(output[i, 1] > 0.5 and risk <= risk_threshold + 0.1),
            "inspectable": bool(output[i, 2] > 0.5),
            "discovered_group": i,
            "similar_groups": [],
        }
    return mask


def choose_action_from_output(output: np.ndarray, obj_features: np.ndarray) -> dict | None:
    n_feats = obj_features.shape[1]
    risk_idx = 5 if n_feats >= 10 else -1
    gain_idx = 2 if n_feats >= 10 else -1
    ind_idx = 3 if n_feats >= 10 else -1
    best_score = -1.0
    best_oid = None
    best_type = "abstain"
    for i in range(output.shape[0]):
        risk = float(obj_features[i, risk_idx]) if risk_idx >= 0 else 0.5
        gain = float(obj_features[i, gain_idx]) if gain_idx >= 0 else 0.5
        ind_ev = float(obj_features[i, ind_idx]) if ind_idx >= 0 else 0.5
        if output[i, 0] > 0.5 and risk <= 0.5:
            score = gain - risk
            if score > best_score:
                best_score = score
                best_oid = i
                best_type = "apply_local_damping"
        if output[i, 1] > 0.5 and risk <= 0.6:
            score = ind_ev - 0.5 * risk
            if score > best_score:
                best_score = score
                best_oid = i
                best_type = "indirect_stabilize"
    if best_oid is None:
        return None
    return {"region_id": best_oid, "action_type": best_type}


def compute_gradients(model: NeuralGenerator, obj_features: np.ndarray,
                      targets: np.ndarray, use_cross: bool = True) -> tuple[float, list[np.ndarray]]:
    eps = 1e-5
    grads = [np.zeros_like(p) for p in model.params()]
    base_out = model.forward(obj_features, use_cross=use_cross)
    base_loss = np.mean((base_out - targets) ** 2)

    for pi, p in enumerate(model.params()):
        flat = p.ravel()
        grad_flat = np.zeros_like(flat)
        for i in range(len(flat)):
            old = flat[i]
            flat[i] = old + eps
            model.set_params(model.params())
            out_plus = model.forward(obj_features, use_cross=use_cross)
            loss_plus = np.mean((out_plus - targets) ** 2)
            flat[i] = old
            grad_flat[i] = (loss_plus - base_loss) / eps
        grads[pi] = grad_flat.reshape(p.shape)
    model.set_params(model.params())
    return base_loss, grads


def train(model: NeuralGenerator, episodes: list[dict],
          lr: float = 0.01, epochs: int = 200,
          mask_train: bool = False) -> list[float]:
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        rng = random.Random(epoch + 777)
        shuffled = list(episodes)
        rng.shuffle(shuffled)
        accum_grads = [np.zeros_like(p) for p in model.params()]
        n_batch = 0
        for ep in shuffled[:min(32, len(shuffled))]:
            feats, targets, _obj_ids = encode_episode(ep)
            n_objs = feats.shape[0]
            if n_objs < 2:
                continue

            masked_feats = feats.copy()
            mask_id = -1
            if mask_train:
                mask_id = rng.randint(0, n_objs - 1)
                masked_feats[mask_id] = 0.0

            loss, grads = compute_gradients(model, masked_feats, targets, use_cross=True)

            if mask_train and mask_id >= 0:
                base_out = model.forward(masked_feats, use_cross=True)
                mask_loss = np.mean((base_out[mask_id] - targets[mask_id]) ** 2).item()
                loss = float(mask_loss)
                grads = _mask_only_grads(model, masked_feats, targets, mask_id)

            epoch_loss += loss
            n_batch += 1
            for gi in range(len(grads)):
                accum_grads[gi] += grads[gi]
        if n_batch > 0:
            epoch_loss /= n_batch
            for gi in range(len(accum_grads)):
                accum_grads[gi] /= n_batch
        losses.append(epoch_loss)

        new_params = []
        for p, g in zip(model.params(), accum_grads):
            new_params.append(p - lr * g)
        model.set_params(new_params)

        if epoch % 50 == 0:
            print(f"    epoch {epoch:3d}  loss={epoch_loss:.4f}" + (" [masked]" if mask_train else ""))
    return losses


def _mask_only_grads(model: NeuralGenerator, obj_features: np.ndarray,
                     targets: np.ndarray, mask_id: int) -> list[np.ndarray]:
    eps = 1e-5
    grads = [np.zeros_like(p) for p in model.params()]
    base_out = model.forward(obj_features, use_cross=True)
    base_loss = float(np.mean((base_out[mask_id] - targets[mask_id]) ** 2).item())
    for pi, p in enumerate(model.params()):
        flat = p.ravel()
        grad_flat = np.zeros_like(flat)
        for i in range(len(flat)):
            old = flat[i]
            flat[i] = old + eps
            model.set_params(model.params())
            out_plus = model.forward(obj_features, use_cross=True)
            loss_plus = float(np.mean((out_plus[mask_id] - targets[mask_id]) ** 2).item())
            flat[i] = old
            grad_flat[i] = (loss_plus - base_loss) / eps
        grads[pi] = grad_flat.reshape(p.shape)
    model.set_params(model.params())
    return grads


CONTRAST_HIDDEN = 24
CONTRAST_MARGIN = 0.3


class ContrastiveScorer:
    def __init__(self, seed: int = 42):
        rng = np.random.Generator(np.random.PCG64(seed))
        self.w_self = _init_weights((N_FEATS, CONTRAST_HIDDEN), rng)
        self.b_self = np.zeros(CONTRAST_HIDDEN, dtype=np.float32)
        self.w_cross = _init_weights((N_FEATS, CONTRAST_HIDDEN), rng)
        self.b_cross = np.zeros(CONTRAST_HIDDEN, dtype=np.float32)
        self.w_head = _init_weights((CONTRAST_HIDDEN * 2, 1), rng)
        self.b_head = np.zeros(1, dtype=np.float32)

    def _relu(self, x):
        return np.maximum(0, x)

    def score_pair(self, feats_a: np.ndarray, feats_b: np.ndarray, use_cross: bool = True) -> float:
        ha = self._relu(feats_a @ self.w_self + self.b_self)
        hb = self._relu(feats_b @ self.w_self + self.b_self)
        if use_cross:
            ca = self._relu(feats_a @ self.w_cross + self.b_cross)
            cb = self._relu(feats_b @ self.w_cross + self.b_cross)
            cross_a = cb
            cross_b = ca
        else:
            cross_a = np.zeros(CONTRAST_HIDDEN, dtype=np.float32)
            cross_b = np.zeros(CONTRAST_HIDDEN, dtype=np.float32)
        combined = np.concatenate([ha + hb, cross_a + cross_b])
        return float((combined @ self.w_head + self.b_head).item())

    def params(self):
        return [self.w_self, self.b_self, self.w_cross, self.b_cross, self.w_head, self.b_head]

    def set_params(self, flat):
        self.w_self, self.b_self, self.w_cross, self.b_cross, self.w_head, self.b_head = flat


def _contrast_grads(model: ContrastiveScorer, a: np.ndarray, b: np.ndarray,
                    loss_val: float, use_cross: bool, is_pos: bool) -> list[np.ndarray]:
    eps = 1e-5
    grads = [np.zeros_like(p) for p in model.params()]
    for pi, p in enumerate(model.params()):
        flat = p.ravel()
        grad_flat = np.zeros_like(flat)
        for i in range(len(flat)):
            old = flat[i]
            flat[i] = old + eps
            model.set_params(model.params())
            new_score = model.score_pair(a, b, use_cross=use_cross)
            if is_pos:
                new_loss = max(0.0, CONTRAST_MARGIN - new_score)
            else:
                new_loss = max(0.0, CONTRAST_MARGIN + new_score)
            flat[i] = old
            grad_flat[i] = (new_loss - loss_val) / eps
        grads[pi] = grad_flat.reshape(p.shape)
    model.set_params(model.params())
    return grads


def train_contrastive(model: ContrastiveScorer, episodes: list[dict],
                      lr: float = 0.02, epochs: int = 300):
    rng = random.Random(42)
    obj_pool: list[tuple[np.ndarray, int]] = []
    for ep_i, ep in enumerate(episodes):
        feats, _, _ = encode_episode(ep)
        for oi in range(feats.shape[0]):
            obj_pool.append((feats[oi].copy(), ep_i))

    for epoch in range(epochs):
        epoch_loss = 0.0
        accum = [np.zeros_like(p) for p in model.params()]
        n_batch = 0
        rng.shuffle(obj_pool)

        for i in range(0, min(128, len(obj_pool) // 2 * 2), 2):
            feats_a, ep_a = obj_pool[i]
            feats_b, ep_b = obj_pool[i + 1]
            is_pos = (ep_a == ep_b)
            score = model.score_pair(feats_a, feats_b, use_cross=True)
            if is_pos:
                loss = max(0.0, CONTRAST_MARGIN - score)
            else:
                loss = max(0.0, CONTRAST_MARGIN + score)
            epoch_loss += loss
            n_batch += 1
            grads = _contrast_grads(model, feats_a, feats_b, loss, use_cross=True, is_pos=is_pos)
            for gi in range(len(grads)):
                accum[gi] += grads[gi]

        if n_batch > 0:
            epoch_loss /= n_batch
            for gi in range(len(accum)):
                accum[gi] /= n_batch

        new_params = []
        for p, g in zip(model.params(), accum):
            new_params.append(p - lr * g)
        model.set_params(new_params)

        if epoch % 50 == 0:
            print(f"    contrast epoch {epoch:3d}  loss={epoch_loss:.4f}")


BOTTLENECK = 2


class BottleneckAE:
    def __init__(self, seed: int = 42):
        rng = np.random.Generator(np.random.PCG64(seed))
        self.w_enc = _init_weights((N_FEATS, 8), rng)
        self.b_enc = np.zeros(8, dtype=np.float32)
        self.w_bn = _init_weights((8, BOTTLENECK), rng)
        self.b_bn = np.zeros(BOTTLENECK, dtype=np.float32)
        self.w_dec0 = _init_weights((BOTTLENECK, 8), rng)
        self.b_dec0 = np.zeros(8, dtype=np.float32)
        self.w_dec1 = _init_weights((8, N_FEATS), rng)
        self.b_dec1 = np.zeros(N_FEATS, dtype=np.float32)

    def _relu(self, x):
        return np.maximum(0, x)

    def encode(self, feats: np.ndarray) -> np.ndarray:
        h = self._relu(feats @ self.w_enc + self.b_enc)
        return h @ self.w_bn + self.b_bn

    def decode(self, latent: np.ndarray) -> np.ndarray:
        h = self._relu(latent @ self.w_dec0 + self.b_dec0)
        return h @ self.w_dec1 + self.b_dec1

    def params(self):
        return [self.w_enc, self.b_enc, self.w_bn, self.b_bn,
                self.w_dec0, self.b_dec0, self.w_dec1, self.b_dec1]

    def set_params(self, flat):
        self.w_enc, self.b_enc, self.w_bn, self.b_bn, \
        self.w_dec0, self.b_dec0, self.w_dec1, self.b_dec1 = flat


def train_autoencoder(model: BottleneckAE, episodes: list[dict],
                      lr: float = 0.01, epochs: int = 200):
    all_feats = []
    for ep in episodes:
        feats, _, _ = encode_episode(ep)
        for oi in range(feats.shape[0]):
            all_feats.append(feats[oi].copy())
    all_feats = np.array(all_feats, dtype=np.float32)
    eps = 1e-5

    for epoch in range(epochs):
        rng = random.Random(epoch + 999)
        indices = list(range(len(all_feats)))
        rng.shuffle(indices)
        batch = indices[:min(64, len(indices))]
        accum = [np.zeros_like(p) for p in model.params()]

        for idx in batch:
            x = all_feats[idx]
            latent = model.encode(x)
            recon = model.decode(latent)
            base_loss = float(np.mean((recon - x) ** 2).item())

            for pi, p in enumerate(model.params()):
                flat_p = p.ravel()
                grad_f = np.zeros_like(flat_p)
                for i in range(len(flat_p)):
                    old = flat_p[i]
                    flat_p[i] = old + eps
                    model.set_params(model.params())
                    new_recon = model.decode(model.encode(x))
                    new_loss = float(np.mean((new_recon - x) ** 2).item())
                    flat_p[i] = old
                    grad_f[i] = (new_loss - base_loss) / eps
                accum[pi] += grad_f.reshape(p.shape)
            model.set_params(model.params())

        n = len(batch)
        for gi in range(len(accum)):
            accum[gi] /= n

        new_params = []
        for p, g in zip(model.params(), accum):
            new_params.append(p - lr * g)
        model.set_params(new_params)

        if epoch % 50 == 0:
            total_loss = 0.0
            for idx in batch:
                x = all_feats[idx]
                recon = model.decode(model.encode(x))
                total_loss += float(np.mean((recon - x) ** 2).item())
            print(f"    ae epoch {epoch:3d}  recon_loss={total_loss / len(batch):.4f}")


TEMP_HIDDEN = 32


class TemporalGenerator:
    def __init__(self, input_dim: int, n_steps: int = 20, seed: int = 42):
        self.input_dim = input_dim
        self.n_steps = n_steps
        flat_in = n_steps * input_dim
        rng = np.random.Generator(np.random.PCG64(seed))
        self.w_self = _init_weights((flat_in, TEMP_HIDDEN), rng)
        self.b_self = np.zeros(TEMP_HIDDEN, dtype=np.float32)
        self.w_cross = _init_weights((flat_in, TEMP_HIDDEN), rng)
        self.b_cross = np.zeros(TEMP_HIDDEN, dtype=np.float32)
        self.w_head = _init_weights((TEMP_HIDDEN * 2, flat_in), rng)
        self.b_head = np.zeros(flat_in, dtype=np.float32)

    def _relu(self, x):
        return np.maximum(0, x)

    def forward(self, trajectories: np.ndarray, use_cross: bool = True) -> np.ndarray:
        n_objs, T, d = trajectories.shape
        flat = trajectories.reshape(n_objs, -1)
        self_h = self._relu(flat @ self.w_self + self.b_self)
        if use_cross and n_objs > 1:
            cross_h = self._relu(flat @ self.w_cross + self.b_cross)
            cross_agg = np.zeros((n_objs, TEMP_HIDDEN), dtype=np.float32)
            for i in range(n_objs):
                m = np.ones(n_objs, dtype=bool)
                m[i] = False
                cross_agg[i] = cross_h[m].mean(axis=0)
        else:
            cross_agg = np.zeros((n_objs, TEMP_HIDDEN), dtype=np.float32)
        combined = np.concatenate([self_h, cross_agg], axis=1)
        pred = combined @ self.w_head + self.b_head
        return pred.reshape(n_objs, T, d)

    def params(self):
        return [self.w_self, self.b_self, self.w_cross, self.b_cross, self.w_head, self.b_head]

    def set_params(self, flat):
        self.w_self, self.b_self, self.w_cross, self.b_cross, self.w_head, self.b_head = flat


def _temp_grads(model: TemporalGenerator, trajs: np.ndarray,
                mask_obj: int, loss_val: float, n_perturb: int = 200) -> list[np.ndarray]:
    eps = 1e-4
    all_p = model.params()
    grads = [np.zeros_like(p) for p in all_p]
    masked_trajs = trajs.copy()
    masked_trajs[mask_obj] = 0.0
    rng = random.Random(mask_obj * 10007 + int(loss_val * 1e6))

    for pi, p in enumerate(all_p):
        flat = p.ravel()
        g = grads[pi].ravel()
        idxs = list(range(len(flat)))
        rng.shuffle(idxs)
        for idx in idxs[:min(n_perturb, len(idxs))]:
            old = flat[idx]
            flat[idx] = old + eps
            model.set_params(all_p)
            pred = model.forward(masked_trajs, use_cross=True)
            l2 = float(np.mean((pred[mask_obj] - trajs[mask_obj]) ** 2).item())
            flat[idx] = old
            g[idx] = (l2 - loss_val) / eps
        grads[pi] = g.reshape(p.shape)
    model.set_params(all_p)
    return grads

