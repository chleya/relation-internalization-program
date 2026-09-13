from __future__ import annotations

import json, math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from g_line.edge_recovery import edge_recovery_metrics
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent

N_OBJ = 6; N_FEAT = 2
HID = 16; NPERT = 60
SDECAY = 0.02; NSTD = 0.01; AMB = 0.5


def make_pairwise_graph(seed):
    rng = random.Random(seed)
    binary = np.zeros((N_OBJ, N_OBJ), dtype=np.int32)
    coef = np.zeros((N_OBJ, N_OBJ), dtype=np.float32)
    for src in range(N_OBJ):
        candidates = [t for t in range(N_OBJ) if t != src]
        rng.shuffle(candidates)
        for tgt in candidates[:rng.randint(1, min(3, N_OBJ - 1))]:
            binary[src, tgt] = 1; coef[src, tgt] = 0.08 + 0.48 * rng.random()
    return binary, coef


def step_pairwise(temps, binary, coef):
    nt = temps.copy()
    for s in range(N_OBJ):
        for t in range(N_OBJ):
            if binary[s, t]: nt[t] += coef[s, t] * (temps[s] - temps[t])
    return nt - SDECAY * (nt - AMB)


def gen_traj_pair(binary, coef, steps=500, seed=0):
    rng = random.Random(seed)
    temps = np.array([AMB + 0.3 * (rng.random() - 0.5) for _ in range(N_OBJ)], dtype=np.float32)
    fs = []
    for _ in range(steps):
        ns = np.array([NSTD * rng.gauss(0, 1) for _ in range(N_OBJ)], dtype=np.float32)
        fs.append(np.stack([temps, temps + ns], axis=1).astype(np.float32))
        temps = step_pairwise(temps, binary, coef)
    return np.array(fs, dtype=np.float32)


def fpairs_pair(binary, coef, steps, seed):
    t = gen_traj_pair(binary, coef, steps, seed); xs, ys = [], []
    for i in range(steps - 1): xs.append(t[i]); ys.append(t[i + 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class EdgeSpecificGNN:
    def __init__(self):
        rng_np = np.random.Generator(np.random.PCG64(42))
        raw_msg = _init_weights((N_OBJ, N_OBJ, N_FEAT, HID), rng_np)
        raw_msg[np.eye(N_OBJ, dtype=bool)] = 0.0
        self.w_msg = raw_msg

        self.w_self = _init_weights((N_FEAT, HID), rng_np)
        self.b_self = np.zeros(HID, dtype=np.float32)
        self.w_out = _init_weights((HID * 2, N_FEAT), rng_np)
        self.b_out = np.zeros(N_FEAT, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)

    def forward(self, feats):
        no = feats.shape[0]
        sh = self._r(feats @ self.w_self + self.b_self)
        ms = np.zeros((no, HID), dtype=np.float32)
        for i in range(no):
            for j in range(no):
                if i == j: continue
                ms[i] += feats[j] @ self.w_msg[j, i]
        combined = np.concatenate([sh, ms], axis=1)
        return combined @ self.w_out + self.b_out

    def param_arrays(self):
        return [self.w_msg, self.w_self, self.b_self, self.w_out, self.b_out]

    def set_all(self, arrs):
        self.w_msg, self.w_self, self.b_self, self.w_out, self.b_out = arrs


class SharedMsgGNN:
    def __init__(self):
        rng_np = np.random.Generator(np.random.PCG64(42))
        self.w_msg = _init_weights((N_FEAT, HID), rng_np)

        self.w_self = _init_weights((N_FEAT, HID), rng_np)
        self.b_self = np.zeros(HID, dtype=np.float32)
        self.w_out = _init_weights((HID * 2, N_FEAT), rng_np)
        self.b_out = np.zeros(N_FEAT, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)

    def forward(self, feats):
        no = feats.shape[0]
        sh = self._r(feats @ self.w_self + self.b_self)
        ms = np.zeros((no, HID), dtype=np.float32)
        for i in range(no):
            for j in range(no):
                if i == j: continue
                ms[i] += feats[j] @ self.w_msg
        combined = np.concatenate([sh, ms], axis=1)
        return combined @ self.w_out + self.b_out

    def param_arrays(self):
        return [self.w_msg, self.w_self, self.b_self, self.w_out, self.b_out]

    def set_all(self, arrs):
        self.w_msg, self.w_self, self.b_self, self.w_out, self.b_out = arrs


def _grads(md, feats, targ, lv):
    eps = 1e-4; all_a = md.param_arrays(); gs = [np.zeros_like(a) for a in all_a]
    rng = random.Random(int(lv * 1e7) + 13007)
    for pi, p in enumerate(all_a):
        flat = p.ravel(); g = gs[pi].ravel()
        idxs = list(range(len(flat))); rng.shuffle(idxs)
        for idx in idxs[:min(NPERT, len(idxs))]:
            old = flat[idx]; flat[idx] = old + eps; md.set_all(all_a)
            l2 = float(np.mean((md.forward(feats) - targ) ** 2).item())
            flat[idx] = old; g[idx] = (l2 - lv) / eps
        gs[pi] = g.reshape(p.shape)
    md.set_all(all_a); return gs


def train_model(md, tx, ty, epochs, lr, name):
    rt = random.Random(777)
    for ep in range(epochs):
        el = 0.0; nb = 0; accum = [np.zeros_like(a) for a in md.param_arrays()]
        idxs = list(range(tx.shape[0])); rt.shuffle(idxs)
        for idx in idxs[:min(32, tx.shape[0])]:
            pr = md.forward(tx[idx]); l = float(np.mean((pr - ty[idx]) ** 2).item())
            if math.isnan(l) or math.isinf(l): continue
            el += l; nb += 1; gs = _grads(md, tx[idx], ty[idx], l)
            for gi in range(len(gs)): accum[gi] += gs[gi]
        if nb > 0:
            el /= nb
            for gi in range(len(accum)): accum[gi] /= nb
        md.set_all([a - lr * g for a, g in zip(md.param_arrays(), accum)])
        if ep % 25 == 0:
            print(f"    ep {ep:3d} loss={el:.6f}")
    return md


print("=" * 53)
print("  EDGE-SPECIFIC GNN")
print("=" * 53)

binary, coef = make_pairwise_graph(42)
n_true = int(binary.sum())
n_strong = int((coef >= 0.35).sum())
strong_mask = binary * (coef >= 0.35)

print(f"  True edges: {n_true}  |  Strong (>0.35): {n_strong}")
for s in range(N_OBJ):
    for t in range(N_OBJ):
        if binary[s, t]:
            tag = "S" if coef[s, t] >= 0.35 else "w"
            print(f"    {s}->{t}: {coef[s, t]:.4f} ({tag})")

tx, ty = fpairs_pair(binary, coef, 400, 10)
print(f"\n  Training EdgeSpecificGNN...")

md_es = EdgeSpecificGNN()
md_es = train_model(md_es, tx, ty, 80, 0.008, "EdgeSpecificGNN")

es_ew = np.zeros((N_OBJ, N_OBJ), dtype=np.float32)
for i in range(N_OBJ):
    for j in range(N_OBJ):
        if i != j:
            es_ew[i, j] = float(np.linalg.norm(md_es.w_msg[j, i]))

es_auc, es_spearman_r, es_avg_precision = edge_recovery_metrics(
    es_ew, coef, N_OBJ, strong_mask)

print(f"  EdgeSpecificGNN: auc={es_auc:.4f}  spearman_r={es_spearman_r:.4f}  avg_precision={es_avg_precision:.4f}")

print(f"\n  Training SharedMsgGNN (baseline)...")

md_bl = SharedMsgGNN()
md_bl = train_model(md_bl, tx, ty, 80, 0.008, "SharedMsgGNN")

w_norm = float(np.linalg.norm(md_bl.w_msg))
bl_ew = np.zeros((N_OBJ, N_OBJ), dtype=np.float32)
bl_ew[~np.eye(N_OBJ, dtype=bool)] = w_norm

bl_auc, bl_spearman_r, bl_avg_precision = edge_recovery_metrics(
    bl_ew, coef, N_OBJ, strong_mask)

print(f"  SharedMsgGNN: auc={bl_auc:.4f}  spearman_r={bl_spearman_r:.4f}  avg_precision={bl_avg_precision:.4f}")

print()
print("=" * 53)
print("  EDGE-SPECIFIC RESULT")
print(f"  es_auc                 = {es_auc:.4f}")
print(f"  es_spearman_r          = {es_spearman_r:.4f}")
print(f"  es_avg_precision        = {es_avg_precision:.4f}")
print(f"  es_baseline_auc         = {bl_auc:.4f}")
print(f"  es_baseline_spearman_r  = {bl_spearman_r:.4f}")
print(f"  es_baseline_avg_prec    = {bl_avg_precision:.4f}")
print(f"  es_delta_auc            = {es_auc - bl_auc:.4f}")
print(f"  es_delta_spearman_r     = {es_spearman_r - bl_spearman_r:.4f}")
print("=" * 53)

metrics = {
    "es_auc": es_auc,
    "es_spearman_r": es_spearman_r,
    "es_avg_precision": es_avg_precision,
    "es_baseline_auc": bl_auc,
    "es_baseline_spearman_r": bl_spearman_r,
    "es_baseline_avg_prec": bl_avg_precision,
    "es_delta_auc": es_auc - bl_auc,
    "es_delta_spearman_r": es_spearman_r - bl_spearman_r,
    "es_edge_weights": es_ew.tolist(),
    "es_baseline_weights": bl_ew.tolist(),
    "true_coef": coef.tolist(),
}

od = plos_dir / "results" / "edge_specific"
od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
