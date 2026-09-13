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

E_EPOCHS = 80; LR = 0.008
OOD_TRAIN_N_OBJ = 5; OOD_TEST_N_OBJ = 6


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


def step_pairwise(temps, binary, coef, n_obj=None):
    n = n_obj if n_obj else temps.shape[0]
    nt = temps.copy()
    for s in range(n):
        for t in range(n):
            if binary[s, t]: nt[t] += coef[s, t] * (temps[s] - temps[t])
    return nt - SDECAY * (nt - AMB)


def gen_traj_pair(binary, coef, steps=500, seed=0, n_obj=None):
    rng = random.Random(seed)
    n = n_obj if n_obj else N_OBJ
    temps = np.array([AMB + 0.3 * (rng.random() - 0.5) for _ in range(n)], dtype=np.float32)
    fs = []
    for _ in range(steps):
        ns = np.array([NSTD * rng.gauss(0, 1) for _ in range(n)], dtype=np.float32)
        fs.append(np.stack([temps, temps + ns], axis=1).astype(np.float32))
        temps = step_pairwise(temps, binary, coef, n_obj=n)
    return np.array(fs, dtype=np.float32)


def fpairs_pair(binary, coef, steps, seed, n_obj=None):
    t = gen_traj_pair(binary, coef, steps, seed, n_obj=n_obj); xs, ys = [], []
    for i in range(steps - 1): xs.append(t[i]); ys.append(t[i + 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


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
        no = feats.shape[0]; sh = self._r(feats @ self.w_self + self.b_self)
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


class HyperEdgeGNN:
    def __init__(self, n_obj=N_OBJ, embed_dim=8):
        rng_np = np.random.Generator(np.random.PCG64(42))
        self.n_obj = n_obj; self.D = embed_dim; self.HH = 16

        self.edge_emb = _init_weights((n_obj, n_obj, embed_dim), rng_np, 0.05)
        for i in range(n_obj): self.edge_emb[i, i] = 0.0

        self.hyper_w1 = _init_weights((embed_dim, self.HH), rng_np)
        self.hyper_b1 = np.zeros(self.HH, dtype=np.float32)
        self.hyper_w2 = _init_weights((self.HH, N_FEAT * HID), rng_np)
        self.hyper_b2 = np.zeros(N_FEAT * HID, dtype=np.float32)

        self.w_self = _init_weights((N_FEAT, HID), rng_np)
        self.b_self = np.zeros(HID, dtype=np.float32)
        self.w_out = _init_weights((HID * 2, N_FEAT), rng_np)
        self.b_out = np.zeros(N_FEAT, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)

    def forward(self, feats):
        no = feats.shape[0]; sh = self._r(feats @ self.w_self + self.b_self)
        ms = np.zeros((no, HID), dtype=np.float32)
        for i in range(no):
            for j in range(no):
                if i == j: continue
                h = self._r(self.edge_emb[j, i] @ self.hyper_w1 + self.hyper_b1)
                w_msg = (h @ self.hyper_w2 + self.hyper_b2).reshape(N_FEAT, HID)
                ms[i] += feats[j] @ w_msg
        combined = np.concatenate([sh, ms], axis=1)
        return combined @ self.w_out + self.b_out

    def param_arrays(self):
        return [self.edge_emb, self.hyper_w1, self.hyper_b1, self.hyper_w2, self.hyper_b2,
                self.w_self, self.b_self, self.w_out, self.b_out]

    def set_all(self, arrs):
        (self.edge_emb, self.hyper_w1, self.hyper_b1, self.hyper_w2, self.hyper_b2,
         self.w_self, self.b_self, self.w_out, self.b_out) = arrs

    def edge_weight_matrix(self, n_eval=None):
        n = n_eval if n_eval else self.n_obj
        ew = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                h = self._r(self.edge_emb[j, i] @ self.hyper_w1 + self.hyper_b1)
                w_msg = (h @ self.hyper_w2 + self.hyper_b2).reshape(N_FEAT, HID)
                ew[j, i] = float(np.linalg.norm(w_msg))
        return ew


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


print("=" * 55)
print("  HYPERNET GNN")
print("=" * 55)

binary, coef = make_pairwise_graph(42)
n_true = int(binary.sum())
strong_mask = binary * (coef >= 0.35)
n_strong = int(strong_mask.sum())

print(f"  True edges: {n_true}  |  Strong (>0.35): {n_strong}")
for s in range(N_OBJ):
    for t in range(N_OBJ):
        if binary[s, t]:
            tag = "S" if coef[s, t] >= 0.35 else "w"
            print(f"    {s}->{t}: {coef[s, t]:.4f} ({tag})")

tx, ty = fpairs_pair(binary, coef, 400, 10)
print(f"\n  Training SharedMsgGNN baseline...")
md_bl = SharedMsgGNN(); md_bl = train_model(md_bl, tx, ty, E_EPOCHS, LR, "SharedMsgGNN")
w_norm = float(np.linalg.norm(md_bl.w_msg))
bl_ew = np.full((N_OBJ, N_OBJ), w_norm, dtype=np.float32); np.fill_diagonal(bl_ew, 0)
bl_auc, bl_spearman_r, bl_avg_precision = edge_recovery_metrics(bl_ew, coef, N_OBJ, strong_mask)
print(f"  SharedMsgGNN: auc={bl_auc:.4f}  spearman_r={bl_spearman_r:.4f}  avg_precision={bl_avg_precision:.4f}")

results = {"baseline": {"auc": bl_auc, "spearman_r": bl_spearman_r, "avg_precision": bl_avg_precision}}

for D in [4, 8]:
    print(f"\n  Training HyperEdgeGNN D={D}...")
    md_h = HyperEdgeGNN(N_OBJ, D); md_h = train_model(md_h, tx, ty, E_EPOCHS, LR, f"HyperEdgeGNN-D{D}")
    h_ew = md_h.edge_weight_matrix(N_OBJ)
    h_auc, h_spearman_r, h_avg_precision = edge_recovery_metrics(h_ew, coef, N_OBJ, strong_mask)
    print(f"  HyperEdgeGNN D={D}: auc={h_auc:.4f}  spearman_r={h_spearman_r:.4f}  avg_precision={h_avg_precision:.4f}")
    results[f"hyper_D{D}"] = {"auc": h_auc, "spearman_r": h_spearman_r, "avg_precision": h_avg_precision, "edge_weights": h_ew.tolist()}

print(f"\n  OOD: {OOD_TRAIN_N_OBJ} -> {OOD_TEST_N_OBJ} objects")
binary_ood, coef_ood = make_pairwise_graph(99)
ood_strong = binary_ood * (coef_ood >= 0.35)
tx_ood, ty_ood = fpairs_pair(binary_ood, coef_ood, 300, 10, n_obj=OOD_TRAIN_N_OBJ)
tx_ood_eval, ty_ood_eval = fpairs_pair(binary_ood, coef_ood, 100, 500, n_obj=OOD_TEST_N_OBJ)

md_h_ood = HyperEdgeGNN(OOD_TRAIN_N_OBJ, 8)
md_h_ood = train_model(md_h_ood, tx_ood, ty_ood, E_EPOCHS, LR, "HyperEdgeGNN-OOD-train")

rng_np = np.random.Generator(np.random.PCG64(999))
new_emb = _init_weights((OOD_TEST_N_OBJ, OOD_TEST_N_OBJ, 8), rng_np, 0.05)
new_emb[:OOD_TRAIN_N_OBJ, :OOD_TRAIN_N_OBJ] = md_h_ood.edge_emb
for i in range(OOD_TEST_N_OBJ): new_emb[i, i] = 0.0
md_h_ood.edge_emb = new_emb; md_h_ood.n_obj = OOD_TEST_N_OBJ

h_ew_ood = md_h_ood.edge_weight_matrix(OOD_TEST_N_OBJ)
h_auc_ood, h_spearman_r_ood, h_avg_precision_ood = edge_recovery_metrics(
    h_ew_ood, coef_ood, OOD_TEST_N_OBJ, ood_strong)
print(f"  OOD edge recovery: auc={h_auc_ood:.4f}  spearman_r={h_spearman_r_ood:.4f}  avg_precision={h_avg_precision_ood:.4f}")
results["hyper_D8_ood"] = {"auc": h_auc_ood, "spearman_r": h_spearman_r_ood, "avg_precision": h_avg_precision_ood}

print()
print("=" * 55)
print("  HYPERNET RESULT")
for k, v in results.items():
    if k == "baseline":
        print(f"  {k:>20s}: auc={v['auc']:.4f}  spearman_r={v['spearman_r']:.4f}  avg_prec={v['avg_precision']:.4f}")
    else:
        print(f"  {k:>20s}: auc={v['auc']:.4f}  spearman_r={v['spearman_r']:.4f}  avg_prec={v['avg_precision']:.4f}  "
              f"delta_r={v['spearman_r']-bl_spearman_r:+.4f}")
print("=" * 55)

results["true_coef"] = coef.tolist(); results["baseline_weights"] = bl_ew.tolist()
results["true_coef_ood"] = coef_ood.tolist()
od = plos_dir / "results" / "hypernet"; od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
