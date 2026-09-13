from __future__ import annotations

import json, math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent

N_OBJ = 6; N_FEAT = 2
SDECAY, NSTD, AMB = 0.02, 0.01, 0.5
HID = 20; NPERT = 60


def make_pairwise_graph(seed):
    rng = random.Random(seed); binary = np.zeros((N_OBJ, N_OBJ), dtype=np.int32)
    coef = np.zeros((N_OBJ, N_OBJ), dtype=np.float32)
    for src in range(N_OBJ):
        candidates = [t for t in range(N_OBJ) if t != src]; rng.shuffle(candidates)
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


class FullGNN:
    def __init__(self, ablated_edge=None):
        rng_np = np.random.Generator(np.random.PCG64(42))
        self.w_self = _init_weights((N_FEAT, HID), rng_np)
        self.b_self = np.zeros(HID, dtype=np.float32)
        self.w_msg = _init_weights((N_FEAT, HID), rng_np)
        self.w_out = _init_weights((HID * 2, N_FEAT), rng_np)
        self.b_out = np.zeros(N_FEAT, dtype=np.float32)
        self.ablated_edge = ablated_edge

    def _r(self, x): return np.maximum(0, x)

    def forward(self, feats):
        no = feats.shape[0]; sh = self._r(feats @ self.w_self + self.b_self)
        ms = np.zeros((no, HID), dtype=np.float32)
        for i in range(no):
            for j in range(no):
                if i == j: continue
                if self.ablated_edge == (j, i): continue
                ms[i] += self._r(feats[j] @ self.w_msg)
        return np.concatenate([sh, ms], axis=1) @ self.w_out + self.b_out

    def param_arrays(self):
        return [self.w_self, self.b_self, self.w_msg, self.w_out, self.b_out]

    def set_all(self, arrs):
        self.w_self, self.b_self, self.w_msg, self.w_out, self.b_out = arrs


def _g(md, feats, targ, lv):
    eps = 1e-4; all_a = md.param_arrays(); gs = [np.zeros_like(a) for a in all_a]
    rng = random.Random(int(lv * 1e7) + 13007)
    for pi, p in enumerate(all_a):
        fl = p.ravel(); g = gs[pi].ravel()
        idxs = list(range(len(fl))); rng.shuffle(idxs)
        for idx in idxs[:min(NPERT, len(idxs))]:
            old = fl[idx]; fl[idx] = old + eps; md.set_all(all_a)
            l2 = float(np.mean((md.forward(feats) - targ) ** 2).item())
            fl[idx] = old; g[idx] = (l2 - lv) / eps
        gs[pi] = g.reshape(p.shape)
    md.set_all(all_a); return gs


def train_full(tx, ty, ablated_edge=None):
    md = FullGNN(ablated_edge=ablated_edge); lr = 0.008; rt = random.Random(777)
    for ep in range(80):
        el = 0.0; nb = 0; accum = [np.zeros_like(a) for a in md.param_arrays()]
        idxs = list(range(tx.shape[0])); rt.shuffle(idxs)
        for idx in idxs[:min(32, tx.shape[0])]:
            l = float(np.mean((md.forward(tx[idx]) - ty[idx]) ** 2).item())
            if math.isnan(l): continue; el += l; nb += 1
            for gi, gg in enumerate(_g(md, tx[idx], ty[idx], l)): accum[gi] += gg
        if nb > 0:
            el /= nb
            for gi in range(len(accum)): accum[gi] /= nb
        md.set_all([a - lr * g for a, g in zip(md.param_arrays(), accum)])
    return md


def eval_loss(md, vx, vy):
    t = 0.0; n = 0
    for i in range(vx.shape[0]):
        l = float(np.mean((md.forward(vx[i]) - vy[i]) ** 2).item())
        if not math.isnan(l): t += l; n += 1
    return t / max(n, 1)


print("=" * 55)
print("  ABLATION IMPORTANCE RANKING")
print("=" * 55)

binary, coef = make_pairwise_graph(42)
n_true = int(binary.sum())
non_diag_pairs = [(i, j) for i in range(N_OBJ) for j in range(N_OBJ) if i != j]
edge_pairs = [(i, j) for i in range(N_OBJ) for j in range(N_OBJ) if binary[i, j]]

print(f"  True edges ({n_true}):")
for s, t in edge_pairs:
    print(f"    {s}->{t}: coef={coef[s,t]:.4f}")

tx, ty = fpairs_pair(binary, coef, 400, 10)
vx, vy = fpairs_pair(binary, coef, 150, 500)

print(f"\n  Training full model (all edges active)...")
md_full = train_full(tx, ty, ablated_edge=None)
loss_full = eval_loss(md_full, vx, vy)
print(f"    Full model loss: {loss_full:.6f}")

print(f"\n  Ablating each edge...")
importance = {}
for si, (s, t) in enumerate(edge_pairs):
    md_ab = train_full(tx, ty, ablated_edge=(s, t))
    loss_ab = eval_loss(md_ab, vx, vy)
    imp = loss_ab - loss_full
    importance[(s, t)] = imp
    print(f"    remove {s}->{t}: loss={loss_ab:.6f}  importance={imp:.6f}")

print(f"\n  Edge importance ranking:")
sorted_edges = sorted(edge_pairs, key=lambda e: -importance[e])
for rank, (s, t) in enumerate(sorted_edges):
    tag = "HIGH" if coef[s,t] >= 0.35 else "LOW"
    print(f"    {rank+1}. {s}->{t}: imp={importance[(s,t)]:.6f}  coef={coef[s,t]:.4f} ({tag})")

high_edges = [e for e in edge_pairs if coef[e[0],e[1]] >= 0.35]
n_high = len(high_edges)
precs = []
for k in range(1, n_high + 1):
    top_k = set(sorted_edges[:k])
    tp = len(top_k & set(high_edges))
    precs.append(tp / k)
avg_prec = float(np.mean(precs))

spearman = 0
for rank, e in enumerate(sorted_edges):
    true_rank = sum(1 for e2 in edge_pairs if coef[e2[0],e2[1]] > coef[e[0],e[1]])
    spearman += (rank - true_rank) ** 2
n_e = len(edge_pairs)
spearman = 1 - 6 * spearman / (n_e * (n_e**2 - 1)) if n_e > 1 else 0

print(f"\n  {'='*45}")
print(f"  ABLATION IMPORTANCE RESULT")
print(f"  ab_n_edges            = {n_e}")
print(f"  ab_n_high_coef        = {n_high}")
print(f"  ab_avg_precision@K    = {avg_prec:.4f}  (1.0 = perfect)")
print(f"  ab_spearman_r         = {spearman:.4f}  (1.0 = perfect)")
if avg_prec > 0.7 and spearman > 0.5:
    print(f"  CONTENT emergence: ablation correctly identifies strong edges!")
elif avg_prec > 0.5:
    print(f"  PARTIAL content emergence.")
else:
    print(f"  NO content emergence from ablation.")

m = {"ab_n_edges": n_e, "ab_n_high": n_high,
     "ab_avg_precision": avg_prec, "ab_spearman_r": spearman,
     "importance": {f"{s},{t}": float(importance[(s,t)]) for s,t in edge_pairs},
     "true_coef": {f"{s},{t}": float(coef[s,t]) for s,t in edge_pairs}}
od = plos_dir / "results" / "ablation_importance"
od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
print(f"  {'='*45}")
