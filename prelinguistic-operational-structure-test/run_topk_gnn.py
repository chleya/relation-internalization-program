from __future__ import annotations

import json, math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent

N_OBJ = 6; N_FEAT = 2
TCOEF, SDECAY, NSTD, AMB = 0.35, 0.02, 0.01, 0.5
HID = 16; NPERT = 60
TOP_K = 14


def make_causal_graph(seed, n=N_OBJ):
    rng = random.Random(seed)
    e = [set() for _ in range(n)]
    for s in range(n):
        c = [t for t in range(n) if t != s]; rng.shuffle(c)
        for t in c[:rng.randint(1, min(3, n - 1))]: e[s].add(t)
    return e


def step_c(temps, edges):
    nt = temps.copy()
    for s in range(N_OBJ):
        for t in edges[s]: nt[t] += TCOEF * (temps[s] - temps[t])
    return nt - SDECAY * (nt - AMB)


def gen_traj(edges, steps=500, seed=0):
    rng = random.Random(seed)
    temps = np.array([AMB + 0.3 * (rng.random() - 0.5) for _ in range(N_OBJ)], dtype=np.float32)
    fs = []
    for _ in range(steps):
        ns = np.array([NSTD * rng.gauss(0, 1) for _ in range(N_OBJ)], dtype=np.float32)
        fs.append(np.stack([temps, temps + ns], axis=1).astype(np.float32))
        temps = step_c(temps, edges)
    return np.array(fs, dtype=np.float32)


def fpairs(edges, steps, seed):
    t = gen_traj(edges, steps, seed); xs, ys = [], []
    for i in range(steps - 1): xs.append(t[i]); ys.append(t[i + 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class TopKGNN:
    def __init__(self, top_k=TOP_K):
        self.top_k = top_k
        rng_np = np.random.Generator(np.random.PCG64(42))
        raw = np.random.randn(N_OBJ, N_OBJ).astype(np.float32) * 0.01
        raw[np.eye(N_OBJ, dtype=bool)] = -10.0
        self.edge_logits = raw.copy()

        self.w_self = _init_weights((N_FEAT, HID), rng_np)
        self.b_self = np.zeros(HID, dtype=np.float32)
        self.w_msg = _init_weights((N_FEAT, HID), rng_np)
        self.w_out = _init_weights((HID * 2, N_FEAT), rng_np)
        self.b_out = np.zeros(N_FEAT, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)

    def forward(self, feats):
        no = feats.shape[0]
        sh = self._r(feats @ self.w_self + self.b_self)

        ew_soft = 1.0 / (1.0 + np.exp(-self.edge_logits))
        n_pairs = N_OBJ * (N_OBJ - 1)
        flat = ew_soft.ravel()
        flat[np.eye(N_OBJ, dtype=bool).ravel()] = -1.0

        cutoff = -np.partition(-flat, self.top_k)[self.top_k - 1]
        mask_flat = (flat >= cutoff - 1e-8)
        ew_hard = flat.copy()
        ew_hard[~mask_flat] = 0.0
        ew_hard = ew_hard.reshape(N_OBJ, N_OBJ)

        ms = np.zeros((no, HID), dtype=np.float32)
        for i in range(no):
            for j in range(no):
                if i == j: continue
                ms[i] += ew_hard[i, j] * self._r(feats[j] @ self.w_msg)

        combined = np.concatenate([sh, ms], axis=1)
        return combined @ self.w_out + self.b_out

    def params(self):
        return [self.w_self, self.b_self, self.w_msg, self.w_out, self.b_out]

    def param_arrays(self):
        return [self.w_self, self.b_self, self.w_msg, self.w_out, self.b_out,
                self.edge_logits]

    def set_params(self, fl):
        (self.w_self, self.b_self, self.w_msg, self.w_out, self.b_out) = fl

    def set_all(self, arrs):
        self.w_self, self.b_self, self.w_msg, self.w_out, self.b_out, self.edge_logits = arrs


def _grads(md, feats, targ, lv):
    eps = 1e-4
    all_a = md.param_arrays()
    gs = [np.zeros_like(a) for a in all_a]
    rng = random.Random(int(lv * 1e7) + 13007)

    for pi, p in enumerate(all_a):
        flat = p.ravel(); g = gs[pi].ravel()
        idxs = list(range(len(flat))); rng.shuffle(idxs)
        for idx in idxs[:min(NPERT, len(idxs))]:
            old = flat[idx]; flat[idx] = old + eps
            md.set_all(all_a)
            pr = md.forward(feats)
            l2 = float(np.mean((pr - targ) ** 2).item())
            flat[idx] = old; g[idx] = (l2 - lv) / eps
        gs[pi] = g.reshape(p.shape)
    md.set_all(all_a)
    return gs


print("=" * 55)
print(f"  Path 1: Top-K GNN (K={TOP_K})")
print("=" * 55)

edges = make_causal_graph(42)
truth_mat = np.zeros((N_OBJ, N_OBJ), dtype=np.int32)
for s in range(N_OBJ):
    for t in edges[s]: truth_mat[s, t] = 1
n_true = int(truth_mat.sum())
print(f"  True edges: {n_true} | Top-K: {TOP_K}")

tx, ty = fpairs(edges, 400, 10)
print(f"  Training... {tx.shape[0]} frames")

md = TopKGNN(top_k=TOP_K)
lr = 0.008; rt = random.Random(777)

for ep in range(100):
    el = 0.0; nb = 0
    accum = [np.zeros_like(a) for a in md.param_arrays()]
    idxs = list(range(tx.shape[0])); rt.shuffle(idxs)
    for idx in idxs[:min(32, tx.shape[0])]:
        pr = md.forward(tx[idx])
        l = float(np.mean((pr - ty[idx]) ** 2).item())
        if math.isnan(l) or math.isinf(l): continue
        el += l; nb += 1
        gs = _grads(md, tx[idx], ty[idx], l)
        for gi in range(len(gs)): accum[gi] += gs[gi]
    if nb > 0:
        el /= nb
        for gi in range(len(accum)): accum[gi] /= nb
    md.set_all([a - lr * g for a, g in zip(md.param_arrays(), accum)])
    if ep % 25 == 0: print(f"    ep {ep:3d} loss={el:.6f}")

print("\n  Final edge weights (soft):")
ew_soft = 1.0 / (1.0 + np.exp(-md.edge_logits))
print(f"    {'':>5s}", end="")
for j in range(N_OBJ): print(f"  {'->'+str(j):>8s}", end="")
print()
for i in range(N_OBJ):
    print(f"    {str(i)+'->':>5s}", end="")
    for j in range(N_OBJ):
        if i == j:
            print(f"  {'-':>8s}", end="")
        else:
            marker = "*" if truth_mat[i, j] else " "
            print(f"  {ew_soft[i,j]:.4f}{marker}", end="")
    print()

flat_ew = []
flat_truth = []
for i in range(N_OBJ):
    for j in range(N_OBJ):
        if i != j:
            flat_ew.append(float(ew_soft[i, j]))
            flat_truth.append(int(truth_mat[i, j]))
flat_ew = np.array(flat_ew, dtype=np.float32)
flat_truth = np.array(flat_truth, dtype=np.int32)

order = np.argsort(-flat_ew)
sorted_truth = flat_truth[order]
precs = [int(sorted_truth[:k].sum()) / k for k in range(1, n_true * 3 + 1)]
auc = float(np.mean(precs))

topk_mask = flat_ew >= np.partition(flat_ew, -TOP_K)[-TOP_K]
tp = int(flat_truth[topk_mask].sum())
precision = tp / TOP_K
recall = tp / n_true if n_true > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n  {'='*45}")
print(f"  TOP-K GNN RESULT")
print(f"  tk_edge_auc           = {auc:.4f}")
print(f"  tk_precision@K        = {precision:.4f}  (top {TOP_K} edges)")
print(f"  tk_recall@K           = {recall:.4f}")
print(f"  tk_f1@K               = {f1:.4f}")
print(f"  (Uniform baseline AUC = 0.54)")

m = {"tk_edge_auc": auc, "tk_precision": precision, "tk_recall": recall,
     "tk_f1": f1, "top_k": TOP_K, "n_true": n_true,
     "edge_weights": ew_soft.tolist(), "truth": truth_mat.tolist()}
od = plos_dir / "results" / "topk_gnn"
od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
print(f"  {'='*45}")
