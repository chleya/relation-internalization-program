from __future__ import annotations

import json, math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent

N_OBJ = 6
N_FEAT = 2
TCOEF, SDECAY, NSTD, AMB = 0.35, 0.02, 0.01, 0.5
HID = 16
NPERT = 60
L1_LAMBDA = 0.005


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
    t = gen_traj(edges, steps, seed)
    xs, ys = [], []
    for i in range(steps - 1): xs.append(t[i]); ys.append(t[i + 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class SparseGNN:
    def __init__(self):
        rng_np = np.random.Generator(np.random.PCG64(42))
        self.edge_logits = np.random.randn(N_OBJ, N_OBJ).astype(np.float32) * 0.01
        self.edge_logits[np.eye(N_OBJ, dtype=bool)] = -10.0

        self.w_self = _init_weights((N_FEAT, HID), rng_np)
        self.b_self = np.zeros(HID, dtype=np.float32)
        self.w_msg = _init_weights((N_FEAT, HID), rng_np)
        self.w_out = _init_weights((HID * 2, N_FEAT), rng_np)
        self.b_out = np.zeros(N_FEAT, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)

    def edge_weights(self):
        return 1.0 / (1.0 + np.exp(-self.edge_logits))

    def forward(self, feats):
        no = feats.shape[0]
        sh = self._r(feats @ self.w_self + self.b_self)
        ew = self.edge_weights()

        ms = np.zeros((no, HID), dtype=np.float32)
        for i in range(no):
            for j in range(no):
                if i == j: continue
                ms[i] += ew[i, j] * self._r(feats[j] @ self.w_msg)

        combined = np.concatenate([sh, ms], axis=1)
        return combined @ self.w_out + self.b_out

    def params(self):
        return [self.w_self, self.b_self, self.w_msg, self.w_out, self.b_out]

    def param_arrays(self):
        return self.params() + [self.edge_logits]

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
        flat = p.ravel()
        g = gs[pi].ravel()
        idxs = list(range(len(flat)))
        rng.shuffle(idxs)
        for idx in idxs[:min(NPERT, len(idxs))]:
            old = flat[idx]
            flat[idx] = old + eps
            md.set_all(all_a)
            pr = md.forward(feats)
            l2 = float(np.mean((pr - targ) ** 2).item())
            ew = md.edge_weights()
            l1 = L1_LAMBDA * float(ew.sum())
            l_new = l2 + l1
            flat[idx] = old
            g[idx] = (l_new - lv) / eps
        gs[pi] = g.reshape(p.shape)
    md.set_all(all_a)
    return gs


print("=" * 55)
print("  Path 1: Sparse GNN with L1 Regularization")
print(f"  L1 lambda = {L1_LAMBDA}")
print("=" * 55)

edges = make_causal_graph(42)
truth_mat = np.zeros((N_OBJ, N_OBJ), dtype=np.int32)
for s in range(N_OBJ):
    for t in edges[s]: truth_mat[s, t] = 1

n_true = int(truth_mat.sum())
print(f"  True edges: {n_true}")
for s in range(N_OBJ):
    ts = ", ".join(str(t) for t in sorted(edges[s]))
    if ts: print(f"    {s} -> {ts}")

tx, ty = fpairs(edges, 400, 10)
print(f"\n  Training... {tx.shape[0]} frames")

md = SparseGNN()
lr = 0.008
rt = random.Random(777)

for ep in range(100):
    el = 0.0; nb = 0
    accum = [np.zeros_like(a) for a in md.param_arrays()]
    idxs = list(range(tx.shape[0])); rt.shuffle(idxs)
    for idx in idxs[:min(32, tx.shape[0])]:
        pr = md.forward(tx[idx])
        l_pred = float(np.mean((pr - ty[idx]) ** 2).item())
        ew = md.edge_weights()
        l1 = L1_LAMBDA * float(ew.sum())
        l = l_pred + l1
        if math.isnan(l) or math.isinf(l): continue
        el += l; nb += 1
        gs = _grads(md, tx[idx], ty[idx], l)
        for gi in range(len(gs)): accum[gi] += gs[gi]
    if nb > 0:
        el /= nb
        for gi in range(len(accum)): accum[gi] /= nb
    new_all = [a - lr * g for a, g in zip(md.param_arrays(), accum)]
    md.set_all(new_all)
    if ep % 25 == 0: print(f"    ep {ep:3d} loss={el:.6f}")

print("\n  Final edge weights:")
final_ew = md.edge_weights()
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
            print(f"  {final_ew[i,j]:.4f}{marker}", end="")
    print()

sparsity = float((final_ew < 0.1).sum()) / (N_OBJ * (N_OBJ - 1))
print(f"\n  Sparsity: {sparsity:.2f} (% edges < 0.1)")

flat_ew = []
flat_truth = []
for i in range(N_OBJ):
    for j in range(N_OBJ):
        if i != j:
            flat_ew.append(float(final_ew[i, j]))
            flat_truth.append(int(truth_mat[i, j]))

flat_ew = np.array(flat_ew, dtype=np.float32)
flat_truth = np.array(flat_truth, dtype=np.int32)
order = np.argsort(-flat_ew)
sorted_truth = flat_truth[order]
precs = []
for k in range(1, n_true * 3 + 1):
    precs.append(int(sorted_truth[:k].sum()) / k)
auc = float(np.mean(precs))

med = float(np.median(flat_ew))
pred_mat = (flat_ew >= med).astype(np.int32).reshape(N_OBJ, N_OBJ)
tp = int((pred_mat * truth_mat).sum())
fp = int((pred_mat * (1 - truth_mat)).sum())
fn = int(((1 - pred_mat) * truth_mat).sum())
prec = tp / (tp + fp) if (tp + fp) > 0 else 0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

print(f"\n  {'='*45}")
print(f"  SPARSE GNN RESULT")
print(f"  sg_edge_auc           = {auc:.4f}  (was 0.543 with uniform)")
print(f"  sg_precision          = {prec:.4f}")
print(f"  sg_recall             = {rec:.4f}")
print(f"  sg_f1                 = {f1:.4f}")
print(f"  sg_sparsity           = {sparsity:.2f}")

m = {"sg_edge_auc": auc, "sg_precision": prec, "sg_recall": rec,
     "sg_f1": f1, "sg_sparsity": sparsity, "edge_weights": final_ew.tolist(),
     "truth": truth_mat.tolist(), "l1_lambda": L1_LAMBDA}
od = plos_dir / "results" / "sparse_gnn"
od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
print(f"  {'='*45}")
