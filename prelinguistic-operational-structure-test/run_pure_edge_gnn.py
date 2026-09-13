from __future__ import annotations

import json, math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent

N_OBJ = 6; N_FEAT = 2
SDECAY, NSTD, AMB = 0.02, 0.01, 0.5
HID = 16; NPERT = 60


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


class PureEdgeGNN:
    def __init__(self):
        rng_np = np.random.Generator(np.random.PCG64(42))
        raw = np.random.randn(N_OBJ, N_OBJ).astype(np.float32) * 0.05
        raw[np.eye(N_OBJ, dtype=bool)] = -10.0
        self.edge_logits = raw.copy()

        self.obj_emb = _init_weights((N_OBJ, HID), rng_np)

        self.w_self = _init_weights((N_FEAT, HID), rng_np)
        self.b_self = np.zeros(HID, dtype=np.float32)
        self.w_out = _init_weights((HID * 2, N_FEAT), rng_np)
        self.b_out = np.zeros(N_FEAT, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)

    def forward(self, feats):
        no = feats.shape[0]; sh = self._r(feats @ self.w_self + self.b_self)
        ew = 1.0 / (1.0 + np.exp(-self.edge_logits))
        emb = self.obj_emb[:no]
        ms = np.zeros((no, HID), dtype=np.float32)
        for i in range(no):
            for j in range(no):
                if i == j: continue
                ms[i] += ew[i, j] * emb[j]
        combined = np.concatenate([sh, ms], axis=1)
        return combined @ self.w_out + self.b_out

    def param_arrays(self):
        return [self.obj_emb, self.w_self, self.b_self, self.w_out, self.b_out,
                self.edge_logits]

    def set_all(self, arrs):
        self.obj_emb, self.w_self, self.b_self, self.w_out, self.b_out, self.edge_logits = arrs


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


print("=" * 55)
print("  PURE EDGE GNN — w_msg removed, object embeddings only")
print("=" * 55)

binary, coef = make_pairwise_graph(42)
n_true = int(binary.sum())
n_strong = int((coef >= 0.35).sum())

print(f"  True edges: {n_true}  |  Strong (>0.35): {n_strong}")
for s in range(N_OBJ):
    for t in range(N_OBJ):
        if binary[s, t]:
            tag = "S" if coef[s,t] >= 0.35 else "w"
            print(f"    {s}->{t}: {coef[s,t]:.4f} ({tag})")

tx, ty = fpairs_pair(binary, coef, 400, 10)
print(f"\n  Training... {tx.shape[0]} frames")

md = PureEdgeGNN(); lr = 0.008; rt = random.Random(777)

for ep in range(100):
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
    if ep % 25 == 0: print(f"    ep {ep:3d} loss={el:.6f}")

print("\n  Learned edge weights:")
ew = 1.0 / (1.0 + np.exp(-md.edge_logits))
print(f"    {'':>5s}", end="")
for j in range(N_OBJ): print(f"  {'->'+str(j):>8s}", end="")
print()
for i in range(N_OBJ):
    print(f"    {str(i)+'->':>5s}", end="")
    for j in range(N_OBJ):
        if i == j:
            print(f"  {'-':>8s}", end="")
        else:
            s = "S" if (binary[i,j] and coef[i,j]>=0.35) else ("*" if binary[i,j] else " ")
            print(f"  {ew[i,j]:.4f}{s}", end="")
    print()

non_diag = ~np.eye(N_OBJ, dtype=bool)
flat_ew = ew[non_diag]
flat_coef = coef[non_diag]
flat_strong = ((binary * (coef >= 0.35))[non_diag]).astype(np.int32)

corr = float(np.corrcoef(flat_ew, flat_coef)[0,1])

order = np.argsort(-flat_ew)
sorted_s = flat_strong[order]
ks = list(range(1, n_strong * 2 + 1))
auc = float(np.mean([int(sorted_s[:k].sum())/k for k in ks]))

topn = min(n_strong * 2, len(sorted_s))
tp = int(sorted_s[:topn].sum()); prec = tp/topn; rec = tp/n_strong
f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0

print(f"\n  {'='*45}")
print(f"  PURE EDGE GNN RESULT")
print(f"  pe_edge_auc           = {auc:.4f}")
print(f"  pe_coef_corr          = {corr:.4f}")
print(f"  pe_prec@top           = {prec:.4f}")
print(f"  pe_recall@top         = {rec:.4f}")
print(f"  pe_f1@top             = {f1:.4f}")
if auc > 0.65 and corr > 0.3:
    print(f"  CONTENT emergence achieved!")
elif auc > 0.55 or corr > 0.2:
    print(f"  PARTIAL content emergence.")
else:
    print(f"  NO content emergence.")

m = {"pe_edge_auc": auc, "pe_coef_corr": corr, "pe_prec": prec,
     "pe_recall": rec, "pe_f1": f1, "edge_weights": ew.tolist(),
     "true_coef": coef.tolist()}
od = plos_dir / "results" / "pure_edge_gnn"
od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
print(f"  {'='*45}")
