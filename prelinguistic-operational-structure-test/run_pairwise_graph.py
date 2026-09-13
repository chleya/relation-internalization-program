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
    rng = random.Random(seed)
    binary = np.zeros((N_OBJ, N_OBJ), dtype=np.int32)
    coef = np.zeros((N_OBJ, N_OBJ), dtype=np.float32)

    for src in range(N_OBJ):
        candidates = [t for t in range(N_OBJ) if t != src]
        rng.shuffle(candidates)
        n_out = rng.randint(1, min(3, N_OBJ - 1))
        for tgt in candidates[:n_out]:
            binary[src, tgt] = 1
            coef[src, tgt] = 0.08 + 0.50 * rng.random()

    return binary, coef


def step_pairwise(temps, binary, coef):
    nt = temps.copy()
    for s in range(N_OBJ):
        for t in range(N_OBJ):
            if binary[s, t]:
                nt[t] += coef[s, t] * (temps[s] - temps[t])
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


class PairwiseGNN:
    def __init__(self):
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
        no = feats.shape[0]; sh = self._r(feats @ self.w_self + self.b_self)
        ew = 1.0 / (1.0 + np.exp(-self.edge_logits))
        ms = np.zeros((no, HID), dtype=np.float32)
        for i in range(no):
            for j in range(no):
                if i == j: continue
                ms[i] += ew[i, j] * self._r(feats[j] @ self.w_msg)
        return np.concatenate([sh, ms], axis=1) @ self.w_out + self.b_out

    def param_arrays(self):
        return [self.w_self, self.b_self, self.w_msg, self.w_out, self.b_out, self.edge_logits]

    def set_all(self, arrs):
        self.w_self, self.b_self, self.w_msg, self.w_out, self.b_out, self.edge_logits = arrs


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
print("  PAIRWISE RANDOM COEFFICIENTS")
print("=" * 55)

binary, coef = make_pairwise_graph(42)
n_true = int(binary.sum())

print(f"  True edges ({n_true}) — per-pair random coef:")
n_strong = 0
for s in range(N_OBJ):
    for t in range(N_OBJ):
        if binary[s, t]:
            tag = "STRONG" if coef[s, t] >= 0.35 else "weak"
            if coef[s, t] >= 0.35: n_strong += 1
            print(f"    {s}->{t}: coef={coef[s,t]:.4f} ({tag})")

print(f"\n  Strong edges (>0.35): {n_strong} | Weak: {n_true - n_strong}")

tx, ty = fpairs_pair(binary, coef, 400, 10)
print(f"\n  Training... {tx.shape[0]} frames")

md = PairwiseGNN(); lr = 0.008; rt = random.Random(777)

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
            s = "S" if (binary[i,j] and coef[i,j] >= 0.35) else ("*" if binary[i,j] else " ")
            print(f"  {ew[i,j]:.4f}{s}", end="")
    print()

flat_ew = []; flat_strong_bin = []
for i in range(N_OBJ):
    for j in range(N_OBJ):
        if i != j:
            flat_ew.append(float(ew[i, j]))
            flat_strong_bin.append(1 if (binary[i,j] and coef[i,j] >= 0.35) else 0)

flat_ew = np.array(flat_ew, dtype=np.float32)
flat_sb = np.array(flat_strong_bin, dtype=np.int32)
order = np.argsort(-flat_ew)
sorted_sb = flat_sb[order]
ks = list(range(1, n_strong * 2 + 1))
auc = float(np.mean([int(sorted_sb[:k].sum())/k for k in ks]))

topn = min(n_strong * 2, len(sorted_sb))
tp = int(sorted_sb[:topn].sum()); prec = tp/topn; rec = tp/n_strong
f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0

non_diag = ~np.eye(N_OBJ, dtype=bool)
corr = float(np.corrcoef(flat_ew, coef[non_diag])[0,1])

print(f"\n  {'='*45}")
print(f"  PAIRWISE RESULT")
print(f"  pw_edge_auc           = {auc:.4f}")
print(f"  pw_prec@top           = {prec:.4f}")
print(f"  pw_recall@top         = {rec:.4f}")
print(f"  pw_f1@top             = {f1:.4f}")
print(f"  pw_coef_corr          = {corr:.4f}")
if auc > 0.65 and corr > 0.3:
    print(f"  CONTENT emergence: pair-specific edge weights learned!")
elif auc > 0.55:
    print(f"  WEAK content emergence.")
else:
    print(f"  NO content emergence.")

m = {"pw_edge_auc": auc, "pw_prec": prec, "pw_recall": rec, "pw_f1": f1,
     "pw_coef_corr": corr, "edge_weights": ew.tolist(), "true_coef": coef.tolist(),
     "true_binary": binary.tolist()}
od = plos_dir / "results" / "pairwise"
od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
print(f"  {'='*45}")
