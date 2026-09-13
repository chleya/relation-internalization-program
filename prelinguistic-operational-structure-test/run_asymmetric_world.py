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

TYPES = ["heater", "cooler", "conductor"] * 2

PAIR_COEF = {
    ("heater", "cooler"): 0.55,
    ("heater", "conductor"): 0.50,
    ("heater", "heater"): 0.05,
    ("cooler", "heater"): 0.55,
    ("cooler", "conductor"): 0.30,
    ("cooler", "cooler"): 0.05,
    ("conductor", "heater"): 0.08,
    ("conductor", "cooler"): 0.08,
    ("conductor", "conductor"): 0.15,
}


def make_type_graph(seed):
    rng = random.Random(seed)
    types = TYPES[:]

    true_binary = np.zeros((N_OBJ, N_OBJ), dtype=np.int32)
    true_coef = np.zeros((N_OBJ, N_OBJ), dtype=np.float32)

    for src in range(N_OBJ):
        candidates = [t for t in range(N_OBJ) if t != src]
        rng.shuffle(candidates)
        n_out = rng.randint(1, min(3, N_OBJ - 1))
        for tgt in candidates[:n_out]:
            true_binary[src, tgt] = 1
            true_coef[src, tgt] = PAIR_COEF.get((types[src], types[tgt]), 0.1)

    return types, true_binary, true_coef


def step_asym(temps, binary, coef):
    nt = temps.copy()
    for s in range(N_OBJ):
        for t in range(N_OBJ):
            if binary[s, t]:
                nt[t] += coef[s, t] * (temps[s] - temps[t])
    return nt - SDECAY * (nt - AMB)


def gen_traj_asym(binary, coef, steps=500, seed=0):
    rng = random.Random(seed)
    temps = np.array([AMB + 0.3 * (rng.random() - 0.5) for _ in range(N_OBJ)], dtype=np.float32)
    fs = []
    for _ in range(steps):
        ns = np.array([NSTD * rng.gauss(0, 1) for _ in range(N_OBJ)], dtype=np.float32)
        fs.append(np.stack([temps, temps + ns], axis=1).astype(np.float32))
        temps = step_asym(temps, binary, coef)
    return np.array(fs, dtype=np.float32)


def fpairs_asym(binary, coef, steps, seed):
    t = gen_traj_asym(binary, coef, steps, seed); xs, ys = [], []
    for i in range(steps - 1): xs.append(t[i]); ys.append(t[i + 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class AsymGNN:
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
        combined = np.concatenate([sh, ms], axis=1)
        return combined @ self.w_out + self.b_out

    def param_arrays(self):
        return [self.w_self, self.b_self, self.w_msg, self.w_out, self.b_out, self.edge_logits]

    def set_all(self, arrs):
        self.w_self, self.b_self, self.w_msg, self.w_out, self.b_out, self.edge_logits = arrs


def _grads(md, feats, targ, lv):
    eps = 1e-4
    all_a = md.param_arrays(); gs = [np.zeros_like(a) for a in all_a]
    rng = random.Random(int(lv * 1e7) + 13007)
    for pi, p in enumerate(all_a):
        flat = p.ravel(); g = gs[pi].ravel()
        idxs = list(range(len(flat))); rng.shuffle(idxs)
        for idx in idxs[:min(NPERT, len(idxs))]:
            old = flat[idx]; flat[idx] = old + eps; md.set_all(all_a)
            pr = md.forward(feats); l2 = float(np.mean((pr - targ) ** 2).item())
            flat[idx] = old; g[idx] = (l2 - lv) / eps
        gs[pi] = g.reshape(p.shape)
    md.set_all(all_a); return gs


print("=" * 55)
print("  ASYMMETRIC CAUSAL WORLD")
print("=" * 55)

types, truth_bin, truth_coef = make_type_graph(42)
n_true = int(truth_bin.sum())

print(f"  Object types: {types}")
print(f"  True edges ({n_true}):")
print(f"    {'src->tgt':>10s} {'type_pair':>18s} {'coef':>6s}")
strong_edges = []
for s in range(N_OBJ):
    for t in range(N_OBJ):
        if truth_bin[s, t]:
            tp = f"{types[s]}->{types[t]}"
            print(f"    {s}->{t:>8}  {tp:>18s}  {truth_coef[s,t]:.4f}")
            if truth_coef[s, t] >= 0.3:
                strong_edges.append((s, t))

print(f"\n  Strong edges (coef>=0.3): {len(strong_edges)}")

tx, ty = fpairs_asym(truth_bin, truth_coef, 400, 10)
print(f"\n  Training... {tx.shape[0]} frames")

md = AsymGNN(); lr = 0.008; rt = random.Random(777)

for ep in range(100):
    el = 0.0; nb = 0
    accum = [np.zeros_like(a) for a in md.param_arrays()]
    idxs = list(range(tx.shape[0])); rt.shuffle(idxs)
    for idx in idxs[:min(32, tx.shape[0])]:
        pr = md.forward(tx[idx]); l = float(np.mean((pr - ty[idx]) ** 2).item())
        if math.isnan(l) or math.isinf(l): continue
        el += l; nb += 1
        gs = _grads(md, tx[idx], ty[idx], l)
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
            s = "S" if truth_coef[i, j] >= 0.3 else ("*" if truth_bin[i, j] else " ")
            print(f"  {ew[i,j]:.4f}{s}", end="")
    print()

flat_ew = []; flat_strong = []
for i in range(N_OBJ):
    for j in range(N_OBJ):
        if i != j:
            flat_ew.append(float(ew[i, j]))
            flat_strong.append(1 if (i, j) in strong_edges else 0)

flat_ew = np.array(flat_ew, dtype=np.float32)
flat_strong = np.array(flat_strong, dtype=np.int32)
n_strong = int(flat_strong.sum())

order = np.argsort(-flat_ew)
sorted_strong = flat_strong[order]
precs = [int(sorted_strong[:k].sum()) / k for k in range(1, n_strong * 2 + 1)]
auc = float(np.mean(precs))

top_n = min(n_strong * 2, len(sorted_strong))
tp_at_top = int(sorted_strong[:top_n].sum())
prec_top = tp_at_top / top_n
rec_top = tp_at_top / n_strong if n_strong > 0 else 0
f1_top = 2 * prec_top * rec_top / (prec_top + rec_top) if (prec_top + rec_top) > 0 else 0

corr = float(np.corrcoef(flat_ew, truth_coef[np.eye(N_OBJ)==0])[0, 1])

print(f"\n  {'='*45}")
print(f"  ASYMMETRIC WORLD RESULT")
print(f"  asym_n_strong         = {n_strong}")
print(f"  asym_edge_auc         = {auc:.4f}  (was 0.54 uniform)")
print(f"  asym_prec@topK        = {prec_top:.4f}")
print(f"  asym_recall@topK      = {rec_top:.4f}")
print(f"  asym_f1@topK          = {f1_top:.4f}")
print(f"  asym_coef_corr        = {corr:.4f}  (corr with true coef)")
print(f"  (Random baseline AUC = 0.50)")
print(f"\n  INTERPRETATION:")
if auc > 0.65 and corr > 0.3:
    print(f"    CONTENT-LEVEL emergence: model learns WHICH edges are strong.")
elif auc > 0.55:
    print(f"    WEAK content emergence: some differentiation, not fully structured.")
else:
    print(f"    NO content emergence: edges still uniform despite asymmetry.")

m = {"asym_n_strong": n_strong, "asym_edge_auc": auc, "asym_prec_topk": prec_top,
     "asym_recall_topk": rec_top, "asym_f1_topk": f1_top, "asym_coef_corr": corr,
     "edge_weights": ew.tolist(), "true_coef": truth_coef.tolist(),
     "true_binary": truth_bin.tolist(), "strong_edges": [(int(s), int(t)) for s, t in strong_edges]}
od = plos_dir / "results" / "asymmetric_world"
od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
print(f"  {'='*45}")
