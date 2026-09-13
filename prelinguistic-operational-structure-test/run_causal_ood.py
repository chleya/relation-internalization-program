from __future__ import annotations

import json, math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent
N_OBJ, N_FEAT = 6, 2
TCOEF, SDECAY, NSTD, AMB = 0.35, 0.02, 0.01, 0.5
HID, NPERT = 16, 60


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


class GNN:
    def __init__(self):
        rng = np.random.Generator(np.random.PCG64(42))
        self.ws = _init_weights((N_FEAT, HID), rng); self.bs = np.zeros(HID, dtype=np.float32)
        self.we = _init_weights((N_FEAT * 2, 1), rng)
        self.wm = _init_weights((N_FEAT, HID), rng)
        self.wo = _init_weights((HID * 2, N_FEAT), rng); self.bo = np.zeros(N_FEAT, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)

    def fw(self, feats, ue=True):
        no = feats.shape[0]; sh = self._r(feats @ self.ws + self.bs)
        ms = np.zeros((no, HID), dtype=np.float32)
        if ue and no > 1:
            for i in range(no):
                for j in range(no):
                    if i == j: continue
                    p = np.concatenate([feats[i], feats[j]])
                    ew = 1.0 / (1.0 + math.exp(-float((p @ self.we).item())))
                    ms[i] += ew * self._r(feats[j] @ self.wm)
        return np.concatenate([sh, ms], axis=1) @ self.wo + self.bo

    def ps(self): return [self.ws, self.bs, self.we, self.wm, self.wo, self.bo]
    def sp(self, fl): (self.ws, self.bs, self.we, self.wm, self.wo, self.bo) = fl


def _g(md, feats, targ, lv):
    eps = 1e-4; ap = md.ps(); gs = [np.zeros_like(p) for p in ap]
    rng = random.Random(int(lv * 1e7) + 13007)
    for pi, p in enumerate(ap):
        fl = p.ravel(); g = gs[pi].ravel(); idxs = list(range(len(fl))); rng.shuffle(idxs)
        for idx in idxs[:min(NPERT, len(idxs))]:
            old = fl[idx]; fl[idx] = old + eps; md.sp(ap)
            l2 = float(np.mean((md.fw(feats, True) - targ) ** 2).item())
            fl[idx] = old; g[idx] = (l2 - lv) / eps
        gs[pi] = g.reshape(p.shape)
    md.sp(ap); return gs


def train_gnn(tx, ty, ne=80, lr=0.008):
    md = GNN(); rt = random.Random(777)
    for ep in range(ne):
        el = 0.0; nb = 0; accum = [np.zeros_like(p) for p in md.ps()]
        idxs = list(range(tx.shape[0])); rt.shuffle(idxs)
        for idx in idxs[:min(32, tx.shape[0])]:
            l = float(np.mean((md.fw(tx[idx], True) - ty[idx]) ** 2).item())
            if math.isnan(l) or math.isinf(l): continue
            el += l; nb += 1; gs = _g(md, tx[idx], ty[idx], l)
            for gi in range(len(gs)): accum[gi] += gs[gi]
        if nb > 0:
            el /= nb
            for gi in range(len(accum)): accum[gi] /= nb
        md.sp([p - lr * g for p, g in zip(md.ps(), accum)])
    return md


def ev(md, dx, dy, ue):
    t = 0.0; n = 0
    for i in range(dx.shape[0]):
        l = float(np.mean((md.fw(dx[i], ue) - dy[i]) ** 2).item())
        if math.isnan(l): continue
        t += l; n += 1
    return t / max(n, 1)


print("=" * 55)
print("  Experiment 3: OOD Generalization")
print("=" * 55)

e_train = make_causal_graph(42)
e_test_a = make_causal_graph(99)
e_test_b = make_causal_graph(77)

print("  Train graph:")
for s in range(N_OBJ):
    ts = ", ".join(str(t) for t in sorted(e_train[s]))
    if ts: print(f"    {s} -> {ts}")

print("\n  Test graph A:")
for s in range(N_OBJ):
    ts = ", ".join(str(t) for t in sorted(e_test_a[s]))
    if ts: print(f"    {s} -> {ts}")

print("\n  Test graph B:")
for s in range(N_OBJ):
    ts = ", ".join(str(t) for t in sorted(e_test_b[s]))
    if ts: print(f"    {s} -> {ts}")

tx, ty = fpairs(e_train, 400, 10)
print(f"\n  Training on graph (42)... {tx.shape[0]} frames")
md = train_gnn(tx, ty)

vx_id, vy_id = fpairs(e_train, 150, 500)
vx_a, vy_a = fpairs(e_test_a, 150, 500)
vx_b, vy_b = fpairs(e_test_b, 150, 500)

print("\n  Evaluating...")
wid = ev(md, vx_id, vy_id, True); nid = ev(md, vx_id, vy_id, False)
wa = ev(md, vx_a, vy_a, True); na = ev(md, vx_a, vy_a, False)
wb = ev(md, vx_b, vy_b, True); nb = ev(md, vx_b, vy_b, False)

bid = nid - wid; ba = na - wa; bb = nb - wb

print(f"\n  {'='*45}")
print(f"  OOD RESULT")
print(f"  In-dist (graph 42):    with={wid:.4f}  no={nid:.4f}  benefit={bid:.4f}")
print(f"  OOD-A  (graph 99):     with={wa:.4f}  no={na:.4f}  benefit={ba:.4f}")
print(f"  OOD-B  (graph 77):     with={wb:.4f}  no={nb:.4f}  benefit={bb:.4f}")
print(f"  OOD retention:         {ba/bid*100:.1f}% / {bb/bid*100:.1f}%")

m = {"ood_id_benefit": bid, "ood_a_benefit": ba, "ood_b_benefit": bb,
     "ood_retention_a_pct": ba / bid * 100 if bid > 0 else 0,
     "ood_retention_b_pct": bb / bid * 100 if bid > 0 else 0}
od = plos_dir / "results" / "causal_ood"
od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
print(f"  {'='*45}")
