from __future__ import annotations

import json, math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent

N_OBJ = 6
N_FEAT = 2
TRANSFER_COEF = 0.35
SELF_DECAY = 0.02
NOISE_STD = 0.01
AMBIENT = 0.5
HID = 20
NPERT = 60


def make_causal_graph(seed, n=N_OBJ):
    rng = random.Random(seed)
    edges = [set() for _ in range(n)]
    for src in range(n):
        candidates = [t for t in range(n) if t != src]
        rng.shuffle(candidates)
        for tgt in candidates[:rng.randint(1, min(3, n - 1))]:
            edges[src].add(tgt)
    return edges


def step_causal(temps, edges):
    n = len(temps); new_t = temps.copy()
    for src in range(n):
        for tgt in edges[src]:
            new_t[tgt] += TRANSFER_COEF * (temps[src] - temps[tgt])
    new_t += -SELF_DECAY * (new_t - AMBIENT)
    return new_t


def generate_trajectory(edges, n_steps=500, seed=0):
    rng = random.Random(seed)
    temps = np.array([AMBIENT + 0.3 * (rng.random() - 0.5) for _ in range(N_OBJ)],
                     dtype=np.float32)
    all_feats = []
    for _ in range(n_steps):
        noise = np.array([NOISE_STD * rng.gauss(0, 1) for _ in range(N_OBJ)], dtype=np.float32)
        all_feats.append(np.stack([temps, temps + noise], axis=1).astype(np.float32))
        temps = step_causal(temps, edges)
    return np.array(all_feats, dtype=np.float32)


def causal_importance(edges):
    imp = np.zeros(N_OBJ, dtype=np.float32)
    in_deg = np.zeros(N_OBJ, dtype=np.int32)
    for src in range(N_OBJ):
        for tgt in edges[src]:
            in_deg[tgt] += 1
    for node in range(N_OBJ):
        out_deg = len(edges[node])
        imp[node] = out_deg + in_deg[node]
    return imp


class PartialGNN:
    def __init__(self):
        rng = np.random.Generator(np.random.PCG64(42))
        self.w_self = _init_weights((N_FEAT + 1, HID), rng)
        self.b_self = np.zeros(HID, dtype=np.float32)
        self.w_cross = _init_weights((N_FEAT + 1, HID), rng)
        self.b_cross = np.zeros(HID, dtype=np.float32)
        self.w_out = _init_weights((HID * 2, N_FEAT), rng)
        self.b_out = np.zeros(N_FEAT, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)

    def forward(self, feats, mask):
        no = feats.shape[0]
        feats_filled = feats.copy()
        feats_filled[~mask] = 0.5
        mask_f = mask.astype(np.float32).reshape(-1, 1)
        augmented = np.concatenate([feats_filled, mask_f], axis=1)

        self_h = self._r(augmented @ self.w_self + self.b_self)
        cross_h = self._r(augmented @ self.w_cross + self.b_cross)
        cross_agg = np.zeros((no, HID), dtype=np.float32)
        for i in range(no):
            m = np.ones(no, dtype=bool); m[i] = False
            if m.sum() > 0:
                cross_agg[i] = cross_h[m].mean(axis=0)

        combined = np.concatenate([self_h, cross_agg], axis=1)
        return combined @ self.w_out + self.b_out

    def params(self):
        return [self.w_self, self.b_self, self.w_cross, self.b_cross,
                self.w_out, self.b_out]

    def set_params(self, fl):
        (self.w_self, self.b_self, self.w_cross, self.b_cross,
         self.w_out, self.b_out) = fl


def make_partial_data(edges, n_steps, seed):
    traj = generate_trajectory(edges, n_steps, seed)
    xs, masks, ys = [], [], []
    rng = random.Random(seed + 77)
    for t in range(n_steps - 1):
        mask = np.ones(N_OBJ, dtype=bool)
        n_hide = rng.randint(1, 2)
        hiders = list(range(N_OBJ)); rng.shuffle(hiders)
        for h in hiders[:n_hide]:
            mask[h] = False
        xs.append(traj[t]); masks.append(mask); ys.append(traj[t + 1])
    return np.array(xs, dtype=np.float32), np.array(masks), np.array(ys, dtype=np.float32)


def _pgrads(md, feats, mask, targ, lv):
    eps = 1e-4; ap = md.params(); gs = [np.zeros_like(p) for p in ap]
    rng = random.Random(int(lv * 1e7) + 13007)
    for pi, p in enumerate(ap):
        fl = p.ravel(); g = gs[pi].ravel(); idxs = list(range(len(fl))); rng.shuffle(idxs)
        for idx in idxs[:min(NPERT, len(idxs))]:
            old = fl[idx]; fl[idx] = old + eps; md.set_params(ap)
            pr = md.forward(feats, mask); l2 = float(np.mean((pr - targ) ** 2).item())
            fl[idx] = old; g[idx] = (l2 - lv) / eps
        gs[pi] = g.reshape(p.shape)
    md.set_params(ap); return gs


print("=" * 55)
print("  Experiment 2: Inspection Budget (O2 core)")
print("=" * 55)

edges = make_causal_graph(42)
importance = causal_importance(edges)

print("  Causal graph & importance:")
for src in range(N_OBJ):
    tgt_str = ", ".join(str(t) for t in sorted(edges[src]))
    in_deg = sum(1 for s in range(N_OBJ) if src in edges[s])
    print(f"    {src}: ->[{tgt_str}]  in={in_deg}  imp={importance[src]:.0f}")

print("\n  Training Partial-GNN (random hiding)...")
tx, tm, ty = make_partial_data(edges, 400, 10)
md = PartialGNN(); lr = 0.008; rt = random.Random(777)

for ep in range(80):
    el = 0.0; nb = 0; accum = [np.zeros_like(p) for p in md.params()]
    idxs = list(range(tx.shape[0])); rt.shuffle(idxs)
    for idx in idxs[:min(32, tx.shape[0])]:
        pr = md.forward(tx[idx], tm[idx])
        l = float(np.mean((pr - ty[idx]) ** 2).item())
        if math.isnan(l) or math.isinf(l): continue
        el += l; nb += 1; gs = _pgrads(md, tx[idx], tm[idx], ty[idx], l)
        for gi in range(len(gs)): accum[gi] += gs[gi]
    if nb > 0:
        el /= nb
        for gi in range(len(accum)): accum[gi] /= nb
    md.set_params([p - lr * g for p, g in zip(md.params(), accum)])
    if ep % 25 == 0: print(f"    ep {ep:3d} loss={el:.6f}")

print("\n  Evaluating inspection choices...")
vx, vm, vy = make_partial_data(edges, 200, 500)

correct_choices = 0
total_eval = 0
choice_dist = np.zeros(N_OBJ, dtype=np.int32)

for fi in range(vx.shape[0]):
    mask = vm[fi].copy()
    hidden = np.where(~mask)[0]
    if len(hidden) == 0:
        continue

    best_var = -1
    best_loss = float("inf")
    for h in hidden:
        test_mask = mask.copy(); test_mask[h] = True
        pr = md.forward(vx[fi], test_mask)
        loss = float(np.mean((pr - vy[fi]) ** 2).item())
        if loss < best_loss:
            best_loss = loss; best_var = h

    choice_dist[best_var] += 1
    total_eval += 1

    if importance[best_var] >= np.median(importance):
        correct_choices += 1

choice_rate = correct_choices / total_eval if total_eval > 0 else 0
random_rate = np.mean(importance >= np.median(importance))
improvement = choice_rate - random_rate

print(f"\n    Inspection choice distribution:")
for oi in range(N_OBJ):
    bar = "#" * (choice_dist[oi] // 2)
    print(f"      obj {oi} (imp={importance[oi]:.0f}): {choice_dist[oi]:3d}  {bar}")

print(f"\n  === INSPECTION BUDGET RESULT ===")
print(f"  ib_total_eval        = {total_eval}")
print(f"  ib_choice_rate       = {choice_rate:.4f}  (% choosing high-importance)")
print(f"  ib_random_baseline   = {random_rate:.4f}")
print(f"  ib_improvement       = {improvement:.4f}  (+ = better than random)")
print(f"\n  Interpretation:")
if improvement > 0.05:
    print(f"    Model O2: PREFERS causally important variables for inspection.")
else:
    print(f"    Model O2: does NOT prefer important variables.")

m = {"ib_total_eval": total_eval, "ib_choice_rate": choice_rate,
     "ib_random_baseline": random_rate, "ib_improvement": improvement,
     "choice_distribution": choice_dist.tolist(),
     "causal_importance": importance.tolist()}
od = plos_dir / "results" / "inspection_budget"
od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
print("=" * 55)
