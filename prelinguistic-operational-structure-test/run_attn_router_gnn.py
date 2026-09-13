from __future__ import annotations

import json, math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from g_line.edge_recovery import edge_recovery_metrics
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent

N_OBJ = 6; N_FEAT = 2
HID = 16; NPERT = 60; N_HEADS = 4
SDECAY = 0.02; NSTD = 0.01; AMB = 0.5
D_R = 4; H_R = 8

E_EPOCHS = 80; LR = 0.008


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


def _softmax(x, ax=-1):
    e = np.exp(x - np.max(x, axis=ax, keepdims=True))
    return e / e.sum(axis=ax, keepdims=True)


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


class AttnRouterGNN:
    def __init__(self, n_heads=N_HEADS):
        rng_np = np.random.Generator(np.random.PCG64(42))
        self.n_heads = n_heads; self.head_dim = HID

        self.edge_emb = _init_weights((N_OBJ, N_OBJ, D_R), rng_np, 0.05)
        for i in range(N_OBJ): self.edge_emb[i, i] = 0.0

        self.route_w1 = _init_weights((D_R, H_R), rng_np)
        self.route_b1 = np.zeros(H_R, dtype=np.float32)
        self.route_w2 = _init_weights((H_R, n_heads), rng_np)
        self.route_b2 = np.zeros(n_heads, dtype=np.float32)

        self.w_in = _init_weights((N_FEAT, HID * n_heads), rng_np)
        self.b_in = np.zeros(HID * n_heads, dtype=np.float32)
        self.w_q = _init_weights((HID * n_heads, HID * n_heads), rng_np)
        self.w_k = _init_weights((HID * n_heads, HID * n_heads), rng_np)
        self.w_v = _init_weights((HID * n_heads, HID * n_heads), rng_np)

        self.w_self = _init_weights((N_FEAT, HID), rng_np)
        self.b_self = np.zeros(HID, dtype=np.float32)
        self.w_out = _init_weights((HID + HID * n_heads, N_FEAT), rng_np)
        self.b_out = np.zeros(N_FEAT, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)

    def forward(self, feats):
        no = feats.shape[0]; nh = self.n_heads; hd = self.head_dim
        sh = self._r(feats @ self.w_self + self.b_self)

        h = self._r(feats @ self.w_in + self.b_in)
        Q = (h @ self.w_q).reshape(no, nh, hd)
        K = (h @ self.w_k).reshape(no, nh, hd)
        V = (h @ self.w_v).reshape(no, nh, hd)

        scale = float(math.sqrt(hd))
        attn = np.zeros((no, nh, no), dtype=np.float32)
        for hi in range(nh):
            scores = (Q[:, hi, :] @ K[:, hi, :].T) / scale
            attn[:, hi, :] = _softmax(scores, ax=-1)

        ms_heads = np.zeros((no, nh, hd), dtype=np.float32)
        for i in range(no):
            for j in range(no):
                if i == j: continue
                route_raw = self._r(self.edge_emb[j, i] @ self.route_w1 + self.route_b1)
                route = _softmax(route_raw @ self.route_w2 + self.route_b2, ax=-1)
                for hi in range(nh):
                    ms_heads[i, hi] += attn[i, hi, j] * route[hi] * V[j, hi]

        ms = ms_heads.reshape(no, HID * nh)
        combined = np.concatenate([sh, ms], axis=1)
        return combined @ self.w_out + self.b_out

    def param_arrays(self):
        return [self.edge_emb, self.route_w1, self.route_b1, self.route_w2, self.route_b2,
                self.w_in, self.b_in, self.w_q, self.w_k, self.w_v,
                self.w_self, self.b_self, self.w_out, self.b_out]

    def set_all(self, arrs):
        (self.edge_emb, self.route_w1, self.route_b1, self.route_w2, self.route_b2,
         self.w_in, self.b_in, self.w_q, self.w_k, self.w_v,
         self.w_self, self.b_self, self.w_out, self.b_out) = arrs

    def edge_weight_matrix(self):
        ew = np.zeros((N_OBJ, N_OBJ), dtype=np.float32)
        for i in range(N_OBJ):
            for j in range(N_OBJ):
                if i == j: continue
                route_raw = self._r(self.edge_emb[j, i] @ self.route_w1 + self.route_b1)
                route = _softmax(route_raw @ self.route_w2 + self.route_b2, ax=-1)
                ew[j, i] = float(np.sum(route))
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
print("  ATTN ROUTER GNN")
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

for nh in [4, 8]:
    print(f"\n  Training AttnRouterGNN heads={nh}...")
    md_a = AttnRouterGNN(nh); md_a = train_model(md_a, tx, ty, E_EPOCHS, LR, f"AttnRouterGNN-h{nh}")
    a_ew = md_a.edge_weight_matrix()
    a_auc, a_spearman_r, a_avg_precision = edge_recovery_metrics(a_ew, coef, N_OBJ, strong_mask)
    print(f"  AttnRouterGNN h={nh}: auc={a_auc:.4f}  spearman_r={a_spearman_r:.4f}  avg_precision={a_avg_precision:.4f}")

    print(f"  Edge route analysis (h={nh}):")
    for s in range(N_OBJ):
        for t in range(N_OBJ):
            if binary[s, t]:
                route_raw = md_a._r(md_a.edge_emb[s, t] @ md_a.route_w1 + md_a.route_b1)
                route = _softmax(route_raw @ md_a.route_w2 + md_a.route_b2, ax=-1)
                rt_str = " ".join(f"{route[hi]:.2f}" for hi in range(nh))
                print(f"    {s}->{t}: coef={coef[s,t]:.4f}  route=[{rt_str}]")

    results[f"attn_h{nh}"] = {"auc": a_auc, "spearman_r": a_spearman_r, "avg_precision": a_avg_precision,
                               "edge_weights": a_ew.tolist()}

print()
print("=" * 55)
print("  ATTN ROUTER RESULT")
for k, v in results.items():
    if k == "baseline":
        print(f"  {k:>20s}: auc={v['auc']:.4f}  spearman_r={v['spearman_r']:.4f}  avg_prec={v['avg_precision']:.4f}")
    else:
        print(f"  {k:>20s}: auc={v['auc']:.4f}  spearman_r={v['spearman_r']:.4f}  avg_prec={v['avg_precision']:.4f}  "
              f"delta_r={v['spearman_r']-bl_spearman_r:+.4f}")
print("=" * 55)

results["true_coef"] = coef.tolist(); results["baseline_weights"] = bl_ew.tolist()
od = plos_dir / "results" / "attn_router"; od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
