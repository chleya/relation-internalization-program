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


def make_causal_graph(seed: int, n: int = N_OBJ) -> list[set[int]]:
    rng = random.Random(seed)
    edges = [set() for _ in range(n)]
    for src in range(n):
        candidates = [t for t in range(n) if t != src]
        rng.shuffle(candidates)
        n_out = rng.randint(1, min(3, n - 1))
        for tgt in candidates[:n_out]:
            edges[src].add(tgt)
    return edges


def step_causal(temps: np.ndarray, edges: list[set[int]]) -> np.ndarray:
    n = len(temps)
    new_t = temps.copy()
    for src in range(n):
        for tgt in edges[src]:
            flow = TRANSFER_COEF * (temps[src] - temps[tgt])
            new_t[tgt] += flow
    new_t += -SELF_DECAY * (new_t - AMBIENT)
    return new_t


def generate_trajectory(edges: list[set[int]], n_steps: int = 500,
                         seed: int = 0) -> np.ndarray:
    rng = random.Random(seed)
    temps = np.array([AMBIENT + 0.3 * (rng.random() - 0.5) for _ in range(N_OBJ)],
                     dtype=np.float32)
    all_feats = []
    for _ in range(n_steps):
        noise = np.array([NOISE_STD * rng.gauss(0, 1) for _ in range(N_OBJ)],
                         dtype=np.float32)
        feat = np.stack([temps, temps + noise], axis=1)
        all_feats.append(feat.astype(np.float32))
        temps = step_causal(temps, edges)
    return np.array(all_feats, dtype=np.float32)


def frame_pairs(edges, n_steps, seed):
    traj = generate_trajectory(edges, n_steps, seed)
    xs, ys = [], []
    for t in range(n_steps - 1):
        xs.append(traj[t])
        ys.append(traj[t + 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


HID = 16
NPERT = 60


def _softmax(x, ax=-1):
    e = np.exp(x - np.max(x, axis=ax, keepdims=True)); return e / e.sum(axis=ax, keepdims=True)


class CausalTF:
    def __init__(self, n_objects=N_OBJ):
        rng = np.random.Generator(np.random.PCG64(42))
        self.w_in = _init_weights((N_FEAT, HID), rng)
        self.b_in = np.zeros(HID, dtype=np.float32)
        self.w_q = _init_weights((HID, HID), rng)
        self.w_k = _init_weights((HID, HID), rng)
        self.w_v = _init_weights((HID, HID), rng)
        self.w_merge = _init_weights((HID, HID), rng)
        self.b_merge = np.zeros(HID, dtype=np.float32)
        self.w_out = _init_weights((HID, N_FEAT), rng)
        self.b_out = np.zeros(N_FEAT, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)

    def forward(self, feats, use_attn=True):
        no = feats.shape[0]; h = self._r(feats @ self.w_in + self.b_in)
        if use_attn and no > 1:
            Q = h @ self.w_q; K = h @ self.w_k; V = h @ self.w_v
            s = Q @ K.T / math.sqrt(HID); a = _softmax(s, -1)
            h = h + self._r(a @ V @ self.w_merge + self.b_merge)
        return h @ self.w_out + self.b_out

    def params(self):
        return [self.w_in, self.b_in, self.w_q, self.w_k, self.w_v,
                self.w_merge, self.b_merge, self.w_out, self.b_out]

    def set_params(self, fl):
        (self.w_in, self.b_in, self.w_q, self.w_k, self.w_v,
         self.w_merge, self.b_merge, self.w_out, self.b_out) = fl


class CausalGNN:
    def __init__(self, n_objects=N_OBJ):
        rng = np.random.Generator(np.random.PCG64(42))
        self.w_self = _init_weights((N_FEAT, HID), rng)
        self.b_self = np.zeros(HID, dtype=np.float32)
        self.w_edge = _init_weights((N_FEAT * 2, 1), rng)
        self.w_msg = _init_weights((N_FEAT, HID), rng)
        self.w_out = _init_weights((HID * 2, N_FEAT), rng)
        self.b_out = np.zeros(N_FEAT, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)

    def forward(self, feats, use_edges=True):
        no = feats.shape[0]; self_h = self._r(feats @ self.w_self + self.b_self)
        msgs = np.zeros((no, HID), dtype=np.float32)
        if use_edges and no > 1:
            for i in range(no):
                for j in range(no):
                    if i == j: continue
                    pair = np.concatenate([feats[i], feats[j]])
                    ew = float(1.0 / (1.0 + math.exp(-float((pair @ self.w_edge).item()))))
                    msg = self._r(feats[j] @ self.w_msg)
                    msgs[i] += ew * msg
        combined = np.concatenate([self_h, msgs], axis=1)
        return combined @ self.w_out + self.b_out

    def params(self):
        return [self.w_self, self.b_self, self.w_edge, self.w_msg, self.w_out, self.b_out]

    def set_params(self, fl):
        (self.w_self, self.b_self, self.w_edge, self.w_msg, self.w_out, self.b_out) = fl


def _make_grads_fn(use_gnn=False):
    def _grads(md, feats, targ, lv):
        eps = 1e-4; ap = md.params(); gs = [np.zeros_like(p) for p in ap]
        rng = random.Random(int(lv * 1e7) + 13007)
        for pi, p in enumerate(ap):
            fl = p.ravel(); g = gs[pi].ravel(); idxs = list(range(len(fl))); rng.shuffle(idxs)
            for idx in idxs[:min(NPERT, len(idxs))]:
                old = fl[idx]; fl[idx] = old + eps; md.set_params(ap)
                kw = {"use_attn": True} if not use_gnn else {"use_edges": True}
                pr = md.forward(feats, **kw); l2 = float(np.mean((pr - targ) ** 2).item())
                fl[idx] = old; g[idx] = (l2 - lv) / eps
            gs[pi] = g.reshape(p.shape)
        md.set_params(ap); return gs
    return _grads


print("=" * 55)
print("  CAUSAL GRAPH WORLD")
print(f"  {N_OBJ} objects | sparse directed edges | heat transfer")
print("=" * 55)

print("\n  Generating causal graph...")
edges_train = make_causal_graph(42)
edges_test = make_causal_graph(999)

print("  Train graph edges:")
for src in range(N_OBJ):
    tgt_str = ", ".join(str(t) for t in sorted(edges_train[src]))
    if tgt_str:
        print(f"    {src} -> {tgt_str}")

print("\n  Computing TE on causal graph trajectory...")
traj_te = generate_trajectory(edges_train, 500, 0)

def granger_causal(traj, src, tgt, nl=2):
    T = traj.shape[0]; pp = T - nl - 1
    if pp < 20: return 0, 0, 0
    Xs = np.zeros((pp, nl * N_FEAT), dtype=np.float32)
    Xf = np.zeros((pp, nl * N_FEAT * 2), dtype=np.float32)
    y = np.zeros((pp, N_FEAT), dtype=np.float32)
    for t in range(pp):
        for l in range(nl):
            bs = l * N_FEAT; bf = l * N_FEAT * 2
            Xs[t, bs:bs+N_FEAT] = traj[t+l, tgt]
            Xf[t, bf:bf+N_FEAT] = traj[t+l, tgt]
            Xf[t, bf+N_FEAT:bf+2*N_FEAT] = traj[t+l, src]
        y[t] = traj[t+nl, tgt]
    Xs_b = np.concatenate([Xs, np.ones((pp, 1), dtype=np.float32)], axis=1)
    Xf_b = np.concatenate([Xf, np.ones((pp, 1), dtype=np.float32)], axis=1)
    try: bs = np.linalg.lstsq(Xs_b, y, rcond=None)[0]; bf = np.linalg.lstsq(Xf_b, y, rcond=None)[0]
    except: return 0, 0, 0
    ms = float(np.mean((y - Xs_b @ bs) ** 2).item())
    mf = float(np.mean((y - Xf_b @ bf) ** 2).item())
    if ms < 1e-12: return 0, 0, 0
    te = max(0, (ms - mf) / ms)
    rng = random.Random(src*100+tgt*7); idxs = list(range(pp)); rng.shuffle(idxs)
    Xsh = Xs.copy(); Xsh[:] = Xf[:, nl*N_FEAT:][idxs]
    Xsh_b = np.concatenate([Xs, Xsh, np.ones((pp, 1), dtype=np.float32)], axis=1)
    try: bsh = np.linalg.lstsq(Xsh_b, y, rcond=None)[0]
    except: return te, 0, 0
    msh = float(np.mean((y - Xsh_b @ bsh) ** 2).item())
    tesh = max(0, (ms - msh) / ms)
    return te, tesh, ms

ALL_TE = []; ALL_SH = []
for src in range(N_OBJ):
    for tgt in range(N_OBJ):
        if src == tgt: continue
        te, sh, _ = granger_causal(traj_te, src, tgt)
        ALL_TE.append(te); ALL_SH.append(sh)

avg_te = float(np.mean(ALL_TE)); avg_sh = float(np.mean(ALL_SH))
te_excess = avg_te - avg_sh
print(f"  TE_avg={avg_te:.4f}  TE_shuf={avg_sh:.4f}  TE_excess={te_excess:.4f}")
print(f"  (Physics world TE excess maxed at 0.058)")

print("\n--- Training Transformer ---")
tx, ty = frame_pairs(edges_train, 400, 10)
vx, vy = frame_pairs(edges_train, 150, 500)
print(f"  Train: {tx.shape[0]} frames")

md_tf = CausalTF(); lr = 0.008; rt = random.Random(777)
tf_grads = _make_grads_fn(use_gnn=False)

for ep in range(80):
    el = 0.0; nb = 0; accum = [np.zeros_like(p) for p in md_tf.params()]
    idxs = list(range(tx.shape[0])); rt.shuffle(idxs)
    for idx in idxs[:min(32, tx.shape[0])]:
        pr = md_tf.forward(tx[idx], use_attn=True)
        l = float(np.mean((pr - ty[idx]) ** 2).item())
        if math.isnan(l) or math.isinf(l): continue
        el += l; nb += 1; gs = tf_grads(md_tf, tx[idx], ty[idx], l)
        for gi in range(len(gs)): accum[gi] += gs[gi]
    if nb > 0:
        el /= nb
        for gi in range(len(accum)): accum[gi] /= nb
    np_ = [p - lr * g for p, g in zip(md_tf.params(), accum)]; md_tf.set_params(np_)
    if ep % 25 == 0: print(f"    tf ep {ep:3d} loss={el:.6f}")


def ev_model(md, dx, dy, use_x, use_kw, label):
    tot = 0.0; n = 0
    for i in range(dx.shape[0]):
        pr = md.forward(dx[i], **(use_kw if not use_x else {"use_attn": not use_x} if isinstance(use_kw, bool) else {}))
        if isinstance(use_kw, bool):
            pr = md.forward(dx[i], **({use_x if use_x is False else "use_attn": use_kw}))
        l = float(np.mean((pr - dy[i]) ** 2).item())
        if math.isnan(l): continue
        tot += l; n += 1
    a = tot / max(n, 1)
    print(f"    {label}: loss={a:.6f}")
    return a


print("\n  Evaluating Transformer...")

def ev_tf(dx, dy, ua):
    tot = 0.0; n = 0
    for i in range(dx.shape[0]):
        pr = md_tf.forward(dx[i], use_attn=ua)
        l = float(np.mean((pr - dy[i]) ** 2).item())
        if math.isnan(l): continue
        tot += l; n += 1
    return tot / max(n, 1)

w_tf = ev_tf(vx, vy, True); n_tf = ev_tf(vx, vy, False)
b_tf = n_tf - w_tf; eff_tf = b_tf / te_excess * 100 if te_excess > 0 else 0
print(f"    tf benefit={b_tf:.4f}  eff={eff_tf:.1f}%")

print("\n--- Training GNN ---")
md_gnn = CausalGNN(); rt2 = random.Random(888)
gnn_grads = _make_grads_fn(use_gnn=True)

for ep in range(80):
    el = 0.0; nb = 0; accum = [np.zeros_like(p) for p in md_gnn.params()]
    idxs = list(range(tx.shape[0])); rt2.shuffle(idxs)
    for idx in idxs[:min(32, tx.shape[0])]:
        pr = md_gnn.forward(tx[idx], use_edges=True)
        l = float(np.mean((pr - ty[idx]) ** 2).item())
        if math.isnan(l) or math.isinf(l): continue
        el += l; nb += 1; gs = gnn_grads(md_gnn, tx[idx], ty[idx], l)
        for gi in range(len(gs)): accum[gi] += gs[gi]
    if nb > 0:
        el /= nb
        for gi in range(len(accum)): accum[gi] /= nb
    np_ = [p - lr * g for p, g in zip(md_gnn.params(), accum)]; md_gnn.set_params(np_)
    if ep % 25 == 0: print(f"    gnn ep {ep:3d} loss={el:.6f}")

print("\n  Evaluating GNN...")

def ev_gnn(dx, dy, ue):
    tot = 0.0; n = 0
    for i in range(dx.shape[0]):
        pr = md_gnn.forward(dx[i], use_edges=ue)
        l = float(np.mean((pr - dy[i]) ** 2).item())
        if math.isnan(l): continue
        tot += l; n += 1
    return tot / max(n, 1)

w_gnn = ev_gnn(vx, vy, True); n_gnn = ev_gnn(vx, vy, False)
b_gnn = n_gnn - w_gnn; eff_gnn = b_gnn / te_excess * 100 if te_excess > 0 else 0
print(f"    gnn benefit={b_gnn:.4f}  eff={eff_gnn:.1f}%")

print(f"\n  {'='*45}")
print(f"  CAUSAL GRAPH RESULT")
print(f"  cg_te_excess          = {te_excess:.4f}")
print(f"  cg_tf_with_attn       = {w_tf:.6f}")
print(f"  cg_tf_no_attn         = {n_tf:.6f}")
print(f"  cg_tf_benefit         = {b_tf:.4f}  (eff={eff_tf:.1f}%)")
print(f"  cg_gnn_with_edges     = {w_gnn:.6f}")
print(f"  cg_gnn_no_edges       = {n_gnn:.6f}")
print(f"  cg_gnn_benefit        = {b_gnn:.4f}  (eff={eff_gnn:.1f}%)")
print(f"\n  vs physics world (best): TE=0.058, benefit=0.014 (24%)")

m = {"cg_te_excess": te_excess, "cg_tf_with_attn": w_tf, "cg_tf_no_attn": n_tf,
     "cg_tf_benefit": b_tf, "cg_tf_efficiency_pct": eff_tf,
     "cg_gnn_with": w_gnn, "cg_gnn_no": n_gnn, "cg_gnn_benefit": b_gnn,
     "cg_gnn_efficiency_pct": eff_gnn}
od = plos_dir / "results" / "causal_graph"
od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
print(f"  {'='*45}")
