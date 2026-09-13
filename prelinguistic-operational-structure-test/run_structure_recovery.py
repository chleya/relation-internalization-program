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
HID = 16
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


def frame_pairs(edges, n_steps, seed):
    traj = generate_trajectory(edges, n_steps, seed)
    xs, ys = [], []
    for t in range(n_steps - 1):
        xs.append(traj[t]); ys.append(traj[t + 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class CausalGNN:
    def __init__(self):
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
                    ew = 1.0 / (1.0 + math.exp(-float((pair @ self.w_edge).item())))
                    msgs[i] += ew * self._r(feats[j] @ self.w_msg)
        combined = np.concatenate([self_h, msgs], axis=1)
        return combined @ self.w_out + self.b_out

    def params(self):
        return [self.w_self, self.b_self, self.w_edge, self.w_msg, self.w_out, self.b_out]

    def set_params(self, fl):
        (self.w_self, self.b_self, self.w_edge, self.w_msg, self.w_out, self.b_out) = fl


def compute_edge_matrix(model, test_x):
    no = N_OBJ; ew_mat = np.zeros((no, no), dtype=np.float32)
    count = 0
    for fi in range(test_x.shape[0]):
        feats = test_x[fi]
        for i in range(no):
            for j in range(no):
                if i == j: continue
                pair = np.concatenate([feats[i], feats[j]])
                ew_mat[i, j] += 1.0 / (1.0 + math.exp(-float((pair @ model.w_edge).item())))
        count += 1
    ew_mat /= count
    return ew_mat


def _gnn_grads(md, feats, targ, lv):
    eps = 1e-4; ap = md.params(); gs = [np.zeros_like(p) for p in ap]
    rng = random.Random(int(lv * 1e7) + 13007)
    for pi, p in enumerate(ap):
        fl = p.ravel(); g = gs[pi].ravel(); idxs = list(range(len(fl))); rng.shuffle(idxs)
        for idx in idxs[:min(NPERT, len(idxs))]:
            old = fl[idx]; fl[idx] = old + eps; md.set_params(ap)
            pr = md.forward(feats, use_edges=True)
            l2 = float(np.mean((pr - targ) ** 2).item())
            fl[idx] = old; g[idx] = (l2 - lv) / eps
        gs[pi] = g.reshape(p.shape)
    md.set_params(ap); return gs


print("=" * 55)
print("  Experiment 1: Causal Graph Structure Recovery")
print("=" * 55)

edges = make_causal_graph(42)
truth_mat = np.zeros((N_OBJ, N_OBJ), dtype=np.int32)
for src in range(N_OBJ):
    for tgt in edges[src]:
        truth_mat[src, tgt] = 1

print("  True causal graph:")
for src in range(N_OBJ):
    tgt_str = ", ".join(str(t) for t in sorted(edges[src]))
    if tgt_str: print(f"    {src} -> {tgt_str}")

print("\n  Training GNN...")
tx, ty = frame_pairs(edges, 400, 10)
md = CausalGNN(); lr = 0.008; rt = random.Random(777)

for ep in range(80):
    el = 0.0; nb = 0; accum = [np.zeros_like(p) for p in md.params()]
    idxs = list(range(tx.shape[0])); rt.shuffle(idxs)
    for idx in idxs[:min(32, tx.shape[0])]:
        pr = md.forward(tx[idx], use_edges=True)
        l = float(np.mean((pr - ty[idx]) ** 2).item())
        if math.isnan(l) or math.isinf(l): continue
        el += l; nb += 1; gs = _gnn_grads(md, tx[idx], ty[idx], l)
        for gi in range(len(gs)): accum[gi] += gs[gi]
    if nb > 0:
        el /= nb
        for gi in range(len(accum)): accum[gi] /= nb
    md.set_params([p - lr * g for p, g in zip(md.params(), accum)])
    if ep % 25 == 0: print(f"    ep {ep:3d} loss={el:.6f}")

print("\n  Computing learned edge matrix...")
vx, vy = frame_pairs(edges, 150, 500)
ew_mat = compute_edge_matrix(md, vx)

print("\n  Learned edge weights (avg over test set):")
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
            print(f"  {ew_mat[i,j]:7.4f}{marker}", end="")
    print()

flat_ew = []
flat_truth = []
for i in range(N_OBJ):
    for j in range(N_OBJ):
        if i != j:
            flat_ew.append(float(ew_mat[i, j]))
            flat_truth.append(int(truth_mat[i, j]))

flat_ew = np.array(flat_ew, dtype=np.float32)
flat_truth = np.array(flat_truth, dtype=np.int32)

order = np.argsort(-flat_ew)
sorted_truth = flat_truth[order]

n_true = int(flat_truth.sum())
precisions = []
recalls = []
for k in range(1, n_true * 3 + 1):
    tp = int(sorted_truth[:k].sum())
    precisions.append(tp / k)
    recalls.append(tp / n_true if n_true > 0 else 0)

auc = float(np.mean(precisions))
top_k = min(n_true * 2, len(sorted_truth))
tp_at_top = int(sorted_truth[:top_k].sum())
recall_at_top = tp_at_top / n_true if n_true > 0 else 0
precision_at_top = tp_at_top / top_k

print(f"\n  === STRUCTURE RECOVERY RESULT ===")
print(f"  sr_n_true_edges       = {n_true}")
print(f"  sr_edge_auc           = {auc:.4f}  (1.0 = perfect ranking)")
print(f"  sr_precision@top{top_k}    = {precision_at_top:.4f}")
print(f"  sr_recall@top{top_k}       = {recall_at_top:.4f}")

thresh = float(np.median(flat_ew))
pred_mat = (ew_mat >= thresh).astype(np.int32)
tp = int((pred_mat * truth_mat).sum())
fp = int((pred_mat * (1 - truth_mat)).sum())
fn = int(((1 - pred_mat) * truth_mat).sum())
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"  sr_precision          = {precision:.4f}")
print(f"  sr_recall             = {recall:.4f}")
print(f"  sr_f1                 = {f1:.4f}")

m = {"sr_n_true_edges": n_true, "sr_edge_auc": auc,
     "sr_precision": precision, "sr_recall": recall, "sr_f1": f1,
     "sr_precision_at_topk": precision_at_top, "sr_recall_at_topk": recall_at_top,
     "edge_matrix": ew_mat.tolist(), "truth_matrix": truth_mat.tolist()}
od = plos_dir / "results" / "causal_structure_recovery"
od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
print("=" * 55)
