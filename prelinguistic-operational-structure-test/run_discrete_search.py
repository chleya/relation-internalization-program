from __future__ import annotations

import json, math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from g_line.edge_recovery import edge_recovery_metrics
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent

N_OBJ = 6; N_FEAT = 2
HID = 8; NPERT = 30
SDECAY = 0.02; NSTD = 0.01; AMB = 0.5
E_EPOCHS = 25; LR = 0.008
MAX_EDGES = 8; N_TRAIN = 200; N_VAL = 50


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


def gen_traj_pair(binary, coef, steps, seed):
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


class SelectiveGNN:
    def __init__(self, active_edges=None):
        rng_np = np.random.Generator(np.random.PCG64(42))
        self.active_edges = set(active_edges) if active_edges else set()
        self.w_msg = _init_weights((N_FEAT, HID), rng_np)
        self.w_self = _init_weights((N_FEAT, HID), rng_np)
        self.b_self = np.zeros(HID, dtype=np.float32)
        self.w_out = _init_weights((HID * 2, N_FEAT), rng_np)
        self.b_out = np.zeros(N_FEAT, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)

    def forward(self, feats):
        no = feats.shape[0]; sh = self._r(feats @ self.w_self + self.b_self)
        ms = np.zeros((no, HID), dtype=np.float32)
        for s, t in self.active_edges:
            ms[t] += feats[s] @ self.w_msg
        combined = np.concatenate([sh, ms], axis=1)
        return combined @ self.w_out + self.b_out

    def add_edge(self, src, tgt):
        self.active_edges.add((src, tgt))

    def edge_weight_matrix(self):
        ew = np.zeros((N_OBJ, N_OBJ), dtype=np.float32)
        for s, t in self.active_edges: ew[s, t] = 1.0
        return ew

    def param_arrays(self):
        return [self.w_msg, self.w_self, self.b_self, self.w_out, self.b_out]

    def set_all(self, arrs):
        self.w_msg, self.w_self, self.b_self, self.w_out, self.b_out = arrs

    def clone(self):
        c = SelectiveGNN(set(self.active_edges))
        c.w_msg = self.w_msg.copy(); c.w_self = self.w_self.copy()
        c.b_self = self.b_self.copy(); c.w_out = self.w_out.copy()
        c.b_out = self.b_out.copy()
        return c


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


def train_model(md, tx, ty, epochs, lr):
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
    return md


def eval_loss(md, tx, ty):
    el = 0.0; nb = 0
    for idx in range(tx.shape[0]):
        pr = md.forward(tx[idx]); l = float(np.mean((pr - ty[idx]) ** 2).item())
        if not math.isnan(l): el += l; nb += 1
    return el / nb if nb > 0 else float("inf")


print("=" * 55)
print("  DISCRETE SEARCH (Greedy Forward Selection)")
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
            print(f"    {s}->{t}: coef={coef[s, t]:.4f} ({tag})")

tx_train, ty_train = fpairs_pair(binary, coef, N_TRAIN + 2, 10)
tx_val, ty_val = fpairs_pair(binary, coef, N_VAL + 2, 500)

md = SelectiveGNN()
md = train_model(md, tx_train, ty_train, E_EPOCHS, LR)
baseline_loss = eval_loss(md, tx_val, ty_val)
print(f"\n  Self-only baseline: val_loss={baseline_loss:.6f}")

all_candidates = [(s, t) for s in range(N_OBJ) for t in range(N_OBJ) if s != t]
selected = []
current_loss = baseline_loss

for step in range(MAX_EDGES):
    remaining = [c for c in all_candidates if c not in selected]
    best_gain = 0.0; best_edge = None
    best_loss = current_loss
    print(f"\n  Step {step + 1}: trying {len(remaining)} candidate edges...")
    for ci, (s, t) in enumerate(remaining):
        trial_md = md.clone(); trial_md.add_edge(s, t)
        trial_md = train_model(trial_md, tx_train, ty_train, E_EPOCHS, LR)
        trial_loss = eval_loss(trial_md, tx_val, ty_val)
        gain = current_loss - trial_loss
        if gain > best_gain:
            best_gain = gain; best_edge = (s, t); best_loss = trial_loss
        if ci % 8 == 0:
            print(f"    testing edge {s}->{t} ({ci+1}/{len(remaining)}): gain={gain:.6f}")
    if best_edge:
        s, t = best_edge; md.add_edge(s, t)
        selected.append((s, t))
        current_loss = best_loss
        tag = "S" if coef[s, t] >= 0.35 else "w"
        print(f"  Step {step + 1}: ADDED {s}->{t}  true_coef={coef[s, t]:.4f} ({tag})  gain={best_gain:.6f}  val_loss={current_loss:.6f}")
    else:
        print(f"  Step {step + 1}: no beneficial edge found, stopping")
        break

print(f"\n  Selected edges ({len(selected)}):")
for i, (s, t) in enumerate(selected):
    tag = "S" if coef[s, t] >= 0.35 else "w"
    in_true = "TRUE" if binary[s, t] else "FALSE"
    print(f"    {i+1}. {s}->{t}: true_coef={coef[s, t]:.4f} ({tag})  [{in_true}]")

ew = md.edge_weight_matrix()
ds_auc, ds_spearman_r, ds_avg_precision = edge_recovery_metrics(ew, coef, N_OBJ, strong_mask)

print()
print("=" * 55)
print("  DISCRETE SEARCH RESULT")
print(f"  ds_auc                 = {ds_auc:.4f}")
print(f"  ds_spearman_r          = {ds_spearman_r:.4f}")
print(f"  ds_avg_precision        = {ds_avg_precision:.4f}")
print(f"  ds_n_selected           = {len(selected)}")
print(f"  ds_n_true_in_selected   = {sum(1 for s, t in selected if binary[s, t])}")
print("=" * 55)

results = {
    "auc": ds_auc, "spearman_r": ds_spearman_r, "avg_precision": ds_avg_precision,
    "n_selected": len(selected),
    "n_true_in_selected": sum(1 for s, t in selected if binary[s, t]),
    "selected_edges": [[s, t] for s, t in selected],
    "true_coef": coef.tolist(), "edge_weights": ew.tolist(),
}
od = plos_dir / "results" / "discrete_search"; od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
