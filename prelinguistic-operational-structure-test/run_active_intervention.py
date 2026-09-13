from __future__ import annotations

import json, math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from g_line.edge_recovery import edge_recovery_metrics
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent

N_OBJ = 6; N_FEAT = 2
HID = 16; NPERT = 60
SDECAY = 0.02; NSTD = 0.01; AMB = 0.5
PUSH_MAGNITUDE = 0.3; N_AFTER = 3
E_EPISODES = 100; LR = 0.008


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


class ActiveInterventionGNN:
    def __init__(self):
        rng_np = np.random.Generator(np.random.PCG64(42))
        self.edge_logits = np.zeros((N_OBJ, N_OBJ), dtype=np.float32)
        for i in range(N_OBJ): self.edge_logits[i, i] = -10.0
        self.w_msg = _init_weights((N_FEAT, HID), rng_np)
        self.w_self = _init_weights((N_FEAT, HID), rng_np)
        self.b_self = np.zeros(HID, dtype=np.float32)
        self.w_out = _init_weights((HID * 2, N_FEAT), rng_np)
        self.b_out = np.zeros(N_FEAT, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)
    def _sigmoid(self, x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def get_edge_weights(self):
        return self._sigmoid(self.edge_logits)

    def forward(self, feats):
        ew = self.get_edge_weights()
        no = feats.shape[0]; sh = self._r(feats @ self.w_self + self.b_self)
        ms = np.zeros((no, HID), dtype=np.float32)
        for i in range(no):
            for j in range(no):
                if i == j: continue
                ms[i] += ew[j, i] * (feats[j] @ self.w_msg)
        combined = np.concatenate([sh, ms], axis=1)
        return combined @ self.w_out + self.b_out

    def param_arrays(self):
        return [self.edge_logits, self.w_msg, self.w_self, self.b_self, self.w_out, self.b_out]

    def set_all(self, arrs):
        self.edge_logits, self.w_msg, self.w_self, self.b_self, self.w_out, self.b_out = arrs


def predict_future(md, current_temps, push_target, binary, coef, n_steps):
    true_t = current_temps.copy()
    true_t[push_target] += PUSH_MAGNITUDE
    futures = []
    for _ in range(n_steps):
        true_t = step_pairwise(true_t, binary, coef)
        ns = np.array([NSTD * random.gauss(0, 1) for _ in range(N_OBJ)], dtype=np.float32)
        futures.append(np.stack([true_t, true_t + ns], axis=1).astype(np.float32))
    return np.array(futures, dtype=np.float32)


def _grads(md, feats_before, futures_actual, push_target, lv):
    eps = 1e-4; all_a = md.param_arrays(); gs = [np.zeros_like(a) for a in all_a]
    rng = random.Random(int(lv * 1e7) + push_target * 13007)
    for pi, p in enumerate(all_a):
        flat = p.ravel(); g = gs[pi].ravel()
        idxs = list(range(len(flat))); rng.shuffle(idxs)
        n_pert = min(NPERT, len(idxs))
        for idx in idxs[:n_pert]:
            old = flat[idx]; flat[idx] = old + eps; md.set_all(all_a)
            total_loss = 0.0
            for step_i in range(N_AFTER):
                pred = md.forward(futures_actual[step_i])
                total_loss += float(np.mean((pred - futures_actual[step_i + 1 if step_i < N_AFTER - 1 else step_i]) ** 2).item())
                # Actually: predict futures_actual[t+1] from futures_actual[t]
            total_loss /= N_AFTER
            flat[idx] = old; g[idx] = (total_loss - lv) / eps
        gs[pi] = g.reshape(p.shape)
    md.set_all(all_a); return gs


def train_active(md, binary, coef, epochs, lr, strategy="random"):
    rt = random.Random(777)
    history = []
    for ep in range(epochs):
        seed = ep * 1000 + 13
        rng = random.Random(seed)
        temps = np.array([AMB + 0.3 * (rng.random() - 0.5) for _ in range(N_OBJ)], dtype=np.float32)
        for _ in range(5):
            temps = step_pairwise(temps, binary, coef)
        ns = np.array([NSTD * rng.gauss(0, 1) for _ in range(N_OBJ)], dtype=np.float32)
        obs = np.stack([temps, temps + ns], axis=1).astype(np.float32)

        if strategy == "random":
            push_target = ep % N_OBJ
        else:
            ew = md.get_edge_weights()
            row_vars = np.array([np.var(ew[i]) for i in range(N_OBJ)])
            push_target = int(np.argmax(row_vars))

        futures = predict_future(md, temps, push_target, binary, coef, N_AFTER + 1)

        md.set_all(md.param_arrays())
        accum = [np.zeros_like(a) for a in md.param_arrays()]
        el = 0.0; nb = 0

        for step_i in range(N_AFTER):
            pred = md.forward(futures[step_i])
            l = float(np.mean((pred - futures[step_i + 1]) ** 2).item())
            if math.isnan(l) or math.isinf(l): continue
            el += l; nb += 1
            gs = _grads_single(md, futures[step_i], futures[step_i + 1], push_target, l)
            for gi in range(len(gs)): accum[gi] += gs[gi]

        if nb > 0:
            el /= nb
            for gi in range(len(accum)): accum[gi] /= nb
        md.set_all([a - lr * g for a, g in zip(md.param_arrays(), accum)])
        history.append(el)
        if ep % 25 == 0:
            print(f"    ep {ep:3d} loss={el:.6f}")
    return md, history


def _grads_single(md, feats, targ, push_target, lv):
    eps = 1e-4; all_a = md.param_arrays(); gs = [np.zeros_like(a) for a in all_a]
    rng = random.Random(int(lv * 1e7) + push_target * 13007)
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
print("  ACTIVE INTERVENTION GNN")
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

results = {}

for strategy in ["random", "uncertainty"]:
    print(f"\n  Training with {strategy} pushes ({E_EPISODES} episodes)...")
    md = ActiveInterventionGNN()
    md, history = train_active(md, binary, coef, E_EPISODES, LR, strategy)
    ew = md.get_edge_weights()
    np.fill_diagonal(ew, 0)
    ai_auc, ai_spearman_r, ai_avg_precision = edge_recovery_metrics(ew, coef, N_OBJ, strong_mask)
    print(f"  {strategy}: auc={ai_auc:.4f}  spearman_r={ai_spearman_r:.4f}  avg_precision={ai_avg_precision:.4f}")
    print(f"  Edge weights:")
    for s in range(N_OBJ):
        for t in range(N_OBJ):
            if binary[s, t]:
                tag = "S" if coef[s, t] >= 0.35 else "w"
                print(f"    {s}->{t}: true={coef[s,t]:.4f} ({tag})  learned={ew[s,t]:.4f}")
    results[strategy] = {
        "auc": ai_auc, "spearman_r": ai_spearman_r, "avg_precision": ai_avg_precision,
        "edge_weights": ew.tolist(),
    }

print()
print("=" * 55)
print("  ACTIVE INTERVENTION RESULT")
for st in ["random", "uncertainty"]:
    v = results[st]
    print(f"  ai_{st:>11s}: auc={v['auc']:.4f}  spearman_r={v['spearman_r']:.4f}  avg_prec={v['avg_precision']:.4f}")
print("=" * 55)

results["true_coef"] = coef.tolist()
od = plos_dir / "results" / "active_intervention"; od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
