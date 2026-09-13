from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from g_line import gn_world
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent

AL_HIDDEN = 16
AL_N_PERTURB = 60
AL_N_STEPS = 10
AL_N_EPOCHS = 150


def _run_forward(state: np.ndarray, n_steps: int,
                 intervene_obj: int) -> np.ndarray:
    cur = state.copy()
    cur[intervene_obj, 2] *= 0.3
    cur[intervene_obj, 3] *= 0.3
    cur[intervene_obj, 4] = gn_world.AMBIENT_T + 0.05
    traj = np.zeros((n_steps, cur.shape[0], gn_world.N_FEATS_CONT), dtype=np.float32)
    for s in range(n_steps):
        cur = gn_world._step_euler(cur)
        traj[s] = gn_world._state_to_features(cur).astype(np.float32)
    return traj


class ActiveModel:
    def __init__(self, n_objects: int = 3):
        self.n_objs = n_objects
        rng = np.random.Generator(np.random.PCG64(42))
        self.w_self = _init_weights((gn_world.N_FEATS_CONT, AL_HIDDEN), rng)
        self.b_self = np.zeros(AL_HIDDEN, dtype=np.float32)
        self.w_cross = _init_weights((gn_world.N_FEATS_CONT, AL_HIDDEN), rng)
        self.b_cross = np.zeros(AL_HIDDEN, dtype=np.float32)
        self.w_act = _init_weights((n_objects, AL_HIDDEN), rng)
        flat_out = AL_N_STEPS * gn_world.N_FEATS_CONT
        self.w_head = _init_weights((AL_HIDDEN * 3, flat_out), rng)
        self.b_head = np.zeros(flat_out, dtype=np.float32)
        self.error_tracker = np.ones(n_objects, dtype=np.float32) * 0.05

    def _relu(self, x):
        return np.maximum(0, x)

    def forward(self, pre_feats: np.ndarray, intervene_obj: int,
                use_cross: bool = True) -> np.ndarray:
        n_o = pre_feats.shape[0]
        self_h = self._relu(pre_feats @ self.w_self + self.b_self)
        if use_cross and n_o > 1:
            cross_h = self._relu(pre_feats @ self.w_cross + self.b_cross)
            cross_agg = np.zeros((n_o, AL_HIDDEN), dtype=np.float32)
            for i in range(n_o):
                m = np.ones(n_o, dtype=bool)
                m[i] = False
                cross_agg[i] = cross_h[m].mean(axis=0)
        else:
            cross_agg = np.zeros((n_o, AL_HIDDEN), dtype=np.float32)
        act_emb = np.zeros(AL_HIDDEN, dtype=np.float32)
        if 0 <= intervene_obj < self.n_objs:
            act_emb = self.w_act[intervene_obj]

        flat_out = AL_N_STEPS * gn_world.N_FEATS_CONT
        preds = np.zeros((n_o, flat_out), dtype=np.float32)
        for i in range(n_o):
            combined = np.concatenate([self_h[i], cross_agg[i], act_emb])
            preds[i] = combined @ self.w_head + self.b_head
        return preds.reshape(n_o, AL_N_STEPS, gn_world.N_FEATS_CONT)

    def params(self):
        return [self.w_self, self.b_self, self.w_cross, self.b_cross,
                self.w_act, self.w_head, self.b_head]

    def set_params(self, flat):
        self.w_self, self.b_self, self.w_cross, self.b_cross, \
        self.w_act, self.w_head, self.b_head = flat

    def choose_action(self, pre_feats: np.ndarray, eps: float = 0.25) -> int:
        if random.random() < eps:
            return random.randint(0, self.n_objs - 1)
        uncer = np.zeros(self.n_objs, dtype=np.float32)
        for oi in range(self.n_objs):
            pred = self.forward(pre_feats, oi, use_cross=True)
            delta_mag = float(np.mean(np.abs(pred)).item())
            uncer[oi] = self.error_tracker[oi] + 0.1 * delta_mag
        return int(np.argmax(uncer))

    def update_error_tracker(self, per_obj_error: np.ndarray, decay: float = 0.85):
        for oi in range(min(self.n_objs, len(per_obj_error))):
            self.error_tracker[oi] = (
                decay * self.error_tracker[oi]
                + (1.0 - decay) * float(per_obj_error[oi])
            )


def _al_grads(model: ActiveModel, pre_feats: np.ndarray,
              intervene_obj: int, target: np.ndarray,
              loss_val: float) -> list[np.ndarray]:
    eps = 1e-4
    all_p = model.params()
    grads = [np.zeros_like(p) for p in all_p]
    rng = random.Random(int(loss_val * 1e7) + intervene_obj * 97 + 7001)

    for pi, p in enumerate(all_p):
        flat = p.ravel()
        g = grads[pi].ravel()
        idxs = list(range(len(flat)))
        rng.shuffle(idxs)
        for idx in idxs[:min(AL_N_PERTURB, len(idxs))]:
            old = flat[idx]
            flat[idx] = old + eps
            model.set_params(all_p)
            pred = model.forward(pre_feats, intervene_obj, use_cross=True)
            l2 = float(np.mean((pred - target) ** 2).item())
            flat[idx] = old
            g[idx] = (l2 - loss_val) / eps
        grads[pi] = g.reshape(p.shape)
    model.set_params(all_p)
    return grads


print("=== Route D: Active Interactive Learning ===")
print("  Starting simulation world...")

n_objects = 3
model = ActiveModel(n_objects=n_objects)
lr = 0.008
rng_st = random.Random(0)

history_active = []
history_random = []

for phase in range(2):
    mode = "active" if phase == 0 else "random"
    phase_model = ActiveModel(n_objects=n_objects)
    print(f"\n  Phase {phase}: mode={mode} ({AL_N_EPOCHS} episodes)")
    losses = []

    for ep in range(AL_N_EPOCHS):
        state = gn_world._init_objects(rng_st, n_objects)
        for _ in range(random.randint(3, 15)):
            state = gn_world._step_euler(state)

        pre_state = state.copy()
        pre_feats = gn_world._state_to_features(pre_state).astype(np.float32)

        if mode == "active":
            intervene_obj = phase_model.choose_action(pre_feats)
        else:
            intervene_obj = random.randint(0, n_objects - 1)

        actual_traj = _run_forward(pre_state, AL_N_STEPS, intervene_obj)
        actual_traj_ = actual_traj.transpose(1, 0, 2)
        pred = phase_model.forward(pre_feats, intervene_obj, use_cross=True)
        loss = float(np.mean((pred - actual_traj_) ** 2).item())

        if math.isnan(loss) or math.isinf(loss):
            continue

        grads = _al_grads(phase_model, pre_feats, intervene_obj, actual_traj_, loss)
        accum = grads
        new_p = [p - lr * g for p, g in zip(phase_model.params(), accum)]
        phase_model.set_params(new_p)

        per_obj = np.array([float(np.mean((pred[oi] - actual_traj_[oi]) ** 2).item())
                           for oi in range(n_objects)], dtype=np.float32)
        phase_model.update_error_tracker(per_obj)

        losses.append(loss)

        if ep % 40 == 0 and ep > 0:
            avg = float(np.mean(losses[-20:]))
            print(f"    ep {ep:3d}  avg_loss={avg:.6f}  last_loss={loss:.6f}")

    final_losses = losses[-40:] if len(losses) >= 40 else losses
    avg_final = float(np.mean(final_losses))
    print(f"    {mode} final_avg_loss = {avg_final:.6f}  total_eps={len(losses)}")

    if mode == "active":
        history_active = losses
    else:
        history_random = losses

    if mode == "active":
        eval_states = []
        eval_acts = []
        eval_trajs = []
        rng_eval = random.Random(999)
        for _ in range(40):
            st = gn_world._init_objects(rng_eval, n_objects)
            for _ in range(8):
                st = gn_world._step_euler(st)
            eval_states.append(gn_world._state_to_features(st).astype(np.float32))
            act = rng_eval.randint(0, n_objects - 1)
            eval_acts.append(act)
            eval_trajs.append(_run_forward(st.copy(), AL_N_STEPS, act).transpose(1, 0, 2))

        cross_loss = 0.0
        no_loss = 0.0
        for idx in range(len(eval_states)):
            p1 = phase_model.forward(eval_states[idx], eval_acts[idx], use_cross=True)
            p0 = phase_model.forward(eval_states[idx], eval_acts[idx], use_cross=False)
            cross_loss += float(np.mean((p1 - eval_trajs[idx]) ** 2).item())
            no_loss += float(np.mean((p0 - eval_trajs[idx]) ** 2).item())
        cross_loss /= len(eval_states)
        no_loss /= len(eval_states)
        active_cross_benefit = no_loss - cross_loss
        print(f"\n    [eval after {mode} phase]")
        print(f"    cross_loss = {cross_loss:.6f}  no_cross = {no_loss:.6f}")
        print(f"    cross_benefit = {active_cross_benefit:.6f}")

final_active = float(np.mean(history_active[-40:])) if len(history_active) >= 40 else float(np.mean(history_active))
final_random = float(np.mean(history_random[-40:])) if len(history_random) >= 40 else float(np.mean(history_random))

print(f"\n  === ACTIVE LEARNING RESULT ===")
print(f"  al_final_active_loss  = {final_active:.6f}")
print(f"  al_final_random_loss  = {final_random:.6f}")
print(f"  al_active_vs_random   = {final_random - final_active:.6f}  (+ = active better)")
print(f"  al_active_cross_benefit = {active_cross_benefit:.6f}")

metrics = {
    "al_final_active_loss": final_active,
    "al_final_random_loss": final_random,
    "al_active_vs_random": final_random - final_active,
    "al_active_cross_benefit": active_cross_benefit,
}
out_dir = plos_dir / "results" / "active_learning"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"  Saved to {out_dir}")
print("=== DONE ===")
