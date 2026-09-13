from __future__ import annotations

import json, math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from g_line import gn_world as sw
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent

XW, XCF, XHT, XDAMP, XCD = 2.0, 10.0, 2.5, 0.05, 0.9
XNO = 8
XNOISE = 0.02
N_CTX = 3

DT, AT, CO = sw.DT, sw.AMBIENT_T, sw.COOLING
NF = sw.N_FEATS_CONT


def _init_xo(rng):
    s = np.zeros((XNO, 5), dtype=np.float32)
    for i in range(XNO):
        a = 2 * math.pi * i / XNO
        r = XW * 0.4 * rng.random()
        s[i, 0] = XW / 2 + r * math.cos(a)
        s[i, 1] = XW / 2 + r * math.sin(a)
        s[i, 2] = (rng.random() - 0.5) * 2.5
        s[i, 3] = (rng.random() - 0.5) * 2.5
        s[i, 4] = AT + rng.random() * 0.9
    return s


def _x_fh(st):
    n = st.shape[0]
    f = np.zeros((n, 2), dtype=np.float32); h = np.zeros(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dx = st[i, 0] - st[j, 0]; dy = st[i, 1] - st[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < XCD and d > 0.005:
                ff = XCF * (XCD - d) / d; fx = ff * dx / d; fy = ff * dy / d
                f[i, 0] += fx; f[i, 1] += fy
                f[j, 0] -= fx; f[j, 1] -= fy
                hf = XHT * (st[j, 4] - st[i, 4]) / (1.0 + d)
                h[i] += hf; h[j] -= hf
    return f, h


def _x_step(st):
    n = st.shape[0]; ns = st.copy()
    fo, he = _x_fh(st)
    for i in range(n):
        ns[i, 0] += st[i, 2] * DT; ns[i, 1] += st[i, 3] * DT
        ns[i, 2] += (fo[i, 0] - XDAMP * st[i, 2]) * DT
        ns[i, 3] += (fo[i, 1] - XDAMP * st[i, 3]) * DT
        ns[i, 4] += (-CO * (st[i, 4] - AT) + he[i]) * DT
    ns[:, 0] = np.clip(ns[:, 0], 0.1, XW - 0.1)
    ns[:, 1] = np.clip(ns[:, 1], 0.1, XW - 0.1)
    ns[:, 4] = np.clip(ns[:, 4], AT - 0.3, AT + 1.2)
    return ns


def _x_feats(st):
    n = st.shape[0]; fs = np.zeros((n, NF), dtype=np.float32)
    for i in range(n):
        fs[i, 0] = st[i, 0] / XW + XNOISE * np.random.randn()
        fs[i, 1] = st[i, 1] / XW + XNOISE * np.random.randn()
        sp = math.sqrt(st[i, 2] ** 2 + st[i, 3] ** 2)
        fs[i, 2] = sw._clamp(sp / 3.0 + XNOISE * np.random.randn())
        fs[i, 3] = sw._clamp(st[i, 4] + XNOISE * np.random.randn())
        fs[i, 4] = sw._clamp(max(0, fs[i, 3] - AT) * 0.8 + fs[i, 2] * 0.4 + XNOISE * np.random.randn())
        fs[i, 5] = sw._clamp(0.15 + XNOISE * np.random.randn())
        fs[i, 6] = sw._clamp(0.10 + XNOISE * np.random.randn())
    return np.clip(fs, 0.0, 1.0)


def gen_temporal_data(n_frames, seed):
    rng = random.Random(seed)
    st = _init_xo(rng)
    all_fs = []
    for _ in range(n_frames):
        all_fs.append(_x_feats(st).astype(np.float32))
        st = _x_step(st)
    all_fs = np.array(all_fs, dtype=np.float32)

    xs, ys = [], []
    for t in range(N_CTX, n_frames):
        ctx = all_fs[t - N_CTX:t]
        xs.append(ctx.transpose(1, 0, 2))
        ys.append(all_fs[t])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


TD = 24
TNP = 60


def _softmax(x, ax=-1):
    e = np.exp(x - np.max(x, axis=ax, keepdims=True))
    return e / e.sum(axis=ax, keepdims=True)


class TempTransformer:
    def __init__(self):
        flat_in = N_CTX * NF
        rng = np.random.Generator(np.random.PCG64(42))
        self.w_in = _init_weights((flat_in, TD), rng)
        self.b_in = np.zeros(TD, dtype=np.float32)
        self.w_q = _init_weights((TD, TD), rng)
        self.w_k = _init_weights((TD, TD), rng)
        self.w_v = _init_weights((TD, TD), rng)
        self.w_merge = _init_weights((TD, TD), rng)
        self.b_merge = np.zeros(TD, dtype=np.float32)
        self.w_out = _init_weights((TD, NF), rng)
        self.b_out = np.zeros(NF, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)

    def forward(self, ctx_feats, use_attn=True):
        no, nctx, nf = ctx_feats.shape
        flat = ctx_feats.reshape(no, -1)
        h = self._r(flat @ self.w_in + self.b_in)
        if use_attn and no > 1:
            Q = h @ self.w_q; K = h @ self.w_k; V = h @ self.w_v
            s = Q @ K.T / math.sqrt(TD)
            a = _softmax(s, -1)
            h = h + self._r(a @ V @ self.w_merge + self.b_merge)
        return h @ self.w_out + self.b_out

    def params(self):
        return [self.w_in, self.b_in, self.w_q, self.w_k, self.w_v,
                self.w_merge, self.b_merge, self.w_out, self.b_out]

    def set_params(self, fl):
        (self.w_in, self.b_in, self.w_q, self.w_k, self.w_v,
         self.w_merge, self.b_merge, self.w_out, self.b_out) = fl


def _tt_grads(md, ctx, targ, lv):
    eps = 1e-4; ap = md.params()
    gs = [np.zeros_like(p) for p in ap]
    rng = random.Random(int(lv * 1e7) + 13007)
    for pi, p in enumerate(ap):
        fl = p.ravel(); g = gs[pi].ravel()
        idxs = list(range(len(fl))); rng.shuffle(idxs)
        for idx in idxs[:min(TNP, len(idxs))]:
            old = fl[idx]; fl[idx] = old + eps
            md.set_params(ap)
            pr = md.forward(ctx, True)
            l2 = float(np.mean((pr - targ) ** 2).item())
            fl[idx] = old; g[idx] = (l2 - lv) / eps
        gs[pi] = g.reshape(p.shape)
    md.set_params(ap)
    return gs


print("=" * 55)
print(f"  TEMPORAL TRANSFORMER on extreme world")
print(f"  Context frames: {N_CTX} | d_model: {TD}")
print("=" * 55)

print("\n  Generating temporal data...")
tx, ty = gen_temporal_data(450, 0)
vex, vey = gen_temporal_data(130, 100)
print(f"  Train: {tx.shape[0]} samples, shape={tx.shape[1:]}")

md = TempTransformer()
lr = 0.006
rt = random.Random(777)

for ep in range(100):
    el = 0.0; nb = 0
    accum = [np.zeros_like(p) for p in md.params()]
    idxs = list(range(tx.shape[0])); rt.shuffle(idxs)
    for idx in idxs[:min(32, tx.shape[0])]:
        pr = md.forward(tx[idx], True)
        l = float(np.mean((pr - ty[idx]) ** 2).item())
        if math.isnan(l) or math.isinf(l): continue
        el += l; nb += 1
        gs = _tt_grads(md, tx[idx], ty[idx], l)
        for gi in range(len(gs)): accum[gi] += gs[gi]
    if nb > 0:
        el /= nb
        for gi in range(len(accum)): accum[gi] /= nb
    np_ = [p - lr * g for p, g in zip(md.params(), accum)]
    md.set_params(np_)
    if ep % 30 == 0: print(f"    ep {ep:3d} loss={el:.6f}")


def ev_tt(data_x, data_y, ua, lab):
    tot = 0.0; n = 0
    for i in range(data_x.shape[0]):
        pr = md.forward(data_x[i], ua)
        l = float(np.mean((pr - data_y[i]) ** 2).item())
        if math.isnan(l): continue
        tot += l; n += 1
    a = tot / max(n, 1)
    print(f"    {lab}: loss={a:.6f}")
    return a


def evs_tt(data_x, data_y, lab):
    oq, ok, ov = md.w_q.copy(), md.w_k.copy(), md.w_v.copy()
    rg = np.random.Generator(np.random.PCG64(555))
    for w in [md.w_q, md.w_k, md.w_v]:
        fl = w.ravel(); rg.shuffle(fl); w[:] = fl.reshape(w.shape)
    tot = 0.0; n = 0
    for i in range(data_x.shape[0]):
        pr = md.forward(data_x[i], True)
        l = float(np.mean((pr - data_y[i]) ** 2).item())
        if math.isnan(l): continue
        tot += l; n += 1
    a = tot / max(n, 1)
    md.w_q[:], md.w_k[:], md.w_v[:] = oq, ok, ov
    print(f"    {lab}: loss={a:.6f}")
    return a


print("\n  Evaluating...")
wl = ev_tt(vex, vey, True, "with_attn ")
nl = ev_tt(vex, vey, False, "no_attn   ")
sl = evs_tt(vex, vey, "shuf_attn ")

ben = nl - wl
sd_ = sl - wl
TE_XS = 0.0579
eff = ben / TE_XS * 100

print(f"\n  {'='*45}")
print(f"  TEMPORAL TRANSFORMER RESULT")
print(f"  tt_with_attn          = {wl:.6f}")
print(f"  tt_no_attn            = {nl:.6f}")
print(f"  tt_attn_benefit       = {ben:.6f}")
print(f"  tt_shuf_drop          = {sd_:.6f}")
print(f"  TE/benefit efficiency = {eff:.1f}% (was 7% on 1-frame)")
print(f"\n  Comparison:")
print(f"    1-frame transformer: benefit=0.0040, efficiency=7.0%")
print(f"    {N_CTX}-frame transformer: benefit={ben:.4f}, efficiency={eff:.1f}%")

m = {"tt_with_attn": wl, "tt_no_attn": nl, "tt_shuf_attn": sl,
     "tt_attn_benefit": ben, "tt_shuf_drop": sd_,
     "te_benefit_efficiency_pct": eff, "n_context": N_CTX}
od = plos_dir / "results" / "temporal_transformer"
od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
print(f"  {'='*45}")
