from __future__ import annotations

import json, math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from g_line import gn_world as sw
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent

XW = 2.0
XCF = 10.0
XHT = 2.5
XDAMP = 0.05
XCD = 0.9
XNO = 8
XNOISE = 0.02


def _init_xo(rng):
    n = XNO
    s = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        a = 2 * math.pi * i / n
        r = XW * 0.4 * rng.random()
        s[i, 0] = XW / 2 + r * math.cos(a)
        s[i, 1] = XW / 2 + r * math.sin(a)
        s[i, 2] = (rng.random() - 0.5) * 2.5
        s[i, 3] = (rng.random() - 0.5) * 2.5
        s[i, 4] = sw.AMBIENT_T + rng.random() * 0.9
    return s


def _x_fh(st):
    n = st.shape[0]
    f = np.zeros((n, 2), dtype=np.float32)
    h = np.zeros(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dx = st[i, 0] - st[j, 0]
            dy = st[i, 1] - st[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < XCD and d > 0.005:
                ff = XCF * (XCD - d) / d
                fx = ff * dx / d
                fy = ff * dy / d
                f[i, 0] += fx; f[i, 1] += fy
                f[j, 0] -= fx; f[j, 1] -= fy
                hf = XHT * (st[j, 4] - st[i, 4]) / (1.0 + d)
                h[i] += hf; h[j] -= hf
    return f, h


def _x_step(st):
    n = st.shape[0]
    fo, he = _x_fh(st)
    ns = st.copy()
    for i in range(n):
        ns[i, 0] += st[i, 2] * sw.DT
        ns[i, 1] += st[i, 3] * sw.DT
        ns[i, 2] += (fo[i, 0] - XDAMP * st[i, 2]) * sw.DT
        ns[i, 3] += (fo[i, 1] - XDAMP * st[i, 3]) * sw.DT
        ns[i, 4] += (-sw.COOLING * (st[i, 4] - sw.AMBIENT_T) + he[i]) * sw.DT
    ns[:, 0] = np.clip(ns[:, 0], 0.1, XW - 0.1)
    ns[:, 1] = np.clip(ns[:, 1], 0.1, XW - 0.1)
    ns[:, 4] = np.clip(ns[:, 4], sw.AMBIENT_T - 0.3, sw.AMBIENT_T + 1.2)
    return ns


def _x_feats(st):
    n = st.shape[0]
    fs = np.zeros((n, sw.N_FEATS_CONT), dtype=np.float32)
    for i in range(n):
        fs[i, 0] = st[i, 0] / XW + XNOISE * np.random.randn()
        fs[i, 1] = st[i, 1] / XW + XNOISE * np.random.randn()
        sp = math.sqrt(st[i, 2] ** 2 + st[i, 3] ** 2)
        fs[i, 2] = sw._clamp(sp / 3.0 + XNOISE * np.random.randn())
        fs[i, 3] = sw._clamp(st[i, 4] + XNOISE * np.random.randn())
        fs[i, 4] = sw._clamp(max(0, fs[i, 3] - sw.AMBIENT_T) * 0.8 + fs[i, 2] * 0.4 + XNOISE * np.random.randn())
        fs[i, 5] = sw._clamp(0.15 + XNOISE * np.random.randn())
        fs[i, 6] = sw._clamp(0.10 + XNOISE * np.random.randn())
    return np.clip(fs, 0.0, 1.0)


def xtraj(steps, seed):
    rng = random.Random(seed)
    st = _init_xo(rng)
    ts = []
    for _ in range(steps):
        ts.append(_x_feats(st).astype(np.float32))
        st = _x_step(st)
    return np.array(ts, dtype=np.float32)


def xtraj_state(steps, seed):
    rng = random.Random(seed)
    st = _init_xo(rng)
    ss = []
    for _ in range(steps):
        ss.append(st.copy())
        st = _x_step(st)
    return np.array(ss, dtype=np.float32)


def granger_t(traj, src, tgt, nl=2):
    T, no, d = traj.shape
    pp = T - nl - 1
    if pp < 20: return 0, 0, 0
    sd = nl * d
    Xs = np.zeros((pp, sd), dtype=np.float32)
    Xf = np.zeros((pp, sd * 2), dtype=np.float32)
    y = np.zeros((pp, d), dtype=np.float32)
    for t in range(pp):
        for l in range(nl):
            Xs[t, l * d:(l + 1) * d] = traj[t + l, tgt]
            Xf[t, l * 2 * d:l * 2 * d + d] = traj[t + l, tgt]
            Xf[t, l * 2 * d + d:l * 2 * d + 2 * d] = traj[t + l, src]
        y[t] = traj[t + nl, tgt]
    Xs_b = np.concatenate([Xs, np.ones((pp, 1), dtype=np.float32)], axis=1)
    Xf_b = np.concatenate([Xf, np.ones((pp, 1), dtype=np.float32)], axis=1)
    try:
        bs = np.linalg.lstsq(Xs_b, y, rcond=None)[0]
        bf = np.linalg.lstsq(Xf_b, y, rcond=None)[0]
    except: return 0, 0, 0
    ms = float(np.mean((y - Xs_b @ bs) ** 2).item())
    mf = float(np.mean((y - Xf_b @ bf) ** 2).item())
    if ms < 1e-10: return 0, 0, 0
    te = max(0, (ms - mf) / ms)
    rng = random.Random(src * 100 + tgt * 7)
    idxs = list(range(pp)); rng.shuffle(idxs)
    Xsh = Xs.copy(); Xsh[:] = Xf[:, sd:][idxs]
    Xsh_b = np.concatenate([Xs, Xsh, np.ones((pp, 1), dtype=np.float32)], axis=1)
    try: bsh = np.linalg.lstsq(Xsh_b, y, rcond=None)[0]
    except: return te, 0, 0
    msh = float(np.mean((y - Xsh_b @ bsh) ** 2).item())
    tesh = max(0, (ms - msh) / ms)
    return te, tesh, ms


print("=" * 55)
print(f"  EXTREME WORLD: {XNO} objects in {XW}x{XW}")
print(f"  Force={XCF} Heat={XHT} Damp={XDAMP}")
print("=" * 55)

print("\n--- TE ---")
t = xtraj(600, 0)
print(f"  Traj: {t.shape}")
tes, shs = [], []
ps = [(i, j) for i in range(XNO) for j in range(XNO) if i != j]
random.Random(42).shuffle(ps)
for src, tgt in ps[:30]:
    te, sh, _ = granger_t(t, src, tgt)
    tes.append(te); shs.append(sh)
ate = float(np.mean(tes)); ash = float(np.mean(shs))
xe = ate - ash
print(f"  TE_avg={ate:.4f}  TE_shuf={ash:.4f}  TE_excess={xe:.4f}")
print(f"  vs small(0.033)={xe/0.0333:.2f}x  vs big_v1(0.054)={xe/0.0535:.2f}x")

TD = 20
TNP = 60


def _softmax(x, ax=-1):
    e = np.exp(x - np.max(x, axis=ax, keepdims=True))
    return e / e.sum(axis=ax, keepdims=True)


class XTF:
    def __init__(self):
        rng = np.random.Generator(np.random.PCG64(42))
        self.wi = _init_weights((sw.N_FEATS_CONT, TD), rng)
        self.bi = np.zeros(TD, dtype=np.float32)
        self.wq = _init_weights((TD, TD), rng)
        self.wk = _init_weights((TD, TD), rng)
        self.wv = _init_weights((TD, TD), rng)
        self.wm = _init_weights((TD, TD), rng)
        self.bm = np.zeros(TD, dtype=np.float32)
        self.wo = _init_weights((TD, sw.N_FEATS_CONT), rng)
        self.bo = np.zeros(sw.N_FEATS_CONT, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)

    def fw(self, feats, ua=True):
        no = feats.shape[0]
        h = self._r(feats @ self.wi + self.bi)
        if ua and no > 1:
            Q = h @ self.wq; K = h @ self.wk; V = h @ self.wv
            s = Q @ K.T / math.sqrt(TD)
            a = _softmax(s, -1)
            h = h + self._r(a @ V @ self.wm + self.bm)
        return h @ self.wo + self.bo

    def ps(self):
        return [self.wi, self.bi, self.wq, self.wk, self.wv, self.wm, self.bm, self.wo, self.bo]

    def sp(self, fl):
        (self.wi, self.bi, self.wq, self.wk, self.wv, self.wm, self.bm, self.wo, self.bo) = fl


def gfp(ns, sd):
    rng = random.Random(sd)
    st = _init_xo(rng)
    xs, ys = [], []
    for _ in range(ns):
        xs.append(_x_feats(st).astype(np.float32))
        ns_ = _x_step(st)
        ys.append(_x_feats(ns_).astype(np.float32))
        st = ns_
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def _x_grads(md, feats, targ, lv):
    eps = 1e-4
    ap = md.ps()
    gs = [np.zeros_like(p) for p in ap]
    rng = random.Random(int(lv * 1e7) + 13007)
    for pi, p in enumerate(ap):
        fl = p.ravel(); g = gs[pi].ravel()
        idxs = list(range(len(fl))); rng.shuffle(idxs)
        for idx in idxs[:min(TNP, len(idxs))]:
            old = fl[idx]; fl[idx] = old + eps
            md.sp(ap)
            pr = md.fw(feats, True)
            l2 = float(np.mean((pr - targ) ** 2).item())
            fl[idx] = old; g[idx] = (l2 - lv) / eps
        gs[pi] = g.reshape(p.shape)
    md.sp(ap)
    return gs


print("\n--- Transformer ---")
tx, ty = gfp(350, 0)
vex, vey = gfp(100, 100)
print(f"  Train: {tx.shape[0]} frames")

md = XTF()
lr = 0.006
rt = random.Random(777)

for ep in range(100):
    el = 0.0; nb = 0
    accum = [np.zeros_like(p) for p in md.ps()]
    idxs = list(range(tx.shape[0])); rt.shuffle(idxs)
    for idx in idxs[:min(32, tx.shape[0])]:
        pr = md.fw(tx[idx], True)
        l = float(np.mean((pr - ty[idx]) ** 2).item())
        if math.isnan(l) or math.isinf(l): continue
        el += l; nb += 1
        gs = _x_grads(md, tx[idx], ty[idx], l)
        for gi in range(len(gs)): accum[gi] += gs[gi]
    if nb > 0:
        el /= nb
        for gi in range(len(accum)): accum[gi] /= nb
    np_ = [p - lr * g for p, g in zip(md.ps(), accum)]
    md.sp(np_)
    if ep % 30 == 0: print(f"    ep {ep:3d} loss={el:.6f}")


def ev(data_x, data_y, ua, lab):
    tot = 0.0; n = 0
    for i in range(data_x.shape[0]):
        pr = md.fw(data_x[i], ua)
        l = float(np.mean((pr - data_y[i]) ** 2).item())
        if math.isnan(l): continue
        tot += l; n += 1
    a = tot / max(n, 1)
    print(f"    {lab}: loss={a:.6f}")
    return a


def evs(data_x, data_y, lab):
    oq, ok, ov = md.wq.copy(), md.wk.copy(), md.wv.copy()
    rg = np.random.Generator(np.random.PCG64(555))
    for w in [md.wq, md.wk, md.wv]:
        fl = w.ravel(); rg.shuffle(fl); w[:] = fl.reshape(w.shape)
    tot = 0.0; n = 0
    for i in range(data_x.shape[0]):
        pr = md.fw(data_x[i], True)
        l = float(np.mean((pr - data_y[i]) ** 2).item())
        if math.isnan(l): continue
        tot += l; n += 1
    a = tot / max(n, 1)
    md.wq[:], md.wk[:], md.wv[:] = oq, ok, ov
    print(f"    {lab}: loss={a:.6f}")
    return a


print("\n  Evaluating...")
wl = ev(vex, vey, True, "with_attn ")
nl = ev(vex, vey, False, "no_attn   ")
sl = evs(vex, vey, "shuf_attn ")
ben = nl - wl
sd_ = sl - wl

print(f"\n  {'='*45}")
print(f"  EXTREME WORLD RESULT")
print(f"  x_te_excess           = {xe:.4f}  (1.6x big_v1, {xe/0.0333:.1f}x small)")
print(f"  x_with_attn           = {wl:.6f}")
print(f"  x_no_attn             = {nl:.6f}")
print(f"  x_attn_benefit        = {ben:.6f}")
print(f"  x_shuf_drop           = {sd_:.6f}")
print(f"  (small benefit=0.0026, big_v1 benefit=0.0030, extreme={ben:.4f})")
print(f"  TE/benefit efficiency = {ben/xe*100:.1f}% (captured % of available TE)")

m = {"x_te_excess": xe, "x_with_attn": wl, "x_no_attn": nl,
     "x_attn_benefit": ben, "x_shuf_drop": sd_,
     "te_benefit_efficiency_pct": ben / xe * 100 if xe > 0 else 0}
od = plos_dir / "results" / "extreme_world"
od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
print(f"  {'='*45}")
