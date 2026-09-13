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
DT, AT, CO = sw.DT, sw.AMBIENT_T, sw.COOLING
NF = sw.N_FEATS_CONT
TD = 24
TNP = 60


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
    n = st.shape[0]; f = np.zeros((n, 2), dtype=np.float32); h = np.zeros(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dx = st[i, 0] - st[j, 0]; dy = st[i, 1] - st[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < XCD and d > 0.005:
                ff = XCF * (XCD - d) / d; fx = ff * dx / d; fy = ff * dy / d
                f[i, 0] += fx; f[i, 1] += fy; f[j, 0] -= fx; f[j, 1] -= fy
                hf = XHT * (st[j, 4] - st[i, 4]) / (1.0 + d)
                h[i] += hf; h[j] -= hf
    return f, h


def _x_step(st):
    n = st.shape[0]; ns = st.copy(); fo, he = _x_fh(st)
    for i in range(n):
        ns[i, 0] += st[i, 2] * DT; ns[i, 1] += st[i, 3] * DT
        ns[i, 2] += (fo[i, 0] - XDAMP * st[i, 2]) * DT
        ns[i, 3] += (fo[i, 1] - XDAMP * st[i, 3]) * DT
        ns[i, 4] += (-CO * (st[i, 4] - AT) + he[i]) * DT
    ns[:, 0] = np.clip(ns[:, 0], 0.1, XW - 0.1); ns[:, 1] = np.clip(ns[:, 1], 0.1, XW - 0.1)
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


def gen_temp_data(n_frames, n_ctx, seed):
    rng = random.Random(seed)
    st = _init_xo(rng)
    all_fs = []
    for _ in range(n_frames):
        all_fs.append(_x_feats(st).astype(np.float32))
        st = _x_step(st)
    all_fs = np.array(all_fs, dtype=np.float32)
    xs, ys = [], []
    for t in range(n_ctx, len(all_fs)):
        xs.append(all_fs[t - n_ctx:t].transpose(1, 0, 2))
        ys.append(all_fs[t])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def _softmax(x, ax=-1):
    e = np.exp(x - np.max(x, axis=ax, keepdims=True)); return e / e.sum(axis=ax, keepdims=True)


class TempTF:
    def __init__(self, n_ctx):
        self.nc = n_ctx; fin = n_ctx * NF
        rng = np.random.Generator(np.random.PCG64(42))
        self.wi = _init_weights((fin, TD), rng); self.bi = np.zeros(TD, dtype=np.float32)
        self.wq = _init_weights((TD, TD), rng); self.wk = _init_weights((TD, TD), rng)
        self.wv = _init_weights((TD, TD), rng)
        self.wm = _init_weights((TD, TD), rng); self.bm = np.zeros(TD, dtype=np.float32)
        self.wo = _init_weights((TD, NF), rng); self.bo = np.zeros(NF, dtype=np.float32)

    def _r(self, x): return np.maximum(0, x)

    def fw(self, ctx, ua=True):
        no, nc, nf = ctx.shape; fl = ctx.reshape(no, -1)
        h = self._r(fl @ self.wi + self.bi)
        if ua and no > 1:
            Q = h @ self.wq; K = h @ self.wk; V = h @ self.wv
            s = Q @ K.T / math.sqrt(TD); a = _softmax(s, -1)
            h = h + self._r(a @ V @ self.wm + self.bm)
        return h @ self.wo + self.bo

    def ps(self): return [self.wi, self.bi, self.wq, self.wk, self.wv, self.wm, self.bm, self.wo, self.bo]

    def sp(self, fl):
        (self.wi, self.bi, self.wq, self.wk, self.wv, self.wm, self.bm, self.wo, self.bo) = fl


def _tt_grads(md, ctx, targ, lv):
    eps = 1e-4; ap = md.ps(); gs = [np.zeros_like(p) for p in ap]
    rng = random.Random(int(lv * 1e7) + 13007)
    for pi, p in enumerate(ap):
        fl = p.ravel(); g = gs[pi].ravel(); idxs = list(range(len(fl))); rng.shuffle(idxs)
        for idx in idxs[:min(TNP, len(idxs))]:
            old = fl[idx]; fl[idx] = old + eps; md.sp(ap)
            pr = md.fw(ctx, True); l2 = float(np.mean((pr - targ) ** 2).item())
            fl[idx] = old; g[idx] = (l2 - lv) / eps
        gs[pi] = g.reshape(p.shape)
    md.sp(ap); return gs


def run_one(n_ctx, label):
    tx, ty = gen_temp_data(450, n_ctx, 0)
    vex, vey = gen_temp_data(130, n_ctx, 100)
    md = TempTF(n_ctx); lr = 0.006; rt = random.Random(777)
    for ep in range(100):
        el = 0.0; nb = 0; accum = [np.zeros_like(p) for p in md.ps()]
        idxs = list(range(tx.shape[0])); rt.shuffle(idxs)
        for idx in idxs[:min(32, tx.shape[0])]:
            pr = md.fw(tx[idx], True); l = float(np.mean((pr - ty[idx]) ** 2).item())
            if math.isnan(l) or math.isinf(l): continue
            el += l; nb += 1; gs = _tt_grads(md, tx[idx], ty[idx], l)
            for gi in range(len(gs)): accum[gi] += gs[gi]
        if nb > 0:
            el /= nb
            for gi in range(len(accum)): accum[gi] /= nb
        np_ = [p - lr * g for p, g in zip(md.ps(), accum)]; md.sp(np_)

    def ev(dx, dy, ua):
        tot = 0.0; n = 0
        for i in range(dx.shape[0]):
            pr = md.fw(dx[i], ua); l = float(np.mean((pr - dy[i]) ** 2).item())
            if math.isnan(l): continue
            tot += l; n += 1
        return tot / max(n, 1)

    wl = ev(vex, vey, True); nl = ev(vex, vey, False)
    return wl, nl, nl - wl


print("=" * 50)
print("  TEMPORAL CONTEXT SWEEP")
print("=" * 50)

TE_XS = 0.0579
print(f"  TE excess baseline: {TE_XS:.4f}")
print(f"\n  {'ctx':>5s} {'w_attn':>8s} {'no_attn':>8s} {'benefit':>8s} {'eff%':>6s}")
print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

results = []
for nc in [1, 3, 5, 7]:
    label = f"ctx{nc}"
    print(f"  Training ctx={nc}...")
    wl, nl, ben = run_one(nc, label)
    eff = ben / TE_XS * 100 if TE_XS > 0 else 0
    print(f"  {nc:5d} {wl:8.4f} {nl:8.4f} {ben:8.4f} {eff:5.1f}%")
    results.append({"ctx": nc, "with": wl, "no": nl, "benefit": ben, "efficiency": eff})

print(f"\n  === SUMMARY ===")
print(f"  TE_excess = {TE_XS:.4f}")
for r in results:
    print(f"  {r['ctx']}-frame: benefit={r['benefit']:.4f}  eff={r['efficiency']:.1f}%")

mdict = {"te_excess": TE_XS, "results": results}
od = plos_dir / "results" / "temporal_sweep"
od.mkdir(parents=True, exist_ok=True)
(od / "metrics.json").write_text(json.dumps(mdict, indent=2), encoding="utf-8")
print(f"  Saved to {od}")
print("=" * 50)
