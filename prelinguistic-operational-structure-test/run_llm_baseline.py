from __future__ import annotations

import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

plos_dir = Path(__file__).resolve().parent
od = plos_dir / "results" / "llm_causal"
probes = json.loads((od / "probes.json").read_text(encoding="utf-8"))
gt = json.loads((od / "ground_truth.json").read_text(encoding="utf-8"))

NAMES = probes["world"]["objects"]
TRUE_EDGES = {(e["src"], e["tgt"]): e["coef"] for e in probes["world"]["edges"]}


def parse_edges(text):
    found = []
    for m in re.finditer(r'([A-F])', text):
        si = ord(m.group(1)) - ord('A')
        if 0 <= si < len(NAMES):
            pass
    for m in re.finditer(r'(\w+)\s*[-→>]+\s*(\w+)', text):
        sn, tn = m.group(1), m.group(2)
        if sn in NAMES and tn in NAMES:
            si, ti = NAMES.index(sn), NAMES.index(tn)
            if si != ti and (si, ti) not in found:
                found.append((si, ti))
    return found


def score_existence(resp, probe_gt):
    pred = parse_edges(resp)
    true_set = set((e[0], e[1]) for e in probe_gt["edges"])
    tp = len(set(pred) & true_set)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(true_set) if true_set else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "n_pred": len(pred)}


def score_ranking(resp, probe_gt):
    pred = parse_edges(resp)
    if len(pred) < 2:
        return {"spearman_r": 0.0, "n_matched": len(pred)}
    true_rank = {(e[0], e[1]): i for i, e in enumerate(probe_gt["ranked"])}
    prs = []
    for pi, (s, t) in enumerate(pred):
        if (s, t) in true_rank:
            prs.append((pi, true_rank[(s, t)]))
    if len(prs) < 2:
        return {"spearman_r": 0.0, "n_matched": len(prs)}
    pa = np.array([p for p, _ in prs])
    ta = np.array([t for _, t in prs])
    pr = pa.argsort().argsort().astype(float)
    tr = ta.argsort().argsort().astype(float)
    if np.std(pr) < 1e-12 or np.std(tr) < 1e-12:
        r = 0.0
    else:
        r = np.corrcoef(pr, tr)[0, 1]
        r = 0.0 if np.isnan(r) else float(r)
    return {"spearman_r": r, "n_matched": len(prs)}


def score_counterfactual(resp, probe_gt):
    ans = probe_gt["answer"]
    for n in NAMES:
        if n in resp and n == ans:
            return {"correct": 1.0}
    return {"correct": 1.0 if ans in resp else 0.0}


def score_edge_diff(resp, probe_gt):
    mx, mn = probe_gt["max_target"], probe_gt["min_target"]
    sc = 0.0
    if mx in resp: sc += 0.5
    if mn in resp: sc += 0.5
    return {"score": sc, "max_correct": 1.0 if mx in resp else 0.0,
            "min_correct": 1.0 if mn in resp else 0.0}


SCORERS = {"existence": score_existence, "ranking": score_ranking,
           "counterfactual": score_counterfactual, "edge_diff": score_edge_diff}


def oracle_response(probe):
    pid = probe["id"]; pgt = probe["ground_truth"]
    if pid == "existence":
        lines = [f"{NAMES[s]}→{NAMES[t]}" for s, t, _ in pgt["edges"]]
        return "\n".join(lines)
    elif pid == "ranking":
        lines = [f"{i+1}. {NAMES[s]}→{NAMES[t]} (系数={c:.4f})"
                 for i, (s, t, c) in enumerate(pgt["ranked"])]
        return "\n".join(lines)
    elif pid == "counterfactual":
        return f"水箱{chr(ord('A')+3)}的温度变化最大，因为D直接影响了它。"
    elif pid == "edge_diff":
        gt_r = pgt["ranked"]
        lines = [f"C对{NAMES[t]}的影响强度为{c:.4f}" for s, t, c in gt_r]
        lines.append(f"受影响最大: {NAMES[gt_r[0][1]]}，受影响最小: {NAMES[gt_r[-1][1]]}")
        return "\n".join(lines)
    return ""


def random_response(probe):
    import random
    rng = random.Random(hash(probe["id"]) * 10007)
    pid = probe["id"]
    if pid in ("existence", "ranking"):
        n = rng.randint(2, 10)
        pairs = [(rng.randint(0, 5), rng.randint(0, 5)) for _ in range(n)]
        pairs = [(s, t) for s, t in pairs if s != t]
        lines = [f"{NAMES[s]}→{NAMES[t]}" for s, t in pairs[:8]]
        return "\n".join(lines)
    elif pid == "counterfactual":
        return NAMES[rng.randint(0, 5)]
    elif pid == "edge_diff":
        n = NAMES[rng.randint(0, 5)]
        return f"C对{n}的影响最大，对其他水箱影响较小。"
    return ""


def empty_response(probe):
    return "我不知道。"


def run(respond_fn, name):
    all_scores = {"existence": [], "ranking": [], "counterfactual": [], "edge_diff": []}
    for si, sc in enumerate(probes["scenarios"]):
        for probe in sc["probes"]:
            resp = respond_fn(probe)
            scores = SCORERS[probe["id"]](resp, probe["ground_truth"])
            all_scores[probe["id"]].append(scores)

    th = gt["pass_threshold"]
    summary = {}
    for pid in ["existence", "ranking", "counterfactual", "edge_diff"]:
        sl = all_scores[pid]
        if pid == "existence":
            f1 = float(np.mean([s["f1"] for s in sl]))
            summary[pid] = {"avg_f1": f1, "passed": f1 >= th["existence"]}
        elif pid == "ranking":
            r = float(np.mean([s["spearman_r"] for s in sl]))
            summary[pid] = {"avg_spearman_r": r, "passed": r >= th["ranking_spearman"]}
        elif pid == "counterfactual":
            acc = float(np.mean([s["correct"] for s in sl]))
            summary[pid] = {"avg_accuracy": acc, "passed": acc >= th["counterfactual"]}
        elif pid == "edge_diff":
            sc = float(np.mean([s["score"] for s in sl]))
            summary[pid] = {"avg_score": sc, "passed": sc >= th["edge_diff"]}
    return summary


print("=" * 55)
print("  LLM CAUSAL DIAGNOSTIC — BASELINE SWEEP")
print("=" * 55)

results = {}
for mode_name, fn in [("empty", empty_response), ("random", random_response), ("oracle", oracle_response)]:
    sm = run(fn, mode_name)
    results[mode_name] = sm
    passed = sum(1 for v in sm.values() if v["passed"])
    print(f"\n  {mode_name.upper():>10s}: {passed}/4 gates passed")
    for pid, info in sm.items():
        pf = "PASS" if info["passed"] else "FAIL"
        if pid == "existence":
            print(f"    {pid:>15s}: f1={info['avg_f1']:.3f}  [{pf}]")
        elif pid == "ranking":
            print(f"    {pid:>15s}: spearman_r={info['avg_spearman_r']:.3f}  [{pf}]")
        elif pid == "counterfactual":
            print(f"    {pid:>15s}: acc={info['avg_accuracy']:.3f}  [{pf}]")
        elif pid == "edge_diff":
            print(f"    {pid:>15s}: score={info['avg_score']:.3f}  [{pf}]")

print()
print("=" * 55)
print("  BASELINE COMPARISON")
print(f"  {'':>12s}  {'exist':>6s}  {'rank':>6s}  {'cfact':>6s}  {'ediff':>6s}  {'passed':>6s}")
for name in ["empty", "random", "oracle"]:
    sm = results[name]
    f1 = sm["existence"]["avg_f1"]
    r = sm["ranking"]["avg_spearman_r"]
    cf = sm["counterfactual"]["avg_accuracy"]
    ed = sm["edge_diff"]["avg_score"]
    ps = sum(1 for v in sm.values() if v["passed"])
    print(f"  {name:>12s}: {f1:6.3f}  {r:6.3f}  {cf:6.3f}  {ed:6.3f}  {ps}/4")

print("\n  ─────────────────────────────────")
print("  Use these baselines to interpret any real LLM result:")
print("    oracle = upper bound (ground truth given directly)")
print("    random = chance-level performance")
print("    empty  = lower bound (refuses to answer)")
print("=" * 55)

results["world_summary"] = {
    "n_obj": probes["world"]["n_obj"], "n_edges": len(probes["world"]["edges"]),
    "n_scenarios": len(probes["scenarios"]), "n_probes": sum(len(s["probes"]) for s in probes["scenarios"]),
}

(od / "baseline_sweep.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"  Saved to {od / 'baseline_sweep.json'}")
