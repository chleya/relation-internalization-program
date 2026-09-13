from __future__ import annotations

import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

plos_dir = Path(__file__).resolve().parent
probes_path = plos_dir / "results" / "llm_causal" / "probes.json"
gt_path = plos_dir / "results" / "llm_causal" / "ground_truth.json"

probes = json.loads(probes_path.read_text(encoding="utf-8"))
ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))

NAMES = probes["world"]["objects"]
TRUE_EDGES = {(e["src"], e["tgt"]): e["coef"] for e in probes["world"]["edges"]}
N_TRUE = len(TRUE_EDGES)


def call_llm(prompt, base_url="http://127.0.0.1:8083", model="local", timeout=120):
    import requests
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.0, "max_tokens": 512},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def parse_edges_from_text(text):
    found = []
    for m in re.finditer(r'(\w+)\s*[-→>]+\s*(\w+)', text):
        src_name, tgt_name = m.group(1), m.group(2)
        if src_name in NAMES and tgt_name in NAMES:
            si = NAMES.index(src_name); ti = NAMES.index(tgt_name)
            if si != ti and (si, ti) not in found:
                found.append((si, ti))
    return found


def score_existence(response, probe_gt):
    pred = parse_edges_from_text(response)
    true_edges_set = set((e[0], e[1]) for e in probe_gt["edges"])
    tp = len(set(pred) & true_edges_set)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true_edges_set) if true_edges_set else 0.0
    return {"precision": precision, "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall > 0 else 0.0,
            "tp": tp, "n_pred": len(pred), "n_true": len(true_edges_set)}


def score_ranking(response, probe_gt):
    pred = parse_edges_from_text(response)
    if len(pred) < 2:
        return {"spearman_r": 0.0, "n_ranked": len(pred), "msg": "too few edges to rank"}
    true_rank = {(e[0], e[1]): i for i, e in enumerate(probe_gt["ranked"])}
    pred_ranks = []
    for pi, (s, t) in enumerate(pred):
        if (s, t) in true_rank:
            pred_ranks.append((pi, true_rank[(s, t)]))
    if len(pred_ranks) < 2:
        return {"spearman_r": 0.0, "n_matched": len(pred_ranks)}
    pr = np.array([p for p, _ in pred_ranks])
    tr = np.array([t for _, t in pred_ranks])
    pr_r = pr.argsort().argsort().astype(float)
    tr_r = tr.argsort().argsort().astype(float)
    if np.std(pr_r) < 1e-12 or np.std(tr_r) < 1e-12:
        r = 0.0
    else:
        r = np.corrcoef(pr_r, tr_r)[0, 1]
        r = 0.0 if np.isnan(r) else float(r)
    return {"spearman_r": r, "n_matched": len(pred_ranks)}


def score_counterfactual(response, probe_gt):
    answer = probe_gt["answer"]
    for name in NAMES:
        if name in response and name == answer:
            return {"correct": 1.0}
    if answer in response:
        return {"correct": 1.0}
    return {"correct": 0.0}


def score_edge_diff(response, probe_gt):
    max_tgt = probe_gt["max_target"]
    min_tgt = probe_gt["min_target"]
    score = 0.0
    if max_tgt in response:
        score += 0.5
    if min_tgt in response:
        score += 0.5
    return {"score": score, "max_correct": 1.0 if max_tgt in response else 0.0,
            "min_correct": 1.0 if min_tgt in response else 0.0}


SCORERS = {"existence": score_existence, "ranking": score_ranking,
           "counterfactual": score_counterfactual, "edge_diff": score_edge_diff}


def run_evaluation(llm_fn, probes_data):
    all_scores = {"existence": [], "ranking": [], "counterfactual": [], "edge_diff": []}
    all_raw = []
    for si, scenario in enumerate(probes_data["scenarios"]):
        for probe in scenario["probes"]:
            pid = probe["id"]
            print(f"  Scenario {si+1}/{len(probes_data['scenarios'])}  {pid} ...", end=" ", flush=True)
            try:
                response = llm_fn(probe["question"])
                scores = SCORERS[pid](response, probe["ground_truth"])
                all_scores[pid].append(scores)
                all_raw.append({"scenario": si, "probe_id": pid, "response": response, "scores": scores})
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")
                all_raw.append({"scenario": si, "probe_id": pid, "error": str(e)})
    return all_scores, all_raw


def summarize(scores):
    th = ground_truth["pass_threshold"]
    summary = {}
    for pid in ["existence", "ranking", "counterfactual", "edge_diff"]:
        slist = scores.get(pid, [])
        if not slist:
            summary[pid] = {"n": 0, "passed": False}
            continue
        if pid == "existence":
            avg_f1 = float(np.mean([s["f1"] for s in slist]))
            passed = avg_f1 >= th["existence"]
            summary[pid] = {"n": len(slist), "avg_f1": avg_f1, "threshold": th["existence"], "passed": passed}
        elif pid == "ranking":
            avg_r = float(np.mean([s["spearman_r"] for s in slist]))
            passed = avg_r >= th["ranking_spearman"]
            summary[pid] = {"n": len(slist), "avg_spearman_r": avg_r, "threshold": th["ranking_spearman"], "passed": passed}
        elif pid == "counterfactual":
            avg_acc = float(np.mean([s["correct"] for s in slist]))
            passed = avg_acc >= th["counterfactual"]
            summary[pid] = {"n": len(slist), "avg_accuracy": avg_acc, "threshold": th["counterfactual"], "passed": passed}
        elif pid == "edge_diff":
            avg_score = float(np.mean([s["score"] for s in slist]))
            passed = avg_score >= th["edge_diff"]
            summary[pid] = {"n": len(slist), "avg_score": avg_score, "threshold": th["edge_diff"], "passed": passed}
    return summary


print("=" * 55)
print("  LLM CAUSAL SCORER")
print("=" * 55)
print(f"  Loaded {len(probes['scenarios'])} scenarios with {sum(len(s['probes']) for s in probes['scenarios'])} probes")

API_URL = "http://127.0.0.1:8083"
API_MODEL = "local"

try:
    import requests
    requests.get(f"{API_URL}/v1/models", timeout=5)
    print(f"\n  LLM API detected at {API_URL}")
    print(f"  Running evaluation with model={API_MODEL}...")
    print()

    llm_fn = lambda p: call_llm(p, API_URL, API_MODEL)
    all_scores, all_raw = run_evaluation(llm_fn, probes)
    summary = summarize(all_scores)

except Exception as e:
    print(f"\n  No LLM API available at {API_URL}: {e}")
    print(f"  Writing manual evaluation template...")
    manual_path = plos_dir / "results" / "llm_causal" / "manual_responses_template.json"
    template = []
    for si, scenario in enumerate(probes["scenarios"]):
        for probe in scenario["probes"]:
            template.append({"scenario": si, "probe_id": probe["id"],
                            "question": probe["question"][:200] + "...",
                            "ground_truth_summary": str(probe["ground_truth"])[:200],
                            "llm_response": "PUT_LLM_RESPONSE_HERE"})
    manual_path.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Template saved to {manual_path}")
    print(f"  Fill in 'llm_response' fields, rename to manual_responses.json, then re-run with --manual")
    all_scores = None; summary = None; all_raw = None

if summary:
    print()
    print("=" * 55)
    print("  LLM CAUSAL DIAGNOSTIC RESULT")
    all_passed = True
    for pid, info in summary.items():
        if pid == "existence":
            flag = "PASS" if info["passed"] else "FAIL"
            if not info["passed"]: all_passed = False
            print(f"  {pid:>15s}: f1={info['avg_f1']:.3f}  (th={info['threshold']})  [{flag}]")
        elif pid == "ranking":
            flag = "PASS" if info["passed"] else "FAIL"
            if not info["passed"]: all_passed = False
            print(f"  {pid:>15s}: spearman_r={info['avg_spearman_r']:.3f}  (th={info['threshold']})  [{flag}]")
        elif pid == "counterfactual":
            flag = "PASS" if info["passed"] else "FAIL"
            if not info["passed"]: all_passed = False
            print(f"  {pid:>15s}: acc={info['avg_accuracy']:.3f}  (th={info['threshold']})  [{flag}]")
        elif pid == "edge_diff":
            flag = "PASS" if info["passed"] else "FAIL"
            if not info["passed"]: all_passed = False
            print(f"  {pid:>15s}: score={info['avg_score']:.3f}  (th={info['threshold']})  [{flag}]")
    final = "LLM HAS CONTENT-LEVEL EMERGENCE" if all_passed else "LLM LACKS CONTENT-LEVEL EMERGENCE"
    print(f"\n  VERDICT: {final}")
    print("=" * 55)

    od = plos_dir / "results" / "llm_causal"
    (od / "eval_results.json").write_text(json.dumps(
        {"summary": summary, "raw": all_raw, "verdict": final}, indent=2, ensure_ascii=False), encoding="utf-8")
    if all_raw:
        (od / "raw_responses.json").write_text(json.dumps(all_raw, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Results saved to {od}")
