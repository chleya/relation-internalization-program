from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

plos_dir = Path(__file__).resolve().parent

PROBE_DIR = plos_dir / "results" / "llm_o1_probe"


def load_data():
    probes = json.loads((PROBE_DIR / "probes.json").read_text(encoding="utf-8"))
    gt = json.loads((PROBE_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    gt_by_id = {g["episode_id"]: g["ground_truth"] for g in gt}
    return probes, gt_by_id


def score_object_identity(answer, truth):
    true_n = truth["n_objects"]
    true_ids = set(truth["object_ids"])
    if isinstance(answer, dict):
        predicted_n = answer.get("n_objects", 0)
        predicted_ids = set(answer.get("object_ids", []))
        n_match = 1.0 if predicted_n == true_n else 0.0
        id_match = len(true_ids & predicted_ids) / len(true_ids) if true_ids else 0.0
        return 0.5 * n_match + 0.5 * id_match
    return 0.0


def score_interaction_detection(answer, truth):
    risk = truth.get("risk_ranking", [])
    has_interaction = len(risk) > 0 and any(r[1] > 0.5 for r in risk[-5:])
    if isinstance(answer, dict):
        predicted_has = answer.get("has_interactions", False)
        return 1.0 if predicted_has == has_interaction else 0.0
    return 0.0


def score_risk_identification(answer, truth):
    risk = truth.get("risk_ranking", [])

    avg_risk = {}
    for obj_id, r in risk:
        avg_risk.setdefault(obj_id, []).append(r)
    avg_risk = {k: float(np.mean(v)) for k, v in avg_risk.items()}
    sorted_truth = sorted(avg_risk.keys(), key=lambda x: -avg_risk[x])

    if isinstance(answer, dict):
        predicted_ranking = answer.get("risk_ranking", [])
        if not predicted_ranking:
            return 0.0

        n_objs = truth["n_objects"]
        score = 0.0
        for i, pred_id in enumerate(predicted_ranking[:n_objs]):
            if pred_id in sorted_truth:
                pos = sorted_truth.index(pred_id)
                score += 1.0 - abs(i - pos) / max(n_objs, 1)
        return score / max(len(predicted_ranking[:n_objs]), 1)
    return 0.0


def score_direct_intervention(answer, truth):
    actionable = set(truth.get("directly_actionable", []))
    high_risk = set(truth.get("high_risk_objects", []))
    correct = actionable | high_risk
    if isinstance(answer, dict):
        chosen = answer.get("chosen_object", -1)
        if not correct:
            abstain_correct = answer.get("would_abstain", False)
            return 1.0 if abstain_correct else 0.0
        return 1.0 if chosen in correct else 0.0
    return 0.0


def score_indirect_intervention(answer, truth):
    indirect = set(truth.get("indirectly_actionable", []))
    if isinstance(answer, dict):
        has_indirect = answer.get("has_indirect_pathway", False)
        if not indirect:
            return 1.0 if not has_indirect else 0.0
        pathway = answer.get("pathway", {})
        target = pathway.get("target_object", -1)
        through = pathway.get("through_object", -1)
        target_correct = target in indirect
        return 1.0 if target_correct else 0.0
    return 0.0


def score_abstain_judgment(answer, truth):
    high_risk = truth.get("high_risk_objects", [])
    all_very_high_risk = len(high_risk) > 0 and len(high_risk) == truth["n_objects"]
    should_abstain = all_very_high_risk

    if isinstance(answer, dict):
        would_abstain = answer.get("would_abstain", "sometimes")
        if would_abstain == "always":
            return 1.0 if should_abstain else 0.3
        elif would_abstain == "sometimes":
            return 0.7
        elif would_abstain == "never":
            return 0.0 if should_abstain else 0.7
    return 0.0


def score_prediction(answer, truth):
    risk = truth.get("risk_ranking", [])
    n_objs = truth["n_objects"]
    many_steps = len(risk)
    if many_steps < 3:
        return 0.0

    last_risks = {}
    for obj_id, r in risk[-3:]:
        last_risks.setdefault(obj_id, []).append(r)
    avg_last = {k: float(np.mean(v)) for k, v in last_risks.items()}

    if isinstance(answer, dict):
        predicted_risks = answer.get("predicted_risks", {})
        errors = []
        for obj_id in range(n_objs):
            true_risk = avg_last.get(obj_id, 0.3)
            pred_risk = predicted_risks.get(str(obj_id), 0.3)
            errors.append(abs(float(pred_risk) - true_risk))
        avg_error = float(np.mean(errors)) if errors else 1.0
        return max(0.0, 1.0 - avg_error)
    return 0.0


SCORERS = {
    "object_identity": score_object_identity,
    "interaction_detection": score_interaction_detection,
    "risk_identification": score_risk_identification,
    "direct_intervention": score_direct_intervention,
    "indirect_intervention": score_indirect_intervention,
    "abstain_judgment": score_abstain_judgment,
    "prediction": score_prediction,
}


def random_baseline_answer(question_id, truth):
    n_o = truth["n_objects"]
    cond = truth.get("condition", "")
    seed_val = hash(question_id) + (ord(cond[0]) if cond else 0)
    rng = random.Random(seed_val)

    base = {}
    if question_id == "object_identity":
        base = {"n_objects": rng.choice([2, 3, 4, 5]),
                "object_ids": list(range(rng.choice([2, 3, 4])))}
    elif question_id == "interaction_detection":
        base = {"has_interactions": rng.choice([True, False])}
    elif question_id == "risk_identification":
        ids = list(range(n_o))
        rng.shuffle(ids)
        base = {"risk_ranking": ids}
    elif question_id == "direct_intervention":
        base = {"chosen_object": rng.randint(0, n_o - 1),
                "would_abstain": rng.choice([True, False])}
    elif question_id == "indirect_intervention":
        base = {"has_indirect_pathway": rng.choice([True, False]),
                "pathway": {"target_object": rng.randint(0, n_o - 1),
                           "through_object": rng.randint(0, n_o - 1)}}
    elif question_id == "abstain_judgment":
        base = {"would_abstain": rng.choice(["always", "sometimes", "never"])}
    elif question_id == "prediction":
        base = {"predicted_risks": {str(i): rng.random() for i in range(n_o)}}
    return base


def rule_baseline_answer(question_id, truth, trajectory_text):
    n_o = truth["n_objects"]
    risk = truth.get("risk_ranking", [])

    avg_r = {}
    for obj_id, r_val in risk:
        avg_r.setdefault(obj_id, []).append(r_val)
    avg_r = {k: float(np.mean(v)) for k, v in avg_r.items()}
    sorted_risk = sorted(avg_r.keys(), key=lambda x: -avg_r[x])
    high_risk_objs = [k for k, v in avg_r.items() if v > 0.5]

    base = {}
    if question_id == "object_identity":
        base = {"n_objects": n_o, "object_ids": list(range(n_o))}
    elif question_id == "interaction_detection":
        has_int = any(v > 0.5 for v in avg_r.values())
        base = {"has_interactions": has_int}
    elif question_id == "risk_identification":
        base = {"risk_ranking": sorted_risk}
    elif question_id == "direct_intervention":
        actionable = truth.get("directly_actionable", [])
        if not high_risk_objs and not actionable:
            base = {"would_abstain": True, "chosen_object": -1}
        else:
            candidates = list(set(actionable) | set(high_risk_objs))
            base = {"chosen_object": candidates[0] if candidates else 0,
                    "would_abstain": False}
    elif question_id == "indirect_intervention":
        indirect = truth.get("indirectly_actionable", [])
        base = {"has_indirect_pathway": len(indirect) > 0,
                "pathway": {"target_object": indirect[0] if indirect else -1,
                           "through_object": 0}}
    elif question_id == "abstain_judgment":
        all_high = len(high_risk_objs) == n_o and n_o > 1
        base = {"would_abstain": "always" if all_high else "sometimes"}
    elif question_id == "prediction":
        base = {"predicted_risks": {str(i): avg_r.get(i, 0.3) for i in range(n_o)}}
    return base


print("=== Line 3: LLM O1 Probe Scorer ===")
probes, gt_by_id = load_data()

question_ids = [q["id"] for q in probes["probe_questions"]]
print(f"  Episodes: {len(probes['episodes'])}")
print(f"  Questions: {len(question_ids)}  ({', '.join(question_ids)})")

random_scores = {q: [] for q in question_ids}
rule_scores = {q: [] for q in question_ids}

for ep_data in probes["episodes"]:
    eid = ep_data["episode_id"]
    truth = gt_by_id[eid]
    traj_text = ep_data.get("trajectory_text", "")
    for qid in question_ids:
        rand_ans = random_baseline_answer(qid, truth)
        rule_ans = rule_baseline_answer(qid, truth, traj_text)

        scorer = SCORERS[qid]
        random_scores[qid].append(scorer(rand_ans, truth))
        rule_scores[qid].append(scorer(rule_ans, truth))

print("\n  Per-question scores:")
print(f"  {'Question':<25s} {'Random':>8s} {'Rule':>8s} {'Gap':>8s}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")

total_rand = 0.0
total_rule = 0.0
results = {}
for qid in question_ids:
    r_avg = float(np.mean(random_scores[qid]))
    ru_avg = float(np.mean(rule_scores[qid]))
    gap = ru_avg - r_avg
    total_rand += r_avg
    total_rule += ru_avg
    results[qid] = {"random": r_avg, "rule": ru_avg, "gap": gap}
    print(f"  {qid:<25s} {r_avg:8.4f} {ru_avg:8.4f} {gap:8.4f}")

avg_rand = total_rand / len(question_ids)
avg_rule = total_rule / len(question_ids)
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'OVERALL':<25s} {avg_rand:8.4f} {avg_rule:8.4f} {avg_rule - avg_rand:8.4f}")

print(f"\n  === O1 PROBE BASELINE ===")
print(f"  o1_random_baseline    = {avg_rand:.4f}")
print(f"  o1_rule_baseline      = {avg_rule:.4f}")
print(f"  o1_rule_over_random   = {avg_rule - avg_rand:.4f}")

metrics = {
    "o1_random_baseline": avg_rand,
    "o1_rule_baseline": avg_rule,
    "o1_rule_over_random": avg_rule - avg_rand,
    "per_question": results,
}

out_dir = plos_dir / "results" / "llm_o1_scorer"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "baselines.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"  Saved to {out_dir}")
print("=== DONE ===")
