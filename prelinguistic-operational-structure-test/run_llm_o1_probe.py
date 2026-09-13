from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from g_line import gn_world

plos_dir = Path(__file__).resolve().parent

print("=== LLM O1 Probe: Generating Test Episodes ===")

datasets = gn_world.make_continuous_datasets(
    n_objects=3, n_train=8, n_test=4, n_ood=4, seed=42,
)

all_eps = []
for cond, eps in datasets.items():
    all_eps.extend(eps)

PROBE_QUESTIONS = [
    {
        "id": "object_identity",
        "question": (
            "How many distinct objects are present in this world? "
            "List them by their object IDs. Describe what makes each object distinct."
        ),
        "evaluates": "object persistence / identity",
    },
    {
        "id": "interaction_detection",
        "question": (
            "Are there any physical interactions between objects? "
            "For each interaction, describe: which objects are involved, "
            "what type of interaction (collision, heat transfer, etc.), "
            "and at which time steps it occurs."
        ),
        "evaluates": "cross-object causal awareness",
    },
    {
        "id": "risk_identification",
        "question": (
            "Which object poses the highest risk? Why? "
            "Rank all objects by risk level from highest to lowest, "
            "with a brief justification for each."
        ),
        "evaluates": "risk perception / prioritization",
    },
    {
        "id": "direct_intervention",
        "question": (
            "If you could apply a local cooling/damping intervention to ONE object, "
            "which would you choose? Explain your reasoning. "
            "What would the expected outcome be?"
        ),
        "evaluates": "direct actionability recognition",
    },
    {
        "id": "indirect_intervention",
        "question": (
            "Is there any object you CANNOT directly intervene on "
            "but COULD influence indirectly through another object? "
            "Describe the indirect pathway: which object you would act on, "
            "and how the effect would propagate to the target."
        ),
        "evaluates": "indirect actionability recognition",
    },
    {
        "id": "abstain_judgment",
        "question": (
            "Is there any situation where you would choose NOT to intervene at all "
            "(abstain)? Under what conditions? Why?"
        ),
        "evaluates": "risk-constrained abstain reasoning",
    },
    {
        "id": "prediction",
        "question": (
            "Based on the observed trajectory, predict what will happen in the "
            "next 5 time steps for each object. Be specific about positions, "
            "speeds, and temperatures."
        ),
        "evaluates": "forward prediction from structure",
    },
]

SYSTEM_PROMPT = (
    "You are observing a 2D continuous physics world. "
    "You receive raw trajectory data: at each time step, each object has "
    "position (pos_x, pos_y normalized to 0-1), speed (normalized 0-1), "
    "temperature (normalized 0-1), and a computed risk score (0-1). "
    "Your task is to answer questions about the structure of this world "
    "based solely on the trajectory data. "
    "Do NOT invent information not present in the data. "
    "If you are uncertain, state your uncertainty explicitly."
)


def format_trajectory(ep: dict) -> str:
    history = ep["model_input"]["interaction_history"]
    obj_ids = sorted(set(int(hi["object_id"]) for hi in history))
    T = gn_world.STEPS_PER_EPISODE

    lines = []
    lines.append(f"Episode: {ep['episode_id']} | Condition: {ep['condition']} | Objects: {len(obj_ids)}")
    lines.append(f"Time steps: {T} | dt: {gn_world.DT}")
    lines.append("")

    header = "Step | " + " | ".join(
        f"Obj{oid}: pos_x pos_y speed temp risk" for oid in obj_ids
    )
    lines.append(header)
    lines.append("-" * len(header))

    for step in range(T):
        row_parts = [f"{step:4d}"]
        for oid in obj_ids:
            matches = [hi for hi in history
                       if int(hi["object_id"]) == oid and int(hi["step"]) == step]
            if matches:
                m = matches[0]
                row_parts.append(
                    f"{float(m['pos_x']):.3f} {float(m['pos_y']):.3f} "
                    f"{float(m['speed']):.3f} {float(m['temperature']):.3f} "
                    f"{float(m['risk']):.3f}"
                )
            else:
                row_parts.append("?.??? ?.??? ?.??? ?.??? ?.???")
        lines.append(" | ".join(row_parts))

    return "\n".join(lines)


def extract_ground_truth(ep: dict) -> dict:
    truth = ep["evaluator_ground_truth"]["regions"]
    obj_set = set()
    direct_set = set()
    indirect_set = set()
    high_risk = []
    low_risk = []

    for t in truth:
        oid = int(t["object_id"])
        obj_set.add(oid)
        if t.get("direct_actionable"):
            direct_set.add(oid)
        if t.get("indirect_actionable"):
            indirect_set.add(oid)
        risk = float(t.get("latent_risk", 0))
        if risk > gn_world.RISK_THRESHOLD:
            high_risk.append((oid, risk))
        else:
            low_risk.append((oid, risk))

    high_risk.sort(key=lambda x: -x[1])
    return {
        "n_objects": len(obj_set),
        "object_ids": sorted(obj_set),
        "directly_actionable": sorted(direct_set),
        "indirectly_actionable": sorted(indirect_set),
        "high_risk_objects": [oid for oid, _ in high_risk],
        "risk_ranking": [(oid, round(r, 3)) for oid, r in high_risk + low_risk],
        "condition": ep["condition"],
    }


probes = []
for ep in all_eps:
    trajectory_text = format_trajectory(ep)
    gt = extract_ground_truth(ep)
    entry = {
        "episode_id": ep["episode_id"],
        "condition": ep["condition"],
        "trajectory_text": trajectory_text,
        "ground_truth": gt,
        "questions": [],
    }
    for q in PROBE_QUESTIONS:
        entry["questions"].append({
            "question_id": q["id"],
            "question": q["question"],
            "evaluates": q["evaluates"],
        })
    probes.append(entry)

OUT = {
    "system_prompt": SYSTEM_PROMPT,
    "n_episodes": len(probes),
    "probe_questions": PROBE_QUESTIONS,
    "episodes": probes,
    "scoring_rubric": {
        "object_identity": "Count correct distinct objects. Score 0-1.",
        "interaction_detection": "Check if collision pairs mentioned. Score 0-1.",
        "risk_identification": "Compare risk ranking with ground truth. Score 0-1.",
        "direct_intervention": "Check if chosen object is in directly_actionable set. Score 0-1.",
        "indirect_intervention": "Check if described pathway exists in ground truth. Score 0-1.",
        "abstain_judgment": "Qualitative: does reasoning reference risk thresholds?",
        "prediction": "Compare predicted trajectory with actual continuation. MSE.",
    },
}

out_dir = plos_dir / "results" / "llm_o1_probe"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "probes.json").write_text(json.dumps(OUT, indent=2, ensure_ascii=False), encoding="utf-8")

ground_truths = [{"episode_id": ep["episode_id"], "condition": ep["condition"],
                   "ground_truth": extract_ground_truth(ep)} for ep in all_eps]
(out_dir / "ground_truth.json").write_text(
    json.dumps(ground_truths, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"  Generated {len(probes)} episodes with {len(PROBE_QUESTIONS)} questions each")
print(f"  Saved to {out_dir}")
print(f"\n  To run: load probes.json, send each episode+question to LLM,")
print(f"  then score responses against ground_truth.json using rubric")
print("=== DONE ===")
