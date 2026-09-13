from __future__ import annotations

import json, math, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from g_line.gn_model import _init_weights

plos_dir = Path(__file__).resolve().parent

N_OBJ = 6; N_FEAT = 2
SDECAY = 0.02; NSTD = 0.01; AMB = 0.5
NAMES = ["水箱A", "水箱B", "水箱C", "水箱D", "水箱E", "水箱F"]


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


def traj_to_text(traj, namelen=20):
    true_temps = traj[:, :, 0]
    lines = []
    for t in range(min(namelen, true_temps.shape[0])):
        vals = "  ".join(f"{NAMES[i]}={true_temps[t, i]:.3f}" for i in range(N_OBJ))
        lines.append(f"时刻{t}: {vals}")
    return "\n".join(lines)


def make_existence_probe(traj_text, edges, coef):
    edge_list = []
    for s, t in edges:
        edge_list.append(f"{NAMES[s]}→{NAMES[t]} (系数={coef[s,t]:.4f})")
    question = (
        f"以下是6个水箱在20个时刻的温度变化记录：\n\n{traj_text}\n\n"
        f"请列出所有存在影响关系的水箱对（格式：A→B 表示A影响B）。\n"
        f"只列出你确定存在的，不要猜测。"
    )
    return {
        "id": "existence",
        "question": question,
        "ground_truth": {
            "edges": [[int(s), int(t), float(coef[s, t])] for s, t in edges],
            "n_true": len(edges),
        }
    }


def make_ranking_probe(traj_text, edges, coef):
    ranked = sorted(edges, key=lambda e: coef[e[0], e[1]], reverse=True)
    question = (
        f"以下是6个水箱在20个时刻的温度变化记录：\n\n{traj_text}\n\n"
        f"请按影响强度从大到小排列所有水箱间的关系。"
    )
    return {
        "id": "ranking",
        "question": question,
        "ground_truth": {
            "ranked": [[int(s), int(t), float(coef[s, t])] for s, t in ranked],
        }
    }


def make_counterfactual_probe(traj_text, edges, coef):
    question = (
        f"以下是6个水箱在20个时刻的温度变化记录：\n\n{traj_text}\n\n"
        f"如果水箱D的温度突然升高0.3度，接下来5个时刻哪个水箱的温度变化最大？\n"
        f"请给出你的推理过程和答案。"
    )
    downstream = [t for s, t in edges if s == 3]
    gt = NAMES[max(downstream, key=lambda x: coef[3, x])] if downstream else "无"
    return {
        "id": "counterfactual",
        "question": question,
        "ground_truth": {"answer": gt, "reasoning": f"D直接影响{', '.join(NAMES[t] for t in downstream)}"},
    }


def make_edge_diff_probe(traj_text, edges, coef):
    c_edges = [(s, t) for s, t in edges if s == 2]
    c_ranked = sorted(c_edges, key=lambda e: coef[e[0], e[1]], reverse=True)
    c_names = ", ".join(NAMES[t] for _, t in c_edges)
    question = (
        f"以下是6个水箱在20个时刻的温度变化记录：\n\n{traj_text}\n\n"
        f"水箱C同时影响{c_names}。它对这几个水箱的影响有什么不同？\n"
        f"哪个受影响最大，哪个受影响最小？请给出推理。"
    )
    return {
        "id": "edge_diff",
        "question": question,
        "ground_truth": {
            "ranked": [[int(s), int(t), float(coef[s, t])] for s, t in c_ranked],
            "max_target": NAMES[c_ranked[0][1]],
            "min_target": NAMES[c_ranked[-1][1]],
        }
    }


print("=" * 55)
print("  LLM CAUSAL DIAGNOSTIC PROBE GENERATOR")
print("=" * 55)

binary, coef = make_pairwise_graph(42)
n_true = int(binary.sum())
edges = [(s, t) for s in range(N_OBJ) for t in range(N_OBJ) if binary[s, t]]

print(f"  True edges: {n_true}")
for s, t in edges:
    tag = "S" if coef[s, t] >= 0.35 else "w"
    print(f"    {s}->{t}: coef={coef[s, t]:.4f} ({tag})")

world_info = {
    "objects": NAMES,
    "edges": [{"src": int(s), "tgt": int(t), "coef": float(coef[s, t])} for s, t in edges],
    "n_obj": N_OBJ,
}

scenarios = []
for si in range(5):
    seed = si * 100 + 7
    traj = gen_traj_pair(binary, coef, 22, seed)
    traj_text = traj_to_text(traj)
    print(f"\n  Scenario {si + 1} (seed={seed}):")
    probes = [
        make_existence_probe(traj_text, edges, coef),
        make_ranking_probe(traj_text, edges, coef),
        make_counterfactual_probe(traj_text, edges, coef),
        make_edge_diff_probe(traj_text, edges, coef),
    ]
    sc = {
        "seed": seed,
        "trajectory": traj[:, :, 0].tolist(),
        "trajectory_text": traj_text,
        "probes": probes,
    }
    scenarios.append(sc)
    print(f"    {len(probes)} probes generated")

probes_data = {"world": world_info, "scenarios": scenarios}

ground_truth = {
    "scoring": {
        "existence": {
            "rule": "precision = |predicted_edges ∩ true_edges| / |predicted_edges|; recall = |predicted_edges ∩ true_edges| / |true_edges|",
            "score_range": [0.0, 1.0],
        },
        "ranking": {
            "rule": "Spearman rank correlation between predicted rank and true coefficient rank",
            "score_range": [-1.0, 1.0],
        },
        "counterfactual": {
            "rule": "1.0 if answer matches ground truth, 0.0 otherwise",
            "score_range": [0.0, 1.0],
        },
        "edge_diff": {
            "rule": "1.0 if max/min targets are correct and order matches, 0.5 if partially correct, 0.0 if wrong",
            "score_range": [0.0, 1.0],
        },
    },
    "pass_threshold": {
        "existence": 0.6,
        "ranking_spearman": 0.4,
        "counterfactual": 0.5,
        "edge_diff": 0.5,
    },
}

od = plos_dir / "results" / "llm_causal"; od.mkdir(parents=True, exist_ok=True)
(od / "probes.json").write_text(json.dumps(probes_data, indent=2, ensure_ascii=False), encoding="utf-8")
(od / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False), encoding="utf-8")

total_probes = len(scenarios) * 4
print(f"\n  Generated {len(scenarios)} scenarios × 4 probe types = {total_probes} probes")
print(f"  Saved to {od}")
print(f"\n  Next step: feed probes to LLM API and evaluate with a scorer script")
