import argparse
from pathlib import Path

import pandas as pd

from .baselines import DecisionTreeBaseline, FittingBaseline, MajorityBaseline, MemoryBaseline, PredictiveBaseline
from .evaluate import evaluate_all
from .relation_model import RelationInternalizationModel, RobustWideRelationInternalizationModel, WideRelationInternalizationModel
from .utils import load_yaml, write_json


def make_agent(name: str, config: dict, seed: int):
    if name == "majority":
        return MajorityBaseline()
    if name == "memory":
        return MemoryBaseline()
    if name == "fitting":
        return FittingBaseline(seed=seed)
    if name == "predictive":
        return PredictiveBaseline(seed=seed)
    if name == "decision_tree":
        return DecisionTreeBaseline(seed=seed)
    if name == "relation":
        return RelationInternalizationModel(seed=seed, **config["relation_model"])
    if name == "wide_relation":
        return WideRelationInternalizationModel(seed=seed, **config["relation_model"])
    if name == "robust_wide_relation":
        return RobustWideRelationInternalizationModel(seed=seed, **config["relation_model"])
    raise ValueError(f"Unknown agent: {name}")


def run(agent_name: str, config_path: str, seed: int) -> dict:
    config = load_yaml(config_path)
    config["seed"] = seed
    agent = make_agent(agent_name, config, seed)
    result = evaluate_all(agent, config)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    prefix = out_dir / f"{agent_name}_seed{seed}"
    write_json(prefix.with_name(prefix.name + "_metrics.json"), result["metrics"])
    write_json(prefix.with_name(prefix.name + "_relations.json"), {"relations": result["relations"]})
    pd.DataFrame(result["records"]).to_csv(prefix.with_name(prefix.name + "_records.csv"), index=False)
    return result["metrics"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--agent",
        required=True,
        choices=[
            "majority",
            "memory",
            "fitting",
            "predictive",
            "decision_tree",
            "relation",
            "wide_relation",
            "robust_wide_relation",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    metrics = run(args.agent, args.config, args.seed)
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
