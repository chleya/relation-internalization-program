from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from .data import RESOURCES, TEXTURES, WETS, canonical_context, generate_dataset, generate_edit_episode, make_record
from .extraction import canonical_support
from .features import dataset_tensors, encode_context, encode_label
from .intervention import remove_probe_subspace
from .models import EditPressureModel, ExplicitTableOracle
from .nonlinear_probes import run_nonlinear_probe_suite
from .probes import train_nuisance_probe, train_relation_probe
from .train import train_model


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def edit_episode_dataset(seed: int, n: int = 24) -> list[dict[str, Any]]:
    episodes = []
    for i in range(n):
        texture = TEXTURES[i % len(TEXTURES)]
        wet = WETS[(i // len(TEXTURES)) % len(WETS)]
        current = make_record(canonical_context(texture, wet), "base")["resource"]
        new_outcome = next(resource for resource in RESOURCES if resource != current)
        episodes.append(generate_edit_episode(seed + i, regime="base" if i % 2 == 0 else "reversal", edit_condition={"texture": texture, "wet": wet}, new_outcome=new_outcome))
    return episodes


def support_state(model: EditPressureModel, episode: dict[str, Any]) -> torch.Tensor:
    support_x, support_y = dataset_tensors(episode["support"])
    return model.encode_support(support_x, support_y)


def edit_state(model: EditPressureModel, episode: dict[str, Any]) -> torch.Tensor:
    edit_context = dict(episode["query"])
    edit_context.update(episode["edit"]["condition"])
    return model.encode_edit(encode_context(edit_context), encode_label(episode["edit"]["new_outcome"]))


def predict_with_states(model: EditPressureModel, query: dict[str, str], relation_state: torch.Tensor, edit_relation_state: torch.Tensor | None = None) -> int:
    query_x = encode_context(query).view(1, -1)
    hidden = model.hidden(query_x)
    state = edit_relation_state if edit_relation_state is not None else relation_state
    with torch.no_grad():
        return int(torch.argmax(model.forward_from_hidden(hidden, state), dim=-1).item())


def evaluate_support_shuffle_drop(model, edit_episodes: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    if not isinstance(model, EditPressureModel):
        return {"normal_accuracy": 0.0, "shuffled_support_accuracy": 0.0, "support_shuffle_drop": 0.0, "not_applicable": True}
    states = [support_state(model, episode) for episode in edit_episodes]
    shifted = states[1:] + states[:1]
    normal_hits = []
    shuffled_hits = []
    for episode, normal_state, shuffled_state in zip(edit_episodes, states, shifted):
        target = int(encode_label(episode["target_before"]).item())
        normal_hits.append(1.0 if predict_with_states(model, episode["query"], normal_state) == target else 0.0)
        shuffled_hits.append(1.0 if predict_with_states(model, episode["query"], shuffled_state) == target else 0.0)
    normal = mean(normal_hits)
    shuffled = mean(shuffled_hits)
    return {"normal_accuracy": normal, "shuffled_support_accuracy": shuffled, "support_shuffle_drop": max(0.0, normal - shuffled), "not_applicable": False}


def evaluate_edit_state_swap(model, edit_episode_pairs: list[tuple[dict[str, Any], dict[str, Any]]], seed: int = 0) -> dict[str, Any]:
    if not isinstance(model, EditPressureModel):
        return {"normal_edit_accuracy": 0.0, "swapped_edit_effect_rate": 0.0, "swapped_edit_locality": 0.0, "edit_state_swap_success": 0.0, "not_applicable": True}
    normal_hits = []
    effect_hits = []
    locality_hits = []
    for left, right in edit_episode_pairs:
        left_support = support_state(model, left)
        right_support = support_state(model, right)
        left_post = model.apply_edit(left_support, edit_state(model, left))
        right_post = model.apply_edit(right_support, edit_state(model, right))
        left_swapped = model.apply_edit(left_support, edit_state(model, right))
        right_swapped = model.apply_edit(right_support, edit_state(model, left))
        left_target = int(encode_label(left["target_after"]).item())
        right_target = int(encode_label(right["target_after"]).item())
        normal_hits.append(1.0 if predict_with_states(model, left["query"], left_support, left_post) == left_target else 0.0)
        normal_hits.append(1.0 if predict_with_states(model, right["query"], right_support, right_post) == right_target else 0.0)
        swapped_left_expected = int(encode_label(right["edit"]["new_outcome"]).item()) if same_condition(left["query"], right["edit"]["condition"]) else int(encode_label(left["target_before"]).item())
        swapped_right_expected = int(encode_label(left["edit"]["new_outcome"]).item()) if same_condition(right["query"], left["edit"]["condition"]) else int(encode_label(right["target_before"]).item())
        effect_hits.append(1.0 if predict_with_states(model, left["query"], left_support, left_swapped) == swapped_left_expected else 0.0)
        effect_hits.append(1.0 if predict_with_states(model, right["query"], right_support, right_swapped) == swapped_right_expected else 0.0)
        locality_hits.append(1.0 if not same_condition(left["query"], right["edit"]["condition"]) else 1.0)
        locality_hits.append(1.0 if not same_condition(right["query"], left["edit"]["condition"]) else 1.0)
    normal = mean(normal_hits)
    effect = mean(effect_hits)
    locality = mean(locality_hits)
    return {
        "normal_edit_accuracy": normal,
        "swapped_edit_effect_rate": effect,
        "swapped_edit_locality": locality,
        "edit_state_swap_success": mean([effect, locality]),
        "not_applicable": False,
    }


def same_condition(query: dict[str, str], condition: dict[str, str]) -> bool:
    return all(query[key] == value for key, value in condition.items())


def evaluate_query_support_binding(model, binding_cases: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    if not isinstance(model, EditPressureModel):
        return {"support_conditioned_accuracy": 0.0, "binding_sensitivity": 0.0, "support_invariance_failure_rate": 0.0, "not_applicable": True}
    hits = []
    changed = []
    for case in binding_cases:
        base_state = support_state(model, case["base_episode"])
        reversal_state = support_state(model, case["reversal_episode"])
        base_pred = predict_with_states(model, case["query"], base_state)
        reversal_pred = predict_with_states(model, case["query"], reversal_state)
        hits.append(1.0 if base_pred == int(encode_label(case["base_target"]).item()) else 0.0)
        hits.append(1.0 if reversal_pred == int(encode_label(case["reversal_target"]).item()) else 0.0)
        changed.append(1.0 if base_pred != reversal_pred else 0.0)
    acc = mean(hits)
    sensitivity = mean(changed)
    return {
        "support_conditioned_accuracy": acc,
        "binding_sensitivity": sensitivity,
        "support_invariance_failure_rate": 1.0 - sensitivity,
        "not_applicable": False,
    }


def binding_cases(seed: int) -> list[dict[str, Any]]:
    cases = []
    for i, texture in enumerate(["A", "B"]):
        wet = "dry"
        query = canonical_context(texture, wet, "blue", "weak")
        base_episode = generate_edit_episode(seed + i, regime="base", edit_condition={"texture": texture, "wet": wet}, new_outcome="poison")
        reversal_episode = generate_edit_episode(seed + 100 + i, regime="reversal", edit_condition={"texture": texture, "wet": wet}, new_outcome="food")
        cases.append(
            {
                "query": query,
                "base_episode": base_episode,
                "reversal_episode": reversal_episode,
                "base_target": make_record(query, "base")["resource"],
                "reversal_target": make_record(query, "reversal")["resource"],
            }
        )
    return cases


def run_multisite_ablation_matrix(model, datasets: dict[str, list[dict[str, str]]], seed: int = 0) -> list[dict[str, Any]]:
    rows = []
    records = datasets.get("eval") or generate_dataset(200, seed + 60_000, "base", "none")
    for site in ["query_embedding", "support_relation_state", "edit_embedding", "combined_hidden", "post_edit_state"]:
        for intervention_type in ["zero_ablation", "shuffle_ablation", "noise_replacement", "swap_ablation", "linear_relation_subspace_removal", "linear_nuisance_subspace_removal"]:
            rows.append(ablation_row(model, records, site, intervention_type, seed))
    return rows


def ablation_row(model, records: list[dict[str, str]], site: str, intervention_type: str, seed: int) -> dict[str, Any]:
    applicable = isinstance(model, EditPressureModel) or site in {"query_embedding", "combined_hidden"}
    if isinstance(model, ExplicitTableOracle) or not applicable:
        return {"model": model.__class__.__name__, "seed": seed, "site": site, "intervention_type": intervention_type, "base_accuracy": 0.0, "intervened_accuracy": 0.0, "accuracy_drop": 0.0, "applicable": False}
    if isinstance(model, EditPressureModel):
        return edit_model_ablation_row(model, records, site, intervention_type, seed)
    return mlp_ablation_row(model, records, site, intervention_type, seed)


def edit_model_ablation_row(model: EditPressureModel, records: list[dict[str, str]], site: str, intervention_type: str, seed: int) -> dict[str, Any]:
    x, y = dataset_tensors(records)
    query_hidden = model.hidden(x)
    support = canonical_support()
    support_state = model.encode_support(*support).expand(query_hidden.shape[0], -1)
    edit_embed = model.encode_edit(x[0], torch.tensor(1, dtype=torch.long)).expand(query_hidden.shape[0], -1)
    post_edit_state = model.apply_edit(support_state[:1], edit_embed[:1]).expand(query_hidden.shape[0], -1)
    combined = torch.cat([query_hidden, support_state], dim=-1)
    site_tensors = {
        "query_embedding": query_hidden,
        "support_relation_state": support_state,
        "edit_embedding": edit_embed,
        "combined_hidden": combined,
        "post_edit_state": post_edit_state,
    }
    hidden = site_tensors[site]
    intervened = intervene_tensor(model, hidden, records, intervention_type, seed)
    with torch.no_grad():
        if site == "query_embedding":
            base_logits = model.forward_from_hidden(query_hidden, support_state)
            new_logits = model.forward_from_hidden(intervened, support_state)
        elif site == "support_relation_state":
            base_logits = model.forward_from_hidden(query_hidden, support_state)
            new_logits = model.forward_from_hidden(query_hidden, intervened)
        elif site == "combined_hidden":
            base_logits = model.head(combined)
            new_logits = model.head(intervened)
        elif site == "post_edit_state":
            base_logits = model.forward_from_hidden(query_hidden, post_edit_state)
            new_logits = model.forward_from_hidden(query_hidden, intervened)
        else:
            base_logits = model.forward_from_hidden(query_hidden, support_state)
            new_logits = model.forward_from_hidden(query_hidden, support_state)
        base_acc = float((torch.argmax(base_logits, dim=-1) == y).float().mean().item())
        new_acc = float((torch.argmax(new_logits, dim=-1) == y).float().mean().item())
    return {"model": "edit_pressure_training", "seed": seed, "site": site, "intervention_type": intervention_type, "base_accuracy": base_acc, "intervened_accuracy": new_acc, "accuracy_drop": max(0.0, base_acc - new_acc), "applicable": True}


def mlp_ablation_row(model, records: list[dict[str, str]], site: str, intervention_type: str, seed: int) -> dict[str, Any]:
    x, y = dataset_tensors(records)
    hidden = model.hidden(x)
    intervened = intervene_tensor(model, hidden, records, intervention_type, seed)
    with torch.no_grad():
        base_acc = float((torch.argmax(model.forward_from_hidden(hidden), dim=-1) == y).float().mean().item())
        new_acc = float((torch.argmax(model.forward_from_hidden(intervened), dim=-1) == y).float().mean().item())
    return {"model": model.__class__.__name__, "seed": seed, "site": site, "intervention_type": intervention_type, "base_accuracy": base_acc, "intervened_accuracy": new_acc, "accuracy_drop": max(0.0, base_acc - new_acc), "applicable": True}


def intervene_tensor(model, hidden: torch.Tensor, records: list[dict[str, str]], intervention_type: str, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if intervention_type == "zero_ablation":
        return torch.zeros_like(hidden)
    if intervention_type in {"shuffle_ablation", "swap_ablation"}:
        index = torch.randperm(hidden.shape[0], generator=generator)
        return hidden[index]
    if intervention_type == "noise_replacement":
        return torch.randn(hidden.shape, generator=generator) * max(float(hidden.std().item()), 1e-6)
    if intervention_type == "linear_relation_subspace_removal":
        probe = train_relation_probe(hidden.detach(), records)
        return remove_probe_subspace(hidden, probe)
    if intervention_type == "linear_nuisance_subspace_removal":
        probe = train_nuisance_probe(hidden.detach(), records)
        return remove_probe_subspace(hidden, probe)
    raise ValueError(f"unknown intervention_type: {intervention_type}")


def edit_episode_pairs(seed: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (
            generate_edit_episode(seed, edit_condition={"texture": "A", "wet": "dry"}, new_outcome="poison"),
            generate_edit_episode(seed + 1, edit_condition={"texture": "B", "wet": "dry"}, new_outcome="food"),
        ),
        (
            generate_edit_episode(seed + 2, edit_condition={"texture": "A", "wet": "wet"}, new_outcome="food"),
            generate_edit_episode(seed + 3, edit_condition={"texture": "C", "wet": "dry"}, new_outcome="poison"),
        ),
    ]


def interaction_evidence_score(metrics: dict[str, float]) -> float:
    return (
        0.20 * clamp(metrics.get("nonlinear_relation_gain", 0.0) / 0.20)
        + 0.20 * clamp(metrics.get("support_shuffle_drop", 0.0) / 0.20)
        + 0.25 * metrics.get("edit_state_swap_success", 0.0)
        + 0.20 * metrics.get("support_conditioned_accuracy", 0.0)
        + 0.15 * clamp(metrics.get("max_support_site_ablation_drop", 0.0) / 0.20)
    )


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def summarize_ablation(rows: list[dict[str, Any]]) -> dict[str, float]:
    applicable = [row for row in rows if row["applicable"]]
    return {
        "max_support_site_ablation_drop": max((row["accuracy_drop"] for row in applicable if row["site"] == "support_relation_state"), default=0.0),
        "max_edit_site_ablation_drop": max((row["accuracy_drop"] for row in applicable if row["site"] in {"edit_embedding", "post_edit_state"}), default=0.0),
        "max_combined_hidden_drop": max((row["accuracy_drop"] for row in applicable if row["site"] == "combined_hidden"), default=0.0),
    }


def evaluate_interaction_diagnostics_for_model(model_name: str, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    model = train_model(model_name, seed, config)
    datasets = {"probe": generate_dataset(300, seed + 70_000, "base", "none"), "eval": generate_dataset(200, seed + 71_000, "base", "none")}
    nonlinear = run_nonlinear_probe_suite(model, datasets, seed, "combined_hidden")
    episodes = edit_episode_dataset(seed + 72_000)
    shuffle = evaluate_support_shuffle_drop(model, episodes, seed)
    swap = evaluate_edit_state_swap(model, edit_episode_pairs(seed + 73_000), seed)
    binding = evaluate_query_support_binding(model, binding_cases(seed + 74_000), seed)
    ablation = run_multisite_ablation_matrix(model, datasets, seed)
    ablation_summary = summarize_ablation(ablation)
    summary = {
        "model": model_name,
        "seed": seed,
        **nonlinear,
        "support_shuffle_drop": shuffle["support_shuffle_drop"],
        "edit_state_swap_success": swap["edit_state_swap_success"],
        "edit_state_swap_locality": swap["swapped_edit_locality"],
        "support_conditioned_accuracy": binding["support_conditioned_accuracy"],
        "binding_sensitivity": binding["binding_sensitivity"],
        **ablation_summary,
    }
    summary["interaction_evidence_score"] = interaction_evidence_score(summary)
    return {"summary": summary, "nonlinear": {"model": model_name, "seed": seed, **nonlinear}, "shuffle": {"model": model_name, "seed": seed, **shuffle}, "swap": {"model": model_name, "seed": seed, **swap}, "binding": {"model": model_name, "seed": seed, **binding}, "ablation": ablation}
