from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .model import hidden_states, predict_from_hidden


def train_probe(hidden: np.ndarray, labels: np.ndarray, seed: int = 0) -> LogisticRegression:
    probe = LogisticRegression(max_iter=1000, random_state=seed)
    probe.fit(hidden, labels)
    return probe


def probe_accuracy(probe: LogisticRegression, hidden: np.ndarray, labels: np.ndarray) -> float:
    return float(accuracy_score(labels, probe.predict(hidden)))


def random_label_control(hidden: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    probe = train_probe(hidden, shuffled, seed=seed)
    return probe_accuracy(probe, hidden, shuffled)


def top_probe_dimensions(probe: LogisticRegression, k: int = 4) -> np.ndarray:
    weights = np.abs(probe.coef_).sum(axis=0)
    return np.argsort(weights)[-k:]


def intervention_drop(model, x: np.ndarray, y: np.ndarray, dims: np.ndarray) -> float:
    hidden = hidden_states(model, x)
    base_pred = predict_from_hidden(model, hidden)
    edited = hidden.copy()
    edited[:, dims] = 0.0
    edited_pred = predict_from_hidden(model, edited)
    base_acc = float(accuracy_score(y, base_pred))
    edited_acc = float(accuracy_score(y, edited_pred))
    return base_acc - edited_acc


def random_intervention_drop(model, x: np.ndarray, y: np.ndarray, k: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    hidden = hidden_states(model, x)
    dims = rng.choice(hidden.shape[1], size=k, replace=False)
    return intervention_drop(model, x, y, dims)


def remove_probe_subspace(hidden: np.ndarray, probe: LogisticRegression) -> np.ndarray:
    weights = probe.coef_
    _, singular_values, vh = np.linalg.svd(weights, full_matrices=False)
    rank = int(np.sum(singular_values > 1e-8))
    if rank == 0:
        return hidden.copy()
    basis = vh[:rank].T
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    projected = centered @ basis @ basis.T
    return hidden - projected


def subspace_intervention_drop(model, x: np.ndarray, y: np.ndarray, probe: LogisticRegression) -> float:
    hidden = hidden_states(model, x)
    base_pred = predict_from_hidden(model, hidden)
    edited = remove_probe_subspace(hidden, probe)
    edited_pred = predict_from_hidden(model, edited)
    base_acc = float(accuracy_score(y, base_pred))
    edited_acc = float(accuracy_score(y, edited_pred))
    return base_acc - edited_acc
