from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def train_mlp(x: np.ndarray, y: np.ndarray, seed: int = 0) -> MLPClassifier:
    clf = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=0.01,
        max_iter=600,
        random_state=seed,
    )
    clf.fit(x, y)
    return clf


def hidden_states(model: MLPClassifier, x: np.ndarray) -> np.ndarray:
    hidden = x @ model.coefs_[0] + model.intercepts_[0]
    return np.maximum(hidden, 0.0)


def predict_from_hidden(model: MLPClassifier, hidden: np.ndarray) -> np.ndarray:
    logits = hidden @ model.coefs_[1] + model.intercepts_[1]
    return np.argmax(logits, axis=1)


def accuracy(model: MLPClassifier, x: np.ndarray, y: np.ndarray) -> float:
    return float(accuracy_score(y, model.predict(x)))
