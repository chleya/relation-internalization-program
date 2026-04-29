from collections import defaultdict

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

from .agents_base import BaseAgent
from .features import FEATURE_ORDER, FEATURE_VALUES, RESOURCES, context_to_tuple, one_hot_context


def optimal_action(resource: str) -> str:
    return "eat" if resource == "food" else "avoid"


class MajorityBaseline(BaseAgent):
    name = "majority"

    def __init__(self):
        self.action_rewards = defaultdict(list)
        self.resource_counts = defaultdict(int)
        self.best_action = "avoid"

    def act(self, context: dict) -> str:
        return self.best_action

    def observe(self, context: dict, action: str, resource: str, reward: float) -> None:
        self.resource_counts[resource] += 1
        for candidate in ["eat", "avoid"]:
            simulated = 1.0 if (candidate, resource) == ("eat", "food") else 0.0
            if (candidate, resource) == ("eat", "poison"):
                simulated = -1.0
            if (candidate, resource) == ("avoid", "food"):
                simulated = -0.2
            if (candidate, resource) == ("avoid", "poison"):
                simulated = 0.2
            self.action_rewards[candidate].append(simulated)
        means = {a: np.mean(v) for a, v in self.action_rewards.items()}
        self.best_action = max(means, key=means.get)

    def infer_resource(self, context: dict) -> str:
        if not self.resource_counts:
            return "neutral"
        return max(self.resource_counts, key=self.resource_counts.get)


class MemoryBaseline(BaseAgent):
    name = "memory"

    def __init__(self):
        self.table = defaultdict(lambda: defaultdict(int))
        self.global_counts = defaultdict(int)

    def act(self, context: dict) -> str:
        return optimal_action(self.infer_resource(context))

    def infer_resource(self, context: dict) -> str:
        key = context_to_tuple(context)
        counts = self.table.get(key)
        if counts:
            return max(counts, key=counts.get)
        if self.global_counts:
            return max(self.global_counts, key=self.global_counts.get)
        return "neutral"

    def observe(self, context: dict, action: str, resource: str, reward: float) -> None:
        self.table[context_to_tuple(context)][resource] += 1
        self.global_counts[resource] += 1

    def describe_relations(self) -> list[dict]:
        return [{"memorized_contexts": len(self.table)}]


class FittingBaseline(BaseAgent):
    name = "fitting"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.x: list[np.ndarray] = []
        self.y: list[str] = []
        self.model = SGDClassifier(loss="log_loss", random_state=seed, max_iter=1000, tol=1e-3)
        self.is_fit = False
        self.classes = np.array(["eat", "avoid"])

    def act(self, context: dict) -> str:
        if not self.is_fit:
            return "avoid"
        pred = self.model.predict([one_hot_context(context)])[0]
        return str(pred)

    def observe(self, context: dict, action: str, resource: str, reward: float) -> None:
        x = one_hot_context(context)
        y = optimal_action(resource)
        self.x.append(x)
        self.y.append(y)
        if len(self.y) == 1:
            self.model.partial_fit([x], [y], classes=self.classes)
        else:
            self.model.partial_fit([x], [y])
        self.is_fit = True

    def train_batch(self, records: list[dict]) -> None:
        self.x = [one_hot_context(r["context"]) for r in records]
        self.y = [optimal_action(r["resource"]) for r in records]
        if len(set(self.y)) > 1:
            self.model.fit(np.vstack(self.x), np.array(self.y))
            self.is_fit = True


class PredictiveBaseline(BaseAgent):
    name = "predictive"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.x: list[np.ndarray] = []
        self.y: list[str] = []
        self.model = SGDClassifier(loss="log_loss", random_state=seed, max_iter=1000, tol=1e-3)
        self.is_fit = False
        self.classes = np.array(RESOURCES)

    def act(self, context: dict) -> str:
        resource = self.infer_resource(context)
        return optimal_action(resource)

    def infer_resource(self, context: dict) -> str:
        if not self.is_fit:
            return "neutral"
        return str(self.model.predict([one_hot_context(context)])[0])

    def observe(self, context: dict, action: str, resource: str, reward: float) -> None:
        x = one_hot_context(context)
        self.x.append(x)
        self.y.append(resource)
        if len(self.y) == 1:
            self.model.partial_fit([x], [resource], classes=self.classes)
            self.is_fit = True
        else:
            self.model.partial_fit([x], [resource])

    def train_batch(self, records: list[dict]) -> None:
        self.x = [one_hot_context(r["context"]) for r in records]
        self.y = [r["resource"] for r in records]
        self.model.fit(np.vstack(self.x), np.array(self.y))
        self.is_fit = True


class DecisionTreeBaseline(BaseAgent):
    name = "decision_tree"

    def __init__(self, seed: int = 0, max_depth: int = 4):
        self.seed = seed
        self.max_depth = max_depth
        self.x: list[np.ndarray] = []
        self.y: list[str] = []
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
        self.is_fit = False

    def act(self, context: dict) -> str:
        return optimal_action(self.infer_resource(context))

    def infer_resource(self, context: dict) -> str:
        if not self.is_fit:
            return "neutral"
        return str(self.model.predict([one_hot_context(context)])[0])

    def observe(self, context: dict, action: str, resource: str, reward: float) -> None:
        self.x.append(one_hot_context(context))
        self.y.append(resource)
        if len(self.y) >= 12 and len(set(self.y)) > 1:
            self.model.fit(np.vstack(self.x), np.array(self.y))
            self.is_fit = True

    def train_batch(self, records: list[dict]) -> None:
        self.x = [one_hot_context(r["context"]) for r in records]
        self.y = [r["resource"] for r in records]
        if len(set(self.y)) > 1:
            self.model.fit(np.vstack(self.x), np.array(self.y))
            self.is_fit = True

    def describe_relations(self) -> list[dict]:
        if not self.is_fit:
            return []
        return [{"tree": export_text(self.model, feature_names=one_hot_feature_names())}]


def one_hot_feature_names() -> list[str]:
    names = []
    for key in FEATURE_ORDER:
        names.extend(f"{key}={value}" for value in FEATURE_VALUES[key])
    return names
