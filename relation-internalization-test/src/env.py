import random

from .features import COLORS, ODORS, OOD_COLORS, OOD_ODORS, TEXTURES, TRAIN_COLORS, TRAIN_ODORS, WETS

REWARD_TABLE = {
    ("eat", "food"): 1.0,
    ("eat", "poison"): -1.0,
    ("eat", "neutral"): 0.0,
    ("avoid", "food"): -0.2,
    ("avoid", "poison"): 0.2,
    ("avoid", "neutral"): 0.0,
}


class RelationWorld:
    def __init__(self, regime: str = "base", seed: int = 0, ood: bool = False):
        self.seed = seed
        self.rng = random.Random(seed)
        self.regime = regime
        self.ood = ood or regime == "ood"

    def sample_context(self) -> dict:
        texture = self._weighted_choice(TEXTURES, [0.55, 0.3, 0.15] if not self.ood else [0.34, 0.33, 0.33])
        wet = self._weighted_choice(WETS, [0.7, 0.3] if not self.ood else [0.45, 0.55])
        resource_regime = "base" if self.regime == "ood" else self.regime
        resource = self._resource_for(texture, wet, resource_regime)

        if not self.ood:
            spurious_resource = self._resource_for(texture, wet, "base")
            color, odor = self._spurious_features(spurious_resource)
        else:
            color = self._weighted_choice(OOD_COLORS, [1.0] * len(OOD_COLORS))
            odor = self._weighted_choice(OOD_ODORS, [1.0] * len(OOD_ODORS))

        return {"texture": texture, "wet": wet, "color": color, "odor": odor}

    def get_resource(self, context: dict) -> str:
        regime = "base" if self.regime == "ood" else self.regime
        return self._resource_for(context["texture"], context["wet"], regime)

    def step(self, action: str, context: dict | None = None) -> dict:
        context = context or self.sample_context()
        resource = self.get_resource(context)
        reward = REWARD_TABLE[(action, resource)]
        return {"context": context, "resource": resource, "action": action, "reward": reward}

    def switch_regime(self, regime: str) -> None:
        self.regime = regime
        self.ood = regime == "ood"

    def _resource_for(self, texture: str, wet: str, regime: str) -> str:
        if regime == "base":
            if texture == "A" and wet == "dry":
                return "food"
            if texture == "A" and wet == "wet":
                return "poison"
            if texture == "B":
                return "poison"
            return "neutral"
        if regime == "reversal":
            if texture == "A" and wet == "dry":
                return "poison"
            if texture == "A" and wet == "wet":
                return "food"
            if texture == "B":
                return "food"
            return "neutral"
        raise ValueError(f"Unknown regime: {regime}")

    def _spurious_features(self, resource: str) -> tuple[str, str]:
        if resource == "food":
            return TRAIN_COLORS[0], TRAIN_ODORS[0]
        if resource == "poison":
            return TRAIN_COLORS[1], TRAIN_ODORS[1]
        return self._weighted_choice(TRAIN_COLORS[2:], [1.0] * len(TRAIN_COLORS[2:])), self._weighted_choice(
            TRAIN_ODORS[2:], [1.0] * len(TRAIN_ODORS[2:])
        )

    def _weighted_choice(self, values: list[str], weights: list[float]) -> str:
        return self.rng.choices(values, weights=weights, k=1)[0]
