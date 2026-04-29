class BaseAgent:
    name = "base"

    def reset(self) -> None:
        pass

    def act(self, context: dict) -> str:
        raise NotImplementedError

    def observe(self, context: dict, action: str, resource: str, reward: float) -> None:
        pass

    def train_batch(self, records: list[dict]) -> None:
        for record in records:
            self.observe(record["context"], record["action"], record["resource"], record["reward"])

    def edit_rule(self, condition: dict, new_outcome: str) -> bool:
        return False

    def describe_relations(self) -> list[dict]:
        return []
