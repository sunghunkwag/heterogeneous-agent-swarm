from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    pass

class DiffusionExplorerAgent:
    def __init__(self, name: str, config: DiffusionConfig):
        self.name = name

    def propose(self, state, memory):
        from ..core.types import Proposal
        import random
        return Proposal(
            action_type="APPEND", action_value=random.randint(0, 9),
            confidence=0.4, predicted_value=0.5, estimated_cost=1.0,
            rationale="random_exploration",
            source_agent=self.name
        )
