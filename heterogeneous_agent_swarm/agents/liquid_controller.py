from dataclasses import dataclass

@dataclass
class LiquidConfig:
    device: str = "cpu"

class LiquidControllerAgent:
    def __init__(self, name: str, config: LiquidConfig):
        self.name = name

    def propose(self, state, memory):
        from ..core.types import Proposal
        import math, random
        
        # Simple oscillation
        t = random.random() * 10 
        mode = math.sin(t)
        
        # In v0.2 tool env, what is "DELETE"?
        # We don't have a delete tool.
        # We only have: write_patch (append), run_tests, summarize.
        # So liquid will just propose write_patch with random values.
        
        return Proposal(
            action_type="APPEND", action_value=random.randint(0, 9),
            confidence=0.3, predicted_value=0.2, estimated_cost=1.0,
            rationale=f"liquid_oscillation_{mode:.2f}",
            source_agent=self.name
        )
