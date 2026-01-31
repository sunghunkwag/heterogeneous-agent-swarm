from dataclasses import dataclass

@dataclass
class JEPAConfig:
    device: str = "cpu"

class JEPAWorldModelAgent:
    def __init__(self, name: str, config: JEPAConfig):
        self.name = name
    
    def propose(self, state, memory):
        from ..core.types import Proposal
        obs = state.raw_obs
        last_test = obs.get("last_test_ok", False)
        
        # If we haven't tested in a while (uncertainty high), suggest TEST
        # Simple heuristic
        if not last_test and len(obs.get("buffer", [])) >= 3:
             return Proposal(
                action_type="TEST", action_value=None,
                confidence=0.85, predicted_value=1.5, estimated_cost=5.0,
                rationale="reduce_uncertainty",
                source_agent=self.name
            )
        
        import random
        return Proposal(
            action_type="APPEND", action_value=random.randint(0, 9),
            confidence=0.1, predicted_value=0.0, estimated_cost=1.0,
            rationale="idle",
            source_agent=self.name
        )
