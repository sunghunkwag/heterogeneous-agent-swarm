from dataclasses import dataclass

@dataclass
class SSMConfig:
    pass

class SSMStabilityAgent:
    def __init__(self, name: str, config: SSMConfig):
        self.name = name

    def propose(self, state, memory):
        from ..core.types import Proposal
        # Conservative
        # If failures are high, propose safe action (summarize/noop)
        obs = state.raw_obs
        failures = obs.get("failures", 0)
        
        # If too many failures, maybe stop thrashing?
        if failures > 5:
             # Wait / Summarize
             return Proposal(
                action_type="summarize", action_value=None,
                confidence=0.8, predicted_value=0.1, estimated_cost=0.5,
                rationale="conserve_resources",
                source_agent=self.name
            )
            
        import random
        return Proposal(
            action_type="APPEND", action_value=random.randint(0, 9),
            confidence=0.2, predicted_value=0.1, estimated_cost=1.0,
            rationale="stable",
            source_agent=self.name
        )
