from dataclasses import dataclass

@dataclass
class SNNConfig:
    pass

class SNNReflexAgent:
    def __init__(self, name: str, config: SNNConfig):
        self.name = name

    def propose(self, state, memory):
        from ..core.types import Proposal
        # Reflex
        # If last tool call failed (cost high, !ok), trigger recoil (summarize/pause)
        # Note: ToolEnv doesn't expose 'last_tool_ok' in obs directly, only last_test_ok
        # But 'step_history' might be in memory?
        # RunnerV2 puts 'last_tool' in memory.
        
        last_tool = memory.get("last_tool")
        if last_tool and not last_tool["ok"]:
             return Proposal(
                action_type="summarize", action_value=None,
                confidence=0.95, predicted_value=1.0, estimated_cost=0.5,
                rationale="reflex_recoil",
                source_agent=self.name
            )
            
        # Deterministic fallback based on state hash
        val = abs(hash(str(state.raw_obs))) % 10
        return Proposal(
            action_type="APPEND", action_value=val,
            confidence=0.1, predicted_value=0.0, estimated_cost=1.0,
            rationale="dormant",
            source_agent=self.name
        )
