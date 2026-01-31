from dataclasses import dataclass

@dataclass
class Policy:
    pass

class NeuroSymbolicVerifierAgent:
    def __init__(self, name: str, policy: Policy):
        self.name = name
        self.artifacts = {"verdict": "allow"}

    def propose(self, state, memory):
        from ..core.types import Proposal
        obs = state.raw_obs
        
        # Invariant: Must run test if buffer len is 5 before doing anything else?
        # Or veto invalid states.
        
        # If buffer len is 5 and NOT tested ok, force TEST.
        # This acts as a veto against "APPEND".
        if len(obs.get("buffer", [])) == 5 and not obs.get("last_test_ok"):
             self.artifacts = {"verdict": "deny", "deny_action": "write_patch", "rule_id": "max_len_verify"}
             return Proposal(
                action_type="run_tests", action_value=None,
                confidence=0.99, predicted_value=5.0, estimated_cost=5.0,
                rationale="RULE:must_verify_max_len",
                source_agent=self.name
            )
        
        self.artifacts = {"verdict": "allow"}
        # Deterministic fallback
        val = abs(hash(str(state.raw_obs))) % 10
        return Proposal(
            action_type="write_patch", action_value=val,
            confidence=0.1, predicted_value=0.0, estimated_cost=1.0,
            rationale="compliant",
            source_agent=self.name
        )
