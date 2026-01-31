from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from ..core.types import EncodedState, Proposal
from ..core.memory import SemanticMemory

@dataclass
class JEPAConfig:
    device: str = "cpu"

class JEPAWorldModelAgent:
    def __init__(self, name: str, config: JEPAConfig):
        self.name = name
        self.config = config
        self.long_term_memory = SemanticMemory()
        self.prev_state_latent = None
    
    def propose(self, state: EncodedState, memory: Dict[str, Any]) -> Proposal:
        # 1. Learning from Success
        obs = state.raw_obs
        last_test = obs.get("last_test_ok", False)
        
        # If the last step led to success (and we have a previous state), store it
        if last_test and self.prev_state_latent is not None:
             last_action_name = memory.get("last_action_name")
             last_action_params = memory.get("last_action_params")

             if last_action_name and last_action_params:
                 # Extract the raw action value from the parameters wrapper
                 # Orchestrator wraps values in {"value": ...}
                 raw_value = last_action_params
                 if isinstance(last_action_params, dict) and "value" in last_action_params:
                     raw_value = last_action_params["value"]

                 # Store (State -> Action)
                 payload = {
                     "action_type": last_action_name,
                     "action_value": raw_value
                 }

                 self.long_term_memory.add(self.prev_state_latent, payload)

        # Update previous state
        self.prev_state_latent = state.task_latent

        # 2. System Thought Integration
        system_thought = np.array(memory.get("system_thought", [0.0]*16))
        thought_power = np.mean(np.abs(system_thought))

        # 3. Retrieval & Prediction
        # Query memory for similar states
        matches = self.long_term_memory.topk(state.task_latent, k=1)

        if matches:
            best_match = matches[0]
            similarity = best_match.get("similarity", 0.0)

            # If similarity is high enough, propose the recalled action
            if similarity > 0.8: # Threshold
                return Proposal(
                    action_type=best_match["action_type"],
                    action_value=best_match["action_value"],
                    confidence=0.9 * similarity + (0.1 * thought_power),
                    predicted_value=1.0,
                    estimated_cost=1.0,
                    rationale=f"Recalled similar situation (sim={similarity:.2f})",
                    source_agent=self.name
                )

        # Fallback / Uncertainty Management
        # If we haven't tested in a while (uncertainty high), suggest TEST (Heuristic kept as "Intuition")
        if not last_test and len(obs.get("buffer", [])) >= 3:
             return Proposal(
                action_type="TEST", action_value=None,
                confidence=0.85, predicted_value=1.5, estimated_cost=5.0,
                rationale="reduce_uncertainty",
                source_agent=self.name
            )

        # If no memory and no heuristic: Wait/Idle (Deterministic)
        return Proposal(
            action_type="summarize", # Safe action
            action_value=None,
            confidence=0.1, # Low confidence
            predicted_value=0.0,
            estimated_cost=1.0,
            rationale="No memory match. Idle.",
            source_agent=self.name
        )
