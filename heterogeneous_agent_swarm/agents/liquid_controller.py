from dataclasses import dataclass, field
import numpy as np
from ..core.types import EncodedState, Proposal

@dataclass
class LiquidConfig:
    device: str = "cpu"

class LiquidControllerAgent:
    def __init__(self, name: str, config: LiquidConfig):
        self.name = name
        self.prev_error = 0.0
        self.integral_error = 0.0
        # PID Gains
        self.kp = 1.0
        self.ki = 0.1
        self.kd = 5.0 # Strong reaction to derivative

    def propose(self, state: EncodedState, memory: dict) -> Proposal:
        system_thought = np.array(memory.get("system_thought", [0.0]*16))
        thought_power = np.mean(np.abs(system_thought))
        
        # Get error_rate from memory (injected by main.py)
        # Default to 0.0 if not found
        current_error = memory.get("error_rate", 0.0)
        
        # PID Calculation (Derivative Check)
        dt = 1.0 # discrete steps
        derivative = (current_error - self.prev_error) / dt
        self.integral_error += current_error * dt
        
        # Control Signal: System Temperature
        # Logic:
        # If error_rate is increasing (derivative > 0) -> Increase "System Temperature" (Exploration).
        # If error_rate is stable (derivative ~ 0) or decreasing -> Decrease "System Temperature" (Exploitation).

        base_temp = 0.5

        # Calculate temperature adjustment
        # Positive derivative (worsening) adds to temp
        # Negative derivative (improving) subtracts from temp
        temp_adjustment = self.kd * derivative

        target_temp = base_temp + temp_adjustment
        target_temp = float(np.clip(target_temp, 0.0, 1.0))

        self.prev_error = current_error

        # Determine Action based on Temperature
        # High Temp -> Exploration Mode
        # Low Temp -> Exploitation Mode

        if target_temp > 0.6:
            # High Temperature: Signal need for change/exploration.
            # Propose "summarize" to trigger reflection/re-planning.
            action_type = "summarize"
            rationale = f"Error Increasing (d={derivative:.2f}). System Temp {target_temp:.2f} -> Exploration."
            # High confidence if we are sure we are failing
            confidence = 0.6 + (0.4 * target_temp)
        else:
            # Low Temperature: Signal stability/exploitation.
            # Propose "run_tests" to verify stability.
            action_type = "run_tests"
            rationale = f"Error Stable/Decreasing (d={derivative:.2f}). System Temp {target_temp:.2f} -> Exploitation."
            # High confidence if we are stable
            confidence = 0.6 + (0.4 * (1.0 - target_temp))

        # Modulation by system thought
        confidence += (0.1 * thought_power)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return Proposal(
            action_type=action_type,
            action_value=None,
            confidence=confidence,
            predicted_value=1.0 - current_error,
            estimated_cost=1.0,
            rationale=rationale,
            source_agent=self.name,
            artifacts={"system_temperature": target_temp}
        )
