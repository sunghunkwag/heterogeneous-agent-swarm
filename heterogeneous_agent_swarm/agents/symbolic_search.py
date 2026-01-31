from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from ..core.types import EncodedState, Proposal
from ..agents.dsl_solver import DSLSolver

@dataclass
class SymbolicConfig:
    device: str = "cpu"

class SymbolicSearchAgent:
    def __init__(self, name: str, config: SymbolicConfig):
        self.name = name
        self.config = config
        self.plan_queue = [] # Queue of pending actions
        self.solver = DSLSolver()
        self.last_solved_task = None
        
    def propose(self, state: EncodedState, bus_memory: Dict[str, Any]) -> Proposal:
        obs = state.raw_obs
        system_thought = np.array(bus_memory.get("system_thought", [0.0]*16))
        thought_power = np.mean(np.abs(system_thought))
        
        # 0. Execute Pending Plan
        if self.plan_queue:
            action = self.plan_queue.pop(0)
            return Proposal(
                action_type="write_patch",
                action_value=action,
                confidence=0.95 + (0.05 * thought_power),
                predicted_value=1.0,
                estimated_cost=1.0,
                rationale=f"Executing DSL Plan ({len(self.plan_queue)} remaining). Thought: {thought_power:.2f}",
                source_agent=self.name
            )
            
        # 1. ARC GRID MODE
        if "input_grid" in obs:
            task_file = obs.get("task_file", "unknown")
            
            # If we haven't solved this task yet, try solving
            if self.last_solved_task != task_file:
                train_ex = obs.get("train_examples", [])
                test_in = obs.get("input_grid", [])
                
                # Attempt Solve
                plan = self.solver.solve(train_ex, test_in)
                
                if plan:
                    self.last_solved_task = task_file
                    self.plan_queue = plan
                    # Return first action immediately
                    action = self.plan_queue.pop(0)
                    return Proposal(
                        action_type="write_patch",
                        action_value=action,
                        confidence=1.0,
                        predicted_value=1.0,
                        estimated_cost=1.0,
                        rationale="ARC: DSL Solution Found!",
                        source_agent=self.name
                    )
                else:
                    self.last_solved_task = task_file # Mark as tried
            
            # Failure signal (No random fallback)
            return Proposal(
                action_type="wait",
                action_value=None,
                confidence=0.0,
                predicted_value=0.0,
                estimated_cost=0.0,
                rationale=f"ARC: Solver Failed. WAITING/IDLE. Thought: {thought_power:.2f}",
                source_agent=self.name
            )

        # 2. SEQUENCE MODE
        buffer = obs.get("buffer", [])
        
        # Simple detection
        next_val = None
        if len(buffer) >= 2:
            # 1. Arithmetic Check
            diff = buffer[-1] - buffer[-2]
            is_arith = True
            for i in range(1, len(buffer)):
                if buffer[i] - buffer[i-1] != diff:
                    is_arith = False
                    break
            if is_arith:
                next_val = buffer[-1] + diff
                
            # 2. Geometric Check (if not arithmetic)
            if next_val is None and buffer[-2] != 0:
                ratio = buffer[-1] / buffer[-2]
                if ratio.is_integer():
                    ratio = int(ratio)
                    is_geo = True
                    for i in range(1, len(buffer)):
                        if buffer[i-1] == 0 or buffer[i] / buffer[i-1] != ratio:
                            is_geo = False
                            break
                    if is_geo:
                        next_val = buffer[-1] * ratio
            
            # 3. Fibonacci Check
            if next_val is None and len(buffer) >= 3:
                is_fib = True
                for i in range(2, len(buffer)):
                    if buffer[i] != buffer[i-1] + buffer[i-2]:
                        is_fib = False
                        break
                if is_fib:
                    next_val = buffer[-1] + buffer[-2]

        if next_val is not None:
             return Proposal(
                action_type="write_patch",
                action_value=next_val,
                confidence=0.9,
                predicted_value=2.0,
                estimated_cost=1.0,
                rationale=f"detected_pattern_next={next_val}",
                source_agent=self.name
            )
        
        # If length match target (5), submit?
        if len(buffer) == 5:
             return Proposal(
                action_type="run_tests",
                action_value=None,
                confidence=0.6,
                predicted_value=10.0,
                estimated_cost=2.0,
                rationale="sequence_length_met_verify",
                source_agent=self.name
            )

        # Failure signal
        return Proposal(
            action_type="wait",
            action_value=None,
            confidence=0.0,
            predicted_value=0.0,
            estimated_cost=1.0,
            rationale="No pattern detected. WAITING/IDLE.",
            source_agent=self.name
        )
