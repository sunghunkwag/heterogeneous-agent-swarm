from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from ..core.types import EncodedState, Proposal
from ..agents.dsl_solver import DSLSolver

@dataclass
class LLMConfig:
    device: str = "cpu"

class TransformerLLMAgent:
    def __init__(self, name: str, config: LLMConfig):
        self.name = name
        self.config = config
        self.plan_queue = [] # Queue of pending actions
        self.solver = DSLSolver()
        self.last_solved_task = None
        
    def propose(self, state: EncodedState, bus_memory: Dict[str, Any]) -> Proposal:
        obs = state.raw_obs
        
        # 0. Execute Pending Plan
        if self.plan_queue:
            action = self.plan_queue.pop(0)
            return Proposal(
                agent_name=self.name,
                confidence=0.95,
                action="write_patch",
                action_value=action,
                reasoning=f"Executing DSL Plan ({len(self.plan_queue)} remaining)"
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
                        agent_name=self.name,
                        confidence=1.0,
                        action="write_patch",
                        action_value=action,
                        reasoning="ARC: DSL Solution Found!"
                    )
                else:
                    self.last_solved_task = task_file # Mark as tried
                    # Fallback to random if failed
            
            # Fallback logic (Random exploration)
            inp = np.array(obs["input_grid"])
            cur = np.array(obs["current_grid"])
            h, w = inp.shape
            import random
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            color = random.randint(0, 9)

            return Proposal(
                agent_name=self.name,
                confidence=0.1,
                action="write_patch",
                action_value={"x": int(x), "y": int(y), "color": int(color)},
                reasoning="ARC: Random Search (Solver Failed)"
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
                action_type="APPEND", action_value=next_val,
                confidence=0.9, predicted_value=2.0, estimated_cost=1.0,
                rationale=f"detected_pattern_next={next_val}",
                source_agent=self.name
            )
        
        # If length match target (5), submit?
        if len(buffer) == 5:
             return Proposal(
                action_type="TEST", action_value=None,
                confidence=0.6, predicted_value=10.0, estimated_cost=2.0,
                rationale="sequence_length_met_verify",
                source_agent=self.name
            )

        import random
        return Proposal(
            action_type="APPEND", action_value=random.randint(0, 9),
            confidence=0.2, predicted_value=0.0, estimated_cost=1.0,
            rationale="random_guess",
            source_agent=self.name
        )
