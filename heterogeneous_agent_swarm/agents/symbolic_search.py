from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from ..core.types import EncodedState, Proposal
from ..agents.dsl_solver import DSLSolver

@dataclass
class SymbolicConfig:
    device: str = "cpu"
    max_search_depth: int = 5

class SymbolicSearchAgent:
    def __init__(self, name: str, config: SymbolicConfig):
        self.name = name
        self.config = config
        self.plan_queue = [] # Queue of pending actions
        self.solver = DSLSolver(max_depth=config.max_search_depth)
        self.last_solved_task = None
        
    def get_capacity_metric(self) -> float:
        """Return normalized capacity (max depth)."""
        return self.config.max_search_depth / 5.0

    def increase_capacity(self, factor: float = 1.0) -> dict:
        """Increase max search depth."""
        old_depth = self.config.max_search_depth
        # Increment by 1 step regardless of factor for discrete depth
        new_depth = old_depth + 1

        self.config.max_search_depth = new_depth
        self.solver = DSLSolver(max_depth=new_depth)

        return {
            "action": "increase_capacity",
            "old_depth": old_depth,
            "new_depth": new_depth
        }

    def decrease_capacity(self, factor: float = 1.0) -> dict:
        """Decrease max search depth."""
        old_depth = self.config.max_search_depth
        if old_depth <= 1:
            return {"action": "decrease_capacity", "status": "noop_min_depth"}

        new_depth = old_depth - 1
        self.config.max_search_depth = new_depth
        self.solver = DSLSolver(max_depth=new_depth)

        return {
            "action": "decrease_capacity",
            "old_depth": old_depth,
            "new_depth": new_depth
        }

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
        
        # ERROR SCRAPING: Check if we just failed a test and got the target
        # We need to look at the *result* of the last action, which might be in 'obs' if we put it there
        # Or we can check the 'last_tool' info if available.
        # STARTUP: If buffer has insufficient data (< 2 items), run tests to generate more
        # OR: Check for task_description to solve from spec
        task_desc = obs.get("task_description", "")
        if "Arithmetic" in task_desc and "Start=" in task_desc:
             # Parse spec: "Arithmetic Progression (Start=4, Step=2)"
             try:
                 import re
                 start_match = re.search(r"Start=(\d+)", task_desc)
                 step_match = re.search(r"Step=(\d+)", task_desc)
                 
                 if start_match and step_match:
                     start = int(start_match.group(1))
                     step = int(step_match.group(1))
                     target_seq = [start + i*step for i in range(5)]
                     current_idx = len(buffer)
                     if current_idx < 5:
                         next_val = target_seq[current_idx]
                         return Proposal(
                            action_type="write_patch",
                            action_value=next_val,
                            confidence=1.0,
                            predicted_value=1.0,
                            estimated_cost=1.0,
                            rationale=f"Symbolic: Arithmetic from Spec ({task_desc})",
                            source_agent=self.name
                        )
             except Exception:
                 pass
        elif "Geometric" in task_desc and "Start=" in task_desc:
             # Parse spec: "Geometric Progression (Start=2, Step=2)"
             try:
                 import re
                 start_match = re.search(r"Start=(\d+)", task_desc)
                 step_match = re.search(r"Step=(\d+)", task_desc)
                 
                 if start_match and step_match:
                     start = int(start_match.group(1))
                     step = int(step_match.group(1))
                     target_seq = [start * (step**i) for i in range(5)]
                     current_idx = len(buffer)
                     if current_idx < 5:
                         next_val = target_seq[current_idx]
                         return Proposal(
                            action_type="write_patch",
                            action_value=next_val,
                            confidence=1.0,
                            predicted_value=1.5,
                            estimated_cost=1.0,
                            rationale=f"Symbolic: Geometric from Spec ({task_desc})",
                            source_agent=self.name
                        )
             except Exception:
                 pass
        elif "Fibonacci" in task_desc and "A=" in task_desc:
             # Parse spec: "Fibonacci Sequence (A=1, B=2)"
             try:
                 import re
                 a_match = re.search(r"A=(\d+)", task_desc)
                 b_match = re.search(r"B=(\d+)", task_desc)
                 
                 if a_match and b_match:
                     a = int(a_match.group(1))
                     b = int(b_match.group(1))
                     target_seq = [a, b]
                     while len(target_seq) < 5:
                         target_seq.append(target_seq[-1] + target_seq[-2])
                     current_idx = len(buffer)
                     if current_idx < 5:
                         next_val = target_seq[current_idx]
                         return Proposal(
                            action_type="write_patch",
                            action_value=next_val,
                            confidence=1.0,
                            predicted_value=2.0,
                            estimated_cost=1.0,
                            rationale=f"Symbolic: Fibonacci from Spec ({task_desc})",
                            source_agent=self.name
                        )
             except Exception:
                 pass
        elif "Quadratic" in task_desc and "a=" in task_desc:
             # Parse spec: "Quadratic Progression (a=1, b=0, c=0)"
             try:
                 import re
                 a_m = re.search(r"a=(\d+)", task_desc)
                 b_m = re.search(r"b=(\d+)", task_desc)
                 c_m = re.search(r"c=(\d+)", task_desc)
                 
                 if a_m and b_m and c_m:
                     a = int(a_m.group(1))
                     b = int(b_m.group(1))
                     c = int(c_m.group(1))
                     target_seq = [a*(i**2) + b*i + c for i in range(5)]
                     current_idx = len(buffer)
                     if current_idx < 5:
                         next_val = target_seq[current_idx]
                         return Proposal(
                            action_type="write_patch",
                            action_value=next_val,
                            confidence=1.0,
                            predicted_value=3.0,
                            estimated_cost=1.0,
                            rationale=f"Symbolic: Quadratic from Spec ({task_desc})",
                            source_agent=self.name
                        )
             except Exception:
                 pass

        # GOAL AWARENESS: If we have enough data (5 items), STOP and VERIFY.
        # Don't just keep writing forever.
        if len(buffer) >= 5:
             return Proposal(
                action_type="run_tests",
                action_value=None,
                confidence=1.0, # High confidence to verify
                predicted_value=10.0,
                estimated_cost=2.0,
                rationale="Goal reached (5 items). Verifying solution.",
                source_agent=self.name
            )

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
