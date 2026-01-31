from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import time


@dataclass
class Blackboard:
    """
    Structured shared state accessible to all agents.
    Acts as a central communication board, distinct from a simple chat log.

    Args:
        episode_id: Unique identifier for the current episode.
        goal_text: High-level goal description.
        task_text: Current task description.
        obs: Current environment observations.
        plan: List of planned tool steps.
        step_history: History of executed tool steps.
        signals: System signals (scores, warnings, flags).
        ts: Timestamp of creation/update.
    """
    episode_id: str
    goal_text: str
    task_text: str
    obs: Dict[str, Any] = field(default_factory=dict)
    plan: List[Dict[str, Any]] = field(default_factory=list)          # ordered tool steps
    step_history: List[Dict[str, Any]] = field(default_factory=list)  # tool exec traces
    signals: Dict[str, Any] = field(default_factory=dict)             # scores, warnings, flags
    ts: float = field(default_factory=lambda: time.time())

    def record_step(self, step: Dict[str, Any]) -> None:
        """
        Append a step execution result to history.

        Args:
            step: Dictionary containing step details.
        """
        self.step_history.append(step)

    def set_signal(self, k: str, v: Any) -> None:
        """
        Set a system signal value.

        Args:
            k: Signal key.
            v: Signal value.
        """
        self.signals[k] = v
