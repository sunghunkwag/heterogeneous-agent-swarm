from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import time


@dataclass
class Blackboard:
    """
    Structured shared state. Not a chat log.
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
        self.step_history.append(step)

    def set_signal(self, k: str, v: Any) -> None:
        self.signals[k] = v
