from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid
import time


@dataclass
class Goal:
    goal_id: str
    text: str
    priority: int = 5
    parent_id: Optional[str] = None
    created_ts: float = field(default_factory=lambda: time.time())
    status: str = "open"   # open|active|done|blocked
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    task_id: str
    goal_id: str
    text: str
    status: str = "queued"  # queued|running|done|failed
    attempts: int = 0
    max_attempts: int = 5
    context: Dict[str, Any] = field(default_factory=dict)


class TaskQueue:
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.tasks: Dict[str, Task] = {}
        self.queue: List[str] = []

    def new_goal(self, text: str, priority: int = 5, parent_id: str | None = None, context: Dict[str, Any] | None = None) -> Goal:
        gid = str(uuid.uuid4())
        g = Goal(goal_id=gid, text=text, priority=priority, parent_id=parent_id, context=context or {})
        self.goals[gid] = g
        return g

    def add_task(self, goal_id: str, text: str, context: Dict[str, Any] | None = None) -> Task:
        tid = str(uuid.uuid4())
        t = Task(task_id=tid, goal_id=goal_id, text=text, context=context or {})
        self.tasks[tid] = t
        self.queue.append(tid)
        return t

    def pop_next(self) -> Task | None:
        # Highest priority goal first
        if not self.queue:
            return None
        self.queue.sort(key=lambda tid: self.goals[self.tasks[tid].goal_id].priority)
        tid = self.queue.pop(0)
        t = self.tasks[tid]
        t.status = "running"
        return t
