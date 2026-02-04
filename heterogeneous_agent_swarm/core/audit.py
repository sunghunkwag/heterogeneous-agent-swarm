import time
from typing import Dict, Any, List
from collections import deque
from dataclasses import dataclass

@dataclass
class AuditRecord:
    ts: float
    event: str
    payload: Dict[str, Any]

class AuditLog:
    """
    Central audit system for tracking significant swarm events.
    Now includes history tracking for meta-learning impact analysis.
    Uses bounded memory to prevent leaks.
    """
    def __init__(self, maxlen: int = 10000):
        # Using deque for O(1) appends and automatic size limiting
        self.records: deque[AuditRecord] = deque(maxlen=maxlen)
        self.meta_train_history: deque[Dict[str, Any]] = deque(maxlen=maxlen)

    def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Record an event.

        Args:
            event_type: Category of event (e.g. "tool_use", "meta_train")
            data: Structured data describing the event
        """
        rec = AuditRecord(ts=time.time(), event=event_type, payload=data)
        self.records.append(rec)

        # Track meta-training events separately for easy lookup
        if event_type == "meta_train" or event_type == "meta_train_impact":
            self.meta_train_history.append(data)

    def tail(self, n: int = 5) -> List[AuditRecord]:
        """
        Return the last n records.
        """
        # deque doesn't support slicing directly, so we convert to list
        # optimizing for common case of small n
        if n <= 0:
            return []
        return list(self.records)[-n:]

    def get_meta_train_impact(self, agent_name: str, window_size: int = 5) -> Dict[str, Any]:
        """
        Retrieve recent meta-train events and their performance correlations.

        Args:
            agent_name: Name of agent to query
            window_size: Number of recent events to retrieve

        Returns:
            Dict containing count and list of recent events
        """
        # Since meta_train_history is a deque, we iterate over it
        recent = [
            e for e in self.meta_train_history
            if e.get("agent") == agent_name
        ][-window_size:]

        return {
            "count": len(recent),
            "events": recent
        }

    def get_logs(self) -> List[AuditRecord]:
        return list(self.records)
