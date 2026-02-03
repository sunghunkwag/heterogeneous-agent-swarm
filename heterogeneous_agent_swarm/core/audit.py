import time
from typing import Dict, Any, List

class AuditLog:
    """
    Central audit system for tracking significant swarm events.
    Now includes history tracking for meta-learning impact analysis.
    """
    def __init__(self):
        self.events = []
        self.meta_train_history = []  # New: track meta-training events

    def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Record an event.

        Args:
            event_type: Category of event (e.g. "tool_use", "meta_train")
            data: Structured data describing the event
        """
        self.events.append({
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        })

        # Track meta-training events separately for easy lookup
        if event_type == "meta_train" or event_type == "meta_train_impact":
            self.meta_train_history.append(data)

    def get_meta_train_impact(self, agent_name: str, window_size: int = 5) -> Dict[str, Any]:
        """
        Retrieve recent meta-train events and their performance correlations.

        Args:
            agent_name: Name of agent to query
            window_size: Number of recent events to retrieve

        Returns:
            Dict containing count and list of recent events
        """
        recent = [
            e for e in self.meta_train_history
            if e.get("agent") == agent_name
        ][-window_size:]

        return {
            "count": len(recent),
            "events": recent
        }

    def get_logs(self) -> List[Dict[str, Any]]:
        return self.events
