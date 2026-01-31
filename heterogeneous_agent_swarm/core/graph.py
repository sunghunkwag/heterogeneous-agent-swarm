from __future__ import annotations
from typing import Dict, List, Any

class AgentGraph:
    def __init__(self):
        self.node_alive: Dict[str, bool] = {}
        self.node_perf: Dict[str, float] = {}
        self.node_cost: Dict[str, float] = {}
        # Simple adjacency if needed, but for now just presence
        self.adjacency: Dict[str, List[str]] = {}

    def ensure_node(self, name: str) -> None:
        if name not in self.node_alive:
            self.node_alive[name] = True
            self.node_perf[name] = 0.5
            self.node_cost[name] = 0.05
            self.adjacency[name] = []

    def alive_nodes(self) -> List[str]:
        return [n for n, alive in self.node_alive.items() if alive]

    def remove_node(self, name: str) -> None:
        self.node_alive[name] = False
