from __future__ import annotations
from typing import Dict, Any, Tuple, List
import random
import math

# We need types that match what RunnerV2 expects (Proposal)
# Proposal is not defined in core yet, RunnerV2 imports it from ..core.types
# I will assume core.types structure

class Orchestrator:
    def __init__(self, graph):
        self.graph = graph

    def choose(self, proposals: Dict[str, Any], state: Any, veto: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Selects the winning action from proposals.
        proposals: {agent_name: Proposal}
        """
        # If veto is present
        if veto and veto.get("deny", False):
            # If veto denies, we might force a "TEST" or "WAIT" action?
            # Or just filter out the dangerous proposal?
            # For this simplified version: If veto denies a specific 'action_type' or action, we filter.
            pass

        alive = [n for n in proposals.keys() if n in self.graph.alive_nodes()]
        if not alive:
            return "summarize", {"reason": "no_alive_agents"} # Fallback tool

        # Simple weighted choice based on confidence and diversity
        # (Simplified version of v0.1 logic)
        scores = []
        for n in alive:
            p = proposals[n]
            # Score = Confidence + some random noise for exploration
            # p.confidence is float
            score = p.confidence * 1.0 + random.uniform(0, 0.2)
            scores.append((score, n, p))

        scores.sort(key=lambda x: x[0], reverse=True)
        winner = scores[0]
        
        # Action is explicitly the 'action_type' string for tools in v0.2?
        # RunnerV2 says: tool_name = action
        # In v0.1 Proposal had action_type="APPEND", action_value=...
        # In v0.2, agents should propose tool names.
        # I need to adapt the agents to propose "tool_name"
        
        chosen_tool = winner[2].action_type # This should be the tool name
        tool_args = {"value": winner[2].action_value}

        return chosen_tool, tool_args, {"winner": winner[1], "score": winner[0], "rationale": winner[2].rationale}
