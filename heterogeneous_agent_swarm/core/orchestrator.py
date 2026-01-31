from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import hashlib
from collections import Counter
from .graph import AgentGraph
from .types import Proposal

class Orchestrator:
    """
    Central decision-making component that selects the best action from agent proposals.
    """
    def __init__(self, graph: AgentGraph):
        """
        Initialize the Orchestrator.

        Args:
            graph: The AgentGraph instance tracking agent states.
        """
        self.graph = graph
        self.veto_threshold = 0.5
        self.selection_strategy = "weighted_perf"

    def update_policy(self, veto_threshold: Optional[float] = None, selection_strategy: Optional[str] = None) -> None:
        """
        Update the orchestration policy parameters.

        Args:
            veto_threshold: New threshold for vetoing actions (0.0 to 1.0).
            selection_strategy: New selection strategy ("weighted_perf", "consensus", "random").
        """
        if veto_threshold is not None:
            self.veto_threshold = max(0.0, min(1.0, veto_threshold))

        if selection_strategy is not None:
            if selection_strategy in ["weighted_perf", "consensus", "random"]:
                self.selection_strategy = selection_strategy

    def choose(self, proposals: Dict[str, Proposal], state: Any, veto: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Select the winning action based on the current policy.

        Args:
            proposals: Dictionary mapping agent names to their Proposals.
            state: Current system state (unused in selection logic currently).
            veto: Optional veto signal from verification agents.

        Returns:
            Tuple containing:
            - tool_name (str)
            - tool_args (dict)
            - debug_info (dict)
        """
        # 1. Veto Check
        if veto:
            veto_score = veto.get("veto_score")
            # If veto_score is present, check against threshold
            if veto_score is not None:
                if veto_score > self.veto_threshold:
                    return "wait", {}, {"reason": "vetoed", "veto_score": veto_score, "threshold": self.veto_threshold}
            # Fallback to binary check if score missing
            elif veto.get("deny", False):
                 return "wait", {}, {"reason": "vetoed_binary"}

        # Filter for alive agents
        alive_proposals = {n: p for n, p in proposals.items()
                           if self.graph.node_alive.get(n, False)}

        if not alive_proposals:
            return "summarize", {"reason": "no_alive_agents"}, {"reason": "no_alive_agents"}

        winner_proposal: Optional[Proposal] = None
        selection_reason = ""
        
        # 2. Strategy Execution
        if self.selection_strategy == "random":
            # Deterministic random: hash-based selection
            # Use state hash to pick deterministically but pseudo-randomly
            if alive_proposals:
                # Sort by name for consistent ordering
                items = sorted(alive_proposals.items(), key=lambda x: x[0])

                # Create deterministic hash from state
                state_str = str(state)
                hash_input = f"{state_str}_{len(items)}".encode()
                hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)

                choice_idx = hash_val % len(items)
                choice_name, winner_proposal = items[choice_idx]
                selection_reason = "deterministic_random_hash"

        elif self.selection_strategy == "consensus":
            # Majority vote on tool_name
            tool_counts = Counter(p.action_type for p in alive_proposals.values())
            total_votes = len(alive_proposals)
            if total_votes > 0:
                # Find tool with max votes
                best_tool, count = tool_counts.most_common(1)[0]

                if count / total_votes > 0.5:
                    # Pick the proposal corresponding to this tool with highest confidence
                    candidates = [p for p in alive_proposals.values() if p.action_type == best_tool]
                    winner_proposal = max(candidates, key=lambda p: p.confidence)
                    selection_reason = f"consensus_agreement_{count}/{total_votes}"
                else:
                    return "wait", {}, {"reason": "no_consensus", "stats": dict(tool_counts)}

        else: # "weighted_perf" (Default)
            # Score = Confidence * Node Performance (if available)
            scored_proposals = []
            for name, p in alive_proposals.items():
                perf = self.graph.node_perf.get(name, 0.5)
                score = p.confidence * perf
                scored_proposals.append((score, name, p))

            # Sort desc by score
            if scored_proposals:
                scored_proposals.sort(key=lambda x: (-x[0], x[1]))
                score, name, winner_proposal = scored_proposals[0]
                selection_reason = f"weighted_perf_score_{score:.2f}"

        # Safety check if no winner selected (should match fallback if logic is correct)
        if not winner_proposal:
             # Should be covered by "if not alive_proposals" but for safety
             return "summarize", {"reason": "no_selection"}, {"reason": "no_winner_selected"}

        # 3. Construct Result
        tool_name = winner_proposal.action_type
        # Preserve existing behavior for tool args
        tool_args = {"value": winner_proposal.action_value}
        
        debug_info = {
            "winner": winner_proposal.source_agent,
            "reason": selection_reason,
            "rationale": winner_proposal.rationale,
            "all_proposals": {n: p.action_type for n, p in alive_proposals.items()},
            "votes": {n: getattr(p, "action_type", "unknown") for n, p in alive_proposals.items()}
        }

        return tool_name, tool_args, debug_info
