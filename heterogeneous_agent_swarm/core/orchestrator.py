from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import hashlib
import numpy as np
import time
import logging
from collections import Counter
from .graph import AgentGraph
from .types import Proposal
from .config import DEFAULT_CONFIG, SwarmConfig

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Central decision-making component that selects the best action from agent proposals.
    Uses centralized configuration for thresholds and parameters.
    """
    def __init__(self, graph: AgentGraph, memory: Optional[Any] = None, config: SwarmConfig = None):
        """
        Initialize the Orchestrator.

        Args:
            graph: The AgentGraph instance tracking agent states.
            memory: Optional EpisodicMemory reference.
            config: SwarmConfig instance for centralized configuration.
        """
        self.config = config or DEFAULT_CONFIG
        self.graph = graph
        self.memory = memory
        self.veto_threshold = self.config.orchestrator.default_veto_threshold
        self.selection_strategy = self.config.orchestrator.default_selection_strategy
        self.step_counter = 0  # Time-varying factor

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
            valid = self.config.orchestrator.valid_strategies
            if selection_strategy in valid:
                self.selection_strategy = selection_strategy
            else:
                logger.warning(f"Invalid strategy '{selection_strategy}', valid options: {valid}")

    def choose(self, proposals: Dict[str, Proposal], state: Any, veto: Optional[Dict[str, Any]] = None,
               force_strategy: Optional[str] = None, system_uncertainty: float = 0.5) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Select the winning action based on the current policy.

        Args:
            proposals: Dictionary mapping agent names to their Proposals.
            state: Current system state (unused in selection logic currently).
            veto: Optional veto signal from verification agents.
            force_strategy: Override selection strategy for this step (e.g. "random" for deadlock recovery).
            system_uncertainty: GNN variance metric (C-Stage).

        Returns:
            Tuple containing:
            - tool_name (str)
            - tool_args (dict)
            - debug_info (dict)
        """
        self.step_counter += 1

        # 1. Veto Check (Traditional)
        if veto:
            veto_score = veto.get("veto_score")
            # If veto_score is present, check against threshold
            if veto_score is not None:
                if veto_score > self.veto_threshold:
                    return "wait", {}, {"reason": "vetoed", "veto_score": veto_score, "threshold": self.veto_threshold}
            # Fallback to binary check if score missing
            elif veto.get("deny", False):
                 return "wait", {}, {"reason": "vetoed_binary"}

        # 0. Pre-filter: Remove Suppressed Agents (Hard Silence)
        alive_proposals = {
            n: p for n, p in proposals.items()
            if self.graph.node_alive.get(n, False) and not self.graph.node_suppressed.get(n, False)
        }

        if not alive_proposals:
            return "summarize", {"reason": "all_agents_dead_or_suppressed"}, {"reason": "no_valid_proposals"}

        # --- C-STAGE: GNN GATING LOGIC ---
        # Thresholds from config
        UNCERTAINTY_HIGH = self.config.uncertainty.high  # High disagreement -> Chaos
        UNCERTAINTY_LOW = self.config.uncertainty.low    # Low disagreement -> Convergence

        gating_decision = "none"

        # Rule 1: High Uncertainty -> Force Verification/Safety
        if system_uncertainty > UNCERTAINTY_HIGH:
            gating_decision = "force_safety"
            # Prioritize Verifier or Stability agents
            safety_agents = ["neurosym_agent", "ssm_agent", "symbolic_agent"]
            safety_proposals = {n: p for n, p in alive_proposals.items() if n in safety_agents}

            if safety_proposals:
                # Override: Pick best safety proposal directly
                winner_proposal = max(safety_proposals.values(), key=lambda p: p.confidence)
                return winner_proposal.action_type, {"value": winner_proposal.action_value}, {
                    "winner": winner_proposal.source_agent,
                    "reason": f"GNN_VETO_HIGH_UNCERTAINTY ({system_uncertainty:.4f})",
                    "mode": "force_safety"
                }

        # Rule 2: Low Uncertainty (Convergence) -> Block Diffusion (Distraction)
        elif system_uncertainty < UNCERTAINTY_LOW:
            gating_decision = "convergence_lock"
            # Mask Diffusion Explorer
            if "diffusion_agent" in alive_proposals:
                del alive_proposals["diffusion_agent"]

            if not alive_proposals:
                 return "wait", {}, {"reason": "convergence_lock_empty"}

        # Memory-based bias initialization
        memory_bias: Dict[str, float] = {}

        # Query memory for similar past experiences
        if self.memory and hasattr(state, '__iter__'):
            try:
                state_vec = np.array(state) if not isinstance(state, np.ndarray) else state
                hits = self.memory.retrieve(state_vec, top_k=5)

                for hit in hits:
                    # Only consider positive experiences
                    reward = hit.metadata.get("reward", 0)
                    positive_threshold = self.config.orchestrator.positive_experience_threshold
                    if reward > positive_threshold:
                        action = hit.metadata.get("action", "")
                        if action and action in [p.action_type for p in alive_proposals.values()]:
                            # Boost score proportional to similarity and reward
                            boost_factor = self.config.orchestrator.memory_similarity_boost_factor
                            boost = hit.similarity * reward * boost_factor
                            memory_bias[action] = memory_bias.get(action, 0.0) + boost
            except Exception as e:
                # Log the error for debugging instead of silently ignoring
                logger.warning(f"Memory query failed: {e}")

        winner_proposal: Optional[Proposal] = None
        selection_reason = ""
        
        # Determine effective strategy
        strategy = force_strategy if force_strategy else self.selection_strategy

        # 2. Strategy Execution
        if strategy == "random":
            # Time-varying random: hash-based selection mixed with step count
            if alive_proposals:
                # Sort by name for consistent ordering
                items = sorted(alive_proposals.items(), key=lambda x: x[0])

                # Create hash from state AND time/step to ensure variety
                state_str = str(state)
                # Include step_counter and system time in hash input
                hash_input = f"{state_str}_{len(items)}_{self.step_counter}_{time.time()}".encode()
                hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)

                choice_idx = hash_val % len(items)
                choice_name, winner_proposal = items[choice_idx]
                selection_reason = "time_varying_random_hash"

        elif strategy == "consensus":
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

        else:  # "weighted_perf" (Default)
            # Score = Confidence * Node Performance + Memory Bias
            scored_proposals = []
            for name, p in alive_proposals.items():
                perf = self.graph.node_perf.get(name, 0.5)
                base_score = p.confidence * perf

                # Add memory bias if this agent's action matches past successes
                action = p.action_type
                bias = memory_bias.get(action, 0.0)
                final_score = base_score + bias

                scored_proposals.append((final_score, name, p, bias))

            # Sort desc by score
            if scored_proposals:
                scored_proposals.sort(key=lambda x: (-x[0], x[1]))
                score, name, winner_proposal, win_bias = scored_proposals[0]
                selection_reason = f"weighted_perf_score_{score:.2f}_mem_{win_bias:.2f}"

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
            "votes": {n: getattr(p, "action_type", "unknown") for n, p in alive_proposals.items()},
            "memory_bias": memory_bias,
            "strategy": strategy,
            "uncertainty": system_uncertainty,
            "gating": gating_decision
        }

        return tool_name, tool_args, debug_info
