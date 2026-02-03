from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import copy
import time
import collections

from .audit import AuditLog
from .graph import AgentGraph
from .orchestrator import Orchestrator

from heterogeneous_agent_swarm.agents.symbolic_search import SymbolicSearchAgent, SymbolicConfig
from heterogeneous_agent_swarm.agents.jepa_world_model import JEPAWorldModelAgent, JEPAConfig
from heterogeneous_agent_swarm.agents.neuro_symbolic import NeuroSymbolicVerifierAgent, Policy
from heterogeneous_agent_swarm.agents.liquid_controller import LiquidControllerAgent, LiquidConfig
from heterogeneous_agent_swarm.agents.diffusion_explorer import DiffusionExplorerAgent, DiffusionConfig
from heterogeneous_agent_swarm.agents.ssm_stability import SSMStabilityAgent, SSMConfig
from heterogeneous_agent_swarm.agents.snn_reflex import SNNReflexAgent, SNNConfig


@dataclass
class ChangeProposal:
    proposal_id: str
    ts: float
    kind: str                    # "drop_agent" | "add_agent" | "policy_update" | "architecture_modification"
    payload: Dict[str, Any]
    rationale: str = ""
    votes: Dict[str, bool] = field(default_factory=dict)  # agent_name -> approve/deny


class MetaKernelV2:
    """
    Manages structural adaptation of the swarm (adding/removing agents, policy updates).
    """
    def __init__(self, graph: AgentGraph, audit: AuditLog, orchestrator: Orchestrator,
                 agents_dict: Dict[str, Any], device: str = "cpu", min_quorum: int = 3):
        """
        Initialize the MetaKernel.

        Args:
            graph: The AgentGraph.
            audit: AuditLog for recording events.
            orchestrator: The system Orchestrator (for policy updates).
            agents_dict: Reference to the main agents dictionary.
            device: Computing device (cpu/cuda).
            min_quorum: Minimum votes required to pass a proposal.
        """
        self.graph = graph
        self.audit = audit
        self.orchestrator = orchestrator
        self.agents_dict = agents_dict
        self.device = device
        self.min_quorum = min_quorum
        self.proposals: Dict[str, ChangeProposal] = {}
        self.consecutive_failures: Dict[str, int] = {}  # Track consecutive poor performance

        self.agent_factory = {
            "symbolic": (SymbolicSearchAgent, SymbolicConfig),
            "jepa": (JEPAWorldModelAgent, JEPAConfig),
            "neurosym": (NeuroSymbolicVerifierAgent, Policy),
            "liquid": (LiquidControllerAgent, LiquidConfig),
            "diffusion": (DiffusionExplorerAgent, DiffusionConfig),
            "ssm": (SSMStabilityAgent, SSMConfig),
            "snn": (SNNReflexAgent, SNNConfig),
        }

    def propose(self, kind: str, payload: Dict[str, Any], rationale: str) -> ChangeProposal:
        """
        Create a new change proposal.
        """
        pid = f"chg_{int(time.time()*1000)}"
        cp = ChangeProposal(proposal_id=pid, ts=time.time(), kind=kind, payload=payload, rationale=rationale)
        self.proposals[pid] = cp
        self.audit.emit("meta_propose", {"id": pid, "kind": kind, "payload": payload, "rationale": rationale})
        return cp

    def vote(self, proposal_id: str, agent_name: str, approve: bool) -> None:
        """
        Cast a vote on a proposal.
        """
        if proposal_id in self.proposals:
            self.proposals[proposal_id].votes[agent_name] = approve

    def _quorum_ok(self, cp: ChangeProposal) -> bool:
        """Check if proposal has enough approval votes."""
        approvals = sum(1 for v in cp.votes.values() if v)
        return approvals >= self.min_quorum

    def _check_connectivity(self, remaining_nodes: List[str]) -> bool:
        """
        Check if the subgraph of remaining nodes is connected.
        If no edges exist in the entire graph, assume valid (implicit connectivity or unused).
        """
        if len(remaining_nodes) <= 1:
            return True

        # Check if any edges exist among remaining nodes
        has_edges = False
        for n in remaining_nodes:
            if self.graph.adjacency.get(n):
                has_edges = True
                break

        if not has_edges:
            return True # No edges managed, so connectivity concept N/A or trivial.

        # BFS
        start_node = remaining_nodes[0]
        queue = collections.deque([start_node])
        visited = {start_node}

        while queue:
            curr = queue.popleft()
            neighbors = self.graph.adjacency.get(curr, [])
            for n in neighbors:
                if n in remaining_nodes and n not in visited:
                    visited.add(n)
                    queue.append(n)

        return len(visited) == len(remaining_nodes)

    def shadow_apply(self, cp: ChangeProposal) -> Union[bool, str]:
        """
        Dry-run for invariants. Returns True if valid, or failure reason string if invalid.
        """
        alive = self.graph.alive_nodes()

        if cp.kind == "drop_agent":
            target = cp.payload.get("name")
            if target not in alive:
                return f"Agent {target} not found or already dead."

            if (len(alive) - 1) < 4:
                return f"Cannot drop {target}: Minimum agent count (4) would be violated."

            # Connectivity check
            remaining = [n for n in alive if n != target]
            if not self._check_connectivity(remaining):
                return f"Cannot drop {target}: Graph would become disconnected."

        elif cp.kind == "add_agent":
            name = cp.payload.get("name")
            if not name:
                return "Missing 'name' in payload."
            if name in alive:
                return f"Agent {name} already exists."
            # Verify config schema
            if "config" not in cp.payload:
                return "Missing 'config' in payload."

        elif cp.kind == "policy_update":
            threshold = cp.payload.get("veto_threshold")
            if threshold is not None:
                if not (0.0 <= threshold <= 1.0):
                    return f"Invalid veto_threshold: {threshold} (must be 0-1)."

            strategy = cp.payload.get("selection_strategy")
            if strategy is not None:
                if strategy not in ["weighted_perf", "consensus", "random"]:
                    return f"Invalid selection_strategy: {strategy}."

        elif cp.kind == "architecture_modification":
            target_agent = cp.payload.get("target_agent")
            if target_agent not in self.agents_dict:
                return f"Target agent {target_agent} not found."

            modification = cp.payload.get("modification")
            if modification not in ["increase_capacity", "decrease_capacity"]:
                return f"Unknown modification type: {modification}"

        return True

    def commit(self, proposal_id: str) -> bool:
        """
        Execute the proposal if valid and approved.
        """
        if proposal_id not in self.proposals:
            return False
        cp = self.proposals[proposal_id]

        if not self._quorum_ok(cp):
            self.audit.emit("meta_reject", {"id": proposal_id, "reason": "no_quorum"})
            return False

        shadow_res = self.shadow_apply(cp)
        if shadow_res is not True:
            # shadow_res is the failure reason string
            self.audit.emit("meta_reject", {"id": proposal_id, "reason": shadow_res})
            return False

        if cp.kind == "drop_agent":
            name = cp.payload.get("name")
            if self.graph.node_alive.get(name, False):
                self.graph.remove_node(name)
                self.audit.emit("meta_commit", {"id": proposal_id, "dropped": name})
                return True

        elif cp.kind == "add_agent":
            name = cp.payload.get("name")
            agent_type = cp.payload.get("agent_type", "symbolic")  # Default to symbolic

            # Create actual agent instance
            if agent_type in self.agent_factory:
                agent_class, config_class = self.agent_factory[agent_type]

                # Instantiate config
                if agent_type in ["symbolic", "jepa", "liquid"]:
                    config = config_class(device=self.device)
                else:
                    config = config_class()

                new_agent = agent_class(name, config)
                self.agents_dict[name] = new_agent  # Add to main agents dict
            else:
                self.audit.emit("meta_reject", {"id": proposal_id, "reason": f"unknown_agent_type_{agent_type}"})
                return False

            self.graph.ensure_node(name)
            self.audit.emit("meta_commit_add_agent", {"id": proposal_id, "added": name, "type": agent_type})
            return True

        elif cp.kind == "policy_update":
            # Update orchestrator
            veto = cp.payload.get("veto_threshold")
            strategy = cp.payload.get("selection_strategy")
            self.orchestrator.update_policy(veto_threshold=veto, selection_strategy=strategy)
            self.audit.emit("meta_commit_policy_update", {"id": proposal_id, "payload": cp.payload})
            return True

        elif cp.kind == "architecture_modification":
            # NEW: Actually modify agent structure
            target_agent_name = cp.payload.get("target_agent")
            modification_type = cp.payload.get("modification", "increase_capacity")

            if target_agent_name in self.agents_dict:
                agent = self.agents_dict[target_agent_name]

                result_meta = {}
                if modification_type == "increase_capacity":
                    if hasattr(agent, 'increase_capacity'):
                        result_meta = agent.increase_capacity()
                    else:
                        result_meta = {"status": "error", "reason": "method_missing"}

                elif modification_type == "decrease_capacity":
                    if hasattr(agent, 'decrease_capacity'):
                        result_meta = agent.decrease_capacity()
                    else:
                        result_meta = {"status": "error", "reason": "method_missing"}

                self.audit.emit("arch_mod_commit", {
                    "agent": target_agent_name,
                    "modification": modification_type,
                    "result": result_meta
                })
                return True

        self.audit.emit("meta_commit_noop", {"id": proposal_id, "kind": cp.kind})
        return True

    def suppress_agent(self, agent_name: str, task_loss: float) -> Optional[Dict[str, Any]]:
        """
        C-Stage Structural Rigor: Enforce performance constraints.
        If an agent consistently fails (task_loss > 0.8), suppress it.

        Args:
            agent_name: Name of agent to evaluate.
            task_loss: Recent task loss (higher = worse performance).

        Returns:
            Dict with suppression data or None if no action.
        """
        if agent_name not in self.agents_dict:
            return None

        # Check failure threshold
        if task_loss > 0.8:
            self.consecutive_failures[agent_name] = self.consecutive_failures.get(agent_name, 0) + 1
        else:
            self.consecutive_failures[agent_name] = 0

        failures = self.consecutive_failures[agent_name]

        # Trigger suppression if 3 consecutive failures
        if failures >= 3:
            # Check if already suppressed
            if not self.graph.node_suppressed.get(agent_name, False):
                self.graph.suppress_node(agent_name)

                impact_data = {
                    "agent": agent_name,
                    "action": "suppressed",
                    "reason": "consecutive_failures",
                    "failures": failures,
                    "task_loss": task_loss,
                    "timestamp": time.time()
                }

                self.audit.emit("meta_suppression", impact_data)
                return impact_data

        return None

    def suggest_architecture_modification(self, agent_name: str) -> Optional[ChangeProposal]:
        """
        Neural Architecture Search (NAS) suggestion:
        If agent consistently underperforms, suggest structural changes.

        Args:
            agent_name: Agent to evaluate

        Returns:
            ChangeProposal or None
        """
        if agent_name not in self.agents_dict:
            return None

        agent = self.agents_dict[agent_name]

        # Get performance history
        perf = self.graph.node_perf.get(agent_name, 0.5)

        # Check if consistently underperforming (threshold < 0.3)
        # Also check if agent supports modification
        if perf < 0.3 and hasattr(agent, 'increase_capacity'):

            # Get current stats if available
            current_cap = 0.0
            if hasattr(agent, 'get_capacity_metric'):
                current_cap = agent.get_capacity_metric()

            # Suggest capacity increase via architecture_modification
            proposal = self.propose(
                kind="architecture_modification",
                payload={
                    "target_agent": agent_name,
                    "modification": "increase_capacity",
                    "current_perf": perf,
                    "current_capacity_metric": current_cap
                },
                rationale=f"Auto-NAS: Agent {agent_name} underperforming (perf={perf:.2f}). Suggesting capacity boost."
            )

            self.audit.emit("nas_suggestion", {
                "agent": agent_name,
                "performance": perf,
                "proposal_id": proposal.proposal_id
            })

            return proposal

        return None
