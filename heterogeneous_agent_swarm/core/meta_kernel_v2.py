from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import copy
import time
import collections

from .audit import AuditLog
from .graph import AgentGraph
from .orchestrator import Orchestrator


@dataclass
class ChangeProposal:
    proposal_id: str
    ts: float
    kind: str                    # "drop_agent" | "add_agent" | "policy_update"
    payload: Dict[str, Any]
    rationale: str = ""
    votes: Dict[str, bool] = field(default_factory=dict)  # agent_name -> approve/deny


class MetaKernelV2:
    """
    Manages structural adaptation of the swarm (adding/removing agents, policy updates).
    """
    def __init__(self, graph: AgentGraph, audit: AuditLog, orchestrator: Orchestrator, min_quorum: int = 3):
        """
        Initialize the MetaKernel.

        Args:
            graph: The AgentGraph.
            audit: AuditLog for recording events.
            orchestrator: The system Orchestrator (for policy updates).
            min_quorum: Minimum votes required to pass a proposal.
        """
        self.graph = graph
        self.audit = audit
        self.orchestrator = orchestrator
        self.min_quorum = min_quorum
        self.proposals: Dict[str, ChangeProposal] = {}

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
            # Graph update only (no logic instantiation as per plan)
            self.graph.ensure_node(name)
            self.audit.emit("meta_commit_add_agent", {"id": proposal_id, "added": name})
            return True

        elif cp.kind == "policy_update":
            # Update orchestrator
            veto = cp.payload.get("veto_threshold")
            strategy = cp.payload.get("selection_strategy")
            self.orchestrator.update_policy(veto_threshold=veto, selection_strategy=strategy)
            self.audit.emit("meta_commit_policy_update", {"id": proposal_id, "payload": cp.payload})
            return True

        self.audit.emit("meta_commit_noop", {"id": proposal_id, "kind": cp.kind})
        return True
