from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import copy
import time

from .audit import AuditLog
from .graph import AgentGraph


@dataclass
class ChangeProposal:
    proposal_id: str
    ts: float
    kind: str                    # "drop_agent" | "add_agent" | "policy_update"
    payload: Dict[str, Any]
    rationale: str = ""
    votes: Dict[str, bool] = field(default_factory=dict)  # agent_name -> approve/deny


class MetaKernelV2:
    def __init__(self, graph: AgentGraph, audit: AuditLog, min_quorum: int = 3):
        self.graph = graph
        self.audit = audit
        self.min_quorum = min_quorum
        self.proposals: Dict[str, ChangeProposal] = {}

    def propose(self, kind: str, payload: Dict[str, Any], rationale: str) -> ChangeProposal:
        pid = f"chg_{int(time.time()*1000)}"
        cp = ChangeProposal(proposal_id=pid, ts=time.time(), kind=kind, payload=payload, rationale=rationale)
        self.proposals[pid] = cp
        self.audit.emit("meta_propose", {"id": pid, "kind": kind, "payload": payload, "rationale": rationale})
        return cp

    def vote(self, proposal_id: str, agent_name: str, approve: bool) -> None:
        if proposal_id in self.proposals:
            self.proposals[proposal_id].votes[agent_name] = approve

    def _quorum_ok(self, cp: ChangeProposal) -> bool:
        approvals = sum(1 for v in cp.votes.values() if v)
        return approvals >= self.min_quorum

    def shadow_apply(self, cp: ChangeProposal) -> bool:
        """
        Dry-run for invariants. (In a fuller system, this would run 1-2 shadow episodes.)
        Here we check minimal invariants: keep >=4 alive agents.
        """
        alive = self.graph.alive_nodes()
        if cp.kind == "drop_agent":
            target = cp.payload.get("name")
            if target in alive and (len(alive) - 1) < 4:
                return False
        return True

    def commit(self, proposal_id: str) -> bool:
        if proposal_id not in self.proposals:
            return False
        cp = self.proposals[proposal_id]
        if not self._quorum_ok(cp):
            self.audit.emit("meta_reject", {"id": proposal_id, "reason": "no_quorum"})
            return False
        if not self.shadow_apply(cp):
            self.audit.emit("meta_reject", {"id": proposal_id, "reason": "shadow_failed"})
            return False

        if cp.kind == "drop_agent":
            name = cp.payload.get("name")
            if self.graph.node_alive.get(name, False):
                self.graph.remove_node(name)
                self.audit.emit("meta_commit", {"id": proposal_id, "dropped": name})
                return True

        # (add_agent/policy_update hooks go here)
        self.audit.emit("meta_commit_noop", {"id": proposal_id, "kind": cp.kind})
        return True
