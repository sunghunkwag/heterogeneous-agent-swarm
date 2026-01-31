import unittest
from unittest.mock import MagicMock, ANY
import time

from heterogeneous_agent_swarm.core.meta_kernel_v2 import MetaKernelV2, ChangeProposal
from heterogeneous_agent_swarm.core.graph import AgentGraph
from heterogeneous_agent_swarm.core.orchestrator import Orchestrator
from heterogeneous_agent_swarm.core.audit import AuditLog

class TestMetaKernelV2(unittest.TestCase):
    def setUp(self):
        self.graph = AgentGraph()
        self.audit = MagicMock(spec=AuditLog)
        self.orch = MagicMock(spec=Orchestrator)
        self.meta = MetaKernelV2(self.graph, self.audit, self.orch, min_quorum=3)

        # Setup initial graph state
        # 5 agents to allow dropping 1 (remaining 4)
        for name in ["A", "B", "C", "D", "E"]:
            self.graph.ensure_node(name)
            # Connectivity: make them all connected to A
            if name != "A":
                self.graph.adjacency["A"].append(name)
                self.graph.adjacency[name].append("A")

    def test_propose(self):
        cp = self.meta.propose("add_agent", {"name": "F", "config": {}}, "rationale")
        self.assertTrue(cp.proposal_id.startswith("chg_"))
        self.assertEqual(cp.kind, "add_agent")
        self.audit.emit.assert_called_with("meta_propose", ANY)

    def test_vote_and_quorum(self):
        cp = self.meta.propose("drop_agent", {"name": "E"}, "rationale")
        self.assertFalse(self.meta._quorum_ok(cp))

        self.meta.vote(cp.proposal_id, "A", True)
        self.meta.vote(cp.proposal_id, "B", True)
        self.assertFalse(self.meta._quorum_ok(cp)) # 2 votes < 3

        self.meta.vote(cp.proposal_id, "C", True)
        self.assertTrue(self.meta._quorum_ok(cp)) # 3 votes >= 3

    def test_shadow_apply_add_agent(self):
        # Valid add
        cp = ChangeProposal("id", 0, "add_agent", {"name": "F", "config": {}})
        self.assertTrue(self.meta.shadow_apply(cp))

        # Duplicate name
        cp_dup = ChangeProposal("id", 0, "add_agent", {"name": "A", "config": {}})
        self.assertNotEqual(self.meta.shadow_apply(cp_dup), True) # Should return error string

    def test_shadow_apply_drop_agent(self):
        # Valid drop (E is connected to A, removing E leaves A-B-C-D connected)
        cp = ChangeProposal("id", 0, "drop_agent", {"name": "E"})
        self.assertTrue(self.meta.shadow_apply(cp))

        # Drop resulting in < 4 agents
        # Remove E first to get to 4
        self.graph.remove_node("E")
        # Now 4 alive: A, B, C, D. Removing D leaves 3.
        cp_low = ChangeProposal("id", 0, "drop_agent", {"name": "D"})
        self.assertNotEqual(self.meta.shadow_apply(cp_low), True)

        # Connectivity check
        # Reset graph to disconnected state
        # A-B, C-D. No path between A and C.
        self.graph.node_alive = {n: True for n in ["A", "B", "C", "D", "E"]}
        self.graph.adjacency = {n: [] for n in ["A", "B", "C", "D", "E"]}
        # Create a bridge: A-B-C-D-E
        self.graph.adjacency["A"] = ["B"]; self.graph.adjacency["B"] = ["A", "C"]
        self.graph.adjacency["C"] = ["B", "D"]; self.graph.adjacency["D"] = ["C", "E"]
        self.graph.adjacency["E"] = ["D"]

        # If we remove C, graph splits into A-B and D-E
        cp_split = ChangeProposal("id", 0, "drop_agent", {"name": "C"})
        self.assertNotEqual(self.meta.shadow_apply(cp_split), True)

    def test_shadow_apply_policy_update(self):
        # Valid
        cp = ChangeProposal("id", 0, "policy_update", {"veto_threshold": 0.8})
        self.assertTrue(self.meta.shadow_apply(cp))

        # Invalid range
        cp_inv = ChangeProposal("id", 0, "policy_update", {"veto_threshold": 1.5})
        self.assertNotEqual(self.meta.shadow_apply(cp_inv), True)

    def test_commit(self):
        # Add Agent
        cp_add = self.meta.propose("add_agent", {"name": "F", "config": {}}, "r")
        self.meta.vote(cp_add.proposal_id, "A", True)
        self.meta.vote(cp_add.proposal_id, "B", True)
        self.meta.vote(cp_add.proposal_id, "C", True)

        self.assertTrue(self.meta.commit(cp_add.proposal_id))
        self.assertTrue(self.graph.node_alive["F"])

        # Policy Update
        cp_pol = self.meta.propose("policy_update", {"veto_threshold": 0.9}, "r")
        self.meta.vote(cp_pol.proposal_id, "A", True)
        self.meta.vote(cp_pol.proposal_id, "B", True)
        self.meta.vote(cp_pol.proposal_id, "C", True)

        self.assertTrue(self.meta.commit(cp_pol.proposal_id))
        self.orch.update_policy.assert_called_with(veto_threshold=0.9, selection_strategy=None)

        # Drop Agent
        cp_drop = self.meta.propose("drop_agent", {"name": "F"}, "r")
        self.meta.vote(cp_drop.proposal_id, "A", True)
        self.meta.vote(cp_drop.proposal_id, "B", True)
        self.meta.vote(cp_drop.proposal_id, "C", True)

        self.assertTrue(self.meta.commit(cp_drop.proposal_id))
        self.assertFalse(self.graph.node_alive["F"])

if __name__ == '__main__':
    unittest.main()
