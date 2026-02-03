import unittest
from unittest.mock import MagicMock
import numpy as np

from heterogeneous_agent_swarm.core.graph import AgentGraph
from heterogeneous_agent_swarm.core.meta_kernel_v2 import MetaKernelV2
from heterogeneous_agent_swarm.core.gnn_brain import LightweightGNN
from heterogeneous_agent_swarm.core.orchestrator import Orchestrator
from heterogeneous_agent_swarm.core.types import Proposal
from heterogeneous_agent_swarm.main import AdvancedAISystem

class TestCStageRigor(unittest.TestCase):

    def test_agent_graph_suppression(self):
        graph = AgentGraph()
        graph.ensure_node("agent_a")

        self.assertTrue(graph.node_alive["agent_a"])
        self.assertFalse(graph.node_suppressed["agent_a"])

        graph.suppress_node("agent_a")
        self.assertTrue(graph.node_suppressed["agent_a"])

        graph.unsuppress_node("agent_a")
        self.assertFalse(graph.node_suppressed["agent_a"])

    def test_metakernel_suppression_logic(self):
        graph = AgentGraph()
        graph.ensure_node("agent_bad")

        audit = MagicMock()
        orch = MagicMock()
        agents = {"agent_bad": MagicMock()}

        meta = MetaKernelV2(graph, audit, orch, agents)

        # Fail 1
        res = meta.suppress_agent("agent_bad", 0.9)
        self.assertIsNone(res)
        self.assertEqual(meta.consecutive_failures["agent_bad"], 1)

        # Fail 2
        res = meta.suppress_agent("agent_bad", 0.9)
        self.assertIsNone(res)

        # Fail 3 -> Trigger
        res = meta.suppress_agent("agent_bad", 0.9)
        self.assertIsNotNone(res)
        self.assertEqual(res["action"], "suppressed")
        self.assertTrue(graph.node_suppressed["agent_bad"])

    def test_gnn_uncertainty(self):
        gnn = LightweightGNN(["a", "b"])
        # Low variance (Convergence)
        gnn.H = np.array([[0.5, 0.5], [0.51, 0.49]])
        unc = gnn.get_system_uncertainty()
        self.assertLess(unc, 0.05)

        # High variance (Chaos)
        gnn.H = np.array([[0.0, 0.0], [1.0, 1.0]])
        unc = gnn.get_system_uncertainty()
        self.assertGreater(unc, 0.2)

    def test_orchestrator_gating(self):
        graph = AgentGraph()
        graph.ensure_node("diffusion_agent")
        graph.ensure_node("neurosym_agent")
        graph.ensure_node("suppressed_agent")
        graph.suppress_node("suppressed_agent")

        orch = Orchestrator(graph)

        proposals = {
            "diffusion_agent": Proposal("write_patch", {}, 0.8, 1.0, 0.1, "explore", "diffusion_agent"),
            "neurosym_agent": Proposal("run_tests", {}, 0.7, 1.0, 0.1, "verify", "neurosym_agent"),
            "suppressed_agent": Proposal("wait", {}, 0.9, 1.0, 0.1, "wait", "suppressed_agent")
        }

        # 1. Normal Mode
        tool, _, dbg = orch.choose(proposals, None, system_uncertainty=0.15)
        self.assertEqual(tool, "write_patch")
        self.assertNotIn("suppressed_agent", dbg["all_proposals"])

        # 2. Convergence Mode
        tool, _, dbg = orch.choose(proposals, None, system_uncertainty=0.01)
        self.assertEqual(tool, "run_tests")
        self.assertNotIn("diffusion_agent", dbg["all_proposals"])

        # 3. Panic Mode
        tool, _, dbg = orch.choose(proposals, None, system_uncertainty=0.4)
        self.assertEqual(tool, "run_tests")
        self.assertEqual(dbg["reason"], "panic_mode_gating")

    def test_jepa_vector_construction(self):
        # We need to mock minimal parts of AdvancedAISystem to avoid full init
        # Create a dummy class that inherits or mocks the method
        # Actually it's an instance method, so we can just grab the unbound method or instantiate
        # But instantiation triggers agent loading.
        # Let's mock the env and work memory attributes

        # We can extract the method or just copy the logic test.
        # But for integration correctness, let's try to instantiate but mock __init__

        with unittest.mock.patch('heterogeneous_agent_swarm.main.AdvancedAISystem.__init__', return_value=None) as mock_init:
            sys_obj = AdvancedAISystem()
            sys_obj.work = MagicMock()

            # Setup inputs
            obs = {"last_test_ok": True, "error": False}
            cost = 0.123
            sys_obj.work.get.return_value = {"output": {"diff_count": 50}}

            # Call method
            vec = AdvancedAISystem._get_env_feedback_vector(sys_obj, obs, cost)

            self.assertEqual(len(vec), 16)
            self.assertEqual(vec[0], 1.0)
            self.assertEqual(vec[1], 0.0)
            self.assertEqual(vec[2], 0.5)
            self.assertEqual(vec[4], 0.123)

if __name__ == '__main__':
    unittest.main()
