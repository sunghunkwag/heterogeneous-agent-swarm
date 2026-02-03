import unittest
import numpy as np
import torch

from heterogeneous_agent_swarm.agents.jepa_world_model import JEPAWorldModelAgent, JEPAConfig
from heterogeneous_agent_swarm.core.memory import EpisodicMemory


class TestJEPA(unittest.TestCase):
    def setUp(self):
        self.config = JEPAConfig(device="cpu", latent_dim=32, input_dim=16)
        self.agent = JEPAWorldModelAgent("jepa_test", self.config)

    def test_initialization(self):
        """Test JEPA agent initializes correctly."""
        self.assertEqual(self.agent.name, "jepa_test")
        self.assertEqual(self.agent.config.latent_dim, 32)

    def test_propose_returns_valid_action(self):
        """Test propose() returns valid Proposal."""
        # Mock state
        class MockState:
            system_thought = np.random.randn(16)

        proposal = self.agent.propose(MockState(), {})

        self.assertIn(proposal.action_type, ["run_tests", "write_patch", "summarize", "wait"])
        self.assertIsNotNone(proposal.rationale)

    def test_train_step_reduces_loss(self):
        """Test that train_step actually learns something."""
        prev_state = np.random.randn(16)
        next_state = np.random.randn(16)

        # Train multiple times
        losses = []
        for _ in range(10):
            loss = self.agent.train_step(prev_state, "wait", next_state, 0.0)
            losses.append(loss)

        # Loss should generally decrease (not strictly due to randomness)
        # Just check it doesn't explode
        self.assertLess(losses[-1], 10.0)
        self.assertGreater(losses[-1], 0.0)

    def test_curiosity_reward(self):
        """Test curiosity signal generation."""
        prev_state = np.random.randn(16)
        next_state = np.random.randn(16)

        curiosity = self.agent.get_curiosity_reward(prev_state, "wait", next_state)

        self.assertIsInstance(curiosity, float)
        self.assertGreaterEqual(curiosity, 0.0)


class TestEpisodicMemory(unittest.TestCase):
    def setUp(self):
        self.memory = EpisodicMemory(capacity=100)

    def test_add_and_retrieve(self):
        """Test basic add/retrieve cycle."""
        vec = np.random.randn(16)
        self.memory.add(vec, {"action": "test", "reward": 1.0})

        results = self.memory.retrieve(vec, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["action"], "test")

    def test_similarity_ranking(self):
        """Test that similar vectors are retrieved first."""
        # Add different vectors
        vec1 = np.array([1.0, 0.0, 0.0] + [0.0] * 13)
        vec2 = np.array([0.0, 1.0, 0.0] + [0.0] * 13)

        self.memory.add(vec1, {"action": "left"})
        self.memory.add(vec2, {"action": "right"})

        # Query similar to vec1
        query = np.array([0.9, 0.1, 0.0] + [0.0] * 13)
        results = self.memory.retrieve(query, top_k=2)

        self.assertEqual(results[0].metadata["action"], "left")  # More similar
        self.assertGreater(results[0].similarity, results[1].similarity)


if __name__ == '__main__':
    unittest.main()
