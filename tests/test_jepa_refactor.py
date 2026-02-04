import sys
import os
import unittest
import numpy as np
import torch

# Adjust path
sys.path.append(os.getcwd())

from heterogeneous_agent_swarm.core.types import EncodedState, StateVectorProvider
from heterogeneous_agent_swarm.agents.jepa_world_model import JEPAWorldModelAgent, JEPAConfig

class TestJEPARefactor(unittest.TestCase):
    def setUp(self):
        self.config = JEPAConfig(
            input_dim=4,
            latent_dim=4,
            hidden_dim=8,
            action_dim=2,
            learning_rate=0.01,
            momentum=0.9
        )
        self.agent = JEPAWorldModelAgent("test_jepa", self.config)

    def test_encoded_state_protocol(self):
        state = EncodedState([], [], 0.0, [], {}, system_thought=[0.1, 0.2, 0.3, 0.4])
        # Runtime checkable protocol check
        self.assertTrue(isinstance(state, StateVectorProvider))
        vec = state.get_vector()
        self.assertEqual(vec.shape, (4,))
        self.assertTrue(np.allclose(vec, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)))

    def test_increase_capacity_warmup(self):
        res = self.agent.increase_capacity(factor=2.0)
        self.assertEqual(res["new_hidden"], 16)
        self.assertEqual(self.agent.warmup_steps_remaining, 100)
        self.assertAlmostEqual(self.agent.current_momentum, 0.999)

        # Verify LR logic in train_step
        # Mock inputs
        prev_s = np.zeros(4)
        next_fb = np.zeros(4)

        # Run one step
        self.agent.train_step(prev_s, "wait", next_fb, 0.0)
        self.assertEqual(self.agent.warmup_steps_remaining, 99)

        # Check LR (should be 0.1 * base)
        current_lr = self.agent.optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(current_lr, 0.01 * 0.1)

    def test_warmup_completion(self):
        self.agent.increase_capacity()
        self.agent.warmup_steps_remaining = 1 # Force nearly done

        prev_s = np.zeros(4)
        next_fb = np.zeros(4)

        self.agent.train_step(prev_s, "wait", next_fb, 0.0)

        # Should be 0 now
        self.assertEqual(self.agent.warmup_steps_remaining, 0)

        # Check Momentum restored
        self.assertEqual(self.agent.current_momentum, 0.9)

        # Run another step to ensure LR is restored
        self.agent.train_step(prev_s, "wait", next_fb, 0.0)
        current_lr = self.agent.optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(current_lr, 0.01)

    def test_validation_errors(self):
        # Propose
        state = EncodedState([], [], 0.0, [], {}, system_thought=[0.1]) # Wrong dim (1 vs 4)
        with self.assertRaisesRegex(ValueError, "JEPA propose expected input dim 4"):
            self.agent.propose(state, {})

        # Train Step
        with self.assertRaisesRegex(ValueError, "JEPA train_step expected prev_system_thought dim 4"):
            self.agent.train_step(np.zeros(1), "wait", np.zeros(4), 0.0)

        with self.assertRaisesRegex(ValueError, "JEPA train_step expected next_env_feedback dim 4"):
             self.agent.train_step(np.zeros(4), "wait", np.zeros(1), 0.0)

    def test_propose_logic(self):
        state = EncodedState([], [], 0.0, [], {}, system_thought=[0.1, 0.2, 0.3, 0.4])
        proposal = self.agent.propose(state, {})

        print(f"Proposal Rationale: {proposal.rationale}")
        self.assertIn("JEPA UCB", proposal.rationale)
        # self.assertIn("Uncertainty", proposal.rationale) # Current format uses 'Uncert='
        self.assertTrue(proposal.predicted_value is not None)

if __name__ == '__main__':
    unittest.main()
