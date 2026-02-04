
import unittest
import numpy as np
import sys
import os

# Ensure proper import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from heterogeneous_agent_swarm.agents.dsl_solver import DSLSolver

class TestDSLSolver(unittest.TestCase):
    def setUp(self):
        self.solver = DSLSolver()

    def test_grid_to_actions_new_shape(self):
        """Test optimization when no input grid is provided (or shape mismatch implied by usage)."""
        # Grid 2x2, diagonal 1s
        grid = np.array([[1, 0], [0, 1]])
        # Should return actions only for non-zero pixels
        actions = self.solver._grid_to_actions(grid)

        self.assertEqual(len(actions), 2, "Should have 2 actions for 2 non-zero pixels")
        self.assertEqual(actions[0]['color'], 1)
        self.assertEqual(actions[1]['color'], 1)

    def test_grid_to_actions_diff_change(self):
        """Test diff optimization: input provided, same shape, pixel changed."""
        input_grid = np.zeros((2, 2))
        # Output: one pixel changed to 5
        grid = np.array([[0, 0], [0, 5]])

        actions = self.solver._grid_to_actions(grid, input_grid=input_grid)

        self.assertEqual(len(actions), 1, "Should have 1 action for changed pixel")
        self.assertEqual(actions[0]['x'], 1)
        self.assertEqual(actions[0]['y'], 1)
        self.assertEqual(actions[0]['color'], 5)

    def test_grid_to_actions_diff_erase(self):
        """Test diff optimization: input provided, same shape, pixel changed to 0."""
        input_grid = np.array([[0, 0], [0, 5]])
        # Output: pixel is 0
        grid = np.zeros((2, 2))

        actions = self.solver._grid_to_actions(grid, input_grid=input_grid)

        self.assertEqual(len(actions), 1, "Should have 1 action for erased pixel")
        self.assertEqual(actions[0]['x'], 1)
        self.assertEqual(actions[0]['y'], 1)
        self.assertEqual(actions[0]['color'], 0)

if __name__ == '__main__':
    unittest.main()
