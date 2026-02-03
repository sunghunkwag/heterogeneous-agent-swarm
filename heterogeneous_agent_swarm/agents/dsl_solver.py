import numpy as np
from typing import List, Dict, Any, Optional, Callable
from ..core.arc_dsl import ARCDSL

class DSLSolver:
    """
    Symbolic Solver for ARC.
    Searches for a program (DSL composition) that solves all training examples.
    """
    def __init__(self, max_depth: int = 2):
        self.primitives = ARCDSL.get_primitives()
        self.max_depth = max_depth
        
    def solve(self, train_examples: List[Dict[str, Any]], test_input: List[List[int]]) -> Optional[List[Dict[str, Any]]]:
        """
        Returns a list of 'write_patch' actions to construct the predicted grid.
        """
        # 1. Convert to numpy
        train_pairs = []
        for ex in train_examples:
            inp = np.array(ex["in"])
            out = np.array(ex["out"])
            train_pairs.append((inp, out))
            
        test_in = np.array(test_input)
        
        # 2. Search for a primitive that satisfies all examples
        best_program = None
        
        # Level 1 Search: Single Primitive
        if self.max_depth >= 1:
            for prim in self.primitives:
                if self._check_program(prim, train_pairs):
                    best_program = prim
                    break
                
        # Level 2 Search: Composition (f(g(x)))
        if not best_program and self.max_depth >= 2:
            for p1 in self.primitives:
                for p2 in self.primitives:
                    def composed(x, _p1=p1, _p2=p2): return _p1(_p2(x))
                    if self._check_program(composed, train_pairs):
                        best_program = composed
                        break
                if best_program: break
        
        # 3. Apply to test input
        if best_program:
            prediction = best_program(test_in)
            return self._grid_to_actions(prediction)
            
        return None

    def _check_program(self, program: Callable, train_pairs: List) -> bool:
        for inp, out in train_pairs:
            try:
                pred = program(inp)
                if pred.shape != out.shape or not np.array_equal(pred, out):
                    return False
            except Exception:
                return False
        return True

    def _grid_to_actions(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        actions = []
        h, w = grid.shape
        for y in range(h):
            for x in range(w):
                color = int(grid[y, x])
                # Optimization: Only write non-zero or if different from input?
                # For safety, write everything for now (or optimize later)
                actions.append({"x": x, "y": y, "color": color})
        return actions
