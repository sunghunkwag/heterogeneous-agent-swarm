import json
import os
import random
import numpy as np
from typing import Dict, Any, Tuple, List

class ARCEnv:
    """
    Real ARC AGI Environment.
    Loads tasks from 'data/ARC/data/training'.
    """
    def __init__(self, data_dir: str = "data/ARC/data/training"):
        self.data_dir = data_dir
        self.task_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        self.reset()
        
    def reset(self, level: int = 1):
        # Pick a random task
        self.current_task_file = random.choice(self.task_files)
        with open(os.path.join(self.data_dir, self.current_task_file), 'r') as f:
            self.task_data = json.load(f)
            
        print(f"[ARC] Loaded Task: {self.current_task_file}")
        
        # ARC tasks have "train" (examples) and "test" (the puzzle)
        # We will use the first "test" pair as the active problem
        self.train_pairs = self.task_data["train"]
        self.test_pair = self.task_data["test"][0]
        
        self.input_grid = np.array(self.test_pair["input"])
        self.output_grid = np.array(self.test_pair["output"])
        self.grid_size = self.input_grid.shape
        
        # Init canvas
        self.current_grid = np.zeros_like(self.input_grid)
        # Usually ARC starts with blank or copy? 
        # For 'editor' mode, let's start with input grid copy to modify, 
        # or blank if it's a construction task. 
        # Let's start with INPUT copy (often useful for transformation/noise removal).
        self.current_grid = self.input_grid.copy() 

    def observe(self):
        return {
            "task_file": self.current_task_file,
            "train_examples": [
                {"in": p["input"], "out": p["output"]} for p in self.train_pairs
            ],
            "input_grid": self.input_grid.tolist(),
            "current_grid": self.current_grid.tolist(),
            "grid_dims": self.grid_size,
            "last_test_ok": False
        }

    def step(self, tool_name: str, tool_args: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if tool_args is None: tool_args = {}
        info = {"cost": 0.1, "info": {}}
        obs = self.observe()
        
        if tool_name == "write_patch": 
            # ARC Tool: various primitives?
            # For now, simple set_pixel(x, y, color)
            x = int(tool_args.get("x", 0))
            y = int(tool_args.get("y", 0))
            c = int(tool_args.get("color", 0))
            
            h, w = self.grid_size
            if 0 <= y < h and 0 <= x < w:
                self.current_grid[y, x] = c
            info["cost"] = 0.5
            
        elif tool_name == "run_tests":
            # Check answer
            ok = np.array_equal(self.current_grid, self.output_grid)
            obs["last_test_ok"] = ok
            if ok:
                info["info"] = {"status": "SOLVED"}
            else:
                diff = np.sum(self.current_grid != self.output_grid)
                info["info"] = {"status": "FAIL", "diff": int(diff)}
            info["cost"] = 1.0

        return obs, type('obj', (object,), info)()
