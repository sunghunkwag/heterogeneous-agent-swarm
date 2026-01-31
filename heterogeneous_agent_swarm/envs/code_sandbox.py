from typing import Dict, Any, Tuple, Optional
import random

class CodeSandboxEnv:
    """
    Simulates a code repository where we try to fix a bug by applying patches.
    Internally uses the 'Arithmetic Progression' logic from v0.1 as the hidden task.
    """
    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        self.reset()
        
    def reset(self, level: int = 1):
        self.level = level
        self.current_code = []
        
        if self.level == 1:
            # Arithmetic: Start + n*Step
            start = random.randint(0, 5)
            step = random.randint(1, 2)
            self.target = [start + (i*step) for i in range(5)]
            self.pattern_name = "Arithmetic Progression"
            
        elif self.level == 2:
            # Geometric: Start * (Step^n)
            # Keep small to avoid huge numbers
            start = random.randint(1, 3)
            step = 2
            self.target = [start * (step**i) for i in range(5)]
            self.pattern_name = "Geometric Progression"
            
        elif self.level == 3:
            # Fibonacci-like: a, b, a+b, ...
            a = random.randint(0, 3)
            b = random.randint(1, 3)
            self.target = [a, b]
            while len(self.target) < 5:
                self.target.append(self.target[-1] + self.target[-2])
            self.pattern_name = "Fibonacci Sequence"
            
        print(f"[Sandbox] Setup Level {self.level} ({self.pattern_name}): Target={self.target}")

    def step(self, action: str, args: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Any]:
        """
        Actions:
        - run_tests: Check if current code matches target.
        - write_patch: Append/Modify code buffer.
        """
        info = {"cost": 0.1, "info": {}}
        obs = {
            "last_test_ok": False,
            "buffer": self.current_code,
            "level": getattr(self, "level", 1)
        }
        
        if action == "run_tests":
            matches = 0
            for i, c in enumerate(self.current_code):
                if i < len(self.target) and c == self.target[i]:
                    matches += 1
            
            ok = (matches == len(self.target) and len(self.current_code) == len(self.target))
            obs["last_test_ok"] = ok
            info["cost"] = 1.0
            info["info"] = {"matches": matches}
            
        elif action == "write_patch":
            val = args.get("value", 0) if args else 0
            self.current_code.append(val)
            info["cost"] = 0.5
            
        elif action == "summarize":
            info["cost"] = 0.05
            
        return obs, type('obj', (object,), info)()

