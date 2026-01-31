from __future__ import annotations
from typing import Dict, Any, Tuple
from dataclasses import dataclass
import time

from ..core.tools import ToolRegistry, ToolResult


@dataclass
class ToolEnvState:
    timestep: int = 0
    failures: int = 0
    last_test_ok: bool = False


class ToolEnv:
    """
    A general tool-based environment wrapper.
    Reality cost is tool cost + time.
    """
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.state = ToolEnvState()

    def observe(self) -> Dict[str, Any]:
        return {
            "timestep": self.state.timestep,
            "failures": self.state.failures,
            "last_test_ok": self.state.last_test_ok,
        }

    def step_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.state.timestep += 1
        t0 = time.time()
        res = self.registry.run(tool_name, tool_args)
        wall = time.time() - t0

        # Update state based on common tool outputs (convention)
        ok = bool(res.ok)
        if tool_name == "run_tests":
            self.state.last_test_ok = ok
            if not ok:
                self.state.failures += 1

        info = {
            "tool": tool_name,
            "args": tool_args,
            "ok": ok,
            "cost": res.cost + 0.02 * min(5.0, wall),
            "output": res.output,
        }
        return self.observe(), info
