from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Callable, Optional


@dataclass
class ToolResult:
    """Result returned by a tool execution."""
    ok: bool
    cost: float
    output: Dict[str, Any]


@dataclass
class ToolSpec:
    """Specification for a registered tool."""
    name: str
    description: str
    fn: Callable[[Dict[str, Any]], ToolResult]


class ToolRegistry:
    """
    Registry of available tools for the swarm.
    """
    def __init__(self):
        self.tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register a new tool."""
        self.tools[spec.name] = spec

    def run(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            name: The tool name.
            args: Arguments for the tool.

        Returns:
            ToolResult containing status, cost, and output.
        """
        if name not in self.tools:
            return ToolResult(ok=False, cost=0.01, output={"error": f"unknown_tool:{name}"})
        return self.tools[name].fn(args)
