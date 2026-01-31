from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Callable, Optional


@dataclass
class ToolResult:
    ok: bool
    cost: float
    output: Dict[str, Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[[Dict[str, Any]], ToolResult]


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def run(self, name: str, args: Dict[str, Any]) -> ToolResult:
        if name not in self.tools:
            return ToolResult(ok=False, cost=0.01, output={"error": f"unknown_tool:{name}"})
        return self.tools[name].fn(args)
