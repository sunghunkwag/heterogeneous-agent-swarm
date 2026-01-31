from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EvalResult:
    """
    Result of a system evaluation.

    Args:
        success: Whether the immediate objectives were met.
        score: Numerical score (-1.0 to 1.0).
        notes: Additional metadata about the evaluation.
    """
    success: bool
    score: float
    notes: Dict[str, Any]


class Evaluator:
    """
    Evaluates outcome signals and produces a scalar score for learning / orchestration.
    """

    def evaluate(self, blackboard: Dict[str, Any]) -> EvalResult:
        """
        Assess the current state from the blackboard.

        Args:
            blackboard: Dictionary representation of the Blackboard.

        Returns:
            EvalResult object containing score and status.
        """
        obs = blackboard.get("obs", {})
        signals = blackboard.get("signals", {})
        last_test_ok = bool(obs.get("last_test_ok", False))
        failures = int(obs.get("failures", 0))

        # Simple but grounded:
        # pass tests => strong success
        # repeated failures => penalty
        score = 0.0
        success = False

        if last_test_ok:
            success = True
            score += 1.0
        score -= 0.10 * failures

        # if we thrashed tool calls too much, penalize
        steps = blackboard.get("step_history", [])
        score -= 0.01 * max(0, len(steps) - 6)

        score = max(-1.0, min(1.0, score))
        return EvalResult(success=success, score=score, notes={"failures": failures, "steps": len(steps)})
