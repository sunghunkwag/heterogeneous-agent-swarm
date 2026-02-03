from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class StateVectorProvider(Protocol):
    def get_vector(self) -> np.ndarray:
        """Returns the vector representation of the state."""
        ...

@dataclass
class Proposal:
    action_type: str  # This maps to tool_name in v0.2
    action_value: Any # Arguments for the tool
    confidence: float
    predicted_value: float
    estimated_cost: float
    rationale: str
    source_agent: str
    artifacts: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EncodedState:
    task_latent: List[float]
    agent_health_latent: List[float]
    uncertainty: float
    risk_flags: List[str]
    raw_obs: Dict[str, Any]
    system_thought: Optional[List[float]] = None

    def get_vector(self) -> np.ndarray:
        if self.system_thought is not None:
            return np.array(self.system_thought, dtype=np.float32)
        # Fallback or empty if not set
        return np.array([], dtype=np.float32)

@dataclass
class AgentConfig:
    name: str
    role: str
    device: str = "cpu"
