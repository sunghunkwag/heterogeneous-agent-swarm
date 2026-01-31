from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

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

@dataclass
class AgentConfig:
    name: str
    role: str
    device: str = "cpu"
