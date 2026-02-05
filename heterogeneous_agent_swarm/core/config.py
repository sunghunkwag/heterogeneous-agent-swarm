"""
Centralized Configuration Module for Heterogeneous Agent Swarm.

This module contains all configurable thresholds and parameters
that were previously scattered as magic numbers throughout the codebase.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
import os


@dataclass
class ControlThresholds:
    """Thresholds for agent control and suppression logic."""
    
    # Agent failure detection
    failure_threshold: float = 0.8  # Loss above this is considered failure
    streak_limit: int = 3  # Consecutive failures before suppression
    recovery_threshold: float = 0.2  # Loss below this triggers unsuppression
    
    # Minimum agent count for safety
    min_agent_count: int = 4


@dataclass
class UncertaintyThresholds:
    """GNN/Orchestrator uncertainty-based gating thresholds."""
    
    # High uncertainty triggers panic mode
    high: float = 0.1
    
    # Low uncertainty triggers convergence lock
    low: float = 0.01


@dataclass 
class OrchestratorConfig:
    """Configuration for orchestrator decision-making."""
    
    default_veto_threshold: float = 0.5
    default_selection_strategy: str = "weighted_perf"
    
    # Memory-based experience replay
    memory_similarity_boost_factor: float = 0.3
    positive_experience_threshold: float = 0.2
    
    # Available strategies
    valid_strategies: tuple = ("weighted_perf", "consensus", "random")


@dataclass
class JEPACapacityConfig:
    """Configuration for JEPA dynamic capacity adjustment."""
    
    # Capacity scaling factors
    increase_factor: float = 1.5
    decrease_factor: float = 0.7
    min_hidden_dim: int = 16
    
    # Warmup after capacity change
    warmup_steps: int = 100
    warmup_lr_multiplier: float = 0.1
    warmup_momentum: float = 0.999
    
    # UCB exploration
    ucb_beta_base: float = 0.1
    ucb_error_scaling: float = 19.0
    
    # Panic mode
    panic_exploration_multiplier: float = 5.0
    panic_wait_penalty: float = 0.1


@dataclass
class MetaKernelConfig:
    """Configuration for MetaKernel structural adaptation."""
    
    min_quorum: int = 3
    
    # NAS suggestion
    underperformance_threshold: float = 0.3
    
    # Emergency rotation scoring
    failure_streak_penalty: float = 0.15


@dataclass
class SwarmConfig:
    """Master configuration containing all sub-configs."""
    
    control: ControlThresholds = field(default_factory=ControlThresholds)
    uncertainty: UncertaintyThresholds = field(default_factory=UncertaintyThresholds)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    jepa: JEPACapacityConfig = field(default_factory=JEPACapacityConfig)
    meta_kernel: MetaKernelConfig = field(default_factory=MetaKernelConfig)
    
    @classmethod
    def from_json(cls, path: str) -> "SwarmConfig":
        """Load configuration from JSON file."""
        if not os.path.exists(path):
            return cls()
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            control=ControlThresholds(**data.get("control", {})),
            uncertainty=UncertaintyThresholds(**data.get("uncertainty", {})),
            orchestrator=OrchestratorConfig(**data.get("orchestrator", {})),
            jepa=JEPACapacityConfig(**data.get("jepa", {})),
            meta_kernel=MetaKernelConfig(**data.get("meta_kernel", {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict
        return asdict(self)
    
    def save_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Global default instance
DEFAULT_CONFIG = SwarmConfig()
