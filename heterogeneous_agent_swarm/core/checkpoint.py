"""
Checkpoint Manager for Heterogeneous Agent Swarm.

Provides save/load functionality for persisting agent states
across sessions, enabling true recursive self-improvement.
"""
from __future__ import annotations
import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import torch

logger = logging.getLogger(__name__)


@dataclass
class SwarmCheckpoint:
    """Container for complete swarm state."""
    
    timestamp: float
    version: str = "1.0.0"
    
    # Agent states (serialized torch state_dicts)
    agent_states: Dict[str, Dict[str, Any]] = None
    
    # Graph state
    node_alive: Dict[str, bool] = None
    node_suppressed: Dict[str, bool] = None
    node_performance: Dict[str, float] = None
    
    # MetaKernel state
    failure_streaks: Dict[str, int] = None
    pending_proposals: List[str] = None
    
    # GNN state
    gnn_weights: Dict[str, Any] = None
    
    # Training metadata
    total_episodes: int = 0
    total_steps: int = 0
    best_reward: float = 0.0
    
    def __post_init__(self):
        if self.agent_states is None:
            self.agent_states = {}
        if self.node_alive is None:
            self.node_alive = {}
        if self.node_suppressed is None:
            self.node_suppressed = {}
        if self.node_performance is None:
            self.node_performance = {}
        if self.failure_streaks is None:
            self.failure_streaks = {}
        if self.pending_proposals is None:
            self.pending_proposals = []
        if self.gnn_weights is None:
            self.gnn_weights = {}


class CheckpointManager:
    """Manages saving and loading of swarm checkpoints."""
    
    DEFAULT_DIR = "checkpoints"
    
    def __init__(self, checkpoint_dir: str = None):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints.
        """
        self.checkpoint_dir = checkpoint_dir or self.DEFAULT_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save(self, 
             agents: Dict[str, Any],
             graph: Any,
             meta_kernel: Any,
             gnn: Any = None,
             metadata: Dict[str, Any] = None) -> str:
        """
        Save complete swarm state to checkpoint.
        
        Args:
            agents: Dictionary of agent instances.
            graph: AgentGraph instance.
            meta_kernel: MetaKernel instance.
            gnn: Optional GNN brain instance.
            metadata: Optional training metadata.
            
        Returns:
            Path to saved checkpoint.
        """
        checkpoint = SwarmCheckpoint(timestamp=time.time())
        
        # 1. Save agent states
        for name, agent in agents.items():
            try:
                if hasattr(agent, 'get_state_dict'):
                    state = agent.get_state_dict()
                    # Convert tensors to CPU and numpy for JSON serialization
                    checkpoint.agent_states[name] = self._serialize_state(state)
                    logger.debug(f"Saved state for agent: {name}")
            except Exception as e:
                logger.warning(f"Failed to save agent {name}: {e}")
        
        # 2. Save graph state
        checkpoint.node_alive = dict(graph.node_alive)
        checkpoint.node_suppressed = dict(graph.node_suppressed)
        checkpoint.node_performance = dict(graph.node_perf)
        
        # 3. Save MetaKernel state
        checkpoint.failure_streaks = dict(meta_kernel.agent_failure_streaks)
        checkpoint.pending_proposals = list(meta_kernel.proposals.keys())
        
        # 4. Save GNN state (optional)
        if gnn is not None and hasattr(gnn, 'get_weights'):
            try:
                checkpoint.gnn_weights = self._serialize_state(gnn.get_weights())
            except Exception as e:
                logger.warning(f"Failed to save GNN: {e}")
        
        # 5. Add metadata
        if metadata:
            checkpoint.total_episodes = metadata.get("episodes", 0)
            checkpoint.total_steps = metadata.get("steps", 0)
            checkpoint.best_reward = metadata.get("best_reward", 0.0)
        
        # Save to files
        checkpoint_name = f"checkpoint_{int(checkpoint.timestamp)}"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save metadata JSON
        meta_path = os.path.join(checkpoint_path, "metadata.json")
        meta_dict = {
            "timestamp": checkpoint.timestamp,
            "version": checkpoint.version,
            "node_alive": checkpoint.node_alive,
            "node_suppressed": checkpoint.node_suppressed,
            "node_performance": checkpoint.node_performance,
            "failure_streaks": checkpoint.failure_streaks,
            "pending_proposals": checkpoint.pending_proposals,
            "total_episodes": checkpoint.total_episodes,
            "total_steps": checkpoint.total_steps,
            "best_reward": checkpoint.best_reward,
            "agents": list(checkpoint.agent_states.keys())
        }
        with open(meta_path, 'w') as f:
            json.dump(meta_dict, f, indent=2)
        
        # Save agent states as torch files
        for name, state in checkpoint.agent_states.items():
            agent_path = os.path.join(checkpoint_path, f"{name}.pt")
            torch.save(state, agent_path)
        
        # Save GNN state
        if checkpoint.gnn_weights:
            gnn_path = os.path.join(checkpoint_path, "gnn.pt")
            torch.save(checkpoint.gnn_weights, gnn_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load(self, checkpoint_path: str) -> SwarmCheckpoint:
        """
        Load checkpoint from path.
        
        Args:
            checkpoint_path: Path to checkpoint directory.
            
        Returns:
            SwarmCheckpoint with loaded data.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load metadata
        meta_path = os.path.join(checkpoint_path, "metadata.json")
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        checkpoint = SwarmCheckpoint(
            timestamp=meta["timestamp"],
            version=meta.get("version", "1.0.0"),
            node_alive=meta.get("node_alive", {}),
            node_suppressed=meta.get("node_suppressed", {}),
            node_performance=meta.get("node_performance", {}),
            failure_streaks=meta.get("failure_streaks", {}),
            pending_proposals=meta.get("pending_proposals", []),
            total_episodes=meta.get("total_episodes", 0),
            total_steps=meta.get("total_steps", 0),
            best_reward=meta.get("best_reward", 0.0)
        )
        
        # Load agent states
        for agent_name in meta.get("agents", []):
            agent_path = os.path.join(checkpoint_path, f"{agent_name}.pt")
            if os.path.exists(agent_path):
                checkpoint.agent_states[agent_name] = torch.load(agent_path, weights_only=False)
                logger.debug(f"Loaded state for agent: {agent_name}")
        
        # Load GNN state
        gnn_path = os.path.join(checkpoint_path, "gnn.pt")
        if os.path.exists(gnn_path):
            checkpoint.gnn_weights = torch.load(gnn_path, weights_only=False)
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def load_latest(self) -> Optional[SwarmCheckpoint]:
        """
        Load the most recent checkpoint.
        
        Returns:
            SwarmCheckpoint or None if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            logger.info("No checkpoints found")
            return None
        
        latest = checkpoints[-1]  # Sorted by timestamp
        return self.load(latest)
    
    def list_checkpoints(self) -> List[str]:
        """
        List all available checkpoints sorted by timestamp.
        
        Returns:
            List of checkpoint paths.
        """
        if not os.path.exists(self.checkpoint_dir):
            return []
        
        checkpoints = []
        for name in os.listdir(self.checkpoint_dir):
            path = os.path.join(self.checkpoint_dir, name)
            meta_path = os.path.join(path, "metadata.json")
            if os.path.isdir(path) and os.path.exists(meta_path):
                checkpoints.append(path)
        
        # Sort by checkpoint name (which contains timestamp)
        checkpoints.sort()
        return checkpoints
    
    def apply_checkpoint(self,
                        checkpoint: SwarmCheckpoint,
                        agents: Dict[str, Any],
                        graph: Any,
                        meta_kernel: Any,
                        gnn: Any = None) -> Dict[str, bool]:
        """
        Apply checkpoint to restore swarm state.
        
        Args:
            checkpoint: SwarmCheckpoint to apply.
            agents: Dictionary of agent instances.
            graph: AgentGraph instance.
            meta_kernel: MetaKernel instance.
            gnn: Optional GNN brain instance.
            
        Returns:
            Dictionary of agent names to load success status.
        """
        results = {}
        
        # 1. Restore agent states
        for name, state in checkpoint.agent_states.items():
            if name in agents:
                try:
                    agent = agents[name]
                    if hasattr(agent, 'load_compatible_state'):
                        agent.load_compatible_state(state)
                        results[name] = True
                        logger.info(f"Restored agent: {name}")
                    elif hasattr(agent, 'load_state_dict'):
                        agent.load_state_dict(state)
                        results[name] = True
                except Exception as e:
                    logger.warning(f"Failed to restore agent {name}: {e}")
                    results[name] = False
        
        # 2. Restore graph state
        for name, alive in checkpoint.node_alive.items():
            graph.node_alive[name] = alive
        for name, suppressed in checkpoint.node_suppressed.items():
            graph.node_suppressed[name] = suppressed
        for name, perf in checkpoint.node_performance.items():
            graph.node_perf[name] = perf
        
        # 3. Restore MetaKernel state
        for name, streak in checkpoint.failure_streaks.items():
            meta_kernel.agent_failure_streaks[name] = streak
        
        # 4. Restore GNN state (optional)
        if gnn is not None and checkpoint.gnn_weights and hasattr(gnn, 'set_weights'):
            try:
                gnn.set_weights(checkpoint.gnn_weights)
            except Exception as e:
                logger.warning(f"Failed to restore GNN: {e}")
        
        logger.info(f"Checkpoint applied: {len(results)} agents restored")
        return results
    
    def _serialize_state(self, state: Dict) -> Dict:
        """Convert state dict tensors to serializable format."""
        serialized = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = value.cpu()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_state(value)
            else:
                serialized[key] = value
        return serialized
