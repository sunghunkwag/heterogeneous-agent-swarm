from __future__ import annotations
import numpy as np
from typing import Dict, List, Any

class LightweightGNN:
    """
    Global Workspace Manager.
    Uses Message Passing to aggregate state from all agents into a unified 'System Thought'.
    """
    def __init__(self, agent_names: List[str], hidden_dim: int = 16):
        self.output_dim = hidden_dim
        self.node_indices = {name: i for i, name in enumerate(agent_names)}
        self.num_nodes = len(agent_names)
        
        # Simple learnable weights (random init for now, but trainable)
        # Message function: W_msg * h_j
        self.W_msg = np.random.randn(hidden_dim, hidden_dim) * 0.1
        # Update function: GRU-like or simple tanh(W_up * [h_i, m_i])
        self.W_up = np.random.randn(hidden_dim * 2, hidden_dim) * 0.1
        
        # Node states (hidden states)
        self.H = np.zeros((self.num_nodes, hidden_dim))

    def update(self, observations: Dict[str, np.ndarray]):
        """
        observations: {agent_name: feature_vector}
        """
        # Save previous state for Hebbian learning
        self.H_prev = self.H.copy()
        
        # 1. Update individual node states with observation (Input injection)
        for name, obs in observations.items():
            if name in self.node_indices:
                idx = self.node_indices[name]
                proj = obs[:self.output_dim] 
                if len(proj) < self.output_dim:
                    proj = np.pad(proj, (0, self.output_dim - len(proj)))
                self.H[idx] += proj * 0.1 # leaky integration

        # 2. Message Passing (All-to-All for global workspace)
        # Msg_i = Sum(W_msg * H_j)
        Messages = np.dot(self.H, self.W_msg) # (N, dim) -> (N, dim)
        Global_Context = np.mean(self.H, axis=0) # (dim,)
        
        # 3. Update State
        # H_new = tanh(W_up * [H_old, Global_Context])
        for i in range(self.num_nodes):
            inp = np.concatenate([self.H[i], Global_Context])
            update = np.tanh(np.dot(inp, self.W_up))
            self.H[i] = 0.9 * self.H[i] + 0.1 * update

        return Global_Context # The "System Thought"

    def train(self, reward: float):
        """
        Online Learning: Hebbian + Reward Modulation.
        If reward > 0, reinforce connections between active nodes.
        If reward < 0, weaken them (Anti-Hebbian).
        
        Delta_W_msg = learning_rate * reward * (H_post.T * H_pre)
        """
        # Simple Hebbian on Message Weights
        # We want to reinforce the pattern H that led to success
        
        # Outer product of H (State correlation)
        # We aggregate correlations across all nodes? 
        # Simplified: Update W_msg based on average correlation
        
        learning_rate = 0.01
        
        # approximate H_pre as H (recurrent)
        # Co-activation matrix
        co_activation = np.dot(self.H.T, self.H) # dim x dim
        
        # Normalize
        co_activation = co_activation / (self.num_nodes + 1e-5)
        
        # Update
        self.W_msg += learning_rate * reward * co_activation
        
        # Clip to prevent explosion
        self.W_msg = np.clip(self.W_msg, -1.0, 1.0)
        
        return np.mean(np.abs(self.W_msg)) # Return mean weight magnitude

    def get_system_state(self):
        return np.mean(self.H, axis=0)
