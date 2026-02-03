from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class JEPAConfig:
    device: str = "cpu"
    learning_rate: float = 1e-3
    latent_dim: int = 64
    input_dim: int = 16
    action_dim: int = 8
    momentum: float = 0.99
    hidden_dim: int = 128  # Added for variable capacity

class JEPAEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class JEPAPredictor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent, action], dim=-1)
        return self.net(x)

class JEPAWorldModelAgent:
    """
    JEPA Agent with self-supervised predictive learning and intrinsic motivation.
    """

    ACTION_TO_IDX = {
        "run_tests": 0, "write_patch": 1, "summarize": 2,
        "wait": 3, "emergency_stop": 4, "none": 5
    }

    def __init__(self, name: str, config: JEPAConfig):
        self.name = name
        self.config = config
        self.device = torch.device(config.device)

        # Networks
        self.encoder = JEPAEncoder(config.input_dim, config.latent_dim, config.hidden_dim).to(self.device)
        self.predictor = JEPAPredictor(config.latent_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_encoder = JEPAEncoder(config.input_dim, config.latent_dim, config.hidden_dim).to(self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=config.learning_rate
        )

        # Action embeddings (learnable)
        self.action_embedding = nn.Embedding(len(self.ACTION_TO_IDX), config.action_dim).to(self.device)

        self.train_step_count = 0
        self.parameter_count = self._count_params()

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad) + \
               sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)

    def get_capacity_metric(self) -> float:
        """Return a scalar metric representing current capacity/utilization."""
        # Normalize arbitrarily for now (e.g. against base of ~5000 params)
        return self._count_params() / 5000.0

    def increase_capacity(self, factor: float = 1.5) -> dict:
        """Increase model capacity by widening hidden layers."""
        old_hidden = self.config.hidden_dim
        new_hidden = int(old_hidden * factor)

        # Create new networks with wider hidden layers
        new_encoder = JEPAEncoder(self.config.input_dim, self.config.latent_dim, new_hidden).to(self.device)
        new_predictor = JEPAPredictor(self.config.latent_dim, self.config.action_dim, new_hidden).to(self.device)

        # Attempt to copy weights where possible (simple slicing)
        # Note: This loses some information but preserves rough initialization scale
        with torch.no_grad():
            # Encoder
            new_encoder.net[0].weight[:old_hidden, :] = self.encoder.net[0].weight
            new_encoder.net[0].bias[:old_hidden] = self.encoder.net[0].bias
            new_encoder.net[2].weight[:, :old_hidden] = self.encoder.net[2].weight

            # Predictor
            new_predictor.net[0].weight[:old_hidden, :] = self.predictor.net[0].weight
            new_predictor.net[0].bias[:old_hidden] = self.predictor.net[0].bias
            new_predictor.net[2].weight[:, :old_hidden] = self.predictor.net[2].weight

        # Update components
        self.encoder = new_encoder
        self.predictor = new_predictor
        self.target_encoder = JEPAEncoder(self.config.input_dim, self.config.latent_dim, new_hidden).to(self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.config.hidden_dim = new_hidden

        # Recreate optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=self.config.learning_rate
        )

        self.parameter_count = self._count_params()

        return {
            "action": "increase_capacity",
            "old_hidden": old_hidden,
            "new_hidden": new_hidden,
            "new_params": self.parameter_count
        }

    def decrease_capacity(self, factor: float = 0.7) -> dict:
        """Decrease model capacity."""
        old_hidden = self.config.hidden_dim
        new_hidden = max(16, int(old_hidden * factor))

        if new_hidden >= old_hidden:
             return {"action": "decrease_capacity", "status": "noop_limit_reached"}

        # Create new narrower networks
        new_encoder = JEPAEncoder(self.config.input_dim, self.config.latent_dim, new_hidden).to(self.device)
        new_predictor = JEPAPredictor(self.config.latent_dim, self.config.action_dim, new_hidden).to(self.device)

        # Slice weights
        with torch.no_grad():
            # Encoder
            new_encoder.net[0].weight = nn.Parameter(self.encoder.net[0].weight[:new_hidden, :])
            new_encoder.net[0].bias = nn.Parameter(self.encoder.net[0].bias[:new_hidden])
            new_encoder.net[2].weight = nn.Parameter(self.encoder.net[2].weight[:, :new_hidden])

            # Predictor
            new_predictor.net[0].weight = nn.Parameter(self.predictor.net[0].weight[:new_hidden, :])
            new_predictor.net[0].bias = nn.Parameter(self.predictor.net[0].bias[:new_hidden])
            new_predictor.net[2].weight = nn.Parameter(self.predictor.net[2].weight[:, :new_hidden])

        # Update
        self.encoder = new_encoder
        self.predictor = new_predictor
        self.target_encoder = JEPAEncoder(self.config.input_dim, self.config.latent_dim, new_hidden).to(self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.config.hidden_dim = new_hidden

        # Recreate optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=self.config.learning_rate
        )

        self.parameter_count = self._count_params()

        return {
            "action": "decrease_capacity",
            "old_hidden": old_hidden,
            "new_hidden": new_hidden,
            "new_params": self.parameter_count
        }

    def _get_action_tensor(self, action_name: str) -> torch.Tensor:
        """Convert action name to embedding tensor."""
        idx = self.ACTION_TO_IDX.get(action_name, 5)  # Default to "none"
        idx_tensor = torch.tensor([idx], device=self.device)
        return self.action_embedding(idx_tensor).squeeze(0)

    def propose(self, state: Any, memory: Dict[str, Any]):
        """
        Agent interface: propose action based on world model prediction.
        Returns Proposal with predicted value of each action.
        """
        from ..core.types import Proposal

        # Get current state vector
        if hasattr(state, 'system_thought'):
            state_vec = np.array(state.system_thought)
        else:
            state_vec = np.zeros(self.config.input_dim)

        # Evaluate each possible action using world model
        action_values = {}
        for action_name in ["run_tests", "write_patch", "summarize", "wait"]:
            with torch.no_grad():
                state_t = torch.FloatTensor(state_vec).to(self.device)
                if state_t.dim() == 1:
                    state_t = state_t.unsqueeze(0)

                latent = self.encoder(state_t)
                action_emb = self._get_action_tensor(action_name).unsqueeze(0)
                next_latent = self.predictor(latent, action_emb)

                # Value = negative prediction error (lower error = better)
                # Plus small bonus for novelty (high variance in latent)
                novelty = torch.var(next_latent).item()
                action_values[action_name] = novelty * 0.1

        # Pick action with highest novelty (curiosity-driven)
        best_action = max(action_values, key=action_values.get)

        return Proposal(
            source_agent=self.name,
            action_type=best_action,
            action_value={},
            confidence=0.5,  # Neutral confidence
            predicted_value=action_values[best_action],
            estimated_cost=0.1,
            rationale=f"JEPA predicts novelty for {best_action}: {action_values[best_action]:.3f}",
            artifacts={"action_values": action_values}
        )

    def train_step(self, prev_obs_vector: np.ndarray, action_name: str,
                   next_obs_vector: np.ndarray, reward: float) -> float:
        """
        Single gradient step on JEPA loss.

        Args:
            prev_obs_vector: Previous observation (e.g., GNN system_thought)
            action_name: Action taken
            next_obs_vector: Resulting observation
            reward: External reward (not used in loss, but logged)

        Returns:
            prediction_error: Float representing curiosity signal
        """
        self.train_step_count += 1
        self.optimizer.zero_grad()

        # Convert to tensors
        s_t = torch.FloatTensor(prev_obs_vector).to(self.device)
        s_t1 = torch.FloatTensor(next_obs_vector).to(self.device)

        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)
        if s_t1.dim() == 1:
            s_t1 = s_t1.unsqueeze(0)

        # Forward
        z_t = self.encoder(s_t)
        z_t1_target = self.target_encoder(s_t1).detach()

        action_emb = self._get_action_tensor(action_name).unsqueeze(0)
        z_t1_pred = self.predictor(z_t, action_emb)

        # Prediction loss (MSE)
        pred_loss = F.mse_loss(z_t1_pred, z_t1_target)

        # Variance loss (prevent collapse)
        # Fix: unbiased=False to suppress warning
        var_loss = torch.mean(F.relu(1.0 - torch.std(z_t, dim=0, unbiased=False))) + \
                   torch.mean(F.relu(1.0 - torch.std(z_t1_target, dim=0, unbiased=False)))

        # Total loss
        loss = pred_loss + 0.1 * var_loss

        # Backward
        loss.backward()
        self.optimizer.step()

        # Momentum update target encoder
        with torch.no_grad():
            m = self.config.momentum
            for param, target_param in zip(self.encoder.parameters(),
                                           self.target_encoder.parameters()):
                target_param.data = m * target_param.data + (1 - m) * param.data

        return pred_loss.item()

    def get_curiosity_reward(self, prev_obs_vector: np.ndarray, action_name: str,
                             next_obs_vector: np.ndarray) -> float:
        """
        Returns prediction error as intrinsic reward (curiosity signal).
        Higher = more surprising = more interesting to explore.
        """
        with torch.no_grad():
            s_t = torch.FloatTensor(prev_obs_vector).to(self.device)
            s_t1 = torch.FloatTensor(next_obs_vector).to(self.device)

            if s_t.dim() == 1:
                s_t = s_t.unsqueeze(0)
            if s_t1.dim() == 1:
                s_t1 = s_t1.unsqueeze(0)

            z_t = self.encoder(s_t)
            z_t1 = self.target_encoder(s_t1)
            action_emb = self._get_action_tensor(action_name).unsqueeze(0)
            z_t1_pred = self.predictor(z_t, action_emb)

            error = F.mse_loss(z_t1_pred, z_t1).item()

        return error
