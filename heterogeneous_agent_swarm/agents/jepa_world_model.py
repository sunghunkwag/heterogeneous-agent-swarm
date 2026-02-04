from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.types import StateVectorProvider, Proposal

@dataclass
class JEPAConfig:
    device: str = "cpu"
    learning_rate: float = 1e-3
    latent_dim: int = 64
    input_dim: int = 16
    action_dim: int = 8
    momentum: float = 0.99
    hidden_dim: int = 128
    num_layers: int = 2 # Added for Structural RSI
    ucb_beta_base: float = 0.1

class JEPAEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class JEPAPredictor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        layers = []
        curr_dim = latent_dim + action_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, latent_dim))
        self.net = nn.Sequential(*layers)

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
        self.encoder = JEPAEncoder(config.input_dim, config.latent_dim, config.hidden_dim, config.num_layers).to(self.device)
        self.predictor = JEPAPredictor(config.latent_dim, config.action_dim, config.hidden_dim, config.num_layers).to(self.device)
        self.target_encoder = JEPAEncoder(config.input_dim, config.latent_dim, config.hidden_dim, config.num_layers).to(self.device)
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

        # Capacity Adjustment Dynamics
        self.warmup_steps_remaining = 0
        self.base_lr = config.learning_rate
        self.base_momentum = config.momentum
        self.current_momentum = config.momentum

    def update_hyperparameters(self, lr: float = None, beta_base: float = None):
        """Dynamic update from MetaMetaOptimizer."""
        if lr is not None:
            self.base_lr = lr
            self.config.learning_rate = lr # Update config for persistence
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        if beta_base is not None:
            self.config.ucb_beta_base = beta_base

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

        # Recreate optimizer (fresh state)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=self.config.learning_rate
        )

        self.parameter_count = self._count_params()

        # Initiate Warmup and Momentum Stabilization
        self.warmup_steps_remaining = 100
        self.current_momentum = 0.999 # Slow down target update temporarily

        return {
            "action": "increase_capacity",
            "old_hidden": old_hidden,
            "new_hidden": new_hidden,
            "new_params": self.parameter_count,
            "warmup_triggered": True
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

        # No warmup needed for capacity decrease (aggressive pruning)

        return {
            "action": "decrease_capacity",
            "old_hidden": old_hidden,
            "new_hidden": new_hidden,
            "new_params": self.parameter_count
        }

    def get_state_dict(self) -> Dict[str, Any]:
        """Return the state dictionary of the agent."""
        return {
            "encoder": self.encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "action_embedding": self.action_embedding.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }

    def load_compatible_state(self, state_dict: Dict[str, Any]):
        """Load state dictionary, skipping mismatched shapes."""
        def load_model_state(model, state_key):
            if state_key not in state_dict:
                return
            model_state = model.state_dict()
            loaded_state = state_dict[state_key]

            compatible_state = {}
            for k, v in loaded_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    compatible_state[k] = v

            model.load_state_dict(compatible_state, strict=False)

        load_model_state(self.encoder, "encoder")
        load_model_state(self.predictor, "predictor")
        load_model_state(self.target_encoder, "target_encoder")
        load_model_state(self.action_embedding, "action_embedding")

        # Optimizer is tricky if shapes changed, usually safer to skip or partial load
        # For simplicity, we skip optimizer state if we are doing structural changes
        # as the momentum buffers would be mismatched.

    def _get_action_tensor(self, action_name: str) -> torch.Tensor:
        """Convert action name to embedding tensor."""
        idx = self.ACTION_TO_IDX.get(action_name, 5)  # Default to "none"
        idx_tensor = torch.tensor([idx], device=self.device)
        return self.action_embedding(idx_tensor).squeeze(0)

    def propose(self, state: StateVectorProvider, memory: Dict[str, Any]) -> Proposal:
        """
        Agent interface: propose action based on world model prediction.
        Returns Proposal with predicted value of each action.
        """

        # Get current state vector via Protocol
        state_vec = state.get_vector()

        # Validation
        if state_vec.shape[-1] != self.config.input_dim:
             raise ValueError(f"JEPA propose expected input dim {self.config.input_dim}, got {state_vec.shape[-1]}")

        # Evaluate each possible action using world model
        action_values = {}
        action_errors = {}

        # TODO: Retrieve average external reward from memory/audit log for baseline
        avg_external_reward = 0.0

        for action_name in ["run_tests", "write_patch", "summarize", "wait"]:
            with torch.no_grad():
                state_t = torch.FloatTensor(state_vec).to(self.device)
                if state_t.dim() == 1:
                    state_t = state_t.unsqueeze(0)

                # Predict next latent state
                latent = self.encoder(state_t)
                action_emb = self._get_action_tensor(action_name).unsqueeze(0)
                next_latent_pred = self.predictor(latent, action_emb)

                # NOTE: In a full JEPA, we would compare against a 'target' prediction or self-consistency.
                # Here, we estimate 'uncertainty' via latent variance or distance to prototypes.
                # Since we don't have the *actual* next state yet, we use a proxy for "Model Confidence".
                # Proxy: Variance of the predicted latent vector (High variance = loose prediction?)
                # Better Proxy: We can't compute prediction error without the target.
                # So we rely on the agent's learned value function (if we had one) or heuristics.

                # REVISED LOGIC based on Instructions:
                # "Use prediction error... (Low error -> exploit, High error -> explore)"
                # But at inference time, we don't have the error because we don't have the outcome.
                # We can only estimate *expected* error if we had a secondary "error predictor" network.
                # Lacking that, we will use the user's suggestion:
                # "Select actions with low error (= high trust)..."
                # This implies we need a way to predict "how wrong will I be?".
                # For this implementation, I will use 'latent variance' as a proxy for UNCERTAINTY.
                # High Variance in latent space typically implies the model is 'stretching' or uncertain?
                # actually, usually Low Variance = collapse.

                # Let's assume we use the 'novelty' metric from before as 'uncertainty'.
                uncertainty = torch.var(next_latent_pred).item()
                action_errors[action_name] = uncertainty

        # Selection Logic: Dynamic Upper Confidence Bound (UCB)
        # Score = Q(action) + beta * Uncertainty(action)
        
        # 1. Get Beta from System State (Error Rate)
        # Error Rate is in memory (passed from SNN/Evaluator)
        # If not present, default to high exploration (assume we are failing if we don't know)
        error_rate = float(memory.get("error_rate", 1.0))
        
        # Beta Mapping:
        # Error 1.0 (Fail) -> Beta = base + (base * 10 * error)
        # Error 0.0 (Success) -> Beta = base
        # This scales exploration relative to the Meta-Meta base setting.
        base = self.config.ucb_beta_base
        beta = base + (base * 19.0 * error_rate) # Scale up to 20x base at max error
        # e.g. base=0.1 -> beta range [0.1, 2.0]

        scores = {}
        for act, uncert in action_errors.items():
            # Safety: Never explore emergency_stop via curiosity
            if act == "emergency_stop":
                scores[act] = -100.0
                continue
                
            # Base Q-Value (predicted reward)
            # In this stub, we use avg external reward, but ideally this comes from the predictor
            # For now, we assume 0.0 base value if unknown, or use memory
            q_val = avg_external_reward
            
            # UCB Score
            score = q_val + beta * uncert
            
            # Action specific bias (Optional: Break ties for 'wait')
            # Penalize 'wait' slightly to encourage action when uncertainty is equal
            if act == "wait":
                 score -= 0.05

            scores[act] = score

        # ===== PANIC EXPLORATION BOOST =====
        consecutive_idle = memory.get("consecutive_idle_count", 0)
        system_panic = memory.get("system_panic", False)

        base_beta = self.config.ucb_beta_base if hasattr(self.config, 'ucb_beta_base') else 1.0

        if system_panic and consecutive_idle >= 2:
            # PANIC MODE: Strongly prefer active actions over waiting
            beta = base_beta * 5.0

            scores["run_tests"] = scores.get("run_tests", 0.0) * 5.0
            scores["write_patch"] = scores.get("write_patch", 0.0) * 5.0
            scores["summarize"] = scores.get("summarize", 0.0) * 1.5
            scores["wait"] = scores.get("wait", 0.0) * 0.1  # 90% discount for waiting

        best_action = max(scores, key=scores.get)
        best_score = scores[best_action]
        best_uncertainty = action_errors[best_action]

        return Proposal(
            source_agent=self.name,
            action_type=best_action,
            action_value={},
            confidence=0.5,
            predicted_value=best_score,
            estimated_cost=0.1,
            rationale=f"JEPA UCB: Beta={beta:.2f}, Uncert={best_uncertainty:.3f}, Score={best_score:.3f}",
            artifacts={"action_scores": scores, "uncertainty": action_errors, "beta": beta}
        )

    def train_step(self, prev_system_thought: np.ndarray, action_name: str,
                   next_env_feedback: np.ndarray, reward: float) -> float:
        """
        Refactored Training Step.

        Args:
            prev_system_thought: The internal GNN state (Context).
            action_name: The action taken.
            next_env_feedback: The PHYSICAL result (Sandbox State: [last_ok, fails, diff...]).
            reward: Scalar reward.

        Returns:
            prediction_error: Intrinsic curiosity signal.
        """
        # Shape Validation
        if prev_system_thought.shape[-1] != self.config.input_dim:
             raise ValueError(f"JEPA train_step expected prev_system_thought dim {self.config.input_dim}, got {prev_system_thought.shape[-1]}")

        # next_env_feedback is also mapped to latent, so it should match input_dim (since we reuse encoder arch)
        # In this specific architecture, target_encoder matches encoder, so input dims must match.
        if next_env_feedback.shape[-1] != self.config.input_dim:
             raise ValueError(f"JEPA train_step expected next_env_feedback dim {self.config.input_dim}, got {next_env_feedback.shape[-1]}")

        self.train_step_count += 1

        # Apply Warmup Logic
        current_lr = self.base_lr
        if self.warmup_steps_remaining > 0:
            current_lr = self.base_lr * 0.1
            self.warmup_steps_remaining -= 1
            if self.warmup_steps_remaining == 0:
                self.current_momentum = self.base_momentum # Restore momentum

        # Set LR for this step
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

        self.optimizer.zero_grad()

        # Inputs
        s_t = torch.FloatTensor(prev_system_thought).to(self.device)
        # Target: Environment Feedback
        env_t1 = torch.FloatTensor(next_env_feedback).to(self.device)

        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)
        if env_t1.dim() == 1:
            env_t1 = env_t1.unsqueeze(0)

        # 1. Encode Context (Thought)
        z_t = self.encoder(s_t)

        # 2. Encode Target (Physical Reality)
        # We use target_encoder to map physical reality to latent space
        z_t1_target = self.target_encoder(env_t1).detach()

        # 3. Predict Latent Consequence
        action_emb = self._get_action_tensor(action_name).unsqueeze(0)
        z_t1_pred = self.predictor(z_t, action_emb)

        # Prediction loss (MSE)
        pred_loss = F.mse_loss(z_t1_pred, z_t1_target)

        # Variance loss (prevent collapse)
        var_loss = torch.mean(F.relu(1.0 - torch.std(z_t, dim=0, unbiased=False))) + \
                   torch.mean(F.relu(1.0 - torch.std(z_t1_target, dim=0, unbiased=False)))

        # Total loss
        loss = pred_loss + 0.1 * var_loss

        # Backward
        loss.backward()
        self.optimizer.step()

        # Momentum update target encoder
        with torch.no_grad():
            m = self.current_momentum # Use dynamic momentum
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
        # Shape Validation (Lightweight)
        if prev_obs_vector.shape[-1] != self.config.input_dim:
             raise ValueError(f"JEPA curiosity expected prev dim {self.config.input_dim}, got {prev_obs_vector.shape[-1]}")

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
