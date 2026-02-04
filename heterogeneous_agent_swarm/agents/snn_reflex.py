import torch
import torch.nn as nn
from typing import Dict, Any
from dataclasses import dataclass
from ..core.types import Proposal

@dataclass
class SNNConfig:
    input_dim: int = 16
    hidden_neurons: int = 128
    output_dim: int = 8
    device: str = "cpu"

class SNNCore(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.thr = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem = torch.zeros(x.shape[0], self.fc1.out_features).to(x.device)
        spikes = []
        for _ in range(5):
            mem = mem * 0.9 + self.fc1(x)
            spike = (mem >= self.thr).float()
            mem[mem >= self.thr] = 0
            spikes.append(spike)
        recalc = torch.stack(spikes).mean(dim=0)
        return self.fc2(recalc)

class SNNReflexAgent:
    def __init__(self, name: str, config: SNNConfig):
        self.name = name
        self.config = config
        self.device = torch.device(config.device)
        self.core = SNNCore(config.input_dim, config.hidden_neurons, config.output_dim).to(self.device)

    def get_state_dict(self) -> Dict[str, Any]:
        """Return the state dictionary of the agent."""
        return {
            "core": self.core.state_dict()
        }

    def load_compatible_state(self, state_dict: Dict[str, Any]):
        """Load state dictionary, skipping mismatched shapes."""
        if "core" in state_dict:
            core_state = self.core.state_dict()
            loaded_state = state_dict["core"]

            compatible_state = {}
            for k, v in loaded_state.items():
                if k in core_state and core_state[k].shape == v.shape:
                    compatible_state[k] = v

            self.core.load_state_dict(compatible_state, strict=False)

    def propose(self, state, memory) -> Proposal:
        last_tool = memory.get("last_tool")
        if last_tool and not last_tool["ok"]:
             return Proposal(
                action_type="summarize", action_value=None,
                confidence=0.99, predicted_value=1.0, estimated_cost=0.5,
                rationale="SNN_RECOIL_EMERGENCY",
                source_agent=self.name
            )
            
        obs_tensor = torch.tensor(state.raw_obs.get("buffer", [0.0]*16)).float().to(self.device)
        if obs_tensor.shape[0] < self.config.input_dim:
            padding = torch.zeros(self.config.input_dim - obs_tensor.shape[0]).to(self.device)
            obs_tensor = torch.cat([obs_tensor, padding])
        
        with torch.no_grad():
            logits = self.core(obs_tensor.unsqueeze(0))
            action_idx = torch.argmax(logits).item()
            conf = torch.sigmoid(torch.max(logits)).item()

        action_map = {0: "run_tests", 1: "write_patch", 2: "summarize", 3: "wait"}
        return Proposal(
            action_type=action_map.get(action_idx % 4, "wait"),
            action_value=None,
            confidence=conf * 0.1,
            predicted_value=1.0,
            estimated_cost=1.0,
            rationale=f"SNN_Reflex_Neurons_{self.config.hidden_neurons}",
            source_agent=self.name
        )
