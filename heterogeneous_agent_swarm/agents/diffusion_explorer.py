from dataclasses import dataclass
import numpy as np
from ..core.types import EncodedState, Proposal

@dataclass
class DiffusionConfig:
    pass

class DiffusionExplorerAgent:
    def __init__(self, name: str, config: DiffusionConfig):
        self.name = name

    def propose(self, state: EncodedState, memory: dict) -> Proposal:
        obs = state.raw_obs
        system_thought = np.array(memory.get("system_thought", [0.0]*16))
        thought_power = np.mean(np.abs(system_thought))

        grid = None
        if "current_grid" in obs:
            grid = np.array(obs["current_grid"])
        elif "input_grid" in obs:
            grid = np.array(obs["input_grid"])

        if grid is not None:
            h, w = grid.shape
            entropy_map = np.zeros((h, w))

            # Calculate local entropy (3x3 window)
            # Pad grid to handle borders
            pad_grid = np.pad(grid, 1, mode='constant', constant_values=0)

            for y in range(h):
                for x in range(w):
                    # 3x3 window centered at y,x (in original grid coordinates)
                    # In padded grid, this is y:y+3, x:x+3
                    window = pad_grid[y:y+3, x:x+3].flatten()
                    # Counts of each color 0-9
                    counts = np.bincount(window, minlength=10)
                    total = np.sum(counts)
                    if total > 0:
                        probs = counts / total
                        # Filter zero probabilities to avoid log(0)
                        probs = probs[probs > 0]
                        entropy = -np.sum(probs * np.log2(probs))
                        entropy_map[y, x] = entropy

            # Find coordinate with maximum entropy
            flat_idx = np.argmax(entropy_map)
            best_y, best_x = np.unravel_index(flat_idx, (h, w))
            max_entropy = entropy_map[best_y, best_x]

            # Determine color: Deterministic choice based on location and entropy
            # Strategy: Pick a color derived from coordinates to ensure variety but determinism
            # Modulo 9 + 1 ensures color is 1-9 (avoiding 0/background)
            color = (int(best_x) + int(best_y) + int(max_entropy * 100)) % 9 + 1

            return Proposal(
                action_type="write_patch",
                action_value={"x": int(best_x), "y": int(best_y), "color": int(color)},
                confidence=0.6 + (0.1 * thought_power),
                predicted_value=0.5,
                estimated_cost=1.0,
                rationale=f"Targeting high entropy region ({max_entropy:.2f}) at {best_x},{best_y}",
                source_agent=self.name
            )

        # Fallback if no grid
        return Proposal(
            action_type="summarize",
            action_value=None,
            confidence=0.1,
            predicted_value=0.0,
            estimated_cost=1.0,
            rationale="No grid observed. Idle.",
            source_agent=self.name
        )
