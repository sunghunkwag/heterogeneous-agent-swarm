from __future__ import annotations
import numpy as np
from typing import List, Dict

class SpikingNeuronLayer:
    """
    LIF (Leaky Integrate-and-Fire) Neurons.
    Detects temporal patterns (sequences of events) to trigger specific interruptions.
    """
    def __init__(self, input_dim: int, num_neurons: int, threshold: float = 1.0, decay: float = 0.9):
        self.v_mem = np.zeros(num_neurons)
        self.threshold = threshold
        self.decay = decay
        # Random synaptic weights
        self.weights = np.random.randn(input_dim, num_neurons) * 0.5

    def step(self, input_spike_vec: np.ndarray) -> List[int]:
        """
        Input: Binary or sparse vector of events.
        Output: Indices of neurons that fired.
        """
        # Integrate
        current = np.dot(input_spike_vec, self.weights)
        self.v_mem = self.v_mem * self.decay + current
        
        # Fire
        fired_indices = np.where(self.v_mem >= self.threshold)[0]
        
        # Reset
        self.v_mem[fired_indices] = 0.0
        
        return fired_indices.tolist()

class EventSNN:
    """
    Monitors system audit logs / signals.
    Fires 'Interrupts' such as 'Trigger Refactor', 'Stop Infinite Loop', 'Alert Budget'.
    """
    def __init__(self):
        # Input features: [Time_since_success, Error_Rate, Budget_slope, Memory_Entropy]
        self.layer = SpikingNeuronLayer(input_dim=4, num_neurons=10)
        self.neuron_labels = {
            0: "emergency_stop",
            1: "force_exploration",
            2: "garbage_collect_memory",
            3: "trigger_sleep_mode",
            8: "high_alert_verify",
            9: "panic_dump"
        }

    def process_signals(self, signals: Dict[str, float]) -> List[str]:
        # Vectorize signals
        vec = np.array([
            signals.get("time_since_success", 0),
            signals.get("error_rate", 0),
            signals.get("budget_slope", 0),
            signals.get("entropy", 0)
        ])
        
        fired = self.layer.step(vec)
        triggers = [self.neuron_labels[i] for i in fired if i in self.neuron_labels]
        return triggers
