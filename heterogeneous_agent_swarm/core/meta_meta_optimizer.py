from __future__ import annotations
import numpy as np
import copy
from typing import Dict, Any, List

class MetaMetaOptimizer:
    """
    Level 3 Intelligence: Autonomous Structural Evolution.
    Uses UCB-based mutation selection to balance exploration of the architecture space.
    Inspired by Bio-Evolutionary strategies (PBT/Novelty Search).
    """
    def __init__(self, population_size: int = 4, generation_window: int = 5):
        self.pop_size = population_size
        self.window = generation_window
        self.current_window_step = 0
        self.generation = 0
        
        # Mutation Registry (UCB Arms)
        self.mutations = [
            "tweak_hyperparams", 
            "deepen_jepa", 
            "increase_snn_neurons", 
            "add_meta_quorum",
            "expand_symbolic_memory"
        ]
        
        # UCB Stats
        self.counts = {m: 1 for m in self.mutations}
        self.values = {m: 0.0 for m in self.mutations}
        self.total_signals = 1
        self.c_exploration = 2.0
        
        # Current Genome
        self.base_genome = {
            "jepa_lr": 1e-3,
            "jepa_depth": 2,
            "snn_neurons": 128,
            "meta_min_quorum": 3,
            "symbolic_depth": 5
        }
        
        self.population = []
        self._init_population()
        self.current_candidate_idx = 0
        self.current_rewards = []

    def _init_population(self):
        """Generate candidates using UCB selection."""
        self.population = [{"genome": copy.deepcopy(self.base_genome), "mutation": "keep"}] # Elitism
        
        for i in range(1, self.pop_size):
            # 1. Select Mutation Arm via UCB
            ucb_scores = {}
            for m in self.mutations:
                exploit = self.values[m]
                explore = self.c_exploration * np.sqrt(np.log(self.total_signals) / self.counts[m])
                ucb_scores[m] = exploit + explore
            
            selected_m = max(ucb_scores, key=ucb_scores.get)
            
            # 2. Apply Mutation
            genome = copy.deepcopy(self.base_genome)
            if selected_m == "tweak_hyperparams":
                genome["jepa_lr"] *= np.random.uniform(0.5, 1.5)
            elif selected_m == "deepen_jepa":
                genome["jepa_depth"] += 1
            elif selected_m == "increase_snn_neurons":
                genome["snn_neurons"] = int(genome["snn_neurons"] * 1.5)
            elif selected_m == "add_meta_quorum":
                genome["meta_min_quorum"] = min(5, genome["meta_min_quorum"] + 1)
            elif selected_m == "expand_symbolic_memory":
                genome["symbolic_depth"] += 2
                
            self.population.append({
                "genome": genome,
                "mutation": selected_m,
                "fitness": 0.0
            })

    def get_current_config(self) -> Dict[str, Any]:
        return self.population[self.current_candidate_idx]["genome"]

    def report_episode_reward(self, reward: float) -> Dict[str, Any]:
        self.current_rewards.append(reward)
        self.current_window_step += 1
        self.total_signals += 1
        
        status = {}
        if self.current_window_step >= self.window:
            avg_reward = np.mean(self.current_rewards)
            candidate = self.population[self.current_candidate_idx]
            candidate["fitness"] = avg_reward
            
            # Update UCB Stats for the mutation used
            m_type = candidate["mutation"]
            if m_type != "keep":
                self.counts[m_type] += 1
                # Delta reward relative to base performance? Standard UCB uses absolute.
                self.values[m_type] += (avg_reward - self.values[m_type]) / self.counts[m_type]
            
            status = {
                "event": "candidate_eval_done",
                "id": self.current_candidate_idx,
                "mutation": m_type,
                "fitness": avg_reward
            }
            
            self.current_candidate_idx += 1
            self.current_window_step = 0
            self.current_rewards = []
            
            if self.current_candidate_idx >= self.pop_size:
                self._evolve()
                status["event"] = "generation_complete"
                status["gen"] = self.generation
                status["best_genome"] = self.base_genome
                self.generation += 1
                self.current_candidate_idx = 0
                
        return status

    def _evolve(self):
        # Elitism: Sort by fitness
        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        self.base_genome = copy.deepcopy(self.population[0]["genome"])
        self._init_population()
