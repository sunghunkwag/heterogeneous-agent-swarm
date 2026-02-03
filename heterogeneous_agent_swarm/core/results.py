import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import os

@dataclass
class EpisodeResult:
    episode_id: int
    level: int
    completed: bool
    reward: float
    total_steps: int
    total_cost: float
    deadlock_count: int
    recovery_count: int
    meta_train_events: int
    arch_mod_events: int

@dataclass
class BenchmarkSummary:
    run_id: str
    episodes: List[EpisodeResult]
    total_runtime: float
    meta_train_total_count: int
    arch_mod_total_count: int
    success_rate: float
    avg_cost_per_episode: float
    deadlock_recovery_rate: float

class ResultsManager:
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.episodes: List[EpisodeResult] = []

    def record_episode(self, ep_result: EpisodeResult) -> None:
        self.episodes.append(ep_result)

    def finalize_benchmark(self, run_id: str, total_runtime: float,
                          meta_train_count: int, arch_mod_count: int,
                          deadlock_recovery_rate: float) -> str:

        # Calculate summary metrics
        total_episodes = len(self.episodes)
        success_count = sum(1 for e in self.episodes if e.completed)
        success_rate = success_count / total_episodes if total_episodes > 0 else 0.0

        total_cost = sum(e.total_cost for e in self.episodes)
        avg_cost = total_cost / total_episodes if total_episodes > 0 else 0.0

        summary = BenchmarkSummary(
            run_id=run_id,
            episodes=self.episodes,
            total_runtime=total_runtime,
            meta_train_total_count=meta_train_count,
            arch_mod_total_count=arch_mod_count,
            success_rate=success_rate,
            avg_cost_per_episode=avg_cost,
            deadlock_recovery_rate=deadlock_recovery_rate
        )

        filename = os.path.join(self.output_dir, f"benchmark_{run_id}.json")
        with open(filename, 'w') as f:
            json.dump(asdict(summary), f, indent=2)

        return filename
