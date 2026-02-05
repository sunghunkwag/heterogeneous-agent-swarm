"""
Hard Benchmark Suite for Heterogeneous Agent Swarm.

This benchmark creates adversarial conditions to test:
1. Agent suppression after consecutive failures
2. Emergency rotation for deadlock recovery
3. Architecture modification (NAS) suggestions
4. Meta-learning adaptation

Run with: python -m heterogeneous_agent_swarm.hard_benchmark
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

# Ensure module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heterogeneous_agent_swarm.main import AdvancedAISystem
from heterogeneous_agent_swarm.core.blackboard import Blackboard
from heterogeneous_agent_swarm.core.config import SwarmConfig


class AdversarialEnvironment:
    """
    Environment that deliberately creates failure conditions
    to test self-modification capabilities.
    """
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.step_count = 0
        self.phase = "chaos"  # chaos, partial_failure, recovery
        self.chaos_duration = 15
        self.failure_duration = 20
        self.forced_failure_agents: List[str] = []
        
    def get_observation(self) -> Dict[str, Any]:
        """Return adversarial observation based on current phase."""
        self.step_count += 1
        
        # Determine phase
        if self.step_count <= self.chaos_duration:
            self.phase = "chaos"
            # High uncertainty environment
            return {
                "last_test_ok": np.random.random() < 0.3,  # 70% failure
                "error": np.random.random() > 0.4,  # 60% errors
                "uncertainty": 0.5 + np.random.random() * 0.5,  # High
                "phase": "chaos"
            }
        elif self.step_count <= self.chaos_duration + self.failure_duration:
            self.phase = "partial_failure"
            # Force specific agents to fail
            return {
                "last_test_ok": np.random.random() < 0.5,
                "error": False,
                "uncertainty": 0.05 + np.random.random() * 0.1,
                "forced_failures": ["jepa_agent", "diffusion_agent"],
                "phase": "partial_failure"
            }
        else:
            self.phase = "recovery"
            # Allow recovery
            return {
                "last_test_ok": True,
                "error": False,
                "uncertainty": 0.02,
                "phase": "recovery"
            }
    
    def get_loss_for_agent(self, agent_name: str, action_success: bool) -> float:
        """Return artificially high loss for targeted agents."""
        if self.phase == "partial_failure":
            if agent_name in ["jepa_agent", "diffusion_agent"]:
                return 0.85 + np.random.random() * 0.1  # Force high loss
        
        if self.phase == "chaos":
            return 0.5 + np.random.random() * 0.4  # Variable high loss
        
        # Normal operation
        return 0.1 + np.random.random() * 0.2 if action_success else 0.6


class HardBenchmarkRunner:
    """Runs adversarial benchmark and collects metrics."""
    
    def __init__(self, total_steps: int = 50):
        self.total_steps = total_steps
        self.config = SwarmConfig()
        self.env = AdversarialEnvironment(self.config)
        self.metrics = {
            "suppression_events": [],
            "recovery_events": [],
            "emergency_rotations": [],
            "arch_modifications": [],
            "phase_transitions": [],
            "agent_performance": {}
        }
        
    def run(self) -> Dict[str, Any]:
        """Execute the hard benchmark."""
        print("=" * 60)
        print("HARD BENCHMARK: Testing Self-Modification Capabilities")
        print("=" * 60)
        
        # Initialize system
        system = AdvancedAISystem(device="cpu")
        bb = Blackboard(
            episode_id="hard_benchmark_001",
            goal_text="Test self-modification capabilities under adversarial conditions",
            task_text="Execute benchmark with deliberate failure injection"
        )
        
        current_phase = None
        
        for step in range(self.total_steps):
            obs = self.env.get_observation()
            
            # Track phase transitions
            if obs["phase"] != current_phase:
                current_phase = obs["phase"]
                self.metrics["phase_transitions"].append({
                    "step": step,
                    "phase": current_phase
                })
                print(f"\n[Step {step}] Phase: {current_phase.upper()}")
            
            # Execute step
            try:
                # step() returns (tool_name, tool_info)
                action, tool_info = system.step(bb)
                
                # Track agent performance using system's last action source
                winner = None
                if hasattr(system, 'last_action'):
                    winner = system.last_action[0]  # tool_name
                
                # Simulate agent-specific loss based on environment
                loss = self.env.get_loss_for_agent(
                    winner or "unknown", 
                    obs.get("last_test_ok", False)
                )
                
                # Get the actual proposing agent from system state
                if hasattr(system, 'agents'):
                    # Pick an agent to test constraints on
                    for agent_name in system.agents.keys():
                        agent_loss = self.env.get_loss_for_agent(agent_name, obs.get("last_test_ok", False))
                        
                        # Update node performance in graph (1.0 - loss gives performance)
                        # This is required for NAS to trigger
                        agent_perf = max(0.0, 1.0 - agent_loss)
                        system.graph.node_perf[agent_name] = agent_perf
                        
                        # Track performance
                        if agent_name not in self.metrics["agent_performance"]:
                            self.metrics["agent_performance"][agent_name] = []
                        self.metrics["agent_performance"][agent_name].append({
                            "step": step,
                            "loss": agent_loss,
                            "perf": agent_perf,
                            "phase": current_phase
                        })
                        
                        # Call enforce_agent_constraints to test suppression
                        if hasattr(system, 'meta'):
                            result = system.meta.enforce_agent_constraints(agent_name, agent_loss)
                            if result:
                                if result["action"] == "suppress":
                                    self.metrics["suppression_events"].append({
                                        "step": step,
                                        "agent": agent_name,
                                        "loss": agent_loss,
                                        "phase": current_phase
                                    })
                                    print(f"  [SUPPRESS] {agent_name} (loss={agent_loss:.3f})")
                                elif result["action"] == "unsuppress":
                                    self.metrics["recovery_events"].append({
                                        "step": step,
                                        "agent": agent_name,
                                        "loss": agent_loss
                                    })
                                    print(f"  [RECOVER] {agent_name} (loss={agent_loss:.3f})")
                
                # Check for deadlock and trigger emergency rotation
                if step % 10 == 0 and step > 0:
                    if hasattr(system, 'meta'):
                        awakened = system.meta.emergency_rotation()
                        if awakened:
                            self.metrics["emergency_rotations"].append({
                                "step": step,
                                "awakened_agent": awakened
                            })
                            print(f"  [EMERGENCY ROTATION] Awakened: {awakened}")
                
                # Execute NAS automatically (with cooldown to prevent runaway growth)
                if hasattr(system, 'meta') and hasattr(system, 'agents'):
                    if not hasattr(self, 'nas_cooldown'):
                        self.nas_cooldown = {}  # Track last NAS step per agent
                    
                    # Only one NAS per step, check agents with cooldown
                    for agent_name in list(system.agents.keys())[:3]:
                        # Skip if agent recently had NAS (10-step cooldown)
                        last_nas = self.nas_cooldown.get(agent_name, -100)
                        if step - last_nas < 10:
                            continue
                        
                        nas_result = system.meta.auto_execute_nas(agent_name)
                        if nas_result and nas_result.get("action") == "nas_executed":
                            self.nas_cooldown[agent_name] = step  # Record cooldown
                            self.metrics["arch_modifications"].append({
                                "step": step,
                                "agent": agent_name,
                                "old_hidden": nas_result.get("old_hidden"),
                                "new_hidden": nas_result.get("new_hidden"),
                                "new_params": nas_result.get("new_params")
                            })
                            print(f"  [NAS EXECUTED] {agent_name}: {nas_result.get('old_hidden')} -> {nas_result.get('new_hidden')} neurons")
                            break  # Only one NAS per step
                
            except Exception as e:
                print(f"  [ERROR] Step {step}: {e}")
                continue
            
            # Progress indicator
            if step % 10 == 0:
                suppressed = sum(1 for _ in self.metrics["suppression_events"])
                recovered = sum(1 for _ in self.metrics["recovery_events"])
                print(f"  Progress: {step}/{self.total_steps} | Suppressions: {suppressed} | Recoveries: {recovered}")
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate final benchmark report."""
        report = {
            "benchmark": "hard_adversarial",
            "timestamp": datetime.now().isoformat(),
            "total_steps": self.total_steps,
            "summary": {
                "total_suppressions": len(self.metrics["suppression_events"]),
                "total_recoveries": len(self.metrics["recovery_events"]),
                "emergency_rotations": len(self.metrics["emergency_rotations"]),
                "nas_suggestions": len(self.metrics["arch_modifications"]),
                "phase_count": len(self.metrics["phase_transitions"])
            },
            "events": self.metrics,
            "success_criteria": {
                "suppression_tested": len(self.metrics["suppression_events"]) > 0,
                "recovery_tested": len(self.metrics["recovery_events"]) > 0,
                "self_modification_active": len(self.metrics["arch_modifications"]) > 0 or 
                                           len(self.metrics["emergency_rotations"]) > 0
            }
        }
        
        # Calculate success rate
        criteria = report["success_criteria"]
        passed = sum(1 for v in criteria.values() if v)
        report["success_criteria"]["pass_rate"] = f"{passed}/{len(criteria)}"
        
        return report


def main():
    """Run hard benchmark and save results."""
    print("\n" + "=" * 60)
    print("HETEROGENEOUS AGENT SWARM - HARD BENCHMARK")
    print("Testing: Suppression, Recovery, Emergency Rotation, NAS")
    print("=" * 60 + "\n")
    
    runner = HardBenchmarkRunner(total_steps=50)
    report = runner.run()
    
    # Save report
    os.makedirs("benchmark_results", exist_ok=True)
    timestamp = int(time.time())
    output_path = f"benchmark_results/hard_benchmark_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")
    print(f"\nSummary:")
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    
    print(f"\nSuccess Criteria:")
    for key, value in report["success_criteria"].items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    
    return report


if __name__ == "__main__":
    main()
