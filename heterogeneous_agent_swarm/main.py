import time
import argparse
import sys
import os
import numpy as np
from datetime import datetime
from collections import Counter
from typing import Dict

# Ensure we can import heterogeneous_agent_swarm
sys.path.append(os.getcwd())

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box
from rich.align import Align

# Import Swarm OS Core
from heterogeneous_agent_swarm.core.tools import ToolRegistry, ToolSpec, ToolResult
from heterogeneous_agent_swarm.envs.code_sandbox import CodeSandboxEnv
from heterogeneous_agent_swarm.envs.tool_env import ToolEnv
from heterogeneous_agent_swarm.envs.arc_env import ARCEnv # New Import
from heterogeneous_agent_swarm.core.graph import AgentGraph
from heterogeneous_agent_swarm.core.blackboard import Blackboard
from heterogeneous_agent_swarm.core.orchestrator import Orchestrator
from heterogeneous_agent_swarm.core.meta_kernel_v2 import MetaKernelV2
from heterogeneous_agent_swarm.core.gnn_brain import LightweightGNN
from heterogeneous_agent_swarm.agents.snn_core import EventSNN
from heterogeneous_agent_swarm.core.audit import AuditLog
from heterogeneous_agent_swarm.core.evaluation import Evaluator
from heterogeneous_agent_swarm.core.memory import WorkingMemory, EpisodicMemory, SemanticMemory
from heterogeneous_agent_swarm.runtime.runner_v2 import SimpleEncoder
from heterogeneous_agent_swarm.core.results import ResultsManager, EpisodeResult

# Import Agents
from heterogeneous_agent_swarm.agents.symbolic_search import SymbolicSearchAgent, SymbolicConfig
from heterogeneous_agent_swarm.agents.jepa_world_model import JEPAWorldModelAgent, JEPAConfig
from heterogeneous_agent_swarm.agents.neuro_symbolic import NeuroSymbolicVerifierAgent, Policy
from heterogeneous_agent_swarm.agents.liquid_controller import LiquidControllerAgent, LiquidConfig
from heterogeneous_agent_swarm.agents.diffusion_explorer import DiffusionExplorerAgent, DiffusionConfig
from heterogeneous_agent_swarm.agents.ssm_stability import SSMStabilityAgent, SSMConfig
from heterogeneous_agent_swarm.agents.snn_reflex import SNNReflexAgent, SNNConfig

class AdvancedAISystem:
    def __init__(self, device="cpu", arc_mode=False):
        self.console = Console()
        self.root = os.getcwd()
        self.arc_mode = arc_mode
        
        # 1. Environment & Tools
        if self.arc_mode:
            self.sandbox = ARCEnv(data_dir=os.path.join(self.root, "data/ARC/data/training"))
        else:
            self.sandbox = CodeSandboxEnv(repo_root=self.root)
            
        self.reg = ToolRegistry()
        self._register_tools()
        self.env = ToolEnv(registry=self.reg)
        
        # 2. Agents
        self.graph = AgentGraph()
        self.agents = {}
        self._init_agents(device)
        
        # 3. Cognitive Core (Brain & Nervous System)
        self.audit = AuditLog()
        self.gnn = LightweightGNN(agent_names=list(self.agents.keys()))
        self.snn = EventSNN()
        
        # 4. Runtime & Memory
        # Episodic Memory for experience storage
        self.episodic = EpisodicMemory(capacity=1000)

        self.orch = Orchestrator(self.graph)
        self.orch.memory = self.episodic  # Late binding to enable memory bias

        self.meta = MetaKernelV2(self.graph, self.audit, self.orch, self.agents, device=device)
        self.encoder = SimpleEncoder()
        self.eval = Evaluator()
        self.work = WorkingMemory()
        
        # State tracking for UI
        self.system_thought = np.zeros(16)
        self.last_spikes = []
        self.last_action = ("None", {})
        self.episode_log = []
        self.consecutive_idle_count = 0

        # Dynamic State Tracking
        self.last_success_time = None
        self.start_time = time.time()
        self.budget_history = []
        self.action_history = []

        # Stats
        self.deadlock_recovery_stats = {
            "total_deadlocks": 0,
            "successful_recoveries": 0,
            "recovery_rate": 0.0
        }

        # Previous state tracking for JEPA training
        self.prev_system_state = None
        self.prev_action_name = None

        # JEPA config update
        self.agents["jepa_agent"].config.input_dim = 16  # Match GNN dim

    def _register_tools(self):
        def tool_run_tests(_args):
            obs, out = self.sandbox.step("run_tests")
            return ToolResult(ok=bool(obs.get("last_test_ok", False)), cost=out.cost, output=out.info)
        def tool_write_patch(args):
            obs, out = self.sandbox.step("write_patch", args)
            return ToolResult(ok=True, cost=out.cost, output=out.info)
        def tool_summarize(_args):
            obs, out = self.sandbox.step("summarize")
            return ToolResult(ok=True, cost=out.cost, output=out.info)
        def tool_wait(_args):
            # Idle action, low cost, no effect
            return ToolResult(ok=True, cost=0.01, output={"status": "waiting"})

        self.reg.register(ToolSpec("run_tests", "Verify System Integrity.", tool_run_tests))
        self.reg.register(ToolSpec("write_patch", "Self-Modify Codebase.", tool_write_patch))
        self.reg.register(ToolSpec("summarize", "Reflect & Conserve.", tool_summarize))
        self.reg.register(ToolSpec("wait", "Idle.", tool_wait))

    def _init_agents(self, device):
        def add(a):
            self.agents[a.name] = a
            self.graph.ensure_node(a.name)

        add(SymbolicSearchAgent("symbolic_agent", SymbolicConfig(device=device)))
        add(JEPAWorldModelAgent("jepa_agent", JEPAConfig(device=device)))
        add(NeuroSymbolicVerifierAgent("neurosym_agent", Policy()))
        add(LiquidControllerAgent("liquid_agent", LiquidConfig(device=device)))
        add(DiffusionExplorerAgent("diffusion_agent", DiffusionConfig()))
        add(SSMStabilityAgent("ssm_agent", SSMConfig()))
        add(SNNReflexAgent("snn_agent", SNNConfig()))

    def _calculate_budget_slope(self):
        if len(self.budget_history) < 2:
            return 0.0
        y = self.budget_history[-5:]
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        return slope

    def _calculate_entropy(self):
        if not self.action_history:
            return 0.0
        actions = self.action_history[-10:]
        counts = Counter(actions)
        total = len(actions)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        return entropy

    def _get_system_state_vector(self) -> np.ndarray:
        """Get current system state as vector for JEPA input (System Thought)."""
        return self.system_thought.copy()

    def _get_env_feedback_vector(self, obs: Dict, cost: float) -> np.ndarray:
        """
        Construct 16-dim environment feedback vector for JEPA target.
        [last_test_ok, failures, diff_count, grid_entropy/buffer, cost, 0...]
        """
        vec = np.zeros(16, dtype=np.float32)

        # 0: last_test_ok
        vec[0] = 1.0 if obs.get("last_test_ok") else 0.0

        # 1: failures (normalized, cap at 10)
        # Assuming failures is tracked in obs or we use a proxy.
        # Using obs.get("error") as boolean for now if failures count isn't explicit in tool output
        vec[1] = 1.0 if obs.get("error") else 0.0

        # 2: diff_count / distance (normalized)
        # Sandbox might return 'diff_count' in info
        info = self.work.get("last_tool", {})
        diff = info.get("output", {}).get("diff_count", 0) if isinstance(info.get("output"), dict) else 0
        vec[2] = min(diff / 100.0, 1.0)

        # 3: Entropy or Buffer Length
        # Proxy: length of output or specific env metric
        # specific logic for ARC vs Code not fully exposed here, using generic proxy
        vec[3] = 0.5 # Placeholder or valid metric if available

        # 4: Cost
        vec[4] = min(cost, 1.0)

        return vec

    def _calculate_combined_reward(self, obs: Dict, prev_state: np.ndarray,
                                   action_name: str, current_state: np.ndarray) -> float:
        """
        Calculate combined extrinsic + intrinsic reward.
        """
        # Extrinsic reward
        if obs.get("last_test_ok"):
            extrinsic = 1.0
        elif obs.get("error"):
            extrinsic = -0.5
        else:
            extrinsic = -0.01  # Small penalty for time passing

        # Intrinsic reward (curiosity from JEPA)
        jepa_agent = self.agents.get("jepa_agent")
        if jepa_agent and hasattr(jepa_agent, 'get_curiosity_reward'):
            intrinsic = jepa_agent.get_curiosity_reward(prev_state, action_name, current_state)
            # Normalize intrinsic to [-1, 1] range approximately
            intrinsic = min(1.0, max(-1.0, intrinsic * 10))
        else:
            intrinsic = 0.0

        # Combined (70% extrinsic, 30% intrinsic for exploration)
        combined = 0.7 * extrinsic + 0.3 * intrinsic

        return combined

    def step(self, bb: Blackboard, force_strategy: str = None):
        """Execute one cognitive cycle."""
        # A. SNN Check (Nervous System)
        current_time = time.time()
        if self.last_success_time:
            t_since = current_time - self.last_success_time
        else:
            t_since = current_time - self.start_time

        snn_inputs = {
            "time_since_success": t_since,
            "error_rate": -1.0 * self.eval.evaluate(bb.__dict__).score,
            "budget_slope": self._calculate_budget_slope(),
            "entropy": self._calculate_entropy()
        }
        interrupts = self.snn.process_signals(snn_inputs)
        self.last_spikes = interrupts
        if "emergency_stop" in interrupts:
            return "emergency_stop", {}

        # Inject error_rate for LiquidControllerAgent
        self.work.set("error_rate", snn_inputs["error_rate"])

        # B. GNN Update (Consciousness)
        gnn_inputs = {}
        for name in self.agents:
            alive = 1.0 if self.graph.node_alive.get(name, False) else 0.0
            perf = self.graph.node_perf.get(name, 0.5)
            gnn_inputs[name] = np.array([alive, perf, 0.0])
        
        self.system_thought = self.gnn.update(gnn_inputs)
        
        # C. Agent Proposals
        state = self.encoder.encode(bb.obs)
        mem = self.work.snapshot()
        mem["system_thought"] = self.system_thought.tolist() # Share consciousness
        
        proposals = {}
        for name, agent in self.agents.items():
            if self.graph.node_alive.get(name, False):
                proposals[name] = agent.propose(state, mem)

        # D. Orchestration
        p_neuro = proposals.get("neurosym_agent")
        veto = None
        if p_neuro and p_neuro.artifacts.get("verdict") == "deny":
            veto = p_neuro.artifacts # Pass full artifacts for veto_score check
        
        # C-Stage: Get System Uncertainty
        sys_uncertainty = self.gnn.get_system_uncertainty()

        # Pass force_strategy to orchestrator
        tool_name, tool_args, dbg = self.orch.choose(proposals, state, veto, force_strategy=force_strategy, system_uncertainty=sys_uncertainty)
        
        # Mapping Removed (Agents now speak Vocabulary)

        # E. Execution
        obs, tool_info = self.env.step_tool(tool_name, tool_args)

        # Update History
        self.budget_history.append(tool_info.get("cost", 0.0))
        self.action_history.append(tool_name)
        if obs.get("last_test_ok"):
            self.last_success_time = time.time()

        bb.obs = obs
        bb.record_step(tool_info)
        
        self.work.set("last_tool", tool_info)
        self.work.set("obs", obs)
        self.work.set("last_action_name", tool_name)
        self.work.set("last_action_params", tool_args)
        self.last_action = (tool_name, tool_args)
        
        # === JEPA Training & Memory Storage ===
        current_system_thought = self._get_system_state_vector()

        # C-Stage: Construct Environment Feedback Vector
        current_cost = tool_info.get("cost", 0.0)
        env_feedback = self._get_env_feedback_vector(obs, current_cost)

        if self.prev_system_state is not None and self.prev_action_name is not None:
            # Train JEPA on transition
            jepa_agent = self.agents.get("jepa_agent")
            if jepa_agent and hasattr(jepa_agent, 'train_step'):
                # Calculate reward for this transition
                # Note: Reward calculation still uses internal thought state for curiosity
                # But prediction target is now external reality (env_feedback)
                reward = self._calculate_combined_reward(
                    obs, self.prev_system_state, self.prev_action_name, current_system_thought
                )

                # JEPA training step
                # Input: Previous System Thought + Action
                # Target: Environment Feedback (Reality)
                pred_error = jepa_agent.train_step(
                    self.prev_system_state,
                    self.prev_action_name,
                    env_feedback,
                    reward
                )

                # Store to episodic memory
                self.episodic.add(
                    vector=self.prev_system_state,
                    metadata={
                        "action": self.prev_action_name,
                        "reward": reward,
                        "pred_error": pred_error,
                        "next_vector": env_feedback.tolist(),
                        "timestamp": time.time()
                    }
                )

        # Update previous state for next iteration (Input is System Thought)
        self.prev_system_state = current_system_thought.copy()
        self.prev_action_name = tool_name

        return tool_name, tool_info


def make_layout() -> Layout:
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=10)
    )
    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    layout["left"].split(
        Layout(name="brain", ratio=1),
        Layout(name="nervous", ratio=1)
    )
    layout["right"].split(
        Layout(name="body", ratio=2),
        Layout(name="meta", ratio=1)
    )
    return layout

def generate_brain_view(gnn_vector):
    # Visualize 16-dim vector as 4x4 grid
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_column()
    grid.add_column()
    grid.add_column()
    
    vec = gnn_vector[:16]
    # Normalize for color
    for i in range(0, 16, 4):
        row_cells = []
        for j in range(4):
            if i+j < len(vec):
                val = vec[i+j]
                color = "green" if val > 0.5 else "blue" if val > 0 else "white"
                row_cells.append(Panel(f"{val:.2f}", style=f"on {color}"))
            else:
                row_cells.append("")
        grid.add_row(*row_cells)
    
    return Panel(grid, title="[bold magenta]GNN System Consciousness[/]", border_style="magenta")

def generate_nervous_view(spikes):
    text = Text()
    if not spikes:
        text.append("Stable.", style="dim green")
    else:
        for s in spikes:
            text.append(f"??{s.upper()} ??n", style="bold red blink")
    return Panel(Align.center(text), title="[bold yellow]SNN Nervous System[/]", border_style="yellow")

def generate_body_view(agents, graph, last_action):
    table = Table(box=box.SIMPLE)
    table.add_column("Agent", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Perf", style="blue")
    
    for name in agents:
        alive = graph.node_alive.get(name, False)
        status = "ACTIVE" if alive else "DORMANT"
        style = "green" if alive else "dim red"
        perf = graph.node_perf.get(name, 0.5)
        table.add_row(name, status, f"{perf:.2f}", style=style)
        
    action_text = f"[bold white]LAST ACTION:[/]\n{last_action[0]}\n{last_action[1]}"
    return Panel(Align.center(table, vertical="middle"), title="[bold Cyan]Swarm Body (Agents)[/]", subtitle=action_text, border_style="cyan")

def generate_log_view(logs):
    text = Text()
    for l in logs[-6:]:
        text.append(l + "\n")
    return Panel(text, title="System Audit Log", style="dim white")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--benchmark", action="store_true", help="Run SOTA-style System Benchmark")
    parser.add_argument("--arc", action="store_true", help="Run Official ARC-AGI 2 Benchmark")
    args = parser.parse_args()

    system = AdvancedAISystem(device=args.device, arc_mode=args.arc)
    layout = make_layout()
    
    # Init results manager
    results_manager = ResultsManager(output_dir="benchmark_results")
    run_start_time = time.time()

    if args.arc:
        title_text = " ADVANCED AI SYSTEM: ARC AGI 2 BENCHMARK "
    elif args.benchmark:
        title_text = " ADVANCED AI SYSTEM: BENCHMARK MODE "
    else:
        title_text = " ADVANCED AI SYSTEM: SINGULARITY MODE "
        
    layout["header"].update(Panel(Align.center(f"[bold white on blue]{title_text}[/]"), box=box.HEAVY))
    
    # Benchmark Config
    episodes = 5 if args.arc else (3 if args.benchmark else 5)
    
    with Live(layout, refresh_per_second=4, screen=False) as live:
        episode = 1
        while episode <= episodes:
            # Benchmark Level Logic
            if args.arc:
                level = 1 # ARCEnv resets to random task regardless of level
                system.sandbox.reset()
                task_name = getattr(system.sandbox, "current_task_file", "Unknown")
                goal_text = f"Solve ARC Task: {task_name}"
            elif args.benchmark:
                level = episode
                system.sandbox.reset(level=level)
                goal_text = f"Solve Level {level} Pattern" 
            else:
                level = 1
                system.sandbox.reset(level=1)
                goal_text = "Achieve Singularity (Solve Pattern)"
            
            task_text = f"Episode {episode}: Optimization Loop"
            
            bb = Blackboard(episode_id=str(episode), goal_text=goal_text, task_text=task_text)
            bb.obs = system.env.observe()
            
            system.episode_log.append(f"--- EPISODE {episode} (Level {level}) START ---")
            
            # Track audit events for this episode
            episode_start_event_count = len(system.audit.events)

            steps_limit = 20
            reward = 0.0 # Default reward
            deadlock_count_this_ep = 0
            recovery_count_this_ep = 0

            step = 0 # Initialize step counter
            for step in range(steps_limit):
                # Run System Step
                tool, info = system.step(bb)
                
                if tool == "emergency_stop":
                    print("[System] SNN triggered Emergency Stop. Halting Episode.")
                    reward = -1.0
                    break # Properly exit the loop

                # Deadlock Prevention
                if tool in ["wait", "summarize"]:
                    system.consecutive_idle_count += 1
                else:
                    system.consecutive_idle_count = 0

                # Improved Deadlock Detection & Recovery
                if system.consecutive_idle_count > 3:
                    deadlock_count_this_ep += 1
                    system.deadlock_recovery_stats["total_deadlocks"] += 1

                    deadlock_info = {
                        "step": step,
                        "action_sequence": system.action_history[-5:],
                        "time_in_deadlock": system.consecutive_idle_count,
                        "episode": episode
                    }
                    system.episode_log.append(f"[DEADLOCK DETECTED] {deadlock_info}")
                    system.audit.emit("deadlock_detected", deadlock_info)

                    # Try recovery: force time-varying random selection
                    print(f"[System] Attempting deadlock recovery (Ep {episode}, Step {step})...")
                    system.episode_log.append("[RECOVERY] Switching to time-varying random selection strategy")

                    system.consecutive_idle_count = 0 # Reset counter to give recovery a chance

                    # Give system 3 more steps to escape using random strategy
                    recovery_steps = 3
                    escaped = False
                    for _ in range(recovery_steps):
                        # Force random strategy
                        r_tool, r_info = system.step(bb, force_strategy="random")

                        # Log it
                        log_msg = f"[Ep{episode}:St{step}+Rec] {r_tool} -> {r_info.get('ok')} (Recovery)"
                        system.episode_log.append(log_msg)

                        if r_tool not in ["wait", "summarize"]:
                            system.episode_log.append("[RECOVERY] SUCCESS - Escaped deadlock")
                            system.deadlock_recovery_stats["successful_recoveries"] += 1
                            recovery_count_this_ep += 1
                            escaped = True
                            break

                        # Update UI during recovery
                        time.sleep(0.1)

                    if escaped:
                         # Calculate current rate
                        total = system.deadlock_recovery_stats["total_deadlocks"]
                        succ = system.deadlock_recovery_stats["successful_recoveries"]
                        system.deadlock_recovery_stats["recovery_rate"] = succ / total if total > 0 else 0.0
                        continue # Continue main loop
                    else:
                        # If still stuck after recovery, then give up
                        print("[System] Recovery failed. Deadlock unresolvable.")
                        reward = -0.5
                        break

                # Logging
                res = "OK" if info["ok"] else "FAIL"
                cost = info["cost"]
                log_msg = f"[Ep{episode}:St{step}] {tool} -> {res} (Cost={cost:.2f})"
                system.episode_log.append(log_msg)
                
                # Update UI
                layout["brain"].update(generate_brain_view(system.system_thought))
                layout["nervous"].update(generate_nervous_view(system.last_spikes))
                layout["body"].update(generate_body_view(system.agents, system.graph, system.last_action))
                layout["footer"].update(generate_log_view(system.episode_log))
                
                time.sleep(0.1) # Faster for benchmark
                
                if bb.obs.get("last_test_ok"):
                    system.episode_log.append(f"[bold green]>>> LEVEL {level} SOLVED <<<[/]")
                    reward = 1.0
                    break
            else:
                reward = -0.1
            
            # === Online Learning (Hebbian) ===
            weight_mag = system.gnn.train(reward)
            system.episode_log.append(f"[BRAIN] Synaptic Update: Reward={reward}, Weights={weight_mag:.4f}")

            # === Meta-Learning: C-Stage Suppression ===
            # Calculate task loss from episode (negative reward)
            task_loss = max(0.0, 1.0 - reward)  # Convert [-1,1] reward to  loss

            # Check for suppression
            meta_suppress_results = {}
            for agent_name in system.agents:
                # Use suppress_agent
                impact = system.meta.suppress_agent(agent_name, task_loss)
                if impact:
                    meta_suppress_results[agent_name] = impact

            # Log summary
            if meta_suppress_results:
                short_res = {k: "SUPPRESSED" for k in meta_suppress_results.keys()}
                system.episode_log.append(f"[META-KERNEL] Suppression: {short_res}")

            # === Neural Architecture Search trigger ===
            # Every 3 episodes, check for underperforming agents
            if episode % 3 == 0:
                for agent_name in system.agents:
                    proposal = system.meta.suggest_architecture_modification(agent_name)
                    if proposal:
                        # Auto-vote for testing (in real system, use actual voting)
                        system.meta.vote(proposal.proposal_id, "orchestrator", True)
                        # Try to commit if quorum met (simplified)
                        if system.meta._quorum_ok(proposal):
                            system.meta.commit(proposal.proposal_id)
            
            # Record episode result
            # Calculate events just for this episode
            ep_events = system.audit.events[episode_start_event_count:]

            ep_result = EpisodeResult(
                episode_id=episode,
                level=level,
                completed=bb.obs.get("last_test_ok", False),
                reward=reward,
                total_steps=step + 1,
                total_cost=sum(system.budget_history[-step:]) if step > 0 else 0,
                deadlock_count=deadlock_count_this_ep,
                recovery_count=recovery_count_this_ep,
                meta_train_events=len([e for e in ep_events if e["type"] in ["meta_train", "meta_train_impact"]]),
                arch_mod_events=len([e for e in ep_events if e["type"] == "arch_mod_commit"])
            )
            results_manager.record_episode(ep_result)

            episode += 1

    # Finalize benchmark
    results_filename = results_manager.finalize_benchmark(
        run_id=f"benchmark_{int(time.time())}",
        total_runtime=time.time() - run_start_time,
        meta_train_count=len([e for e in system.audit.events if e["type"] in ["meta_train", "meta_train_impact"]]),
        arch_mod_count=len([e for e in system.audit.events if e["type"] == "arch_mod_commit"]),
        deadlock_recovery_rate=system.deadlock_recovery_stats.get("recovery_rate", 0.0)
    )
    print(f"Benchmark results saved to: {results_filename}")
                
    print(f"System Halted. Benchmark Complete." if args.benchmark else "System Halted. Optimization Complete.")

if __name__ == "__main__":
    main()
