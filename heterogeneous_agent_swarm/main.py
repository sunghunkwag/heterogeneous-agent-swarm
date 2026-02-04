import time
import argparse
import sys
import os
import numpy as np
from datetime import datetime
from collections import Counter
from typing import Dict, Any

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
from heterogeneous_agent_swarm.envs.arc_env import ARCEnv 
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
        self.device = device
        
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
        self.episodic = EpisodicMemory(capacity=1000)
        self.orch = Orchestrator(self.graph)
        self.orch.memory = self.episodic  

        self.meta = MetaKernelV2(self.graph, self.audit, self.orch, self.agents, device=device)
        self.encoder = SimpleEncoder()
        self.eval = Evaluator()
        self.work = WorkingMemory()
        
        # State tracking
        self.system_thought = np.zeros(16)
        self.last_spikes = []
        self.last_action = ("None", {})
        self.episode_log = []
        self.consecutive_idle_count = 0
        self.last_success_time = None
        self.start_time = time.time()
        self.budget_history = []
        self.action_history = []

        self.deadlock_recovery_stats = {
            "total_deadlocks": 0,
            "successful_recoveries": 0,
            "recovery_rate": 0.0
        }

        self.prev_system_state = None
        self.prev_action_name = None

        # Warmup for JEPA input dim
        self.agents["jepa_agent"].config.input_dim = 16  

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
        add(LiquidControllerAgent("liquid_agent", LiquidConfig(device=device)))
        add(DiffusionExplorerAgent("diffusion_agent", DiffusionConfig()))
        add(SSMStabilityAgent("ssm_agent", SSMConfig()))
        add(SNNReflexAgent("snn_agent", SNNConfig()))

    def _calculate_budget_slope(self):
        if len(self.budget_history) < 2: return 0.0
        y = self.budget_history[-5:]
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        return slope

    def _calculate_entropy(self):
        if not self.action_history: return 0.0
        actions = self.action_history[-10:]
        counts = Counter(actions)
        total = len(actions)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        return entropy

    def _get_system_state_vector(self) -> np.ndarray:
        return self.system_thought.copy()

    def _get_env_feedback_vector(self, obs: Dict, cost: float) -> np.ndarray:
        vec = np.zeros(16, dtype=np.float32)
        vec[0] = 1.0 if obs.get("last_test_ok") else 0.0
        vec[1] = 1.0 if obs.get("error") else 0.0
        info = self.work.get("last_tool", {})
        diff = info.get("output", {}).get("diff_count", 0) if isinstance(info.get("output"), dict) else 0
        vec[2] = min(diff / 100.0, 1.0)
        vec[3] = 0.5 
        vec[4] = min(cost, 1.0)
        return vec

    def _calculate_combined_reward(self, obs: Dict, prev_state: np.ndarray,
                                   action_name: str, current_state: np.ndarray) -> float:
        if obs.get("last_test_ok"): extrinsic = 1.0
        elif obs.get("error"): extrinsic = -0.5
        else: extrinsic = -0.01  

        jepa_agent = self.agents.get("jepa_agent")
        if jepa_agent and hasattr(jepa_agent, 'get_curiosity_reward'):
            intrinsic = jepa_agent.get_curiosity_reward(prev_state, action_name, current_state)
            intrinsic = min(1.0, max(-1.0, intrinsic * 10))
        else:
            intrinsic = 0.0
        return 0.7 * extrinsic + 0.3 * intrinsic

    def step(self, bb: Blackboard, force_strategy: str = None):
        current_time = time.time()
        t_since = current_time - (self.last_success_time if self.last_success_time else self.start_time)

        snn_inputs = {
            "time_since_success": t_since,
            "error_rate": -1.0 * self.eval.evaluate(bb.__dict__).score,
            "budget_slope": self._calculate_budget_slope(),
            "entropy": self._calculate_entropy()
        }
        interrupts = self.snn.process_signals(snn_inputs)
        self.last_spikes = interrupts
        self.work.set("error_rate", snn_inputs["error_rate"])

        gnn_inputs = {name: np.array([1.0 if self.graph.node_alive.get(name, False) else 0.0, 
                                      self.graph.node_perf.get(name, 0.5), 0.0]) for name in self.agents}
        self.system_thought = self.gnn.update(gnn_inputs)
        
        state = self.encoder.encode(bb.obs)
        state.system_thought = self.system_thought.tolist()

        mem = self.work.snapshot()
        mem["system_thought"] = self.system_thought.tolist() 
        
        proposals = {name: agent.propose(state, mem) for name, agent in self.agents.items() if self.graph.node_alive.get(name, False)}
        
        sys_uncertainty = self.gnn.get_system_uncertainty()
        tool_name, tool_args, _ = self.orch.choose(proposals, state, None, force_strategy=force_strategy, system_uncertainty=sys_uncertainty)
        
        obs, tool_info = self.env.step_tool(tool_name, tool_args)
        obs = {**obs, **self.sandbox.observe()}

        self.budget_history.append(tool_info.get("cost", 0.0))
        self.action_history.append(tool_name)
        if obs.get("last_test_ok"): self.last_success_time = time.time()

        bb.obs = obs
        bb.record_step(tool_info)
        self.work.set("last_tool", tool_info)
        self.work.set("obs", obs)
        self.last_action = (tool_name, tool_args)
        
        # JEPA Training
        current_thought = self._get_system_state_vector()
        env_feedback = self._get_env_feedback_vector(obs, tool_info.get("cost", 0.0))

        if self.prev_system_state is not None and self.prev_action_name is not None:
            jepa_agent = self.agents.get("jepa_agent")
            if jepa_agent and hasattr(jepa_agent, 'train_step'):
                reward = self._calculate_combined_reward(obs, self.prev_system_state, self.prev_action_name, current_thought)
                pred_error = jepa_agent.train_step(self.prev_system_state, self.prev_action_name, env_feedback, reward)
                self.episodic.add(vector=self.prev_system_state, metadata={"action": self.prev_action_name, "reward": reward, "pred_error": pred_error})

        self.prev_system_state = current_thought.copy()
        self.prev_action_name = tool_name
        return tool_name, tool_info

def make_layout() -> Layout:
    layout = Layout(name="root")
    layout.split(Layout(name="header", size=3), Layout(name="main", ratio=1), Layout(name="footer", size=10))
    layout["main"].split_row(Layout(name="left"), Layout(name="right"))
    layout["left"].split(Layout(name="brain", ratio=1), Layout(name="nervous", ratio=1))
    layout["right"].split(Layout(name="body", ratio=2), Layout(name="meta", ratio=1))
    return layout

def generate_brain_view(gnn_vector):
    grid = Table.grid(expand=True)
    for _ in range(4): grid.add_column()
    vec = gnn_vector[:16]
    for i in range(0, 16, 4):
        grid.add_row(*[Panel(f"{vec[i+j]:.2f}", style=f"on {'green' if vec[i+j]>0.5 else 'blue' if vec[i+j]>0 else 'white'}") for j in range(4)])
    return Panel(grid, title="[bold magenta]GNN System Consciousness[/]", border_style="magenta")

def generate_nervous_view(spikes):
    text = Text("Stable.", style="dim green") if not spikes else Text("".join([f"??{s.upper()} ??n" for s in spikes]), style="bold red blink")
    return Panel(Align.center(text), title="[bold yellow]SNN Nervous System[/]", border_style="yellow")

def generate_body_view(agents, graph, last_action):
    table = Table(box=box.SIMPLE)
    table.add_column("Agent", style="cyan"); table.add_column("Status", style="green"); table.add_column("Perf", style="blue")
    for name in agents:
        alive = graph.node_alive.get(name, False)
        table.add_row(name, "ACTIVE" if alive else "DORMANT", f"{graph.node_perf.get(name, 0.5):.2f}", style="green" if alive else "dim red")
    return Panel(Align.center(table, vertical="middle"), title="[bold Cyan]Swarm Body (Agents)[/]", subtitle=f"[bold white]LAST ACTION:[/]\n{last_action[0]}", border_style="cyan")

def generate_log_view(logs):
    text = Text("".join([l + "\n" for l in logs[-6:]]))
    return Panel(text, title="System Audit Log", style="dim white")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--arc", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    system = AdvancedAISystem(device=args.device, arc_mode=args.arc)
    layout = make_layout()
    results_manager = ResultsManager(output_dir="benchmark_results")
    run_start_time = time.time()
    
    layout["header"].update(Panel(Align.center(f"[bold white on blue] ADVANCED AI SYSTEM: {'ARC' if args.arc else 'BENCHMARK' if args.benchmark else 'SINGULARITY'} MODE [/]"), box=box.HEAVY))
    
    episodes = args.episodes
    from heterogeneous_agent_swarm.core.meta_meta_optimizer import MetaMetaOptimizer
    meta_meta = MetaMetaOptimizer(population_size=4, generation_window=5)
    
    current_level = 1
    consecutive_successes = 0
    last_applied_genome = None

    with Live(layout, refresh_per_second=4, screen=False) as live:
        episode = 1
        while episode <= episodes:
            if args.arc: system.sandbox.reset(); goal_text = f"Solve ARC Task: {getattr(system.sandbox, 'current_task_file', 'Unknown')}"
            else: system.sandbox.reset(level=episode if args.benchmark else current_level); goal_text = f"Solve Level {episode if args.benchmark else current_level}"
            
            from heterogeneous_agent_swarm.envs.tool_env import ToolEnvState
            system.env.state = ToolEnvState()

            curr_conf = meta_meta.get_current_config()
            task_text = f"Ep {episode}: [bold cyan]G{meta_meta.generation}:C{meta_meta.current_candidate_idx}[/]"
            
            # --- OPTIMIZED STRUCTURAL RSI: Only Re-init if Genome Changed ---
            if curr_conf != last_applied_genome:
                # 1. JEPA
                if "jepa_agent" in system.agents:
                    agent = system.agents["jepa_agent"]
                    if agent.config.num_layers != curr_conf["jepa_depth"]:
                        from heterogeneous_agent_swarm.agents.jepa_world_model import JEPAConfig, JEPAWorldModelAgent
                        system.agents["jepa_agent"] = JEPAWorldModelAgent("jepa_agent", JEPAConfig(device=system.device, learning_rate=curr_conf["jepa_lr"], num_layers=curr_conf["jepa_depth"]))
                    else: agent.update_hyperparameters(lr=curr_conf["jepa_lr"])

                # 2. SNN
                if "snn_agent" in system.agents:
                    if system.agents["snn_agent"].config.hidden_neurons != curr_conf["snn_neurons"]:
                        from heterogeneous_agent_swarm.agents.snn_reflex import SNNConfig, SNNReflexAgent
                        system.agents["snn_agent"] = SNNReflexAgent("snn_agent", SNNConfig(hidden_neurons=curr_conf["snn_neurons"]))
                
                # 3. Symbolic
                if "symbolic_agent" in system.agents:
                    if system.agents["symbolic_agent"].config.max_search_depth != curr_conf["symbolic_depth"]:
                        from heterogeneous_agent_swarm.agents.symbolic_search import SymbolicConfig, SymbolicSearchAgent
                        system.agents["symbolic_agent"] = SymbolicSearchAgent("symbolic_agent", SymbolicConfig(max_search_depth=curr_conf["symbolic_depth"]))

                if hasattr(system.meta, 'update_min_quorum'):
                    system.meta.update_min_quorum(curr_conf["meta_min_quorum"])
                
                last_applied_genome = copy.deepcopy(curr_conf)
                system.episode_log.append(f"[RSI] Architecture Reconfigured for Candidate {meta_meta.current_candidate_idx}")

            bb = Blackboard(episode_id=str(episode), goal_text=goal_text, task_text=task_text)
            bb.obs = {**system.env.observe(), **system.sandbox.observe()}
            system.episode_log.append(f"--- EPISODE {episode} START ---")
            
            episode_start_event_count = len(system.audit.events)
            steps_limit = 20
            reward = 0.0 
            deadlock_count = 0; recovery_count = 0

            for step in range(steps_limit):
                tool, info = system.step(bb)
                if tool == "emergency_stop": reward = -1.0; break
                
                system.consecutive_idle_count = system.consecutive_idle_count + 1 if tool in ["wait", "summarize"] else 0
                if system.consecutive_idle_count > 3:
                    deadlock_count += 1
                    system.deadlock_recovery_stats["total_deadlocks"] += 1
                    system.episode_log.append("[DEADLOCK] Attempting Recovery...")
                    system.consecutive_idle_count = 0
                    escaped = False
                    for _ in range(3):
                        r_tool, r_info = system.step(bb, force_strategy="random")
                        if r_tool not in ["wait", "summarize"]:
                            system.deadlock_recovery_stats["successful_recoveries"] += 1
                            recovery_count += 1; escaped = True; break
                    if not escaped: reward = -0.5; break

                layout["brain"].update(generate_brain_view(system.system_thought))
                layout["nervous"].update(generate_nervous_view(system.last_spikes))
                layout["body"].update(generate_body_view(system.agents, system.graph, system.last_action))
                layout["footer"].update(generate_log_view(system.episode_log))
                
                if bb.obs.get("last_test_ok"):
                    system.episode_log.append(">>> SOLVED <<<")
                    reward = 1.0
                    if meta_meta.current_candidate_idx == 0:
                        consecutive_successes += 1
                        if consecutive_successes >= 3 and current_level < 4:
                            current_level += 1; consecutive_successes = 0
                    break
            else:
                reward = -0.1
                if meta_meta.current_candidate_idx == 0: consecutive_successes = 0
            
            mm_status = meta_meta.report_episode_reward(reward)
            
            # --- PORTABLE RSI LOGGING: Use relative path ---
            try:
                import json
                genome_path = os.path.join(os.getcwd(), "current_genome.json")
                with open(genome_path, "w") as f:
                    json.dump({
                        "generation": meta_meta.generation, "candidate": meta_meta.current_candidate_idx,
                        "level": current_level, "genome": curr_conf,
                        "ucb": {k: {"n": meta_meta.counts[k], "v": meta_meta.values[k]} for k in meta_meta.mutations}
                    }, f, indent=2)
            except Exception: pass

            weight_mag = system.gnn.train(reward)
            task_loss = max(0.0, 1.0 - reward)
            for agent_name in system.agents:
                if agent_name != "symbolic_agent": system.meta.enforce_agent_constraints(agent_name, task_loss)

            if episode % 3 == 0:
                for agent_name in system.agents:
                    prop = system.meta.suggest_architecture_modification(agent_name)
                    if prop:
                        system.meta.vote(prop.proposal_id, "orchestrator", True)
                        if system.meta._quorum_ok(prop): system.meta.commit(prop.proposal_id)
            
            ep_events = system.audit.events[episode_start_event_count:]
            results_manager.record_episode(EpisodeResult(
                episode_id=episode, level=episode if args.benchmark else current_level,
                completed=bb.obs.get("last_test_ok", False), reward=reward, total_steps=step + 1,
                total_cost=sum(system.budget_history[-step:]) if step > 0 else 0,
                deadlock_count=deadlock_count, recovery_count=recovery_count,
                meta_train_events=len([e for e in ep_events if e["type"] in ["meta_train", "meta_train_impact"]]),
                arch_mod_events=len([e for e in ep_events if e["type"] == "arch_mod_commit"])
            ))
            episode += 1

    results_filename = results_manager.finalize_benchmark(
        run_id=f"benchmark_{int(time.time())}", total_runtime=time.time() - run_start_time,
        meta_train_count=len([e for e in system.audit.events if e["type"] in ["meta_train", "meta_train_impact"]]),
        arch_mod_count=len([e for e in system.audit.events if e["type"] == "arch_mod_commit"]),
        deadlock_recovery_rate=system.deadlock_recovery_stats.get("recovery_rate", 0.0)
    )
    print(f"Results: {results_filename}\nHalted.")

if __name__ == "__main__":
    main()
