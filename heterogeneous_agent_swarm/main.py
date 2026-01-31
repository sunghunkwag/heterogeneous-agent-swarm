import time
import argparse
import sys
import os
import numpy as np
from datetime import datetime

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
        self.orch = Orchestrator(self.graph)
        self.meta = MetaKernelV2(self.graph, self.audit)
        self.encoder = SimpleEncoder()
        self.eval = Evaluator()
        self.work = WorkingMemory()
        
        # State tracking for UI
        self.system_thought = np.zeros(16)
        self.last_spikes = []
        self.last_action = ("None", {})
        self.episode_log = []
        self.consecutive_idle_count = 0

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

    def step(self, bb: Blackboard):
        """Execute one cognitive cycle."""
        # A. SNN Check (Nervous System)
        snn_inputs = {
            "time_since_success": 0.1, 
            "error_rate": -1.0 * self.eval.evaluate(bb.__dict__).score,
            "budget_slope": -0.1,
            "entropy": abs(np.sin(len(self.episode_log))) # Deterministic entropy proxy
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
            veto = {"deny": True}
        
        tool_name, tool_args, dbg = self.orch.choose(proposals, state, veto)
        
        # Mapping Removed (Agents now speak Vocabulary)

        # E. Execution
        obs, tool_info = self.env.step_tool(tool_name, tool_args)
        bb.obs = obs
        bb.record_step(tool_info)
        
        self.work.set("last_tool", tool_info)
        self.work.set("obs", obs)
        self.work.set("last_action_name", tool_name)
        self.work.set("last_action_params", tool_args)
        self.last_action = (tool_name, tool_args)
        
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
            
            steps_limit = 20
            reward = 0.0 # Default reward
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

                if system.consecutive_idle_count > 3:
                    print("[System] Deadlock Detected (>3 Idle Steps). Breaking.")
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
            
            # Online Learning (Hebbian)
            weight_mag = system.gnn.train(reward)
            system.episode_log.append(f"[BRAIN] Synaptic Update: Reward={reward}, Weights={weight_mag:.4f}")
            
            episode += 1
                
    print(f"System Halted. Benchmark Complete." if args.benchmark else "System Halted. Optimization Complete.")

if __name__ == "__main__":
    main()
