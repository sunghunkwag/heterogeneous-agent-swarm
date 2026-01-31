from __future__ import annotations
from typing import Dict, Any, List
import uuid

from rich.console import Console

from ..core.blackboard import Blackboard
from ..core.tasking import TaskQueue
from ..core.evaluation import Evaluator
from ..core.memory import WorkingMemory, EpisodicMemory, SemanticMemory, EpisodicRecord
from ..core.audit import AuditLog
from ..core.graph import AgentGraph
from ..core.orchestrator import Orchestrator
from ..core.meta_kernel_v2 import MetaKernelV2
from ..core.types import EncodedState, Proposal
from ..core.gnn_brain import LightweightGNN
from ..agents.snn_core import EventSNN
import numpy as np

# Reuse or define SimpleEncoder
class SimpleEncoder:
    def encode(self, obs):
        return EncodedState(
            task_latent=[], 
            agent_health_latent=[], 
            uncertainty=0.5, 
            risk_flags=[], 
            raw_obs=obs
        )

class RunnerV2:
    def __init__(self, agents: Dict[str, Any], graph: AgentGraph, device: str = "cpu"):
        self.console = Console()
        self.encoder = SimpleEncoder()
        self.audit = AuditLog()
        self.graph = graph
        self.orch = Orchestrator(self.graph)
        self.meta = MetaKernelV2(self.graph, self.audit, min_quorum=3)
        
        # New Brains
        self.gnn = LightweightGNN(agent_names=list(agents.keys()))
        self.snn = EventSNN()

        self.agents = agents

        self.eval = Evaluator()
        self.work = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()

        self.tasks = TaskQueue()

    def _collect_proposals(self, state: EncodedState) -> Dict[str, Proposal]:
        proposals: Dict[str, Proposal] = {}
        mem = self.work.snapshot()
        
        # 1. Update GNN with "Mental State" of agents (Pseudo-telepathy)
        # For this prototype, we treat agent's *output confidence* from previous step as input feature?
        # Or simple random signature.
        gnn_inputs = {}
        for name in self.agents:
            # Feature: [Alive(1/0), Perf, Cost]
            alive = 1.0 if self.graph.node_alive.get(name, False) else 0.0
            perf = self.graph.node_perf.get(name, 0.5)
            gnn_inputs[name] = np.array([alive, perf, 0.0]) # dim 3
            
        system_thought = self.gnn.update(gnn_inputs)
        # Inject system thought into shared memory for agents to see?
        mem["system_thought"] = system_thought.tolist()

        for name, agent in self.agents.items():
            if not self.graph.node_alive.get(name, False):
                continue
            proposals[name] = agent.propose(state, mem)
        return proposals

    def _extract_veto(self, proposals: Dict[str, Proposal]) -> Dict[str, Any] | None:
        p = proposals.get("neurosym_agent")
        if not p:
            return None
        verdict = p.artifacts.get("verdict")
        if verdict == "deny":
            return {"deny": True, "deny_action": p.artifacts.get("deny_action"), "reason": p.artifacts.get("rule_id")}
        return {"deny": False}

    def run_one_task(self, env, goal_text: str, task_text: str, max_tool_steps: int = 12) -> Dict[str, Any]:
        episode_id = str(uuid.uuid4())
        bb = Blackboard(episode_id=episode_id, goal_text=goal_text, task_text=task_text)

        obs = env.observe()
        bb.obs = obs

        # Loop: propose -> choose -> execute tool -> update bb -> repeat
        for _ in range(max_tool_steps):
            # Check SNN Interrupts
            snn_signals = {
                "time_since_success": 0.0, # Stub
                "error_rate": self.eval.evaluate(bb.__dict__).score * -1, 
                "budget_slope": -0.1,
                "entropy": 0.5
            }
            interrupts = self.snn.process_signals(snn_signals)
            if "emergency_stop" in interrupts:
                self.console.print("[SNN] EMERGENCY STOP TRIGGERED!")
                break
            
            state = self.encoder.encode(bb.obs)
            proposals = self._collect_proposals(state)
            veto = self._extract_veto(proposals)

            # Updated Orchestrator returns (tool_name, tool_args, debug)
            tool_name, tool_args, dbg = self.orch.choose(proposals, state, veto=veto)
            self.audit.emit("choose", dbg)

            # Map the high-level action to the tool name if needed?
            # v0.1 agents proposed "APPEND", "DELETE", "TEST"
            # v0.2 ToolRegistry has "write_patch", "run_tests".
            # I need a translation layer or update Agent proposals.
            # I will assume Agents Propose "write_patch" instead of "APPEND".
            
            # Simple Mapping for v0.1 compatibility:
            if tool_name == "APPEND":
                tool_name = "write_patch"
                # tool_args has "value"
            elif tool_name == "DELETE":
                tool_name = "write_patch"
                # DELETE usually means remove last. 
                # Sandbox "write_patch" doesn't strictly support DELETE in my mock?
                # I'll ignore DELETE or map it to no-op for now in this demo.
                pass 
            elif tool_name == "TEST":
                tool_name = "run_tests"
            
            obs, tool_info = env.step_tool(tool_name, tool_args)
            bb.obs = obs
            bb.record_step(tool_info)
            
            # Record outcome for GNN/SNN inputs next step
            # (Implicitly done via memory update)

            # Update working memory
            self.work.set("last_tool", tool_info)
            self.work.set("obs", obs)

            # Stop if solved
            if obs.get("last_test_ok", False):
                break

        # Evaluate
        eval_res = self.eval.evaluate(bb.__dict__)
        bb.set_signal("eval", {"success": eval_res.success, "score": eval_res.score, "notes": eval_res.notes})

        # Update episodic memory
        self.episodic.add(
            EpisodicRecord(
                ts=bb.ts,
                goal=goal_text,
                task=task_text,
                summary=f"steps={len(bb.step_history)} last_ok={bb.obs.get('last_test_ok')}",
                success=eval_res.success,
                score=eval_res.score,
                artifacts={"audit_tail": [e.payload for e in self.audit.tail(5)]},
            )
        )

        # Meta-kernel can propose structural changes based on sustained badness
        alive = self.graph.alive_nodes()
        if len(alive) > 4:
            worst = None
            for n in alive:
                if n in ("neurosym_agent", "snn_agent"):
                    continue
                perf = self.graph.node_perf.get(n, 0.5)
                cost = self.graph.node_cost.get(n, 0.05)
                badness = (0.60 - perf) + 1.1 * max(0.0, cost - 0.10)
                if worst is None or badness > worst[0]:
                    worst = (badness, n, perf, cost)

            if worst and worst[0] > 0.60:
                cp = self.meta.propose("drop_agent", {"name": worst[1]}, rationale=f"badness={worst[0]:.2f}")
                voters = [x for x in alive if x != worst[1]][:3]
                for v in voters:
                    self.meta.vote(cp.proposal_id, v, approve=True)
                self.meta.commit(cp.proposal_id)

        return {
            "episode_id": episode_id,
            "success": eval_res.success,
            "score": eval_res.score,
            "alive": self.graph.alive_nodes(),
            "steps": bb.step_history,
            "signals": bb.signals,
        }
