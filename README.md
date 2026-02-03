# Heterogeneous Agent Swarm

Multi-agent system with structural self-modification.

## Agents

- SymbolicSearch, JEPAWorldModel, NeuroSymbolic, LiquidController, DiffusionExplorer, SSMStability, SNNReflex

## Components

- **Orchestrator**: Proposal aggregation, uncertainty-driven gating
- **MetaKernel**: Agent suppression, capacity modification
- **GNN**: State aggregation

## Features

- Agent suppression after 3 failures (loss > 0.8), recovery at loss < 0.2
- Panic mode (uncertainty > 0.1) blocks risky agents
- Runtime capacity adjustment with LR warmup
- Protocol-based interfaces, type validation

## Updates  

- JEPA: prediction-error selection, warmup/momentum stabilization
- C-Stage: failure tracking, uncertainty gating
- Protocol decoupling, full type hints

## Usage

```bash
python -m heterogeneous_agent_swarm.main --benchmark



