# Heterogeneous Agent Swarm

Multi-agent system with structural self-modification and cost-aware coordination.

## Quick Start

```bash
pip install -r requirements.txt
python -m heterogeneous_agent_swarm.main --benchmark
```

## Agents

SymbolicSearch, JEPAWorldModel, LiquidController, DiffusionExplorer, SSMStability, SNNReflex

## Core Components

| Component | Function |
|-----------|----------|
| **Orchestrator** | Proposal aggregation, uncertainty-driven gating |
| **MetaKernel** | Agent suppression/recovery, capacity modification |
| **GNN Brain** | State aggregation, system consciousness |
| **Config** | Centralized thresholds and parameters |

## Self-Modification Features

- **Agent Suppression**: Auto-suppress after 3 consecutive failures (loss > 0.8)
- **Agent Recovery**: Auto-recover when loss < 0.2
- **Emergency Rotation**: Deadlock mitigation via agent awakening
- **NAS Execution**: Automatic architecture modification for underperformers
- **Panic Mode**: High uncertainty (> 0.1) blocks risky exploration agents
- **Checkpointing**: Save/load swarm state across sessions

## Benchmarks

```bash
# Standard benchmark (5 episodes)
python -m heterogeneous_agent_swarm.main --benchmark

# Hard adversarial benchmark (tests self-modification)
python -m heterogeneous_agent_swarm.hard_benchmark
```

### Hard Benchmark Results

| Metric | Verified |
|--------|----------|
| Suppression | ✓ |
| Recovery | ✓ |
| Emergency Rotation | ✓ |
| NAS Execution | ✓ |

## Recent Updates

- Centralized configuration via `config.py`
- Kaiming initialization for JEPA capacity changes
- Improved error handling with logging
- Hard benchmark for self-modification verification
- Full test coverage (26/26 passing)

## Project Structure

```
heterogeneous_agent_swarm/
├── agents/          # 6 heterogeneous agents
├── core/            # Orchestrator, MetaKernel, GNN, Config
├── envs/            # Sandbox environments
├── runtime/         # Execution runtime
├── main.py          # Standard benchmark
└── hard_benchmark.py # Adversarial benchmark
```

## License

MIT
