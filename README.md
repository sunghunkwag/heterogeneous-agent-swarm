# Heterogeneous Agent Swarm

This repository contains a **research prototype** for experimenting with heterogeneous multi-agent swarms under explicit cost, budget, and control constraints.

⚠️ **Status:** Work-in-progress research code.  
This is **not** a production system and **not** a benchmarked AI model.

## Overview

The system runs multiple heterogeneous agents in parallel, each proposing actions based on different inductive biases (control, constraints, exploration, reflex).

A central orchestrator arbitrates between proposals using confidence, cost, conflict, and diversity signals.  
A meta-level kernel can modify system behavior (e.g., mode switching) under budget pressure or stagnation.

## What This Is

- A conceptual architecture prototype
- A sandbox for studying cost-aware multi-agent coordination
- An exploration of meta-control and structural adaptation

## Recent Fixes & Improvements

- **Bug Fixes:** Resolved `TypeError` in meta-learning param access and silenced PyTorch `std()` warnings.
- **RSI Foundation:** Implemented actual architecture modifications (e.g., neural capacity expansion) and meta-learning impact tracking.
- **Resilience:** Added deadlock recovery mechanism using time-varying random selection strategies.
- **Metrics:** Added structured JSON logging for benchmark results.

## Usage

Run the benchmark mode:

```bash
python -m heterogeneous_agent_swarm.main --benchmark
```

### Interpreting Results

Results are saved to `benchmark_results/benchmark_<TIMESTAMP>.json`. Key metrics include:

- `success_rate`: Percentage of episodes solved.
- `deadlock_recovery_rate`: Success rate of escaping detected deadlocks.
- `meta_train_total_count`: Number of times agent learning rates were adapted.
- `arch_mod_total_count`: Number of times agent architectures were modified (NAS).

## License

MIT
