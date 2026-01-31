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


## Usage

```bash
python -m heterogeneous_agent_swarm.main --episodes 10
```

## License

MIT
