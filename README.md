# DLHRL: An Efficient Multi-Task Path Planning Framework for Robot Navigation

We propose DLHRL, a hierarchical reinforcement learning (HRL) framework for efficient multi-task path planning in robot navigation. This framework decomposes the multi-task path planning problem into stable task allocation and safe motion control, and integrates three core innovations:(i) The establishment of the Spatio-Temporal Awareness Environment (STAE), which adopts a dual-temporal graph attention network (DTGAT) for the real-time quantification of task urgency and collision risk.(ii) The proposal of the Dual Lyapunov Optimization (DLO) algorithm, which comprises Lyapunov Allocation Optimization (LAO) for robot task sequencing under energy constraints and Lyapunov Safety Optimization (LSO) for collision-free motion control.(iii) The introduction of a drift-regularized meta-gradient (DRMG) mechanism within the hierarchical architecture, which enables fine-tuning of reward parameters while maintaining the stability guarantees of LAO and the safety constraints of LSO.


## Quick Start

### Installation

```bash
pip install torch numpy matplotlib tqdm cvxpy gym>=0.21.0
```

- **cvxpy**: QCQP solver for LSO module
- **gym**: Environment interface

### Usage

**Training:**
```bash
python main_dlhrl.py --mode train --episodes 1500
```

**Evaluation:**
```bash
python main_dlhrl.py --mode evaluate
```

**Training + Evaluation:**
```bash
python main_dlhrl.py --mode both
```

**Visualization:**
```bash
python visualize_dlhrl.py
```

## Project Structure

```
experiments/
├── ablation_staff/              # STAFF ablation experiments
│   ├── model_unified.py
│   └── results/models/...
├── dlhrl_core/                  # DLHRL core modules
│   ├── __init__.py
│   ├── stae_interface.py        # Spatial-Temporal Attribute Encoder
│   ├── lao_module.py            # Look-Ahead Optimizer
│   ├── lso_module.py            # Low-Level Safety Optimizer
│   ├── drmg_module.py           # Dynamic State Memory Graph
│   ├── actor_critic_hierarchical.py
│   ├── replay_buffer.py
│   └── dlhrl_agent.py
├── environment/                 # Simulation environment
│   ├── __init__.py
│   ├── grid_world.py
│   ├── robot_dynamics.py
│   └── task_generator.py
├── train_dlhrl.py               # Training script
├── evaluate_dlhrl.py            # Evaluation script
├── main_dlhrl.py                # Main entry point
├── visualize_dlhrl.py           # Visualization script
└── results/
    ├── dlhrl_models/            # Trained models
    ├── dlhrl_logs/              # Training logs
    └── dlhrl_figures/           # Visualization results
```

## Core Features

- **Hierarchical Decomposition**: Stable task allocation (LAO) + safe motion control (LSO)
- **Real-Time Awareness**: Dual-temporal graph attention for urgent task and risk quantification
- **Energy-Aware Planning**: Lyapunov-based task sequencing under energy constraints
- **Safety Guarantee**: Convex optimization for collision-free motion with stability preservation
- **Adaptive Learning**: DRMG mechanism for reward parameter tuning while maintaining guarantees
