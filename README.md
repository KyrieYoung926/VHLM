# VHLM: Vision-based Humanoid Loco-Manipulation

A hierarchical control system for humanoid robots based on Genesis simulator, (will be) supporting vision-guided locomotion and manipulation task.

## Project Structure

```
VHLM/
├── assets/          # Model and policy files
│   ├── g1.xml      # G1 robot MJCF model
│   ├── high_level.jit  # High-level policy network
│   └── low_level.pt    # Low-level locomotion policy
├── cfg/            # Configuration files
│   └── robot_config.py  # Robot joint mappings and default configs
├── core/           # Core control modules
│   ├── controller_new.py  # Hierarchical controller main logic
│   └── ik_control.py      # Differential IK solver
├── utils/          # Utility functions
│   └── math_ops.py  # Math operations (quaternion, rotation, etc.)
└── scripts/        # Execution scripts
    └── deploy_genesis.py  # Deployment entry point
```

## Quick Start

### Requirements

Tested on Ubuntu 20.04 with:

- Python 3.10.18
- Genesis 0.3.3
- RTX 4090

### Installation

Follow Genesis installation instructions: https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html#

### Run Simulation

```bash
# Basic run (default parameters)
python scripts/deploy_genesis.py

# Custom parameters
python scripts/deploy_genesis.py \
  --policy assets/low_level.pt \
  --high-policy assets/high_level.jit \
  --robot assets/g1.xml \
  --duration 300 \
  --device cuda:0
```

### Main Parameters

- `--policy`: Path to low-level locomotion policy (.pt file)
- `--high-policy`: Path to high-level task policy (.jit file)
- `--robot`: Path to robot MJCF model
- `--duration`: Simulation duration (seconds)
- `--dt`: Simulation time step (default 0.02s)
- `--no-render`: Disable visualization
- `--device`: Computing device (cuda:0 / cpu)

## Core Features

### Hierarchical Control Architecture

- **High-level Policy**: (TODO) Vision input → End-effector target poses
- **Low-level Policy**: Target poses → Joint position/velocity control
- **IK Solver**: End-effector control via differential Jacobian

### Supported Robots

- Unitree G1 humanoid robot (29 DOF)


## License

MIT
