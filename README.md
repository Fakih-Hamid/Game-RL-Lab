# Reinforcement Learning for Game Environments

PPO implementation on a custom grid-based game environment using PyTorch.

## Problem Formulation

Grid world navigation task: agent collects goals while avoiding hazards.

**State Space**: Flattened grid representation (grid_size × grid_size)
- 0: Empty cell
- 1: Agent position
- 2: Goal
- 3: Hazard
- 4: Wall

**Action Space**: Discrete {0: Up, 1: Right, 2: Down, 3: Left, 4: Stay}

**Rewards**:
- Goal: +10.0
- Hazard: -10.0
- Step: -0.1
- Termination: Episode ends when all goals collected or max steps (200) reached

## Environment

`GridWorldEnv` implements Gym interface:
- `reset()`: Random initialization
- `step(action)`: Returns (observation, reward, done, info)
- `render()`: ASCII visualization

Goals are removed when collected. Episodes terminate on completion or step limit.

## Algorithm

PPO with clipped surrogate objective. Policy and value networks are 3-layer MLPs (128 hidden units).

**Hyperparameters**:
- Policy LR: 3e-4
- Value LR: 3e-4
- γ: 0.99
- GAE λ: 0.95
- Clip ε: 0.2
- Value coefficient: 0.5
- Entropy coefficient: 0.01
- Update frequency: 2048 steps
- Update epochs: 10
- Batch size: 64

## Training

```bash
python train.py --num_episodes 1000
```

Checkpoints saved every 100 episodes to `./checkpoints/`.

## Evaluation

```bash
python evaluate.py --checkpoint ./checkpoints/ppo_final.pth
```

Compares PPO agent against random and heuristic baselines. Reports average reward, success rate, episode length, and goals collected.

## Project Structure

```
Game-RL-Lab/
├── env/
│   ├── __init__.py
│   └── grid_world.py
├── agents/
│   ├── __init__.py
│   └── ppo_agent.py
├── utils/
│   └── __init__.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## References

- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
