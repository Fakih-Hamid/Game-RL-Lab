# Instructions de Test

## Test Rapide

```bash
python test.py
```

Vérifie que l'environnement, l'agent et l'entraînement fonctionnent correctement.

## Test Complet

### 1. Entraînement (court)

```bash
python train.py --num_episodes 100 --update_frequency 512
```

Entraîne l'agent sur 100 épisodes avec des mises à jour plus fréquentes.

### 2. Entraînement Complet

```bash
python train.py --num_episodes 1000
```

Entraîne l'agent sur 1000 épisodes. Les checkpoints sont sauvegardés dans `./checkpoints/`.

### 3. Évaluation

```bash
python evaluate.py --checkpoint ./checkpoints/ppo_final.pth --num_episodes 50
```

Compare l'agent entraîné avec les baselines (random et heuristic).

## Test Manuel de l'Environnement

```python
from env.grid_world import GridWorldEnv

env = GridWorldEnv(grid_size=5, num_goals=1, num_hazards=2)
state = env.reset()
env.render()

for _ in range(10):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    print(f"Reward: {reward}, Done: {done}, Goals: {info['goals_collected']}")
    if done:
        break
```

