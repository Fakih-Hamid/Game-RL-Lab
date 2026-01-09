import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
from typing import Tuple, Dict, Any


class GridWorldEnv(gym.Env):
    """Grid-based navigation environment.
    
    Grid: 0=empty, 1=agent, 2=goal, 3=hazard, 4=wall
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, grid_size: int = 10, num_goals: int = 1, num_hazards: int = 3, 
                 hazard_reward: float = -10.0, goal_reward: float = 10.0, 
                 step_reward: float = -0.1, max_steps: int = 200):
        super(GridWorldEnv, self).__init__()
        
        self.grid_size = grid_size
        self.num_goals = num_goals
        self.num_hazards = num_hazards
        self.hazard_reward = hazard_reward
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.max_steps = max_steps
        
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(grid_size * grid_size,), dtype=np.int32
        )
        self.grid = None
        self.agent_pos = None
        self.goal_positions = []
        self.step_count = 0
        self.goals_collected = 0
        
    def reset(self) -> np.ndarray:
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        while True:
            self.agent_pos = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            )
            if self.grid[self.agent_pos] == 0:
                break
        
        self.grid[self.agent_pos] = 1
        
        self.goal_positions = []
        for _ in range(self.num_goals):
            while True:
                pos = (
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                )
                if self.grid[pos] == 0:
                    self.grid[pos] = 2
                    self.goal_positions.append(pos)
                    break
        
        for _ in range(self.num_hazards):
            while True:
                pos = (
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                )
                if self.grid[pos] == 0:
                    self.grid[pos] = 3
                    break
        
        self.step_count = 0
        self.goals_collected = 0
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_count += 1
        
        new_pos = list(self.agent_pos)
        
        if action == 0:
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        elif action == 2:
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        elif action == 3:
            new_pos[1] = max(0, new_pos[1] - 1)
        
        new_pos = tuple(new_pos)
        reward = self.step_reward
        
        if self.grid[new_pos] != 4:
            if self.grid[self.agent_pos] == 1:
                self.grid[self.agent_pos] = 0
            
            self.agent_pos = new_pos
            cell_value = self.grid[new_pos]
            
            if cell_value == 2:
                reward += self.goal_reward
                self.goals_collected += 1
                self.grid[new_pos] = 1
                if new_pos in self.goal_positions:
                    self.goal_positions.remove(new_pos)
            elif cell_value == 3:
                reward += self.hazard_reward
                self.grid[new_pos] = 1
            else:
                self.grid[new_pos] = 1
        
        done = False
        info = {}
        
        if self.goals_collected >= self.num_goals:
            done = True
            info['termination_reason'] = 'goal_reached'
        elif self.step_count >= self.max_steps:
            done = True
            info['termination_reason'] = 'max_steps'
        
        info['goals_collected'] = self.goals_collected
        info['step_count'] = self.step_count
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        return self.grid.flatten()
    
    def render(self, mode: str = 'human'):
        if mode == 'human':
            print(f"\nStep: {self.step_count}, Goals collected: {self.goals_collected}")
            print("-" * (self.grid_size * 2 + 1))
            for i in range(self.grid_size):
                row = "|"
                for j in range(self.grid_size):
                    cell = self.grid[i, j]
                    if cell == 0:
                        row += " |"
                    elif cell == 1:
                        row += "A|"
                    elif cell == 2:
                        row += "G|"
                    elif cell == 3:
                        row += "H|"
                    elif cell == 4:
                        row += "#|"
                print(row)
                print("-" * (self.grid_size * 2 + 1))
    
    def get_state_info(self) -> Dict[str, Any]:
        return {
            'agent_pos': self.agent_pos,
            'goal_positions': self.goal_positions,
            'step_count': self.step_count,
            'goals_collected': self.goals_collected
        }

