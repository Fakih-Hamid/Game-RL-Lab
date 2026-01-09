import numpy as np
import torch
import argparse
import os

from env.grid_world import GridWorldEnv
from agents.ppo_agent import PPOAgent


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def select_action(self, state):
        return self.action_space.sample()


class HeuristicAgent:
    def __init__(self, grid_size, num_goals):
        self.grid_size = grid_size
        self.num_goals = num_goals
    
    def select_action(self, state):
        grid = state.reshape(self.grid_size, self.grid_size)
        
        agent_pos = np.where(grid == 1)
        if len(agent_pos[0]) == 0:
            return 4
        
        agent_pos = (agent_pos[0][0], agent_pos[1][0])
        
        goal_positions = np.where(grid == 2)
        if len(goal_positions[0]) == 0:
            return 4
        
        min_dist = float('inf')
        nearest_goal = None
        
        for i in range(len(goal_positions[0])):
            goal_pos = (goal_positions[0][i], goal_positions[1][i])
            dist = abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest_goal = goal_pos
        
        if nearest_goal is None:
            return 4
        
        dr = nearest_goal[0] - agent_pos[0]
        dc = nearest_goal[1] - agent_pos[1]
        
        if abs(dr) > abs(dc):
            if dr > 0:
                return 2
            else:
                return 0
        else:
            if dc > 0:
                return 1
            else:
                return 3


def evaluate_agent(env, agent, num_episodes=100, max_steps=200, deterministic=True):
    episode_rewards = []
    episode_lengths = []
    episode_goals = []
    success_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            if isinstance(agent, PPOAgent):
                action, _, _ = agent.select_action(state, deterministic=deterministic)
            elif isinstance(agent, (RandomAgent, HeuristicAgent)):
                action = agent.select_action(state)
            else:
                raise ValueError(f"Unknown agent type: {type(agent)}")
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        goals_collected = info.get('goals_collected', 0)
        episode_goals.append(goals_collected)
        
        if goals_collected >= env.num_goals:
            success_count += 1
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_goals': episode_goals,
        'success_rate': success_count / num_episodes,
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'avg_goals': np.mean(episode_goals)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO agent')
    
    parser.add_argument('--grid_size', type=int, default=10, help='Grid size')
    parser.add_argument('--num_goals', type=int, default=1, help='Number of goals')
    parser.add_argument('--num_hazards', type=int, default=3, help='Number of hazards')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/ppo_final.pth', help='Checkpoint path')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    env = GridWorldEnv(
        grid_size=args.grid_size,
        num_goals=args.num_goals,
        num_hazards=args.num_hazards
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    ppo_agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=args.device
    )
    
    if os.path.exists(args.checkpoint):
        ppo_agent.load(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")
        print("Evaluating untrained agent...")
    
    random_agent = RandomAgent(env.action_space)
    heuristic_agent = HeuristicAgent(args.grid_size, args.num_goals)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\n1. PPO Agent (Trained):")
    print("-" * 60)
    ppo_results = evaluate_agent(env, ppo_agent, args.num_episodes, deterministic=True)
    print(f"   Average Reward: {ppo_results['avg_reward']:.2f} ± {ppo_results['std_reward']:.2f}")
    print(f"   Average Length: {ppo_results['avg_length']:.1f}")
    print(f"   Average Goals: {ppo_results['avg_goals']:.2f}/{args.num_goals}")
    print(f"   Success Rate: {ppo_results['success_rate']*100:.1f}%")
    
    print("\n2. Random Agent (Baseline):")
    print("-" * 60)
    random_results = evaluate_agent(env, random_agent, args.num_episodes)
    print(f"   Average Reward: {random_results['avg_reward']:.2f} ± {random_results['std_reward']:.2f}")
    print(f"   Average Length: {random_results['avg_length']:.1f}")
    print(f"   Average Goals: {random_results['avg_goals']:.2f}/{args.num_goals}")
    print(f"   Success Rate: {random_results['success_rate']*100:.1f}%")
    
    print("\n3. Heuristic Agent (Baseline):")
    print("-" * 60)
    heuristic_results = evaluate_agent(env, heuristic_agent, args.num_episodes)
    print(f"   Average Reward: {heuristic_results['avg_reward']:.2f} ± {heuristic_results['std_reward']:.2f}")
    print(f"   Average Length: {heuristic_results['avg_length']:.1f}")
    print(f"   Average Goals: {heuristic_results['avg_goals']:.2f}/{args.num_goals}")
    print(f"   Success Rate: {heuristic_results['success_rate']*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"PPO vs Random - Reward improvement: {ppo_results['avg_reward'] - random_results['avg_reward']:.2f}")
    print(f"PPO vs Random - Success rate improvement: {(ppo_results['success_rate'] - random_results['success_rate'])*100:.1f}%")
    print(f"PPO vs Heuristic - Reward improvement: {ppo_results['avg_reward'] - heuristic_results['avg_reward']:.2f}")
    print(f"PPO vs Heuristic - Success rate improvement: {(ppo_results['success_rate'] - heuristic_results['success_rate'])*100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()

