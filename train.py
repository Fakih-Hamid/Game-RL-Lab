import numpy as np
import torch
import argparse
import os
from collections import deque
import time

from env.grid_world import GridWorldEnv
from agents.ppo_agent import PPOAgent


def train(
    env: GridWorldEnv,
    agent: PPOAgent,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 200,
    update_frequency: int = 2048,
    save_frequency: int = 100,
    save_dir: str = './checkpoints',
    log_frequency: int = 10
):
    os.makedirs(save_dir, exist_ok=True)
    
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_goals = deque(maxlen=100)
    
    total_steps = 0
    episode_count = 0
    
    recent_rewards = []
    recent_lengths = []
    recent_goals = []
    
    print("Starting training...")
    print(f"Environment: {env.grid_size}x{env.grid_size} grid")
    print(f"Goals: {env.num_goals}, Hazards: {env.num_hazards}")
    print(f"Update frequency: {update_frequency} steps")
    print("-" * 60)
    
    start_time = time.time()
    
    while episode_count < num_episodes:
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps_per_episode:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, log_prob, value, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            if total_steps % update_frequency == 0 and len(agent.states) > 0:
                if not done:
                    _, _, next_value = agent.select_action(next_state, deterministic=True)
                else:
                    next_value = 0.0
                
                agent.update(epochs=10, batch_size=64)
                print(f"Updated agent at step {total_steps}")
        
        episode_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_goals.append(info.get('goals_collected', 0))
        
        recent_rewards.append(episode_reward)
        recent_lengths.append(episode_length)
        recent_goals.append(info.get('goals_collected', 0))
        
        if episode_count % log_frequency == 0:
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            avg_goals = np.mean(recent_goals)
            success_rate = np.mean([g >= env.num_goals for g in recent_goals])
            
            elapsed_time = time.time() - start_time
            steps_per_sec = total_steps / elapsed_time if elapsed_time > 0 else 0
            
            print(f"Episode {episode_count}/{num_episodes}")
            print(f"  Avg Reward (last {log_frequency}): {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Avg Goals Collected: {avg_goals:.1f}/{env.num_goals}")
            print(f"  Success Rate: {success_rate*100:.1f}%")
            print(f"  Total Steps: {total_steps}")
            print(f"  Steps/sec: {steps_per_sec:.1f}")
            print("-" * 60)
            
            recent_rewards = []
            recent_lengths = []
            recent_goals = []
        
        if episode_count % save_frequency == 0:
            checkpoint_path = os.path.join(save_dir, f'ppo_checkpoint_ep{episode_count}.pth')
            agent.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    if len(agent.states) > 0:
        agent.update(epochs=10, batch_size=64)
    
    final_path = os.path.join(save_dir, 'ppo_final.pth')
    agent.save(final_path)
    print(f"\nTraining completed!")
    print(f"Final model saved: {final_path}")
    print(f"Total episodes: {episode_count}")
    print(f"Total steps: {total_steps}")
    
    if len(episode_rewards) > 0:
        print(f"\nFinal Statistics (last 100 episodes):")
        print(f"  Average Reward: {np.mean(episode_rewards):.2f}")
        print(f"  Average Length: {np.mean(episode_lengths):.1f}")
        print(f"  Average Goals: {np.mean(episode_goals):.1f}/{env.num_goals}")
        print(f"  Success Rate: {np.mean([g >= env.num_goals for g in episode_goals])*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent on GridWorld')
    
    parser.add_argument('--grid_size', type=int, default=10, help='Grid size')
    parser.add_argument('--num_goals', type=int, default=1, help='Number of goals')
    parser.add_argument('--num_hazards', type=int, default=3, help='Number of hazards')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=200, help='Max steps per episode')
    parser.add_argument('--update_frequency', type=int, default=2048, help='Update frequency in steps')
    parser.add_argument('--lr_policy', type=float, default=3e-4, help='Policy learning rate')
    parser.add_argument('--lr_value', type=float, default=3e-4, help='Value learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    env = GridWorldEnv(
        grid_size=args.grid_size,
        num_goals=args.num_goals,
        num_hazards=args.num_hazards
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        gamma=args.gamma,
        clip_epsilon=args.clip_epsilon,
        device=args.device
    )
    
    train(
        env=env,
        agent=agent,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        update_frequency=args.update_frequency,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()

