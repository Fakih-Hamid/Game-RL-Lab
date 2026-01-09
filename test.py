import numpy as np
import torch

from env.grid_world import GridWorldEnv
from agents.ppo_agent import PPOAgent

def test_environment():
    print("Testing environment...")
    env = GridWorldEnv(grid_size=5, num_goals=1, num_hazards=2)
    
    state = env.reset()
    assert state.shape == (25,), f"Expected shape (25,), got {state.shape}"
    print("[OK] Environment reset works")
    
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    assert next_state.shape == (25,), f"Expected shape (25,), got {next_state.shape}"
    print("[OK] Environment step works")
    
    print("Environment test passed!\n")

def test_agent():
    print("Testing PPO agent...")
    state_dim = 25
    action_dim = 5
    
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    print("[OK] Agent initialization works")
    
    state = np.random.randint(0, 5, size=(state_dim,))
    action, log_prob, value = agent.select_action(state)
    assert 0 <= action < action_dim, f"Action {action} out of range"
    print("[OK] Action selection works")
    
    agent.store_transition(state, action, 1.0, log_prob, value, False)
    print("[OK] Transition storage works")
    
    print("Agent test passed!\n")

def test_training_step():
    print("Testing training step...")
    env = GridWorldEnv(grid_size=5, num_goals=1, num_hazards=2)
    agent = PPOAgent(state_dim=25, action_dim=5)
    
    state = env.reset()
    for _ in range(10):
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, log_prob, value, done)
        state = next_state
        if done:
            break
    
    if len(agent.states) > 0:
        agent.update(epochs=1, batch_size=5)
        print("[OK] Training update works")
    
    print("Training step test passed!\n")

if __name__ == '__main__':
    print("=" * 60)
    print("Running tests...")
    print("=" * 60 + "\n")
    
    try:
        test_environment()
        test_agent()
        test_training_step()
        
        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()

