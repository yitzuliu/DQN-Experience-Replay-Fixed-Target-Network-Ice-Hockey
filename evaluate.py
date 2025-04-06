import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

import env_wrappers
import config
from dqn_agent import DQNAgent

def evaluate(agent, env, num_episodes=config.EVAL_EPISODES, render=False):
    """
    Evaluate a trained agent over several episodes
    
    Args:
        agent: The trained DQN agent
        env: The environment to evaluate in
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment during evaluation
        
    Returns:
        list: Rewards for each evaluation episode
    """
    rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Select action (no exploration during evaluation)
            action = agent.select_action(state, evaluate=True)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update state and statistics
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Render if requested
            if render:
                env.render()
                time.sleep(0.01)  # Small delay to make rendering visible
        
        rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"Evaluation Episode {episode+1}/{num_episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Steps: {steps}")
    
    # Print summary statistics
    print("\n===== Evaluation Summary =====")
    print(f"Episodes: {num_episodes}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f}")
    print("==============================\n")
    
    return rewards

def visualize_episode(agent, env, max_steps=1000):
    """
    Run a single episode with visualization
    
    Args:
        agent: The trained DQN agent
        env: The environment with rendering enabled
        max_steps: Maximum steps to run
    """
    # Make sure env has render_mode set to "human"
    if env.render_mode != "human":
        env.close()
        env = env_wrappers.make_env(render_mode="human")
    
    state, _ = env.reset()
    episode_reward = 0
    steps = 0
    done = False
    
    print("Starting visualization... Press Ctrl+C to stop.")
    
    try:
        while not done and steps < max_steps:
            # Select action (no exploration during visualization)
            action = agent.select_action(state, evaluate=True)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update state and statistics
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Small delay to make it viewable
            time.sleep(0.01)
            
            # Print current step info
            print(f"\rStep: {steps} | Reward: {episode_reward:.2f}", end="")
    
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    
    print(f"\nEpisode finished after {steps} steps with reward {episode_reward:.2f}")
    
    return episode_reward

if __name__ == "__main__":
    # This will run if the script is executed directly
    import argparse
    import utils
    
    parser = argparse.ArgumentParser(description="Evaluate DQN agent")
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--episodes', type=int, default=config.EVAL_EPISODES, help='Number of episodes')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--visualize', action='store_true', help='Run single episode with visualization')
    
    args = parser.parse_args()
    
    # Create environment
    env = env_wrappers.make_env(render_mode="human" if args.render or args.visualize else None)
    
    # 使用utils中的設備設置函數
    device = utils.setup_device()
    agent = DQNAgent(env, device=device)
    
    # Load model
    if not agent.load_model(args.model):
        print(f"Failed to load model from {args.model}")
        exit(1)
    
    if args.visualize:
        # Run a single episode with visualization
        visualize_episode(agent, env)
    else:
        # Run evaluation
        evaluate(agent, env, num_episodes=args.episodes, render=args.render)
    
    env.close()

