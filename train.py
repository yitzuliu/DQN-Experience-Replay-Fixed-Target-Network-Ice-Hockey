# ===== DQN Training Loop Implementation =====
#
# This implementation follows the pseudocode provided:
# 1. Initialize replay memory D with capacity N
# 2. Initialize action-value network Q (θ₁) with random weights
# 3. Initialize target network Q_target (θ₂) ← θ₁
#
# 4. For each episode = 1 to M:
#     5. Initialize initial state S₁
#
#     6. For t = 1 to T:
#         7. With probability ε, select a random action Aₜ (exploration)
#         8. Otherwise, select Aₜ = argmaxₐ Q(Sₜ, a; θ₁) (exploitation)
#
#         9. Execute action Aₜ, observe reward Rₜ₊₁ and next state Sₜ₊₁
#
#         10. Store transition (Sₜ, Aₜ, Rₜ₊₁, Sₜ₊₁) into replay buffer D
#
#         11. Sample a random minibatch of transitions (Sⱼ, Aⱼ, Rⱼ₊₁, Sⱼ₊₁) from D
#
#         12. For each sample j in the minibatch:
#             If Sⱼ₊₁ is terminal:
#                 yⱼ ← Rⱼ₊₁
#             Else:
#                 yⱼ ← Rⱼ₊₁ + γ * maxₐ' Q_target(Sⱼ₊₁, a'; θ₂)
#
#         13. Perform gradient descent step to minimize:
#             L = (yⱼ - Q(Sⱼ, Aⱼ; θ₁))²
#
#         14. Every C steps:
#             Update target network: θ₂ ← θ₁
#
# Additionally, this implementation includes these optional enhancements:
# - [OPTIONAL] Pre-filling replay memory before training
# - [OPTIONAL] Statistics tracking and visualization
# - [OPTIONAL] Model saving and evaluation

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import env_wrappers
from dqn_agent import DQNAgent
import config

def train(agent, env, num_episodes=config.TRAINING_EPISODES, render=False):
    """
    Train a DQN agent in the given environment
    
    Args:
        agent: The DQN agent to train
        env: The environment to train in
        num_episodes: Number of episodes to train for
        render: Whether to render the environment during training
        
    Returns:
        dict: Training statistics
    """
    # [OPTIONAL] Initialize statistics tracking
    stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_losses': [],
        'eval_rewards': [],
        'epsilons': []
    }
    
    # [OPTIONAL] Create directory for saving models
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # [OPTIONAL] Time tracking
    start_time = time.time()
    
    # [OPTIONAL] Pre-fill replay memory with random experiences
    # 這部分不在原始偽代碼中，但可以提高訓練初期的穩定性
    print("Pre-filling replay memory with random experiences...")
    state, _ = env.reset()
    for _ in tqdm(range(config.LEARNING_STARTS)):
        # Take random action
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store experience
        agent.store_experience(state, action, reward, next_state, done)
        
        if done:
            state, _ = env.reset()
        else:
            state = next_state
    
    print(f"Starting training for {num_episodes} episodes...")
    
    # === Core DQN training loop based on the provided pseudocode ===
    
    # For each episode = 1 to M [Pseudocode step 4]
    for episode in range(num_episodes):
        # Initialize initial state S₁ [Pseudocode step 5]
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        episode_length = 0
        done = False
        
        # Episode loop [Pseudocode step 6]
        while not done:
            # With probability ε, select random action, otherwise select best action [Pseudocode steps 7-8]
            action = agent.select_action(state)
            
            # Execute action, observe reward and next state [Pseudocode step 9]
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition in replay memory [Pseudocode step 10]
            agent.store_experience(state, action, reward, next_state, done)
            
            # Learn from experience if sufficient samples exist
            # This handles steps 11-14 of the pseudocode
            if agent.memory.can_sample(agent.batch_size):
                loss = agent.learn()
                if loss is not None:
                    episode_loss.append(loss)
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # [OPTIONAL] Render if requested
            if render:
                env.render()
        
        # [OPTIONAL] Update epsilon after each episode
        agent.update_epsilon()
        
        # === End of core DQN training loop ===
        
        # [OPTIONAL] Record episode statistics
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_length)
        stats['epsilons'].append(agent.epsilon)
        if episode_loss:
            stats['episode_losses'].append(np.mean(episode_loss))
        else:
            stats['episode_losses'].append(0)
        
        # [OPTIONAL] Print progress
        avg_reward = np.mean(stats['episode_rewards'][-100:]) if len(stats['episode_rewards']) >= 100 else np.mean(stats['episode_rewards'])
        avg_loss = np.mean(stats['episode_losses'][-100:]) if len(stats['episode_losses']) >= 100 else np.mean(stats['episode_losses'])
        
        print(f"Episode {episode+1}/{num_episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Avg Reward (100): {avg_reward:.2f} | "
              f"Loss: {avg_loss:.4f} | "
              f"Epsilon: {agent.epsilon:.4f} | "
              f"Memory: {len(agent.memory)}")
        
        # [OPTIONAL] Periodically save model
        if (episode + 1) % config.SAVE_FREQUENCY == 0:
            save_path = os.path.join(models_dir, f"dqn_episode_{episode+1}.pth")
            agent.save_model(save_path)
            print(f"Model saved to {save_path}")
            
            # [OPTIONAL] Save training statistics
            with open(os.path.join(models_dir, "training_stats.pkl"), "wb") as f:
                pickle.dump(stats, f)
            
            # [OPTIONAL] Plot training progress
            plot_training_progress(stats, os.path.join(models_dir, "training_progress.png"))
            
        # [OPTIONAL] Periodically evaluate the model
        if (episode + 1) % config.EVAL_FREQUENCY == 0:
            print("\nEvaluating current policy...")
            eval_rewards = evaluate(agent, env, num_episodes=config.EVAL_EPISODES)
            stats['eval_rewards'].append((episode + 1, np.mean(eval_rewards)))
            print(f"Evaluation: Avg Reward over {config.EVAL_EPISODES} episodes: {np.mean(eval_rewards):.2f}\n")
    
    # [OPTIONAL] Save final model
    save_path = os.path.join(models_dir, "dqn_final.pth")
    agent.save_model(save_path)
    
    # [OPTIONAL] Final statistics
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Final average reward (100 episodes): {np.mean(stats['episode_rewards'][-100:]):.2f}")
    
    return stats

# [OPTIONAL] Evaluation function - 用於定期評估代理，不在原始偽代碼中
def evaluate(agent, env, num_episodes=5, render=False):
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
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action (no exploration during evaluation)
            action = agent.select_action(state, evaluate=True)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Render if requested
            if render:
                env.render()
        
        rewards.append(episode_reward)
        print(f"Evaluation Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f}")
    
    return rewards

# [OPTIONAL] Visualization function - 用於繪製訓練進度圖表，不在原始偽代碼中
def plot_training_progress(stats, save_path=None):
    """
    Plot training metrics
    
    Args:
        stats: Dictionary containing training statistics
        save_path: Path to save the plot, or None to display
    """
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(stats['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot smoothed rewards
    plt.subplot(2, 2, 2)
    window_size = min(100, len(stats['episode_rewards']))
    if window_size > 0:
        smoothed_rewards = np.convolve(stats['episode_rewards'], np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_rewards)
        plt.title(f'Smoothed Rewards (Window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Reward')
    
    # Plot losses
    plt.subplot(2, 2, 3)
    plt.plot(stats['episode_losses'])
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # Plot epsilon decay
    plt.subplot(2, 2, 4)
    plt.plot(stats['epsilons'])
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Create environment
    env = env_wrappers.make_env()
    print(f"Environment: {config.ENV_NAME}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create agent
    agent = DQNAgent(env)
    print(f"DQN Agent created on device: {agent.device}")
    
    # Train agent
    try:
        stats = train(agent, env)
        
        # [OPTIONAL] Plot final training progress
        plot_training_progress(stats)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Clean up
        env.close()
        print("Environment closed")
