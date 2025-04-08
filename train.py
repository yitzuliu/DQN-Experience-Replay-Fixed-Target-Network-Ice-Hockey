"""
DQN Training Implementation

This module implements the core training loop for the Deep Q-Network (DQN) algorithm.
The implementation follows the DQN algorithm described in the paper "Human-level control
through deep reinforcement learning" by Mnih et al. (2015).

The training process consists of:
1. Environment setup and preprocessing
2. Agent initialization
3. Experience collection
4. Batch sampling and learning
5. Periodic model saving and statistics tracking

Each step is thoroughly documented to help beginners understand the DQN algorithm.
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from datetime import datetime

import env_wrappers
from dqn_agent import DQNAgent
from replay_memory import OptimizedArrayReplayMemory, ArrayReplayMemory, ListReplayMemory
import config
import utils


def create_replay_memory(memory_type, capacity, state_shape):
    """
    Create the appropriate replay memory based on configuration.

    Replay memory stores agent experiences (state, action, reward, next_state, done)
    for later training. Different implementations offer trade-offs between
    memory usage and computational efficiency.
    
    Args:
        memory_type (str): Type of memory implementation to use
        capacity (int): Maximum memory capacity
        state_shape (tuple): Shape of state observations
        
    Returns:
        Replay memory instance
    """
    if memory_type.lower() == "list":
        print(f"Using list-based replay memory with capacity {capacity}")
        return ListReplayMemory(capacity=capacity)
    elif memory_type.lower() == "array":
        print(f"Using array-based replay memory with capacity {capacity}")
        return ArrayReplayMemory(capacity=capacity, state_shape=state_shape)
    else:  # Default to optimized implementation
        print(f"Using optimized array-based replay memory with capacity {capacity}")
        return OptimizedArrayReplayMemory(capacity=capacity, state_shape=state_shape)


def train(device=None, render_training=False, output_dir=None):
    """
    Train a DQN agent on Atari Ice Hockey.
    
    This function implements the complete DQN training algorithm with careful
    consideration for GPU optimization and follows the pseudocode from the
    original DQN paper.
    
    Args:
        device (torch.device, optional): Device to use for training (auto-detected if None)
        render_training (bool): Whether to render training episodes (slower)
        output_dir (str, optional): Directory to save outputs (auto-generated if None)
        
    Returns:
        tuple: (trained agent, training statistics)
    """
    # ===== SETUP PHASE =====
    
    # 1. Setup device (CPU/GPU) - Optimize for GPU when available
    if device is None:
        device = utils.get_device()
    print(f"Training on device: {device}")
    
    # Enable CUDA optimization if available
    if device.type == 'cuda':
        # Benchmark mode can improve performance when input sizes don't change
        torch.backends.cudnn.benchmark = True
        # Deterministic mode for reproducibility (comment out for max speed)
        # torch.backends.cudnn.deterministic = True
        print(f"CUDA optimizations enabled with {torch.cuda.get_device_name()}")
        
        # Set memory allocation strategy for better GPU memory management
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory_stats'):
            print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    # 2. Setup result directories
    if output_dir is None:
        # Create timestamped directories
        directories = utils.create_directories()
        output_dir = directories["run"]
        model_dir = directories["models"]
        log_dir = directories["logs"]
        viz_dir = directories["viz"]
    else:
        # Use provided directory
        model_dir = os.path.join(output_dir, "models")
        log_dir = os.path.join(output_dir, "logs")
        viz_dir = os.path.join(output_dir, "visualizations")
        
        # Ensure directories exist
        for directory in [output_dir, model_dir, log_dir, viz_dir]:
            os.makedirs(directory, exist_ok=True)
    
    # 3. Save configuration for reproducibility
    config_file = os.path.join(output_dir, "config.txt")
    with open(config_file, "w") as f:
        for attr in dir(config):
            if not attr.startswith("__"):
                value = getattr(config, attr)
                if not callable(value):
                    f.write(f"{attr} = {value}\n")
    
    # 4. Create environment - matching PSEUDOCODE SETUP
    print("Creating environment...")
    render_mode = "human" if render_training else None
    env = env_wrappers.make_atari_env(
        render_mode=render_mode,
        clip_rewards=True,
        episode_life=True,
        force_training_mode=True,
        gpu_acceleration=device.type in ['cuda', 'mps']  # Use GPU for preprocessing when available
    )
    
    # Get environment info
    state_shape = env.observation_space.shape  # Should be (C, H, W) in PyTorch format
    n_actions = env.action_space.n
    print(f"Environment created: {config.ENV_NAME}")
    print(f"State shape: {state_shape}, Action space: {n_actions}")
    
    # 5. Initialize replay memory - PSEUDOCODE LINE 1
    print("Initializing replay memory...")
    memory = create_replay_memory(
        memory_type=config.MEMORY_IMPLEMENTATION,
        capacity=config.MEMORY_CAPACITY,
        state_shape=state_shape
    )
    
    # 6. Initialize agent - PSEUDOCODE LINES 2-3
    print("Initializing DQN agent...")
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        memory=memory,
        device=device
    )
    
    # 7. Initialize statistics tracking
    stats = {
        "episode_rewards": [],        # Total reward per episode
        "episode_lengths": [],        # Number of steps per episode
        "episode_losses": [],         # Average loss per episode
        "episode_q_values": [],       # Average Q-values per episode
        "epsilons": [],               # Epsilon values at the end of each episode
        "learning_rate": config.LEARNING_RATE,
        "start_time": time.time()     # Track training duration
    }
    
    # ===== TRAINING LOOP =====
    
    # 8. Training loop - PSEUDOCODE LINE 4: For each episode = 1 to M
    print(f"Starting training for {config.TRAINING_EPISODES} episodes...")
    print(f"Will start learning after {config.LEARNING_STARTS} steps")
    print(f"Target network will update every {config.TARGET_UPDATE_FREQUENCY} steps")
    step_count = 0
    best_reward = float("-inf")
    total_time_start = time.time()
    
    # Use tqdm for progress bar
    episode_iterator = tqdm(range(1, config.TRAINING_EPISODES + 1), desc="Training Progress")
    
    for episode in episode_iterator:
        episode_start_time = time.time()
        episode_reward = 0
        episode_length = 0
        episode_loss = []
        
        # Reset environment - PSEUDOCODE LINE 5: Initialize initial state S₁
        state, _ = env.reset()
        
        # Episode loop - PSEUDOCODE LINE 6: For t = 1 to T
        done = False
        while not done:
            # Select action using epsilon-greedy - PSEUDOCODE LINES 7-8
            action = agent.select_action(state)
            
            # Execute action in environment - PSEUDOCODE LINE 9
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition in replay memory - PSEUDOCODE LINE 10
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            step_count += 1
            
            # Learn from experiences when memory has enough samples - PSEUDOCODE LINES 11-13
            if step_count > config.LEARNING_STARTS and step_count % config.UPDATE_FREQUENCY == 0:
                loss = agent.learn()
                if loss is not None:
                    episode_loss.append(loss)
            
            # Update target network periodically - PSEUDOCODE LINE 14
            if step_count % config.TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_network()
                print(f"Episode {episode}: Updated target network at step {step_count}")
            
            # Move to next state
            state = next_state
        
        # End of episode processing
        episode_duration = time.time() - episode_start_time
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        
        # Track statistics
        stats["episode_rewards"].append(episode_reward)
        stats["episode_lengths"].append(episode_length)
        stats["episode_losses"].append(avg_loss)
        stats["epsilons"].append(agent.epsilon)
        
        # Get Q-values (last 1000 only to save memory)
        if agent.avg_q_values:
            avg_q = np.mean(agent.avg_q_values[-1000:])
            stats["episode_q_values"].append(avg_q)
        
        # Update progress bar description
        episode_iterator.set_description(
            f"Ep {episode}: Reward={episode_reward:.1f}, Loss={avg_loss:.4f}, ε={agent.epsilon:.2f}"
        )
        
        # Print more detailed progress occasionally
        if episode % 10 == 0:
            print(f"\nEpisode {episode}/{config.TRAINING_EPISODES} - "
                  f"Reward: {episode_reward:.2f}, Length: {episode_length}, "
                  f"Loss: {avg_loss:.6f}, Epsilon: {agent.epsilon:.4f}, "
                  f"Time: {episode_duration:.2f}s")
            print(f"Memory size: {len(agent.memory)}/{config.MEMORY_CAPACITY}, Learning threshold: {config.LEARNING_STARTS}")
            print(f"Steps done: {agent.steps_done}, Updates performed: {max(0, (agent.steps_done - config.LEARNING_STARTS) // config.UPDATE_FREQUENCY)}")
            
            # Display recent losses if available
            if len(agent.losses) > 0:
                recent_losses = agent.losses[-10:]  # Last 10 loss values
                print(f"Recent losses: {[f'{l:.6f}' for l in recent_losses]}")
            
            # Display GPU memory usage if available
            if device.type == 'cuda' and hasattr(torch.cuda, 'memory_allocated'):
                allocated = torch.cuda.memory_allocated() / 1e6
                reserved = torch.cuda.memory_reserved() / 1e6
                print(f"GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(os.path.join(model_dir, "best_model.pth"))
            print(f"Saved new best model with reward {best_reward:.2f}")
        
        # Periodically save checkpoint and visualize progress
        if episode % config.SAVE_FREQUENCY == 0:
            # Save model checkpoint
            agent.save_model(os.path.join(model_dir, f"model_ep{episode}.pth"))
            
            # Save statistics (using pickle for efficiency)
            stats_file = os.path.join(log_dir, "training_stats.pkl")
            with open(stats_file, "wb") as f:
                pickle.dump(stats, f)
            
            # Update plots
            if len(stats["episode_rewards"]) > 0:
                # Plot rewards
                utils.plot_learning_curve(
                    values=stats["episode_rewards"],
                    window_size=100,
                    title="Episode Rewards",
                    xlabel="Episode",
                    ylabel="Reward",
                    save_path=os.path.join(viz_dir, "rewards.png")
                )
                
                # Plot losses if any
                if any(stats["episode_losses"]):
                    utils.plot_learning_curve(
                        values=stats["episode_losses"],
                        window_size=100,
                        title="Training Loss",
                        xlabel="Episode",
                        ylabel="Loss",
                        save_path=os.path.join(viz_dir, "losses.png")
                    )
    
    # ===== END OF TRAINING =====
    
    # Calculate and report training metrics
    total_time = time.time() - total_time_start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Save final model
    final_model_path = os.path.join(model_dir, "final_model.pth")
    agent.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save final statistics
    final_stats_path = os.path.join(log_dir, "final_stats.pkl")
    with open(final_stats_path, "wb") as f:
        pickle.dump(stats, f)
    
    # Generate final visualizations
    utils.plot_episode_stats(stats, save_dir=viz_dir, show=False)
    
    # Clean up resources
    env.close()
    
    # Free GPU memory if available
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("GPU memory cleared")
    
    print(f"Training completed. To evaluate the trained models, use:")
    print(f"  python evaluate.py evaluate {final_model_path} --render")
    print(f"  python evaluate.py evaluate {os.path.join(model_dir, 'best_model.pth')} --render")
    
    return agent, stats


if __name__ == "__main__":
    # This setup allows running the training module directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DQN agent on Atari Ice Hockey")
    parser.add_argument("--render", action="store_true", help="Render training episodes")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for outputs")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--episodes", type=int, default=config.TRAINING_EPISODES, help="Number of episodes to train")
    parser.add_argument("--learning_starts", type=int, default=config.LEARNING_STARTS, 
                       help="Steps before starting learning")
    args = parser.parse_args()
    
    # Override config if specified through command line
    if args.episodes != config.TRAINING_EPISODES:
        print(f"Overriding training episodes from {config.TRAINING_EPISODES} to {args.episodes}")
        config.TRAINING_EPISODES = args.episodes
        
    if args.learning_starts != config.LEARNING_STARTS:
        print(f"Overriding learning start threshold from {config.LEARNING_STARTS} to {args.learning_starts}")
        config.LEARNING_STARTS = args.learning_starts
    
    # Select device based on arguments
    if args.gpu and args.cpu:
        print("Error: Cannot specify both --gpu and --cpu")
        exit(1)
    elif args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
        else:
            print("Warning: GPU requested but no compatible GPU found, using CPU instead")
            device = torch.device("cpu")
    elif args.cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as requested")
    else:
        device = utils.get_device()  # Auto-detect
    
    # Print training configuration summary
    print("\n===== TRAINING CONFIGURATION =====")
    print(f"Environment: {config.ENV_NAME}")
    print(f"Episodes: {config.TRAINING_EPISODES}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Learning starts after: {config.LEARNING_STARTS} steps")
    print(f"Epsilon: {config.EPSILON_START} → {config.EPSILON_END} over {config.EPSILON_DECAY} steps")
    print(f"Target update frequency: Every {config.TARGET_UPDATE_FREQUENCY} steps")
    print(f"Memory capacity: {config.MEMORY_CAPACITY//1000}K transitions")
    print(f"Network architecture: {1 if config.USE_ONE_CONV_LAYER else (2 if config.USE_TWO_CONV_LAYERS else 3)} conv layers")
    print(f"Device: {device}")
    print("=================================\n")
    
    # Run training
    try:
        trained_agent, training_stats = train(
            device=device,
            render_training=args.render,
            output_dir=args.output_dir
        )
        
        print("Training complete!")
        
        # Display training summary
        final_reward = training_stats["episode_rewards"][-1]
        best_reward = max(training_stats["episode_rewards"])
        total_steps = sum(training_stats["episode_lengths"])
        
        print(f"\nTraining Summary:")
        print(f"Total episodes: {len(training_stats['episode_rewards'])}")
        print(f"Total steps: {total_steps:,}")
        print(f"Final epsilon: {trained_agent.epsilon:.4f}")
        print(f"Best reward: {best_reward:.2f}")
        print(f"Final reward: {final_reward:.2f}")
        
        # Optional: Show final training curves
        if input("Show training curves? (y/n): ").lower() == 'y':
            utils.plot_episode_stats(training_stats, show=True)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
        # Save current model on interrupt
        if 'agent' in locals() and 'model_dir' in locals():
            interrupted_path = os.path.join(model_dir, "interrupted_model.pth")
            agent.save_model(interrupted_path)
            print(f"Interrupted model saved to {interrupted_path}")
