"""
Resume Training Module for DQN

This module provides functionality to resume training from a previously saved
checkpoint. This is useful when training was interrupted or when you want to
continue training a model for additional episodes.

Usage:
    python resume.py --checkpoint path/to/checkpoint.pth [--options]
"""

import os
import torch
import argparse
import json
import time

import config
import utils
from train import train
from dqn_agent import DQNAgent
from replay_memory import OptimizedArrayReplayMemory, ArrayReplayMemory, ListReplayMemory
import env_wrappers
from logger import Logger  # Import the new minimal logger


def create_replay_memory(memory_type, capacity, state_shape):
    """
    Create the appropriate replay memory based on configuration.
    
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


def resume_training(checkpoint_path, output_dir=None, episodes=None, render_training=False, device=None, enable_recovery=True):
    """
    Resume DQN training from a saved checkpoint.
    
    Args:
        checkpoint_path (str): Path to the saved model checkpoint
        output_dir (str, optional): Directory to save new outputs
            If None, will create a new timestamped directory
        episodes (int, optional): Number of episodes to train for, overrides config value
        render_training (bool): Whether to render training episodes (slower)
        device (torch.device, optional): Device to use for training (auto-detected if None)
        enable_recovery (bool): Enable automatic error recovery and checkpointing
    
    Returns:
        tuple: (trained_agent, training_statistics)
    """
    # Setup device
    if device is None:
        device = utils.get_device()
    print(f"Resuming training on device: {device}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None, None
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create environment to get state shape and action space
    render_mode = "human" if render_training else None
    env = env_wrappers.make_atari_env(render_mode=render_mode)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    env.close()  # Close the temporary environment
    
    # Create memory with same parameters as in training
    memory = create_replay_memory(
        memory_type=config.MEMORY_IMPLEMENTATION,
        capacity=config.MEMORY_CAPACITY,
        state_shape=state_shape
    )
    
    # Create agent
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        memory=memory,
        device=device
    )
    
    # Load agent state
    success = agent.load_model(checkpoint_path)
    if not success:
        print(f"Error loading agent state from checkpoint")
        return None, None
    
    # Retrieve training progress information
    steps_done = agent.steps_done
    epsilon = agent.epsilon
    
    print(f"Successfully loaded checkpoint with:")
    print(f"- Training steps completed: {steps_done}")
    print(f"- Current epsilon: {epsilon:.4f}")
    
    # Try to load additional training stats if available
    stats = None
    start_episode = None
    checkpoint_dir = os.path.dirname(checkpoint_path)
    possible_stats_files = [
        os.path.join(checkpoint_dir, "..", "logs", "training_stats.pkl"),
        os.path.join(checkpoint_dir, "..", "logs", "final_stats.pkl")
    ]
    
    for stats_file in possible_stats_files:
        if os.path.exists(stats_file):
            print(f"Found training statistics at {stats_file}")
            stats = utils.load_object(stats_file)
            if stats:
                episodes_completed = len(stats.get('episode_rewards', []))
                print(f"Loaded stats with {episodes_completed} episodes recorded")
                start_episode = episodes_completed + 1
                break
    
    if stats is None:
        print("No previous training statistics found, starting with empty statistics")
        stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_losses": [],
            "episode_q_values": [],
            "epsilons": [],
            "learning_rate": config.LEARNING_RATE,
            "start_time": time.time()
        }
    
    # Setup output directory for resumed training
    if output_dir is None:
        # Create new timestamped output directory
        directories = utils.create_directories()
        output_dir = directories["run"]
        print(f"Will save resumed training outputs to: {output_dir}")
        model_dir = os.path.join(output_dir, "models")
        log_dir = os.path.join(output_dir, "logs")
        
        # Ensure directories exist
        for directory in [model_dir, log_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Save a reference to the original checkpoint
        with open(os.path.join(output_dir, "resumed_from.txt"), "w") as f:
            f.write(f"Training resumed from: {os.path.abspath(checkpoint_path)}\n")
            f.write(f"at step {steps_done} with epsilon {epsilon:.4f}\n")
            f.write(f"on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Override number of episodes if specified
    original_episodes = config.TRAINING_EPISODES
    if episodes is not None:
        print(f"Overriding training episodes from {config.TRAINING_EPISODES} to {episodes}")
        config.TRAINING_EPISODES = episodes
    
    # Resume training
    if start_episode:
        print(f"Resuming training from episode {start_episode} for {config.TRAINING_EPISODES} total episodes...")
    else:
        print(f"Resuming training for {config.TRAINING_EPISODES} more episodes...")
    
    # Call the train function with the loaded agent and stats
    trained_agent, new_stats = train(
        device=device,
        render_training=render_training,
        output_dir=output_dir,
        agent=agent,  # Pass the loaded agent
        training_stats=stats,  # Pass loaded statistics
        enable_recovery=enable_recovery,
        start_episode=start_episode  # Pass the starting episode number
    )
    
    # Restore original episodes setting
    if episodes is not None:
        config.TRAINING_EPISODES = original_episodes
    
    return trained_agent, new_stats


if __name__ == "__main__":
    # Setup argument parser for command-line usage
    parser = argparse.ArgumentParser(description="Resume DQN training from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, 
                      help="Path to the checkpoint file (.pth)")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Directory to save outputs (new timestamped dir if not specified)")
    parser.add_argument("--episodes", type=int, default=None,
                      help="Number of episodes to train for (default: use config value)")
    parser.add_argument("--render", action="store_true", 
                      help="Render training episodes (slower)")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--enable_recovery", action="store_true",
                      help="Enable automatic recovery checkpoints")
    args = parser.parse_args()
    
    # Determine device (CPU/GPU)
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
    
    # Execute resume training
    resume_training(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        episodes=args.episodes,
        render_training=args.render,
        device=device,
        enable_recovery=args.enable_recovery
    )
