"""
Resume Training Module for DQN

This module provides functionality to resume training from a previously saved
checkpoint. This is useful when training was interrupted or when you want to
continue training a model for additional episodes.

Usage:
    python resume.py --checkpoint path/to/checkpoint.pth [--output_dir new/output/dir]
    or
    python main.py resume --checkpoint path/to/checkpoint.pth
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


def create_replay_memory(memory_type, capacity, state_shape):
    """Create the appropriate replay memory instance based on configuration."""
    if memory_type.lower() == "list":
        return ListReplayMemory(capacity=capacity)
    elif memory_type.lower() == "array":
        return ArrayReplayMemory(capacity=capacity, state_shape=state_shape)
    else:  # Default to optimized implementation
        return OptimizedArrayReplayMemory(capacity=capacity, state_shape=state_shape)


def resume_training(checkpoint_path, output_dir=None):
    """
    Resume DQN training from a saved checkpoint.
    
    Args:
        checkpoint_path (str): Path to the saved model checkpoint
        output_dir (str, optional): Directory to save new outputs
            If None, will create a new timestamped directory
    
    Returns:
        tuple: (trained_agent, training_statistics)
    """
    # Setup device
    device = utils.get_device()
    print(f"Resuming training on device: {device}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None, None
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create environment to get state shape and action space
    env = env_wrappers.make_atari_env(render_mode=None)
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
                print(f"Loaded stats with {len(stats.get('episode_rewards', []))} episodes recorded")
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
        
        # Save a reference to the original checkpoint
        with open(os.path.join(output_dir, "resumed_from.txt"), "w") as f:
            f.write(f"Training resumed from: {os.path.abspath(checkpoint_path)}\n")
            f.write(f"at step {steps_done} with epsilon {epsilon:.4f}\n")
            f.write(f"on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Resume training
    print(f"Resuming training for {config.TRAINING_EPISODES} more episodes...")
    trained_agent, new_stats = train(
        device=device,
        output_dir=output_dir,
        agent=agent,  # Pass the loaded agent
        training_stats=stats,  # Pass loaded statistics
        enable_recovery=True
    )
    
    return trained_agent, new_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume DQN training from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, 
                      help="Path to the checkpoint file (.pth)")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Directory to save outputs (new timestamped dir if not specified)")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()
    
    # Execute resume training
    resume_training(args.checkpoint, args.output_dir)
