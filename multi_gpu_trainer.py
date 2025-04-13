"""
Multi-GPU Training Module for DQN

This module provides functionality to train DQN models using multiple GPUs.
It leverages PyTorch's DataParallel for training efficiency.
This implementation aims to minimize changes to the existing codebase.

Usage:
    python multi_gpu_trainer.py --num_gpus 2 --batch_size 512 --output_dir results/multi_gpu_run
"""

import os
import sys
import argparse
import torch
import time
from datetime import datetime

# Import project modules
import config
from train import train
from dqn_agent import DQNAgent
from q_network import create_q_network
import utils
from replay_memory import OptimizedArrayReplayMemory


class MultiGPUTrainer:
    """
    MultiGPUTrainer manages training DQN models across multiple GPUs.
    
    This class uses DataParallel: Simple approach where batches are split across GPUs
    """
    
    def __init__(self, num_gpus=None):
        """
        Initialize the multi-GPU trainer.
        
        Args:
            num_gpus (int): Number of GPUs to use. If None, use all available GPUs.
        """
        # Check GPU availability
        if not torch.cuda.is_available():
            print("CUDA is not available. Multi-GPU training not possible.")
            print("Falling back to CPU training.")
            self.num_gpus = 0
            self.device = torch.device("cpu")
            return
            
        total_available_gpus = torch.cuda.device_count()
        
        # Set number of GPUs based on user specification and availability
        if num_gpus is None:
            # Use all available GPUs
            self.num_gpus = total_available_gpus
        else:
            # Use the specified number of GPUs, but don't exceed available ones
            self.num_gpus = min(num_gpus, total_available_gpus)
            if num_gpus > total_available_gpus:
                print(f"Warning: Requested {num_gpus} GPUs but only {total_available_gpus} available.")
                print(f"Using {self.num_gpus} GPUs.")
        
        # Set device to the first GPU
        self.device = torch.device("cuda:0")
        
        print(f"Setting up training with {self.num_gpus} GPUs")
        
        # Print GPU information for all GPUs we're using
        for i in range(self.num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name}, Memory: {gpu_mem:.2f} GB")
    
    def create_parallel_agent(self, state_shape, n_actions, memory):
        """
        Create a DQN agent with networks distributed across multiple GPUs.
        
        Args:
            state_shape (tuple): Shape of state observations
            n_actions (int): Number of possible actions
            memory: Experience replay memory instance
            
        Returns:
            DQNAgent: Agent with GPU-parallelized networks
        """
        # First create a normal agent
        agent = DQNAgent(
            state_shape=state_shape,
            n_actions=n_actions,
            memory=memory,
            device=self.device
        )
        
        if self.num_gpus <= 1:
            # No parallelization needed
            return agent
        
        # Create device IDs list for DataParallel
        device_ids = list(range(self.num_gpus))
        
        # Make sure the model is on the correct device before wrapping with DataParallel
        agent.q_network = agent.q_network.to(self.device)
        agent.target_network = agent.target_network.to(self.device)
        
        # Parallelize the networks using DataParallel with explicit device_ids
        agent.q_network = torch.nn.DataParallel(agent.q_network, device_ids=device_ids)
        agent.target_network = torch.nn.DataParallel(agent.target_network, device_ids=device_ids)
        
        return agent
    
    def parallel_train(self, output_dir=None, batch_size=None):
        """
        Train the DQN model using multiple GPUs.
        
        Args:
            output_dir (str): Directory to save outputs
            batch_size (int): Batch size for training (will be adjusted per GPU)
            
        Returns:
            tuple: (trained_agent, training_stats)
        """
        # Adjust batch size based on number of GPUs
        if batch_size is None:
            # Use batch size from config, adjusted for number of GPUs
            effective_batch_size = config.BATCH_SIZE
            if self.num_gpus > 1:
                # Make sure batch size is divisible by number of GPUs
                effective_batch_size = (config.BATCH_SIZE // self.num_gpus) * self.num_gpus
                if effective_batch_size != config.BATCH_SIZE:
                    print(f"Adjusting batch size from {config.BATCH_SIZE} to {effective_batch_size} "
                          f"to be divisible by {self.num_gpus} GPUs")
                    config.BATCH_SIZE = effective_batch_size
        else:
            # Set batch size to specified value, adjusted for number of GPUs
            effective_batch_size = (batch_size // self.num_gpus) * self.num_gpus
            if effective_batch_size != batch_size and self.num_gpus > 1:
                print(f"Adjusting batch size from {batch_size} to {effective_batch_size} "
                      f"to be divisible by {self.num_gpus} GPUs")
            config.BATCH_SIZE = effective_batch_size
        
        # Create a timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # If no output directory specified, create one based on timestamp
        if output_dir is None:
            output_dir = os.path.join("results", f"multi_gpu_{self.num_gpus}_{timestamp}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save multi-GPU configuration
        with open(os.path.join(output_dir, "multi_gpu_config.txt"), "w") as f:
            f.write(f"Number of GPUs: {self.num_gpus}\n")
            f.write(f"Batch size: {config.BATCH_SIZE}\n")
            f.write(f"GPU devices:\n")
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                f.write(f"  GPU {i}: {gpu_name}, Memory: {gpu_mem:.2f} GB\n")
        
        # Start DataParallel training
        print(f"Starting DataParallel training with {self.num_gpus} GPUs")
        
        # Call the regular training function
        # The agent will be created with DataParallel in train()
        return train(
            device=self.device,
            render_training=False,
            output_dir=output_dir,
            enable_recovery=True,
            multi_gpu_trainer=self  # Pass self to train() for agent creation
        )


def train_with_multiple_gpus(num_gpus=None, batch_size=None, output_dir=None):
    """
    High-level function to train DQN with multiple GPUs.
    
    Args:
        num_gpus (int): Number of GPUs to use
        batch_size (int): Batch size for training
        output_dir (str): Directory to save outputs
        
    Returns:
        tuple: (trained_agent, training_stats)
    """
    # Create the multi-GPU trainer
    trainer = MultiGPUTrainer(num_gpus=num_gpus)
    
    # Start training
    return trainer.parallel_train(output_dir=output_dir, batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN with multiple GPUs")
    parser.add_argument("--num_gpus", type=int, default=None, 
                        help="Number of GPUs to use (default: all available)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (will be adjusted to be divisible by number of GPUs)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of episodes to train (default: from config)")
    args = parser.parse_args()
    
    # Update config if episodes is specified
    if args.episodes is not None:
        config.TRAINING_EPISODES = args.episodes
    
    # Start time
    start_time = time.time()
    
    # Start multi-GPU training
    agent, stats = train_with_multiple_gpus(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    # Training duration
    duration = time.time() - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Report final performance
    if stats is not None:
        print("\nTraining Summary:")
        print(f"Total episodes: {len(stats['episode_rewards'])}")
        print(f"Best reward: {max(stats['episode_rewards']):.2f}")
        print(f"Final reward: {stats['episode_rewards'][-1]:.2f}")
