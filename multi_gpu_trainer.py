"""
Multi-GPU Training Coordinator for DQN

This module implements parallel training across multiple GPUs using data parallelism.
It coordinates the distribution of environments and models across available GPUs
and aggregates results for more efficient training.
"""

import os
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import threading
import queue
from collections import deque

import config
from dqn_agent import DQNAgent
import env_wrappers
from replay_memory import OptimizedArrayReplayMemory
import utils


class ExperienceCollector(threading.Thread):
    """
    Thread that collects experiences using multiple environments.
    """
    def __init__(self, gpu_id, memory_queue, state_shape, n_actions, num_envs=8, stop_event=None):
        """
        Initialize experience collector thread.
        
        Args:
            gpu_id (int): GPU ID to use for this collector
            memory_queue (queue.Queue): Queue to put collected experiences
            state_shape (tuple): Shape of state observations
            n_actions (int): Number of possible actions
            num_envs (int): Number of parallel environments to run
            stop_event (threading.Event): Event to signal thread to stop
        """
        super(ExperienceCollector, self).__init__()
        self.daemon = True
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        self.memory_queue = memory_queue
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.num_envs = num_envs
        self.stop_event = stop_event or threading.Event()
        
        # Create environments
        self.envs = []
        for _ in range(num_envs):
            env = env_wrappers.make_atari_env(
                render_mode=None,
                clip_rewards=True,
                episode_life=True,
                force_training_mode=True,
                gpu_acceleration=(self.device.type == 'cuda')
            )
            self.envs.append(env)
        
        # Create local memory for experiences
        self.local_batch = []
        self.batch_size = config.BATCH_SIZE // 4  # Split across GPUs
        
        # Agent for action selection (shares weights with main agent)
        self.memory = OptimizedArrayReplayMemory(
            capacity=1000,  # Small capacity, just for action selection
            state_shape=state_shape
        )
        self.agent = DQNAgent(
            state_shape=state_shape,
            n_actions=n_actions,
            memory=self.memory,
            device=self.device
        )
        
        # Track episodes and statistics
        self.episode_rewards = [0] * num_envs
        self.episode_lengths = [0] * num_envs
        self.completed_episodes = 0  # Track completed episodes counter
        self.states = [None] * num_envs
        
        # Reset all environments
        for i in range(num_envs):
            self.states[i], _ = self.envs[i].reset()
    
    def update_agent(self, agent_state_dict):
        """Update local agent with main agent weights."""
        self.agent.q_network.load_state_dict(agent_state_dict)
    
    def run(self):
        """Collect experiences from environments in parallel."""
        print(f"Experience collector started on GPU {self.gpu_id}")
        
        while not self.stop_event.is_set():
            # Collect experiences from all environments
            for env_idx in range(self.num_envs):
                # Select action
                action = self.agent.select_action(self.states[env_idx])
                
                # Execute action in environment
                next_state, reward, terminated, truncated, _ = self.envs[env_idx].step(action)
                done = terminated or truncated
                
                # Store experience
                self.local_batch.append((
                    self.states[env_idx], action, reward, next_state, done
                ))
                
                # Update statistics
                self.episode_rewards[env_idx] += reward
                self.episode_lengths[env_idx] += 1
                
                # Handle episode end
                if done:
                    # Record episode stats
                    reward = self.episode_rewards[env_idx]
                    length = self.episode_lengths[env_idx]
                    
                    # Increment completed episodes counter
                    self.completed_episodes += 1
                    
                    # Reset episode stats
                    self.episode_rewards[env_idx] = 0
                    self.episode_lengths[env_idx] = 0
                    
                    # Reset environment
                    self.states[env_idx], _ = self.envs[env_idx].reset()
                else:
                    # Update state
                    self.states[env_idx] = next_state
            
            # Send experiences to main process when batch is full
            if len(self.local_batch) >= self.batch_size:
                self.memory_queue.put(self.local_batch[:self.batch_size])
                self.local_batch = self.local_batch[self.batch_size:]
        
        # Clean up environments
        for env in self.envs:
            env.close()
        
        print(f"Experience collector on GPU {self.gpu_id} stopped")


class LearnerThread(threading.Thread):
    """
    Thread that performs learning using experiences from the queue.
    """
    def __init__(self, gpu_id, agent, memory_queue, param_queue, batch_size, stop_event=None):
        """
        Initialize learner thread.
        
        Args:
            gpu_id (int): GPU ID to use for this learner
            agent (DQNAgent): Agent to perform learning
            memory_queue (queue.Queue): Queue to get experiences from
            param_queue (queue.Queue): Queue to put updated parameters
            batch_size (int): Batch size for learning
            stop_event (threading.Event): Event to signal thread to stop
        """
        super(LearnerThread, self).__init__()
        self.daemon = True
        self.gpu_id = gpu_id
        self.agent = agent
        self.memory_queue = memory_queue
        self.param_queue = param_queue
        self.batch_size = batch_size
        self.stop_event = stop_event or threading.Event()
        
        # Statistics
        self.updates = 0
        self.losses = []
        
    def run(self):
        """Perform learning using experiences from the queue."""
        print(f"Learner started on GPU {self.gpu_id}")
        
        while not self.stop_event.is_set():
            try:
                # Get experiences from queue with timeout
                experiences = self.memory_queue.get(timeout=1.0)
                
                # Store experiences in agent memory
                for state, action, reward, next_state, done in experiences:
                    self.agent.store_transition(state, action, reward, next_state, done)
                
                # Learn from experiences if enough samples
                if self.agent.memory.can_sample(self.batch_size):
                    loss = self.agent.learn()
                    if loss is not None:
                        self.losses.append(loss)
                        self.updates += 1
                    
                    # Periodically put model parameters in queue
                    if self.updates % 10 == 0:
                        self.param_queue.put((
                            self.gpu_id, 
                            self.agent.q_network.state_dict(),
                            self.agent.steps_done,
                            self.losses[-100:] if len(self.losses) > 0 else []
                        ))
            except queue.Empty:
                # Queue empty, continue waiting
                continue
                
        print(f"Learner on GPU {self.gpu_id} stopped")


class MultiGPUTrainer:
    """
    Coordinates training across multiple GPUs.
    """
    def __init__(self, state_shape, n_actions, num_gpus=None):
        """
        Initialize multi-GPU trainer.
        
        Args:
            state_shape (tuple): Shape of state observations
            n_actions (int): Number of possible actions
            num_gpus (int, optional): Number of GPUs to use, defaults to all available minus one
        """
        # Determine number of GPUs to use
        self.num_gpus = num_gpus or torch.cuda.device_count()
        if self.num_gpus < 1:
            raise ValueError("No GPUs available for multi-GPU training")
        available_gpus = torch.cuda.device_count()
        if num_gpus is None:
            self.num_gpus = max(1, available_gpus - 1)  # Ensure at least 1 GPU is used
        else:
            self.num_gpus = min(num_gpus, available_gpus)  # Don't exceed available GPUs
        print(f"Initializing multi-GPU trainer with {self.num_gpus} GPUs (out of {available_gpus} available)")
        
        # Set up queues for communication
        self.memory_queues = [queue.Queue(maxsize=100) for _ in range(self.num_gpus)]
        self.param_queue = queue.Queue()
        
        # Create stop event for all threads
        self.stop_event = threading.Event()
        
        # Track training progress
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.global_steps = 0
        self.episodes_completed = 0
        self.best_reward = float('-inf')
        
        # Create main agent on device 0
        device = torch.device(f"cuda:0")
        memory = OptimizedArrayReplayMemory(
            capacity=config.MEMORY_CAPACITY // self.num_gpus,
            state_shape=state_shape
        )
        self.main_agent = DQNAgent(
            state_shape=state_shape,
            n_actions=n_actions,
            memory=memory,
            device=device
        )
        
        # Create collectors and learners
        self.collectors = []
        self.learners = []
        
        # Environments per GPU (scale based on available GPUs)
        envs_per_gpu = config.NUM_PARALLEL_ENVS // self.num_gpus
        
        # Create collectors and learners for each GPU
        for gpu_id in range(self.num_gpus):
            # Create memory for this GPU
            gpu_memory = OptimizedArrayReplayMemory(
                capacity=config.MEMORY_CAPACITY // self.num_gpus,
                state_shape=state_shape
            )
            
            # Create agent for this GPU
            gpu_device = torch.device(f"cuda:{gpu_id}")
            gpu_agent = DQNAgent(
                state_shape=state_shape,
                n_actions=n_actions,
                memory=gpu_memory,
                device=gpu_device
            )
            
            # Copy weights from main agent
            gpu_agent.q_network.load_state_dict(self.main_agent.q_network.state_dict())
            gpu_agent.target_network.load_state_dict(self.main_agent.target_network.state_dict())
            
            # Create collector
            collector = ExperienceCollector(
                gpu_id=gpu_id,
                memory_queue=self.memory_queues[gpu_id],
                state_shape=state_shape,
                n_actions=n_actions,
                num_envs=envs_per_gpu,
                stop_event=self.stop_event
            )
            self.collectors.append(collector)
            
            # Create learner
            learner = LearnerThread(
                gpu_id=gpu_id,
                agent=gpu_agent,
                memory_queue=self.memory_queues[gpu_id],
                param_queue=self.param_queue,
                batch_size=config.BATCH_SIZE // self.num_gpus,
                stop_event=self.stop_event
            )
            self.learners.append(learner)
        
        # Target network update thread
        self.last_target_update = 0
    
    def start(self):
        """Start all threads."""
        print("Starting multi-GPU training...")
        
        # Start collector and learner threads
        for collector in self.collectors:
            collector.start()
        for learner in self.learners:
            learner.start()
        
        # Start parameter synchronization thread
        self.param_sync_thread = threading.Thread(target=self._sync_parameters)
        self.param_sync_thread.daemon = True
        self.param_sync_thread.start()
    
    def _sync_parameters(self):
        """Synchronize parameters between GPUs."""
        while not self.stop_event.is_set():
            try:
                # Get updated parameters from queue
                gpu_id, params, steps, losses = self.param_queue.get(timeout=1.0)
                
                # Update global step count
                self.global_steps = max(self.global_steps, steps)
                
                # Update target network if needed
                if self.global_steps - self.last_target_update >= config.TARGET_UPDATE_FREQUENCY:
                    self.main_agent.update_target_network()
                    self.last_target_update = self.global_steps
                    print(f"Updated target network at step {self.global_steps}")
                
                # Update main agent with parameters from GPU
                self.main_agent.q_network.load_state_dict(params)
                
                # Broadcast updated parameters to all other GPUs
                for i, collector in enumerate(self.collectors):
                    if i != gpu_id:
                        collector.update_agent(params)
            except queue.Empty:
                # Queue empty, continue waiting
                continue
    
    def stop(self):
        """Stop all threads."""
        print("Stopping multi-GPU training...")
        self.stop_event.set()
        
        # Wait for all threads to complete
        for collector in self.collectors:
            collector.join(timeout=5)
        for learner in self.learners:
            learner.join(timeout=5)
        self.param_sync_thread.join(timeout=5)
        
        print("Multi-GPU training stopped")
    
    def save_model(self, filepath):
        """Save the trained model."""
        return self.main_agent.save_model(filepath)
    
    def get_training_stats(self):
        """Get current training statistics from all collectors and learners."""
        stats = {
            "completed_episodes": sum(collector.completed_episodes for collector in self.collectors),
            "global_steps": self.global_steps,
            "avg_loss": 0.0,
            "losses": []
        }
        
        # Collect losses from all learners
        all_losses = []
        for learner in self.learners:
            if learner.losses:
                all_losses.extend(learner.losses[-100:])  # Only use last 100 losses
        
        if all_losses:
            stats["avg_loss"] = np.mean(all_losses)
            stats["losses"] = all_losses[-10:]  # Last 10 losses
        
        return stats


def train_multi_gpu(num_episodes=config.TRAINING_EPISODES, output_dir=None, num_gpus=None):
    """
    Train DQN using multiple GPUs.
    
    Args:
        num_episodes (int): Number of episodes to train
        output_dir (str): Directory to save outputs
        num_gpus (int): Number of GPUs to use
    
    Returns:
        tuple: (trained agent, training statistics)
    """
    trainer = None
    stats = None
    model_dir = None
    
    try:
        # Setup result directories
        if output_dir is None:
            directories = utils.create_directories()
            output_dir = directories["run"]
            model_dir = directories["models"]
            log_dir = directories["logs"]
            viz_dir = directories["viz"]
        else:
            model_dir = os.path.join(output_dir, "models")
            log_dir = os.path.join(output_dir, "logs")
            viz_dir = os.path.join(output_dir, "visualizations")
            
            # Ensure directories exist
            for directory in [output_dir, model_dir, log_dir, viz_dir]:
                os.makedirs(directory, exist_ok=True)
        
        # Create sample environment to get shapes
        env = env_wrappers.make_atari_env()
        state_shape = env.observation_space.shape
        n_actions = env.action_space.n
        env.close()
        
        # Create multi-GPU trainer
        trainer = MultiGPUTrainer(
            state_shape=state_shape,
            n_actions=n_actions,
            num_gpus=num_gpus
        )
        
        # Start training
        print("\nStarting multi-GPU training...")
        print(f"Training for {num_episodes} episodes across {trainer.num_gpus} GPUs")
        print(f"Progress will be displayed every 30 seconds")
        print("Press Ctrl+C to stop training gracefully and save the model\n")
        
        start_time = time.time()
        trainer.start()
        
        progress_interval = 30  # seconds
        last_progress_time = time.time()
        
        try:
            # Wait for training to complete
            for episode_check in range(1, num_episodes + 1, 10):  # Check every 10 episodes
                time.sleep(5)  # Wait 5 seconds between checks
                
                # Get current status from collector threads
                stats = trainer.get_training_stats()
                completed_episodes = stats["completed_episodes"]
                
                # Display progress periodically
                current_time = time.time()
                if current_time - last_progress_time >= progress_interval:
                    # Calculate training rate
                    elapsed_hours = (current_time - start_time) / 3600
                    episodes_per_hour = completed_episodes / max(0.1, elapsed_hours)
                    
                    # Estimate completion time
                    remaining_episodes = num_episodes - completed_episodes
                    est_hours_left = remaining_episodes / max(1, episodes_per_hour)
                    
                    print(f"Progress: ~{completed_episodes}/{num_episodes} episodes "
                          f"({completed_episodes/num_episodes*100:.1f}%)")
                    print(f"Total steps: {stats['global_steps']:,}")
                    print(f"Training rate: {episodes_per_hour:.1f} episodes/hour")
                    print(f"Estimated time remaining: {est_hours_left:.1f} hours")
                    
                    if stats.get("avg_loss", 0) > 0:
                        print(f"Average loss: {stats['avg_loss']:.6f}")
                    
                    last_progress_time = current_time
                
                # Save model periodically
                if episode_check % config.SAVE_FREQUENCY == 0:
                    model_path = os.path.join(model_dir, f"model_ep{episode_check}.pth")
                    trainer.save_model(model_path)
                    print(f"Saved model checkpoint at episode {episode_check}")
                
                # Check if we've reached the target number of episodes
                if completed_episodes >= num_episodes:
                    break
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving progress...")
            interrupted = True
        
        # Save final model
        final_model_path = os.path.join(model_dir, "final_model.pth")
        trainer.save_model(final_model_path)
        print(f"Model saved to {final_model_path}")
        
        # Calculate and report final statistics
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n" + "="*60)
        print(f"MULTI-GPU TRAINING COMPLETED - {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print("="*60)
        print(f"Total episodes completed: {completed_episodes}/{num_episodes}")
        print(f"Total environment steps:  {stats['global_steps']:,}")
        if "avg_loss" in stats and stats["avg_loss"] > 0:
            print(f"Final average loss:      {stats['avg_loss']:.6f}")
        print("-"*60)
        print(f"Final model saved to: {final_model_path}")
        print(f"Used {trainer.num_gpus} GPUs for training")
        print("="*60 + "\n")
        
        if trainer is not None:
            stats = trainer.get_training_stats()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
        if trainer is not None and model_dir is not None:
            # Save the model on interrupt
            interrupt_path = os.path.join(model_dir, "interrupted_model.pth")
            try:
                trainer.save_model(interrupt_path)
                print(f"Interrupted model saved to {interrupt_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
            
            # Get final stats
            stats = trainer.get_training_stats()
    
    except Exception as e:
        print(f"\nUnexpected error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop training threads if trainer exists
        if trainer is not None:
            try:
                trainer.stop()
            except Exception as e:
                print(f"Error stopping trainer: {e}")
    
    return trainer.main_agent if trainer is not None else None, stats


if __name__ == "__main__":
    # Check for available GPUs
    available_gpus = torch.cuda.device_count()
    
    # Use all GPUs minus one, ensuring at least one is used
    usable_gpus = max(1, available_gpus - 1)
    
    if available_gpus < 1:
        print(f"WARNING: No GPUs found. Multi-GPU training requires at least 1 GPU.")
        print("Falling back to CPU training...")
        from train import train
        agent, stats = train()
    else:
        print(f"Found {available_gpus} GPUs. Using {usable_gpus} for training (leaving 1 GPU free if available).")
        agent, stats = train_multi_gpu(num_gpus=usable_gpus)
    
    print("Training complete!")
