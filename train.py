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

Pseudocode:
1. Initialize replay memory D with capacity N
2. Initialize action-value network Q (θ₁) with random weights
3. Initialize target network Q_target (θ₂) ← θ₁
4. For each episode = 1 to M:
   5. Initialize initial state S₁
   6. For t = 1 to T:
      7. With probability ε, select a random action Aₜ (exploration)
      8. Otherwise, select Aₜ = argmaxₐ Q(Sₜ, a; θ₁) (exploitation)
      9. Execute action Aₜ, observe reward Rₜ₊₁ and next state Sₜ₊₁
      10. Store transition (Sₜ, Aₜ, Rₜ₊₁, Sₜ₊₁) into replay buffer D
      11. Sample a random minibatch of transitions from D
      12. For each sample j in the minibatch:
          If Sⱼ₊₁ is terminal:
              yⱼ ← Rⱼ₊₁
          Else:
              yⱼ ← Rⱼ₊₁ + γ * maxₐ' Q_target(Sⱼ₊₁, a'; θ₂)
      13. Perform gradient descent step to minimize:
          L = (yⱼ - Q(Sⱼ, Aⱼ; θ₁))²
      14. Every C steps:
          Update target network: θ₂ ← θ₁
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from datetime import datetime
import json

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


def train(device=None, render_training=False, output_dir=None, enable_recovery=True, agent=None, training_stats=None, start_episode=None):
    """
    Train a DQN agent on Atari Ice Hockey.
    
    This function implements the complete DQN training algorithm with careful
    consideration for GPU optimization and follows the pseudocode from the
    original DQN paper.
    
    Args:
        device (torch.device, optional): Device to use for training (auto-detected if None)
        render_training (bool): Whether to render training episodes (slower)
        output_dir (str, optional): Directory to save outputs (auto-generated if None)
        enable_recovery (bool): Enable automatic error recovery and checkpointing
        agent (DQNAgent, optional): Pre-initialized agent (for resuming training)
        training_stats (dict, optional): Pre-loaded training statistics (for resuming training)
        start_episode (int, optional): Episode to start from (for resuming training)
        
    Returns:
        tuple: (trained agent, training statistics)
    """
    # ===== SETUP PHASE =====
    
    # 1. Setup device (CPU/GPU) - Optimize for GPU when available
    if device is None:
        device = utils.get_device()
    print(f"Training on device: {device}")
    
    # Enable standard CUDA optimization if available
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
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
    
    # 5. Initialize replay memory and agent only if not provided (new training session)
    if agent is None:
        # PSEUDOCODE LINE 1: Initialize replay memory D with capacity N
        print("Initializing replay memory...")
        memory = create_replay_memory(
            memory_type=config.MEMORY_IMPLEMENTATION,
            capacity=config.MEMORY_CAPACITY,
            state_shape=state_shape
        )
        
        # PSEUDOCODE LINES 2-3: Initialize Q-network and target network
        print("Initializing DQN agent...")
        agent = DQNAgent(
            state_shape=state_shape,
            n_actions=n_actions,
            memory=memory,
            device=device
        )
    else:
        print("Using pre-initialized agent (resuming training)")
    
    # 7. Initialize or use provided statistics tracking
    if training_stats is None:
        stats = {
            "episode_rewards": [],        # Total reward per episode
            "episode_lengths": [],        # Number of steps per episode
            "episode_losses": [],         # Average loss per episode
            "episode_q_values": [],       # Average Q-values per episode
            "epsilons": [],               # Epsilon values at the end of each episode
            "learning_rate": config.LEARNING_RATE,
            "start_time": time.time()     # Track training duration
        }
    else:
        stats = training_stats
        print("Using pre-loaded training statistics (resuming training)")

    # Determine the starting episode
    current_episode = 1
    if start_episode is not None and start_episode > 1:
        current_episode = start_episode
        print(f"Resuming training from episode {current_episode}")
    elif training_stats is not None and "episode_rewards" in training_stats:
        current_episode = len(training_stats["episode_rewards"]) + 1
        print(f"Continuing from episode {current_episode} based on loaded statistics")

    # Calculate remaining episodes
    remaining_episodes = config.TRAINING_EPISODES - (current_episode - 1)
    if remaining_episodes <= 0:
        print("Warning: Already completed all episodes. Setting to train 100 more episodes.")
        remaining_episodes = 100

    # Ensure we save checkpoints more frequently for long training runs
    checkpoint_interval = min(50, config.SAVE_FREQUENCY)
    recovery_checkpoint_path = None
    
    # Add a flag to indicate graceful shutdown requested
    shutdown_requested = False

    # ===== TRAINING LOOP =====
    
    # 8. Training loop - PSEUDOCODE LINE 4: For each episode = 1 to M
    print(f"Starting training for {remaining_episodes} more episodes...")
    print(f"Will start learning after {config.LEARNING_STARTS} steps")
    print(f"Target network will update every {config.TARGET_UPDATE_FREQUENCY} steps")
    step_count = agent.steps_done if agent else 0
    best_reward = float("-inf")
    if stats.get("episode_rewards"):
        best_reward = max(stats["episode_rewards"])
        print(f"Previous best reward: {best_reward:.2f}")
    total_time_start = time.time()
    
    # Use tqdm for progress bar
    episode_iterator = tqdm(range(current_episode, current_episode + remaining_episodes), desc="Training Progress", 
                           disable=True)  # Disable tqdm progress bar
    try:
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
                try:
                    if step_count > config.LEARNING_STARTS and step_count % config.UPDATE_FREQUENCY == 0:
                        loss = agent.learn()
                        if loss is not None:
                            episode_loss.append(loss)
                except KeyboardInterrupt:
                    # Handle keyboard interrupt by breaking the episode loop and the training loop
                    print("\nTraining interrupted by user. Preparing for safe shutdown...")
                    shutdown_requested = True
                    done = True  # Force episode to end
                
                # Update target network periodically - PSEUDOCODE LINE 14
                if step_count % config.TARGET_UPDATE_FREQUENCY == 0:
                    agent.update_target_network()
                    
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
            
            print(f"Episode {episode}/{current_episode+remaining_episodes-1} - "
                  f"Total Steps: {step_count}, Reward: {episode_reward:.2f}, "
                  f"Loss: {avg_loss:.6f}, Epsilon: {agent.epsilon:.4f}, "
                  f"Time: {episode_duration:.2f}s")
            
            # Modify the best model message to be more minimal
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save_model(os.path.join(model_dir, "best_model.pth"))
            
            # Additional auto-recovery checkpoints every checkpoint_interval episodes
            if enable_recovery and episode % checkpoint_interval == 0:
                recovery_checkpoint_path = os.path.join(model_dir, f"recovery_checkpoint.pth")
                agent.save_model(recovery_checkpoint_path)
            
            # Periodically clean GPU memory if using CUDA
            if device.type == 'cuda' and episode % 100 == 0:
                torch.cuda.empty_cache()
            
            # Periodically save checkpoint and visualize progress
            if episode % config.SAVE_FREQUENCY == 0:
                try:
                    # Save model checkpoint
                    checkpoint_path = os.path.join(model_dir, f"model_ep{episode}.pth")
                    if agent.save_model(checkpoint_path):
                        print(f"Model checkpoint saved to {checkpoint_path}")
                    
                    # Save statistics (using pickle for efficiency)
                    stats_file = os.path.join(log_dir, "training_stats.pkl")
                    if utils.save_object(stats, stats_file):
                        print(f"Training stats saved to {stats_file}")
                    
                    # Also save as JSON for easier inspection
                    try:
                        # Convert numpy arrays to lists for JSON serialization
                        json_stats = {}
                        for key, value in stats.items():
                            if isinstance(value, list) and len(value) > 0:
                                # If list contains numpy values, convert them
                                if hasattr(value[0], 'item'):
                                    json_stats[key] = [float(x) for x in value]
                                else:
                                    json_stats[key] = value
                            elif hasattr(value, 'item'):
                                json_stats[key] = float(value)
                            else:
                                json_stats[key] = value
                        
                        # Save only most recent values to keep file size manageable
                        for key in ['episode_rewards', 'episode_lengths', 'episode_losses', 'episode_q_values', 'epsilons']:
                            if key in json_stats and len(json_stats[key]) > 1000:
                                json_stats[key + "_recent"] = json_stats[key][-1000:]
                                
                        json_file = os.path.join(log_dir, "training_stats.json")
                        with open(json_file, 'w') as f:
                            json.dump(json_stats, f, indent=2)
                    except Exception as e:
                        print(f"Warning: Could not save stats as JSON: {e}")
                    
                    # Update plots
                    if len(stats["episode_rewards"]) > 0:
                        try:
                            # Plot rewards
                            utils.plot_learning_curve(
                                values=stats["episode_rewards"],
                                window_size=100,
                                title=f"Episode Rewards (Episode {episode})",
                                xlabel="Episode",
                                ylabel="Reward",
                                save_path=os.path.join(viz_dir, "rewards.png")
                            )
                            
                            # Plot losses if any
                            if any(stats["episode_losses"]):
                                utils.plot_learning_curve(
                                    values=stats["episode_losses"],
                                    window_size=100,
                                    title=f"Training Loss (Episode {episode})",
                                    xlabel="Episode",
                                    ylabel="Loss",
                                    save_path=os.path.join(viz_dir, "losses.png")
                                )
                                
                            # Generate combined plots
                            utils.plot_episode_stats(stats, save_dir=viz_dir, show=False)
                            print(f"Plots updated in {viz_dir}")
                        except Exception as e:
                            print(f"Warning: Error generating plots: {e}")
                except Exception as e:
                    print(f"Warning: Error during periodic saving at episode {episode}: {e}")
            
            # Check if shutdown was requested
            if shutdown_requested:
                print("Gracefully shutting down training as requested...")
                break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
        # Save current model on interrupt
        if 'agent' in locals():
            try:
                interrupted_path = os.path.join(model_dir, "interrupted_model.pth")
                print(f"Saving interrupted model to {interrupted_path}...")
                # Save to a temp file first to avoid corruption
                temp_path = interrupted_path + ".tmp"
                agent.save_model(temp_path)
                if os.path.exists(temp_path):
                    if os.path.exists(interrupted_path):
                        os.remove(interrupted_path)
                    os.rename(temp_path, interrupted_path)
                    print(f"Interrupted model saved to {interrupted_path}")
            except Exception as e:
                print(f"Warning: Failed to save model: {e}")
    except Exception as e:
        print(f"\nUnexpected error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Calculate and report training metrics
        total_time = time.time() - total_time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
        # Save final model
        final_model_path = os.path.join(model_dir, "final_model.pth")
        agent.save_model(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Save final statistics with error handling
        try:
            final_stats_path = os.path.join(log_dir, "final_stats.pkl")
            if utils.save_object(stats, final_stats_path):
                print(f"Final statistics saved to {final_stats_path}")
                
            # Generate final visualizations with error handling
            try:
                utils.plot_episode_stats(stats, save_dir=viz_dir, show=False)
                print(f"Final plots saved to {viz_dir}")
            except Exception as e:
                print(f"Warning: Could not generate final plots: {e}")
        except Exception as e:
            print(f"Warning: Error saving final statistics: {e}")
        
        # Clean up resources
        env.close()
        
        # Free GPU memory if available
        if device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                print("GPU memory cleared")
            except Exception as e:
                print(f"Warning: Could not clear GPU memory: {e}")
        
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
    parser.add_argument("--enable_recovery", action="store_true", help="Enable automatic recovery mechanism")
    parser.add_argument("--start_episode", type=int, default=None, help="Episode to start from (for resuming training)")
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
            output_dir=args.output_dir,
            enable_recovery=args.enable_recovery,
            start_episode=args.start_episode
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
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources in all cases
        if 'trained_agent' in locals():
            try:
                # Create reference to env closure from train function
                local_env = None

                # Close environment if it exists through trained_agent
                if hasattr(trained_agent, 'env'):
                    trained_agent.env.close()
                    print("Environment closed successfully")
            except:
                pass
        
        if 'device' in locals() and device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                print("GPU memory cleared")
            except:
                pass
