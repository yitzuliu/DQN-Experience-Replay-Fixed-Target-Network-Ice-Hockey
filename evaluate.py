"""
DQN Model Evaluation

This module provides functionality to evaluate trained DQN models on the
Atari Ice Hockey environment. It allows for:
1. Visual evaluation with on-screen rendering
2. Performance metrics calculation
3. Comparison between different trained models
4. Recording of evaluation results
"""

import os
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import pickle
import cv2

import env_wrappers
import config
from dqn_agent import DQNAgent
from replay_memory import OptimizedArrayReplayMemory
import utils


def evaluate_model(model_path, num_episodes=10, render=True, record_video=False, device=None):
    """
    Evaluate a trained DQN model on the Ice Hockey environment.
    
    Args:
        model_path (str): Path to the saved model file
        num_episodes (int): Number of episodes to evaluate
        render (bool): Whether to render the environment
        record_video (bool): Whether to record video of gameplay
        device (torch.device): Device to run evaluation on
        
    Returns:
        dict: Evaluation results containing rewards, episode lengths, etc.
    """
    # Setup device
    if device is None:
        device = utils.get_device()
    print(f"Evaluating on device: {device}")
    
    # Create environment for evaluation (different settings than training)
    render_mode = "human" if render else None
    env = env_wrappers.make_atari_env(
        render_mode=render_mode,
        clip_rewards=False,  # Don't clip rewards during evaluation
        episode_life=False,  # Don't end episode on life loss
        gpu_acceleration=device.type in ['cuda', 'mps']
    )
    
    # Setup video recording if requested
    video_recorder = None
    if record_video:
        output_dir = os.path.dirname(model_path)
        video_path = os.path.join(output_dir, f"gameplay_{os.path.basename(model_path).split('.')[0]}.mp4")
        fps = 30
        size = (config.FRAME_WIDTH * 4, config.FRAME_HEIGHT * 4)  # 4x upscaling for better visibility
        video_recorder = cv2.VideoWriter(
            video_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            size
        )
        print(f"Recording video to {video_path}")
    
    # Get environment info
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # Create dummy memory (not used during evaluation, but required for agent initialization)
    memory = OptimizedArrayReplayMemory(
        capacity=100,  # Small capacity since it won't be used
        state_shape=state_shape
    )
    
    # Create and load agent
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        memory=memory,
        device=device
    )
    
    # Load trained model
    success = agent.load_model(model_path)
    if not success:
        print(f"Failed to load model from {model_path}")
        return None
    
    print(f"Model loaded from {model_path}")
    print(f"Evaluating for {num_episodes} episodes...")
    
    # Initialize metrics
    results = {
        "episode_rewards": [],
        "episode_lengths": [],
        "action_frequencies": np.zeros(n_actions),
        "model_path": model_path,
        "steps": agent.steps_done,
        "epsilon": agent.epsilon
    }
    
    # Get action meanings for better reporting
    action_meanings = env.unwrapped.get_action_meanings()
    
    # Evaluation loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Start timer for FPS calculation
        start_time = time.time()
        frames_processed = 0
        
        while not done:
            # Always use evaluate=True for pure exploitation (no random actions)
            action = agent.select_action(state, evaluate=True)
            
            # Update action frequency counter
            results["action_frequencies"][action] += 1
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record frame if video recording is enabled
            if video_recorder is not None and render_mode != "human":
                # Get rendered frame in RGB format
                if hasattr(env.unwrapped, "render"):
                    frame = env.unwrapped.render()
                    # Upscale for better visibility
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)
                    # Convert to BGR for OpenCV
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_recorder.write(frame)
            
            # Update stats
            episode_reward += reward
            episode_length += 1
            frames_processed += 1
            
            # Move to next state
            state = next_state
        
        # End of episode
        episode_duration = time.time() - start_time
        fps = frames_processed / episode_duration if episode_duration > 0 else 0
        
        # Record metrics
        results["episode_rewards"].append(episode_reward)
        results["episode_lengths"].append(episode_length)
        
        # Print episode results
        print(f"Episode {episode+1}/{num_episodes}: "
              f"Reward = {episode_reward:.2f}, "
              f"Length = {episode_length}, "
              f"FPS = {fps:.1f}")
    
    # Close environment and video recorder
    env.close()
    if video_recorder is not None:
        video_recorder.release()
    
    # Calculate summary statistics
    results["mean_reward"] = np.mean(results["episode_rewards"])
    results["std_reward"] = np.std(results["episode_rewards"])
    results["min_reward"] = np.min(results["episode_rewards"])
    results["max_reward"] = np.max(results["episode_rewards"])
    results["mean_length"] = np.mean(results["episode_lengths"])
    
    # Normalize action frequencies to percentage
    total_actions = np.sum(results["action_frequencies"])
    if total_actions > 0:
        results["action_frequencies"] = results["action_frequencies"] / total_actions * 100
    
    # Print summary statistics
    print("\nEvaluation Summary:")
    print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Min/Max reward: {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(f"Mean episode length: {results['mean_length']:.2f}")
    
    # Print most frequent actions
    print("\nMost frequent actions:")
    sorted_actions = np.argsort(results["action_frequencies"])[::-1]
    for i in range(min(5, n_actions)):
        action_idx = sorted_actions[i]
        print(f"  {action_meanings[action_idx]}: {results['action_frequencies'][action_idx]:.1f}%")
    
    # Save results
    results_dir = os.path.dirname(model_path)
    results_path = os.path.join(results_dir, f"eval_{os.path.basename(model_path).split('.')[0]}.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Evaluation results saved to {results_path}")
    
    return results


def plot_evaluation_results(results_path):
    """
    Plot evaluation results from a saved results file.
    
    Args:
        results_path (str): Path to the saved results file
    """
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    
    plt.figure(figsize=(12, 8))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(results["episode_rewards"], 'b-')
    plt.axhline(y=results["mean_reward"], color='r', linestyle='--', label=f"Mean: {results['mean_reward']:.2f}")
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    
    # Plot episode lengths
    plt.subplot(2, 2, 2)
    plt.plot(results["episode_lengths"], 'g-')
    plt.axhline(y=results["mean_length"], color='r', linestyle='--', label=f"Mean: {results['mean_length']:.2f}")
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()
    
    # Plot action frequencies
    plt.subplot(2, 2, 3)
    action_indices = np.arange(len(results["action_frequencies"]))
    plt.bar(action_indices, results["action_frequencies"])
    plt.title("Action Frequencies")
    plt.xlabel("Action")
    plt.ylabel("Frequency (%)")
    
    # Add model information
    plt.subplot(2, 2, 4)
    plt.axis('off')
    info_text = (
        f"Model: {os.path.basename(results['model_path'])}\n"
        f"Training steps: {results['steps']}\n"
        f"Epsilon: {results['epsilon']:.4f}\n\n"
        f"Evaluation metrics:\n"
        f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n"
        f"Min reward: {results['min_reward']:.2f}\n"
        f"Max reward: {results['max_reward']:.2f}\n"
        f"Mean episode length: {results['mean_length']:.2f}"
    )
    plt.text(0.1, 0.5, info_text, fontsize=10, va='center')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = results_path.replace('.pkl', '.png')
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"Plot saved to {plot_path}")


def compare_models(model_paths, num_episodes=10, device=None):
    """
    Compare multiple models by evaluating them on the same environment.
    
    Args:
        model_paths (list): List of paths to saved model files
        num_episodes (int): Number of episodes to evaluate each model
        device (torch.device): Device to run evaluation on
    """
    # Setup device
    if device is None:
        device = utils.get_device()
    
    # Evaluate each model
    results_list = []
    for model_path in model_paths:
        print(f"\nEvaluating model: {model_path}")
        results = evaluate_model(
            model_path=model_path,
            num_episodes=num_episodes,
            render=False,  # No rendering when comparing models
            device=device
        )
        if results is not None:
            results_list.append(results)
    
    # If no models were successfully evaluated, return
    if not results_list:
        print("No models were successfully evaluated.")
        return
    
    # Compare results
    print("\nModel Comparison:")
    print("-" * 80)
    print(f"{'Model':30s} | {'Mean Reward':12s} | {'Std Dev':8s} | {'Min':6s} | {'Max':6s} | {'Steps':10s}")
    print("-" * 80)
    
    for results in results_list:
        model_name = os.path.basename(results['model_path'])
        print(f"{model_name:30s} | {results['mean_reward']:12.2f} | {results['std_reward']:8.2f} | "
              f"{results['min_reward']:6.2f} | {results['max_reward']:6.2f} | {results['steps']:10d}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Plot mean rewards
    plt.subplot(1, 2, 1)
    model_names = [os.path.basename(r['model_path']).split('.')[0] for r in results_list]
    mean_rewards = [r['mean_reward'] for r in results_list]
    std_rewards = [r['std_reward'] for r in results_list]
    
    bars = plt.bar(model_names, mean_rewards, yerr=std_rewards, capsize=5)
    plt.title("Mean Rewards by Model")
    plt.xlabel("Model")
    plt.ylabel("Mean Reward")
    plt.xticks(rotation=45, ha='right')
    
    # Add reward values on top of bars
    for bar, reward in zip(bars, mean_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{reward:.2f}", ha='center', va='bottom')
    
    # Plot training progress
    plt.subplot(1, 2, 2)
    steps = [r['steps'] for r in results_list]
    plt.scatter(steps, mean_rewards, s=100)
    
    # Add model names as annotations
    for i, (x, y, name) in enumerate(zip(steps, mean_rewards, model_names)):
        plt.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points')
        
    plt.title("Reward vs. Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Reward")
    
    plt.tight_layout()
    
    # Save comparison plot
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    comparison_path = os.path.join(os.path.dirname(model_paths[0]), f"model_comparison_{timestamp}.png")
    plt.savefig(comparison_path, dpi=300)
    plt.show()
    print(f"Comparison plot saved to {comparison_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained DQN models")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a single model")
    eval_parser.add_argument("model", type=str, help="Path to the model file (.pth)")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    eval_parser.add_argument("--render", action="store_true", help="Render environment")
    eval_parser.add_argument("--video", action="store_true", help="Record video")
    eval_parser.add_argument("--gpu", action="store_true", help="Force GPU usage")
    eval_parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Plot evaluation results")
    plot_parser.add_argument("results", type=str, help="Path to the results file (.pkl)")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("models", type=str, nargs="+", help="Paths to model files (.pth)")
    compare_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per model")
    compare_parser.add_argument("--gpu", action="store_true", help="Force GPU usage")
    compare_parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    
    # Determine device
    if hasattr(args, "gpu") and hasattr(args, "cpu"):
        if args.gpu and args.cpu:
            print("Error: Cannot specify both --gpu and --cpu")
            exit(1)
        elif args.gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif args.cpu:
            device = torch.device("cpu")
        else:
            device = utils.get_device()
    else:
        device = utils.get_device()
    
    # Execute the appropriate command
    if args.command == "evaluate":
        evaluate_model(
            model_path=args.model,
            num_episodes=args.episodes,
            render=args.render,
            record_video=args.video,
            device=device
        )
    elif args.command == "plot":
        plot_evaluation_results(args.results)
    elif args.command == "compare":
        compare_models(
            model_paths=args.models,
            num_episodes=args.episodes,
            device=device
        )
    else:
        parser.print_help()
