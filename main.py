import argparse
import os
import torch
import numpy as np

import env_wrappers
from dqn_agent import DQNAgent
from train import train
import config
import utils
from evaluate import evaluate  # this will be implemented separately

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DQN for Atari Ice Hockey")
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'evaluate', 'visualize'],
                        help='Mode: train, evaluate, or visualize')
    
    # Training settings
    parser.add_argument('--episodes', type=int, default=config.TRAINING_EPISODES,
                        help='Number of training episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training/evaluation')
    
    # Model settings
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file for evaluation or continuing training')
    
    # Evaluation settings
    parser.add_argument('--eval_episodes', type=int, default=config.EVAL_EPISODES,
                        help='Number of evaluation episodes')
    
    # Visualization settings
    parser.add_argument('--stats', type=str, default='models/training_stats.pkl',
                        help='Path to training statistics file for visualization')
    
    return parser.parse_args()

def setup_environment_and_agent(args):
    """Setup environment and agent"""
    # Create environment
    env = env_wrappers.make_env(render_mode="human" if args.render else None)
    print(f"Environment: {config.ENV_NAME}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create agent
    agent = DQNAgent(env, device=device)
    print(f"DQN Agent created")
    
    # Load model if specified
    if args.model and os.path.exists(args.model):
        agent.load_model(args.model)
    elif args.model:
        print(f"Warning: Model file {args.model} not found, using new model")
    
    return env, agent

def train_agent(args):
    """Train DQN agent"""
    env, agent = setup_environment_and_agent(args)
    
    print(f"Starting training for {args.episodes} episodes...")
    try:
        stats = train(agent, env, num_episodes=args.episodes, render=args.render)
        
        # Plot and save training results
        utils.plot_training_metrics(stats)
        utils.print_training_summary(stats)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        env.close()
        print("Training finished")

def evaluate_agent(args):
    """Evaluate trained agent"""
    if not args.model:
        print("Error: No model specified for evaluation")
        return
    
    env, agent = setup_environment_and_agent(args)
    
    print(f"Evaluating model for {args.eval_episodes} episodes...")
    rewards = evaluate(agent, env, num_episodes=args.eval_episodes, render=args.render)
    
    # Print evaluation results
    print("\n===== Evaluation Results =====")
    print(f"Episodes: {args.eval_episodes}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Min reward: {np.min(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")
    print("==============================\n")
    
    env.close()

def visualize_training(args):
    """Visualize training results"""
    stats = utils.load_training_stats(args.stats)
    
    if stats:
        utils.plot_training_metrics(stats)
        utils.print_training_summary(stats)
    else:
        print("No training statistics available to visualize")

def main():
    """Main function"""
    # 修改為不傳入參數時自動使用預設值（train模式）
    args = parse_args()
    
    # 顯示當前運行模式
    print(f"Running in mode: {args.mode}")
    
    # Execute selected mode
    if args.mode == 'train':
        train_agent(args)
    elif args.mode == 'evaluate':
        # 暫時注釋掉評估功能，但保留代碼以便將來啟用
        print("Evaluation mode is currently disabled.")
        # evaluate_agent(args)  # 暫時注釋掉，不是刪除
    elif args.mode == 'visualize':
        visualize_training(args)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
