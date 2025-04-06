"""
Logger module for tracking and visualizing DQN training progress.

This module provides:
1. Tracking of metrics (rewards, losses, epsilon values, etc.)
2. Logging to file (CSV format)
3. Real-time console feedback
4. Visualization of training progress through plots
5. Hyperparameter logging for experiment tracking

The Logger class acts as a central hub for all monitoring and visualization needs
during DQN training, helping to analyze model performance and debug issues.
"""

import os
import time
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class Logger:
    """
    Logger class for tracking DQN training progress.
    """
    
    def __init__(self, log_dir="logs", exp_name=None):
        """
        Initialize the logger with the specified log directory.
        
        Args:
            log_dir (str): Directory where logs will be stored
            exp_name (str): Name of the experiment (defaults to timestamp)
        """
        # Create a unique experiment name if not provided
        if exp_name is None:
            exp_name = f"dqn_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.exp_name = exp_name
        self.log_dir = os.path.join(log_dir, exp_name)
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Paths for different log files
        self.metrics_path = os.path.join(self.log_dir, "metrics.csv")
        self.config_path = os.path.join(self.log_dir, "config.json")
        self.summary_path = os.path.join(self.log_dir, "summary.txt")
        
        # Initialize metrics dictionary
        self.metrics = {
            'episode': [],
            'episode_reward': [],
            'episode_length': [],
            'loss': [],
            'epsilon': [],
            'avg_q_value': [],
            'eval_reward': [],
            'time_elapsed': []
        }
        
        # Initialize CSV file with header
        self._initialize_csv()
        
        # Track start time
        self.start_time = time.time()
        
        print(f"Logger initialized. Experiment: {self.exp_name}")
        print(f"Logs will be saved to: {self.log_dir}")
    
    def _initialize_csv(self):
        """Initialize the CSV file with header row."""
        with open(self.metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.metrics.keys())
    
    def log_hyperparameters(self, config_dict):
        """
        Log hyperparameters to a JSON file.
        
        Args:
            config_dict (dict): Dictionary of hyperparameters
        """
        # Save configuration to JSON file
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        print(f"Hyperparameters logged to {self.config_path}")
    
    def log_metrics(self, episode, episode_reward, episode_length, loss=None, 
                    epsilon=None, avg_q_value=None, eval_reward=None):
        """
        Log metrics for a single training episode.
        
        Args:
            episode (int): Episode number
            episode_reward (float): Total reward for the episode
            episode_length (int): Number of steps in the episode
            loss (float, optional): Average loss for the episode
            epsilon (float, optional): Current epsilon value
            avg_q_value (float, optional): Average Q-value for the episode
            eval_reward (float, optional): Evaluation reward if available
        """
        # Update metrics dictionary
        self.metrics['episode'].append(episode)
        self.metrics['episode_reward'].append(episode_reward)
        self.metrics['episode_length'].append(episode_length)
        self.metrics['loss'].append(loss if loss is not None else "N/A")
        self.metrics['epsilon'].append(epsilon if epsilon is not None else "N/A")
        self.metrics['avg_q_value'].append(avg_q_value if avg_q_value is not None else "N/A")
        self.metrics['eval_reward'].append(eval_reward if eval_reward is not None else "N/A")
        self.metrics['time_elapsed'].append(time.time() - self.start_time)
        
        # Append to CSV file
        with open(self.metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, 
                episode_reward, 
                episode_length, 
                loss if loss is not None else "N/A",
                epsilon if epsilon is not None else "N/A",
                avg_q_value if avg_q_value is not None else "N/A",
                eval_reward if eval_reward is not None else "N/A",
                time.time() - self.start_time
            ])
    
    def log_summary(self, agent_stats=None, additional_info=None):
        """
        Log a summary of the training run.
        
        Args:
            agent_stats (dict, optional): Statistics from the agent
            additional_info (dict, optional): Additional information to log
        """
        with open(self.summary_path, 'w') as f:
            f.write(f"DQN Training Summary\n")
            f.write(f"====================\n")
            f.write(f"Experiment: {self.exp_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total episodes: {len(self.metrics['episode'])}\n")
            f.write(f"Total training time: {self._format_time(self.metrics['time_elapsed'][-1] if self.metrics['time_elapsed'] else 0)}\n\n")
            
            # Metrics summary
            f.write(f"Metrics Summary:\n")
            f.write(f"- Final epsilon: {self.metrics['epsilon'][-1] if self.metrics['epsilon'] and self.metrics['epsilon'][-1] != 'N/A' else 'N/A'}\n")
            
            # Calculate average reward for last 100 episodes
            if len(self.metrics['episode_reward']) >= 100:
                last_100_rewards = [r for r in self.metrics['episode_reward'][-100:] if r != 'N/A']
                avg_last_100 = sum(last_100_rewards) / len(last_100_rewards) if last_100_rewards else 'N/A'
                f.write(f"- Average reward (last 100 episodes): {avg_last_100 if avg_last_100 != 'N/A' else 'N/A'}\n")
            
            # Add agent statistics if provided
            if agent_stats:
                f.write(f"\nAgent Statistics:\n")
                for key, value in agent_stats.items():
                    f.write(f"- {key}: {value}\n")
            
            # Add additional info if provided
            if additional_info:
                f.write(f"\nAdditional Information:\n")
                for key, value in additional_info.items():
                    f.write(f"- {key}: {value}\n")
        
        print(f"Training summary saved to {self.summary_path}")
    
    def _format_time(self, seconds):
        """Format time in seconds to hours, minutes, seconds."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
    
    def plot_metrics(self, save=True, show=True):
        """
        Plot training metrics.
        
        Args:
            save (bool): Whether to save plots to files
            show (bool): Whether to display the plots
        """
        # Create plots directory
        plots_dir = os.path.join(self.log_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Define common x-axis (episodes)
        episodes = self.metrics['episode']
        if not episodes:
            print("No data to plot")
            return
        
        # 1. Plot rewards
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, self.metrics['episode_reward'], label='Episode Reward')
        
        # Plot evaluation rewards if available
        eval_rewards = [(ep, reward) for ep, reward in zip(episodes, self.metrics['eval_reward']) 
                         if reward != 'N/A']
        if eval_rewards:
            eval_episodes, eval_reward_values = zip(*eval_rewards)
            plt.scatter(eval_episodes, eval_reward_values, color='red', marker='o', 
                        label='Evaluation Reward')
        
        # Add smoothed rewards (moving average)
        if len(episodes) >= 10:
            window_size = min(100, len(episodes) // 10)
            reward_values = [r for r in self.metrics['episode_reward'] if r != 'N/A']
            if reward_values:
                smoothed_rewards = np.convolve(reward_values, 
                                              np.ones(window_size)/window_size, 
                                              mode='valid')
                plt.plot(episodes[window_size-1:window_size-1+len(smoothed_rewards)],
                         smoothed_rewards, color='orange', 
                         label=f'Smoothed Reward (window={window_size})')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save:
            reward_path = os.path.join(plots_dir, "rewards.png")
            plt.savefig(reward_path)
            print(f"Rewards plot saved to {reward_path}")
        
        # 2. Plot loss
        plt.figure(figsize=(10, 6))
        loss_values = [(ep, loss) for ep, loss in zip(episodes, self.metrics['loss']) 
                        if loss != 'N/A']
        
        if loss_values:
            loss_episodes, loss_data = zip(*loss_values)
            plt.plot(loss_episodes, loss_data)
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True, alpha=0.3)
            if save:
                loss_path = os.path.join(plots_dir, "loss.png")
                plt.savefig(loss_path)
                print(f"Loss plot saved to {loss_path}")
        
        # 3. Plot epsilon decay
        plt.figure(figsize=(10, 6))
        epsilon_values = [(ep, eps) for ep, eps in zip(episodes, self.metrics['epsilon']) 
                           if eps != 'N/A']
        
        if epsilon_values:
            eps_episodes, eps_data = zip(*epsilon_values)
            plt.plot(eps_episodes, eps_data)
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.title('Exploration Rate (Epsilon)')
            plt.grid(True, alpha=0.3)
            if save:
                eps_path = os.path.join(plots_dir, "epsilon.png")
                plt.savefig(eps_path)
                print(f"Epsilon plot saved to {eps_path}")
        
        # 4. Combined metrics plot
        plt.figure(figsize=(15, 10))
        
        # Rewards subplot
        plt.subplot(2, 2, 1)
        plt.plot(episodes, self.metrics['episode_reward'], label='Episode Reward')
        if eval_rewards:
            plt.scatter(eval_episodes, eval_reward_values, color='red', marker='o', 
                        label='Evaluation')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss subplot
        plt.subplot(2, 2, 2)
        if loss_values:
            plt.plot(loss_episodes, loss_data)
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True, alpha=0.3)
        
        # Epsilon subplot
        plt.subplot(2, 2, 3)
        if epsilon_values:
            plt.plot(eps_episodes, eps_data)
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.title('Exploration Rate')
            plt.grid(True, alpha=0.3)
        
        # Q-values subplot
        plt.subplot(2, 2, 4)
        q_values = [(ep, q) for ep, q in zip(episodes, self.metrics['avg_q_value']) 
                     if q != 'N/A']
        if q_values:
            q_episodes, q_data = zip(*q_values)
            plt.plot(q_episodes, q_data)
            plt.xlabel('Episode')
            plt.ylabel('Average Q-Value')
            plt.title('Q-Values')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            combined_path = os.path.join(plots_dir, "combined_metrics.png")
            plt.savefig(combined_path)
            print(f"Combined metrics plot saved to {combined_path}")
        
        if show:
            plt.show()
        else:
            plt.close('all')
    
    def print_episode_summary(self, episode, reward, avg_reward, loss, epsilon, steps):
        """
        Print a summary of the current episode to the console.
        
        Args:
            episode (int): Current episode number
            reward (float): Reward for this episode
            avg_reward (float): Average reward over recent episodes
            loss (float): Average loss for this episode
            epsilon (float): Current epsilon value
            steps (int): Number of steps in this episode
        """
        time_elapsed = self._format_time(time.time() - self.start_time)
        
        # Format the output
        output = (f"Episode {episode} | "
                 f"Reward: {reward:.2f} | "
                 f"Avg Reward: {avg_reward:.2f} | "
                 f"Loss: {loss:.6f} | "
                 f"Epsilon: {epsilon:.4f} | "
                 f"Steps: {steps} | "
                 f"Time: {time_elapsed}")
        
        print(output)


# Test code for this module
if __name__ == "__main__":
    # Create a test logger
    logger = Logger(log_dir="test_logs")
    
    # Log some test hyperparameters
    logger.log_hyperparameters({
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "batch_size": 32,
        "memory_capacity": 10000,
        "target_update_frequency": 1000
    })
    
    # Generate some fake metrics
    import random
    
    for episode in range(1, 101):
        # Simulate some training metrics
        reward = episode * 0.1 + random.uniform(-1, 1)
        length = random.randint(100, 500)
        loss = 1.0 / (episode + 10) + random.uniform(0, 0.1)
        epsilon = max(0.1, 1.0 - episode * 0.01)
        q_value = episode * 0.05 + random.uniform(0, 0.5)
        
        # Log metrics
        eval_reward = episode + random.uniform(-2, 2) if episode % 10 == 0 else None
        logger.log_metrics(
            episode=episode,
            episode_reward=reward,
            episode_length=length,
            loss=loss,
            epsilon=epsilon,
            avg_q_value=q_value,
            eval_reward=eval_reward
        )
        
        # Print episode summary
        if episode % 10 == 0:
            avg_reward = sum([logger.metrics['episode_reward'][i] for i in range(-10, 0)]) / 10
            logger.print_episode_summary(
                episode=episode,
                reward=reward,
                avg_reward=avg_reward,
                loss=loss,
                epsilon=epsilon,
                steps=length
            )
    
    # Log summary
    logger.log_summary(
        agent_stats={
            "total_steps": 50000,
            "final_epsilon": 0.1,
            "memory_utilization": "75%"
        },
        additional_info={
            "environment": "IceHockey-v5",
            "hardware": "NVIDIA RTX 3080"
        }
    )
    
    # Plot metrics
    logger.plot_metrics(save=True, show=False)
    
    print("Logger test completed successfully!")
