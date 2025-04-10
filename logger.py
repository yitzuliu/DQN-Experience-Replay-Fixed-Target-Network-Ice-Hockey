"""
Simplified Logger for DQN Training

This module provides basic visualization and metrics tracking for DQN training.
It focuses on:
1. Essential metrics tracking (rewards, losses, etc.)
2. Training progress visualization
3. Saving data needed for evaluation
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import pickle


class Logger:
    """
    Simplified logger for tracking and visualizing DQN training progress.
    """
    
    def __init__(self, log_dir=None, experiment_name=None):
        """
        Initialize the logger with minimal setup.
        
        Args:
            log_dir (str): Directory to save logs and visualizations
            experiment_name (str): Name of the current experiment/run
        """
        # Setup basic logging
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.experiment_name = experiment_name or f"dqn_run_{timestamp}"
        self.log_dir = log_dir or os.path.join("logs", self.experiment_name)
        
        # Create directory structure
        self.viz_dir = os.path.join(self.log_dir, "plots")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Initialize metrics dictionary
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        
        print(f"Simplified logger initialized. Saving to: {self.log_dir}")
    
    def log_episode(self, episode_num, episode_reward, episode_length, epsilon, loss=None, q_value=None):
        """
        Log basic metrics for a completed episode.
        
        Args:
            episode_num (int): Episode number
            episode_reward (float): Total reward for the episode
            episode_length (int): Number of steps in the episode
            epsilon (float): Current epsilon value
            loss (float, optional): Average loss for the episode
            q_value (float, optional): Average Q-value for the episode
        """
        # Store essential metrics
        self.metrics["episode"].append(episode_num)
        self.metrics["reward"].append(episode_reward)
        self.metrics["length"].append(episode_length)
        self.metrics["epsilon"].append(epsilon)
        
        if loss is not None:
            self.metrics["loss"].append(loss)
        
        if q_value is not None:
            self.metrics["q_value"].append(q_value)
    
    def log_eval_episode(self, train_episode, eval_rewards):
        """
        Log evaluation episode results.
        
        Args:
            train_episode (int): Training episode number when evaluation was performed
            eval_rewards (list): List of rewards from evaluation episodes
        """
        # Store evaluation data for later use
        if "eval_episode" not in self.metrics:
            self.metrics["eval_episode"] = []
            self.metrics["eval_reward"] = []
        
        self.metrics["eval_episode"].append(train_episode)
        self.metrics["eval_reward"].append(np.mean(eval_rewards))
        
        # Print summary
        print(f"Evaluation at episode {train_episode}: Mean reward = {np.mean(eval_rewards):.2f}")
    
    def save_metrics(self, filename="training_metrics.pkl"):
        """
        Save current metrics to a file.
        
        Args:
            filename (str): Name of the file to save metrics to
        """
        # Save metrics to pickle file for later analysis
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "wb") as f:
            pickle.dump(self.metrics, f)
        
        # Also save a JSON summary with key statistics
        summary = {
            "episodes_completed": len(self.metrics["episode"]),
            "max_reward": max(self.metrics["reward"]) if self.metrics["reward"] else None,
            "final_reward": self.metrics["reward"][-1] if self.metrics["reward"] else None,
            "avg_reward_last_100": np.mean(self.metrics["reward"][-100:]) if len(self.metrics["reward"]) >= 100 else None,
            "final_epsilon": self.metrics["epsilon"][-1] if self.metrics["epsilon"] else None,
            "training_duration_hours": (time.time() - self.start_time) / 3600,
        }
        
        with open(os.path.join(self.log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    
    def plot_training_curves(self, save=True, show=False):
        """
        Generate and save/show essential training curves.
        
        Args:
            save (bool): Whether to save plots to files
            show (bool): Whether to display plots
        """
        if not self.metrics["episode"]:
            print("No metrics to plot.")
            return
        
        # Create a 2x2 grid of plots for key metrics
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'DQN Training Progress - {self.experiment_name}', fontsize=16)
        
        # 1. Reward curve (with moving average)
        self._plot_with_moving_avg(axs[0, 0], 
                               self.metrics["episode"], 
                               self.metrics["reward"],
                               "Episode Rewards", "Episode", "Reward")
        
        # 2. Epsilon decay curve
        axs[0, 1].plot(self.metrics["episode"], self.metrics["epsilon"])
        axs[0, 1].set_title("Exploration Rate (ε)")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Epsilon")
        axs[0, 1].grid(True, alpha=0.3)
        
        # 3. Episode length curve
        self._plot_with_moving_avg(axs[1, 0],
                               self.metrics["episode"],
                               self.metrics["length"],
                               "Episode Lengths", "Episode", "Steps")
        
        # 4. Loss curve (if available)
        if "loss" in self.metrics and self.metrics["loss"]:
            self._plot_with_moving_avg(axs[1, 1],
                                   self.metrics["episode"],
                                   self.metrics["loss"],
                                   "Training Loss", "Episode", "Loss")
        else:
            axs[1, 1].set_visible(False)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the figure
        if save:
            plt.savefig(os.path.join(self.viz_dir, "training_curves.png"), dpi=300)
        
        # Show or close
        if show:
            plt.show()
        else:
            plt.close()
        
        # If we have evaluation data, plot that too
        if "eval_episode" in self.metrics and self.metrics["eval_episode"]:
            self._plot_eval_curve(save, show)
    
    def _plot_eval_curve(self, save=True, show=False):
        """Plot evaluation results."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["eval_episode"], self.metrics["eval_reward"], 'bo-')
        plt.title("Evaluation Rewards")
        plt.xlabel("Training Episode")
        plt.ylabel("Mean Evaluation Reward")
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.viz_dir, "eval_rewards.png"), dpi=300)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _plot_with_moving_avg(self, ax, x, y, title, xlabel, ylabel, window_size=100):
        """Plot a metric with its moving average."""
        # Plot raw data
        ax.plot(x, y, alpha=0.3, color='blue', label='Raw')
        
        # Plot moving average if we have enough data
        if len(y) >= window_size:
            moving_avg = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
            ax.plot(x[window_size-1:], moving_avg, color='red', 
                   label=f'Moving Avg (n={window_size})')
        
        # Set labels and grid
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def get_eval_data(self):
        """
        Get evaluation data for external use.
        
        Returns:
            dict: Dictionary containing evaluation episodes and rewards
        """
        if "eval_episode" not in self.metrics:
            return {"episodes": [], "rewards": []}
        
        return {
            "episodes": self.metrics["eval_episode"],
            "rewards": self.metrics["eval_reward"]
        }


def verify_logger(log_dir=None):
    """
    Function to verify logger functionality and diagnose issues.
    
    Args:
        log_dir (str): Directory to check for logs
    
    Returns:
        bool: True if verification passes
    """
    import os
    import glob
    
    print("\n===== LOGGER VERIFICATION =====")
    
    # If log_dir not provided, try some common locations
    if log_dir is None:
        possible_dirs = [
            "./logs",
            "./results",
            os.path.join(".", "results", "run_*"),
            os.path.join(".", "results", "*", "logs")
        ]
        
        log_files_found = []
        for pattern in possible_dirs:
            matching_dirs = glob.glob(pattern)
            for directory in matching_dirs:
                if os.path.isdir(directory):
                    log_files = glob.glob(os.path.join(directory, "*.pkl")) + \
                               glob.glob(os.path.join(directory, "*.json")) + \
                               glob.glob(os.path.join(directory, "*.png"))
                    if log_files:
                        log_files_found.extend(log_files)
                        print(f"Found {len(log_files)} log files in {directory}")
                    
        if not log_files_found:
            print("No log files found in standard locations")
            print("Try specifying a directory or checking permissions")
            return False
        else:
            print(f"Found a total of {len(log_files_found)} log files")
            for file in sorted(log_files_found)[:10]:  # Show first 10 files
                print(f" - {file} ({os.path.getsize(file)} bytes)")
            if len(log_files_found) > 10:
                print(f" - ... and {len(log_files_found) - 10} more files")
    else:
        # Check specific directory
        if not os.path.isdir(log_dir):
            print(f"Directory does not exist: {log_dir}")
            return False
        
        log_files = glob.glob(os.path.join(log_dir, "*.pkl")) + \
                   glob.glob(os.path.join(log_dir, "*.json")) + \
                   glob.glob(os.path.join(log_dir, "*.png"))
        
        if not log_files:
            print(f"No log files found in {log_dir}")
            print("Try checking permissions or if the directory is correct")
            return False
        else:
            print(f"Found {len(log_files)} log files in {log_dir}")
            for file in sorted(log_files)[:10]:  # Show first 10 files
                print(f" - {file} ({os.path.getsize(file)} bytes)")
            if len(log_files) > 10:
                print(f" - ... and {len(log_files) - 10} more files")
    
    print("\nVerification complete.")
    return True


# Simple test if run directly
if __name__ == "__main__":
    import random
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify and debug logger functionality")
    parser.add_argument("--check", action="store_true", help="Check for existing log files")
    parser.add_argument("--test", action="store_true", help="Run a test logger instance")
    parser.add_argument("--dir", type=str, default=None, help="Directory to check or use for test")
    args = parser.parse_args()
    
    if args.check:
        verify_logger(args.dir)
    
    if args.test:
        # Create a test logger and generate some dummy data
        test_dir = args.dir or "./logs/test_logger"
        print(f"Creating test logger in {test_dir}")
        
        test_logger = Logger(log_dir=test_dir, experiment_name="logger_test")
        
        # Generate dummy data
        import random
        import numpy as np
        
        for episode in range(1, 101):
            # Simulate increasing rewards with noise
            reward = -10 + episode * 0.2 + random.uniform(-2, 2)
            length = int(300 + episode * 0.5 + random.uniform(-20, 20))
            epsilon = max(0.1, 1.0 - (episode / 100))
            loss = max(0.01, 1.0 - (episode / 200))
            
            # Log episode
            test_logger.log_episode(episode, reward, length, epsilon, loss)
            
            # Occasional evaluation
            if episode % 25 == 0:
                eval_rewards = [reward + random.uniform(-1, 3) for _ in range(5)]
                test_logger.log_eval_episode(episode, eval_rewards)
        
        # Save metrics and generate plots
        test_logger.save_metrics()
        test_logger.plot_training_curves(save=True, show=False)
        
        print(f"Test logger data saved to {test_dir}")
        verify_logger(test_dir)
    
    # Only run this code if neither --check nor --test is specified
    if not (args.check or args.test):
        print("Running default logger test...")
        logger = Logger(experiment_name="default_test")
        
        # Simulate 200 training episodes
        for ep in range(1, 201):
            # Log random metrics
            reward = -15 + ep * 0.1 + random.uniform(-5, 5)  # Gradually increasing with noise
            length = random.randint(500, 1000)
            epsilon = max(0.1, 1.0 - (ep / 200))
            loss = max(0.01, 1.0 * (0.98 ** ep))
            
            logger.log_episode(ep, reward, length, epsilon, loss)
            
            # Occasional evaluation
            if ep % 50 == 0:
                eval_rewards = [reward + random.uniform(-2, 5) for _ in range(5)]
                logger.log_eval_episode(ep, eval_rewards)
        
        # Save metrics and generate plots
        logger.save_metrics()
        logger.plot_training_curves(show=True)
        
        print(f"Test complete. Check logs in: {logger.log_dir}")
