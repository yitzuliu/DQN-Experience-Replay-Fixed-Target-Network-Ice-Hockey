"""
Ultra-Minimal Logger for DQN Training

This module focuses only on:
1. Generating visualization plots (saved as PNG images only)
2. Maintaining minimal resume data in memory
3. Minimizing disk writes to absolute minimum

No serialized training data is saved by default - just images and tiny resume JSON.
"""

import os
import json

class Logger:
    """
    Ultra-minimal logger for DQN that only maintains data for plots and resumption.
    Prioritizes system stability over data preservation.
    """
    def __init__(self, run_dir, save_frequency=200):
        """
        Args:
            run_dir (str): Base directory for this training run.
            save_frequency (int): How often to save resume data (in episodes)
        """
        self.run_dir = run_dir
        self.log_dir = os.path.join(run_dir, "logs")
        self.viz_dir = os.path.join(run_dir, "visualizations")
        self.save_frequency = save_frequency
        self.last_save_episode = 0
        
        # Create directories only when needed
        try:
            os.makedirs(self.viz_dir, exist_ok=True)  # For plots only
        except Exception as e:
            print(f"Warning: Could not create viz directory: {e}")
        
        # Minimal tracked metrics - kept in memory only
        self.episode_count = 0
        self.best_reward = float('-inf')
        self.last_reward = None
        
        # In-memory data for plotting - never saved to disk except as images
        self.plot_data = {
            "rewards": [],
            "lengths": [],
            "losses": [],
            "epsilons": []
        }
        
        # Moving average calculation
        self.reward_window = []
        self.window_size = 100

    def log_episode(self, reward, length, loss, epsilon):
        """
        Record essential data for one training episode.
        Everything stays in memory only.
        """
        self.episode_count += 1
        self.last_reward = reward
        
        # Update best reward
        if reward > self.best_reward:
            self.best_reward = reward
            
        # Update moving average calculation
        self.reward_window.append(reward)
        if len(self.reward_window) > self.window_size:
            self.reward_window.pop(0)
            
        # Save data for plots (in memory only)
        self.plot_data["rewards"].append(reward)
        self.plot_data["lengths"].append(length)
        self.plot_data["losses"].append(loss if loss is not None else 0)
        self.plot_data["epsilons"].append(epsilon)
        
        # Auto-save resume data based on frequency (minimal writes)
        if self.episode_count - self.last_save_episode >= self.save_frequency:
            self.save_resume_data()
            self.last_save_episode = self.episode_count

    def save_resume_data(self):
        """
        Save minimal data needed to resume training.
        This is the only data actually written to disk apart from plots.
        """
        try:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, exist_ok=True)
                
            # Ultra-minimal data - this JSON file will be tiny (<100 bytes)
            resume_data = {
                "episode_count": self.episode_count,
                "best_reward": self.best_reward
            }
            
            # Use atomic write pattern to prevent corruption
            filepath = os.path.join(self.log_dir, "resume_data.json")
            temp_filepath = filepath + ".tmp"
            
            with open(temp_filepath, 'w') as f:
                json.dump(resume_data, f)
                
            # Atomic rename to minimize corruption risk
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_filepath, filepath)
                
            return True
        except Exception as e:
            print(f"Warning: Error saving resume data: {e}")
            return False

    def plot(self):
        """
        Generate plots and save as PNG files.
        Does not save the underlying data to disk.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend to reduce errors
            
            # Only create visualizations directory when actually needed
            if not os.path.exists(self.viz_dir):
                os.makedirs(self.viz_dir, exist_ok=True)
            
            # Create reward plot
            self._create_plot(
                data=self.plot_data["rewards"],
                title=f"Episode Rewards (Best: {self.best_reward:.1f})",
                y_label="Reward",
                filename="rewards.png",
                show_moving_avg=True
            )
            
            # Create loss plot if we have meaningful data
            if self.plot_data["losses"] and not all(l == 0 for l in self.plot_data["losses"]):
                self._create_plot(
                    data=self.plot_data["losses"],
                    title="Training Loss",
                    y_label="Loss",
                    filename="losses.png",
                    color='orange',
                    show_moving_avg=False
                )
            
            # Create combined plot
            self._create_combined_plot()
            
            return True
        except Exception as e:
            print(f"Warning: Error generating plots: {e}")
            return False
    
    def _create_plot(self, data, title, y_label, filename, color='blue', show_moving_avg=False):
        """Helper method to create individual plots"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(data, alpha=0.3, color=color)
            
            # Add moving average if requested
            if show_moving_avg and len(self.reward_window) >= 10:
                # Calculate moving average
                window_size = min(100, max(10, len(data) // 10))
                avg_data = []
                
                for i in range(len(data)):
                    if i < window_size:
                        # For early points, use available data only
                        avg = sum(data[:i+1]) / (i+1)
                    else:
                        # For later points, use sliding window
                        avg = sum(data[i-window_size+1:i+1]) / window_size
                    avg_data.append(avg)
                
                plt.plot(avg_data, color='red', label=f'Moving Avg (window={window_size})')
                plt.legend()
            
            plt.title(title)
            plt.xlabel("Episode")
            plt.ylabel(y_label)
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(self.viz_dir, filename)
            plt.savefig(save_path, dpi=150)  # Lower DPI to reduce file size
            plt.close()
        except Exception as e:
            print(f"Warning: Error creating {filename}: {e}")
    
    def _create_combined_plot(self):
        """Create a combined plot with all metrics"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('DQN Training Progress', fontsize=16)
            
            # Plot rewards
            axes[0, 0].plot(self.plot_data["rewards"], alpha=0.3, color='blue')
            # Add moving average for rewards
            if len(self.reward_window) >= 10:
                window_size = min(100, max(10, len(self.plot_data["rewards"]) // 10))
                avg_data = []
                for i in range(len(self.plot_data["rewards"])):
                    if i < window_size:
                        avg = sum(self.plot_data["rewards"][:i+1]) / (i+1)
                    else:
                        avg = sum(self.plot_data["rewards"][i-window_size+1:i+1]) / window_size
                    avg_data.append(avg)
                axes[0, 0].plot(avg_data, color='red')
            axes[0, 0].set_title(f'Rewards (Best: {self.best_reward:.1f})')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            
            # Plot episode lengths
            axes[0, 1].plot(self.plot_data["lengths"], color='green')
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            
            # Plot losses
            axes[1, 0].plot(self.plot_data["losses"], color='orange')
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            
            # Plot epsilon
            axes[1, 1].plot(self.plot_data["epsilons"], color='purple')
            axes[1, 1].set_title('Exploration Rate (ε)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = os.path.join(self.viz_dir, "training_stats.png")
            plt.savefig(save_path, dpi=150)  # Lower DPI to reduce file size
            plt.close()
        except Exception as e:
            print(f"Warning: Error creating combined plot: {e}")

    @classmethod
    def load(cls, run_dir, save_frequency=50):
        """
        Load minimal resume data.
        
        Returns:
            tuple: (logger, episode_count) 
        """
        try:
            # Create a new logger instance
            logger = cls(run_dir, save_frequency=save_frequency)
            
            # Try to load resume data if it exists
            resume_path = os.path.join(run_dir, "logs", "resume_data.json")
            if not os.path.exists(resume_path):
                print("No resume data found. Starting fresh.")
                return logger, 1
                
            with open(resume_path, 'r') as f:
                resume_data = json.load(f)
                
            # Load essential data
            logger.episode_count = resume_data.get("episode_count", 0)
            logger.best_reward = resume_data.get("best_reward", float('-inf'))
            logger.last_reward = resume_data.get("last_reward")
            
            next_episode = logger.episode_count + 1
            print(f"Resuming from episode {next_episode} with best reward {logger.best_reward}")
            return logger, next_episode
            
        except Exception as e:
            print(f"Warning: Error loading resume data: {e}")
            print("Starting with a fresh logger.")
            return cls(run_dir, save_frequency=save_frequency), 1


# Simple test if run directly
if __name__ == "__main__":
    import random
    import argparse

    parser = argparse.ArgumentParser(description="Test ultra-minimal logger")
    parser.add_argument("--test", action="store_true", help="Run a test logger instance")
    parser.add_argument("--dir", type=str, default=None, help="Directory to use for test")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    args = parser.parse_args()

    if args.test:
        # Create a test logger and generate some dummy data
        test_dir = args.dir or "./logs/test_logger"
        print(f"Creating test logger in {test_dir}")

        test_logger = Logger(run_dir=test_dir, save_frequency=args.save_freq)

        # Generate dummy data
        for episode in range(1, 101):
            # Simulate increasing rewards with noise
            reward = -10 + episode * 0.2 + random.uniform(-2, 2)
            length = int(300 + episode * 0.5 + random.uniform(-20, 20))
            epsilon = max(0.1, 1.0 - (episode / 100))
            loss = max(0.01, 1.0 - (episode / 200))

            # Log episode
            test_logger.log_episode(reward, length, loss, epsilon)

        # Generate plots
        test_logger.plot()
        
        print(f"Test complete. Logged {test_logger.episode_count} episodes.")
        print(f"Best reward: {test_logger.best_reward:.2f}")
        print(f"Resume data saved every {args.save_freq} episodes")
        print(f"Plots saved to: {test_logger.viz_dir}")
