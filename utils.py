import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
import time

def get_device():
    """
    Determine if GPU is available and return appropriate device.
    
    Returns:
        torch.device: CPU or GPU device for tensor operations
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        print("GPU not available, using CPU")
        return torch.device("cpu")

def create_directories(base_dir="./results"):
    """
    Create necessary directories for saving models, logs, and visualizations.
    
    Args:
        base_dir (str): Base directory for all outputs
    
    Returns:
        dict: Dictionary containing paths to created directories
    """
    # Create timestamp for unique run identification
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create directory structure
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    model_dir = os.path.join(run_dir, "models")
    log_dir = os.path.join(run_dir, "logs")
    viz_dir = os.path.join(run_dir, "visualizations")
    
    # Ensure directories exist
    for directory in [run_dir, model_dir, log_dir, viz_dir]:
        os.makedirs(directory, exist_ok=True)
    
    paths = {
        "run": run_dir,
        "models": model_dir,
        "logs": log_dir,
        "viz": viz_dir
    }
    
    return paths

def plot_learning_curve(values, window_size=100, title="", xlabel="", ylabel="", save_path=None, max_points=5000):
    """
    Plot learning curves with moving average for training visualization.
    
    Args:
        values (list): Values to plot (e.g., rewards or losses)
        window_size (int): Size of the moving average window
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        save_path (str): Path to save the figure, or None to display
        max_points (int): Maximum number of points to plot (down-samples if exceeded)
    """
    if not values:
        print(f"Warning: No data to plot for {title}")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Down-sample if too many points (for efficiency)
    if len(values) > max_points:
        skip = len(values) // max_points
        indices = np.arange(0, len(values), skip)
        values = [values[i] for i in indices]
    
    # Plot raw values
    plt.plot(values, alpha=0.3, color='blue', label='Raw')
    
    # Calculate and plot moving average if enough data points
    if len(values) >= window_size:
        moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(values)), moving_avg, color='red', 
                label=f'Moving Avg (window={window_size})')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
            plt.close()
    else:
        plt.show()

def save_object(obj, filepath):
    """
    Save a Python object to disk using pickle.
    
    Args:
        obj: Python object to save
        filepath (str): Path to save the object
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        return True
    except Exception as e:
        print(f"Error saving object to {filepath}: {e}")
        return False

def load_object(filepath):
    """
    Load a Python object from disk using pickle.
    
    Args:
        filepath (str): Path to the saved object
        
    Returns:
        The loaded Python object or None if loading fails
    """
    try:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None
            
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading object from {filepath}: {e}")
        return None

def plot_episode_stats(stats, save_dir=None, show=True):
    """
    Plot various statistics from Deep Q-Learning training.
    
    Args:
        stats (dict): Dictionary containing statistics lists:
                     - 'episode_rewards': List of total rewards per episode
                     - 'episode_lengths': List of steps per episode
                     - 'episode_losses': List of average losses per episode
                     - 'episode_q_values': List of average Q-values per episode
        save_dir (str): Directory to save plots, or None to not save
        show (bool): If True, display the plots
    """
    # Check if we have data to plot
    has_data = False
    for key in ['episode_rewards', 'episode_lengths', 'episode_losses', 'episode_q_values']:
        if key in stats and len(stats[key]) > 0:
            has_data = True
            break
            
    if not has_data:
        print("No training statistics data to plot")
        return
        
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Deep Q-Learning Training Statistics', fontsize=16)
    
    # Plot episode rewards
    if 'episode_rewards' in stats and len(stats['episode_rewards']) > 0:
        axes[0, 0].plot(stats['episode_rewards'])
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards')
        
        # Add moving average
        if len(stats['episode_rewards']) > 100:
            moving_avg = np.convolve(stats['episode_rewards'], 
                                    np.ones(100)/100, 
                                    mode='valid')
            axes[0, 0].plot(np.arange(100-1, len(stats['episode_rewards'])), 
                           moving_avg, 
                           color='red',
                           label='Moving Average (100 episodes)')
            axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'No reward data', horizontalalignment='center',
                      verticalalignment='center', transform=axes[0, 0].transAxes)
    
    # Plot episode lengths
    if 'episode_lengths' in stats and len(stats['episode_lengths']) > 0:
        axes[0, 1].plot(stats['episode_lengths'])
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Lengths')
    else:
        axes[0, 1].text(0.5, 0.5, 'No episode length data', horizontalalignment='center',
                      verticalalignment='center', transform=axes[0, 1].transAxes)
    
    # Plot episode losses
    if 'episode_losses' in stats and len(stats['episode_losses']) > 0:
        axes[1, 0].plot(stats['episode_losses'])
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
    else:
        axes[1, 0].text(0.5, 0.5, 'No loss data', horizontalalignment='center',
                      verticalalignment='center', transform=axes[1, 0].transAxes)
    
    # Plot episode average Q-values
    if 'episode_q_values' in stats and len(stats['episode_q_values']) > 0:
        axes[1, 1].plot(stats['episode_q_values'])
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Q-Value')
        axes[1, 1].set_title('Q-Value Estimates')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Q-value data', horizontalalignment='center',
                      verticalalignment='center', transform=axes[1, 1].transAxes)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if requested
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'training_stats.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving training statistics plot: {e}")
    
    # Show if requested, otherwise close
    if show:
        plt.show()
    else:
        plt.close()

