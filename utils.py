import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
import platform  # For detecting operating system
import multiprocessing as mp  # For getting CPU core count
import time  # For timestamp creation

def get_device(force_cpu=False):
    """
    Determine the optimal device for training (CPU or GPU).
    
    This function detects available hardware acceleration:
    - CUDA for NVIDIA GPUs (with multi-GPU awareness)
    - MPS for Apple Silicon (M1/M2/M3) Macs
    - Falls back to CPU if no acceleration is available
    
    Args:
        force_cpu (bool): If True, will always return CPU even if GPU is available
        
    Returns:
        torch.device: The device to use for tensor operations
    """
    if force_cpu:
        return torch.device("cpu")
    
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            # Use the first available GPU, leaving at least one free
            device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
            print(f"Using CUDA GPU: {gpu_name} ({gpu_mem:.2f} GB), leaving {num_gpus - 1} GPU(s) free")
        else:
            print("Only one GPU available, using it fully")
            device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        return device
    
    # Check for MPS (Apple Silicon M1/M2/M3)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and platform.processor() == 'arm':
        device = torch.device("mps")
        print("Using M-series Mac GPU (Metal)")
        return device
    
    # Fall back to CPU
    else:
        # Leave at least one CPU core free
        num_cores = os.cpu_count()
        if num_cores > 1:
            torch.set_num_threads(num_cores - 1)
            print(f"Using CPU: {num_cores - 1} cores (1 core left free)")
        else:
            torch.set_num_threads(1)
            print("Using CPU: 1 core (no cores left free)")
        return torch.device("cpu")

def get_system_info():
    """
    Get information about the system for logging and optimization decisions.
    
    Returns:
        dict: Dictionary containing system information
    """
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu_count": mp.cpu_count(),
        "cpu_type": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    # Add NVIDIA GPU details if available
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
        
        # Add GPU memory information
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
        info["gpu_memory_gb"] = f"{total_memory:.2f}"
    
    # Add Apple Silicon (MPS) details if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps_available"] = True
        info["gpu_type"] = "Apple Silicon"
        if platform.processor() == 'arm':
            info["apple_silicon"] = True
    
    # Add number of CPU threads being used
    info["cpu_threads_used"] = torch.get_num_threads()
    
    return info

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
    plt.figure(figsize=(10, 6))
    
    # Down-sample if too many points (for efficiency)
    if len(values) > max_points:
        skip = len(values) // max_points
        indices = np.arange(0, len(values), skip)
        values = [values[i] for i in indices]
        print(f"Down-sampled from {len(values)*skip} to {len(values)} points for plotting efficiency")
    
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_object(filepath):
    """
    Load a Python object from disk using pickle.
    
    Args:
        filepath (str): Path to the saved object
        
    Returns:
        The loaded Python object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def plot_episode_stats(stats, save_dir=None, show=True):
    """
    Plot various statistics from training.
    
    Args:
        stats (dict): Dictionary containing statistics lists:
                     - 'episode_rewards': List of total rewards per episode
                     - 'episode_lengths': List of steps per episode
                     - 'episode_losses': List of average losses per episode
                     - 'episode_q_values': List of average Q-values per episode
        save_dir (str): Directory to save plots, or None to not save
        show (bool): If True, display the plots
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Statistics', fontsize=16)
    
    # Plot episode rewards
    if 'episode_rewards' in stats:
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
    
    # Plot episode lengths
    if 'episode_lengths' in stats:
        axes[0, 1].plot(stats['episode_lengths'])
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Lengths')
    
    # Plot episode losses
    if 'episode_losses' in stats:
        axes[1, 0].plot(stats['episode_losses'])
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
    
    # Plot episode average Q-values
    if 'episode_q_values' in stats:
        axes[1, 1].plot(stats['episode_q_values'])
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Q-Value')
        axes[1, 1].set_title('Q-Value Estimates')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if requested
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_stats.png'), 
                   dpi=300, 
                   bbox_inches='tight')
    
    # Show if requested, otherwise close
    if show:
        plt.show()
    else:
        plt.close()

