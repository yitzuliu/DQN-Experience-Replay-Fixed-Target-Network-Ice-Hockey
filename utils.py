import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
import platform  # For detecting operating system
import time
import gc  # Garbage collection for memory management
import psutil


def get_device():
    """
    Determine if GPU is available and return appropriate device.
    This function checks for NVIDIA GPU (CUDA), Apple Silicon GPU (MPS),
    or defaults to CPU if no GPU is available.
    
    Returns:
        torch.device: CPU or GPU device for tensor operations
    """
    # Check for NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        try:
            # Test CUDA availability with a small tensor operation
            test_tensor = torch.tensor([1.0, 2.0], device='cuda')
            test_result = test_tensor + test_tensor
            device = torch.device("cuda")
            print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
            # Free memory after test
            del test_tensor
            del test_result
            torch.cuda.empty_cache()
            return device
        except Exception as e:
            print(f"CUDA reported as available but encountered error: {e}")
            print("Falling back to CPU")
            return torch.device("cpu")
    
    # Check for Apple Silicon (M1/M2/M3) with MPS acceleration
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Test MPS availability with a small tensor operation
            test_tensor = torch.tensor([1.0, 2.0], device='mps')
            test_result = test_tensor + test_tensor
            device = torch.device("mps")
            print("Using Apple Silicon GPU (M1/M2/M3)")
            # Free memory after test
            del test_tensor
            del test_result
            return device
        except Exception as e:
            print(f"MPS reported as available but encountered error: {e}")
            print("Falling back to CPU")
            return torch.device("cpu")
    
    # If no GPU is detected, use CPU
    else:
        print("No GPU detected, using CPU")
        return torch.device("cpu")

def get_system_info():
    """
    Get system information for logging and optimization.
    
    Returns:
        dict: Dictionary with system information
    """
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    # CPU information
    info["cpu_count"] = os.cpu_count()
    info["cpu_type"] = platform.processor() or platform.machine()
    
    # NVIDIA GPU information
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
        
        # GPU memory information
        if hasattr(torch.cuda, 'get_device_properties'):
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
            info["gpu_memory_gb"] = f"{total_memory:.2f}"
    
    # Apple Silicon (MPS) information
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps_available"] = True
        # Check if it's Apple Silicon (arm architecture)
        if platform.machine().startswith('arm'):
            info["apple_silicon"] = True
            info["gpu_type"] = "Apple Silicon"
            # Try to detect model (M1/M2/M3)
            try:
                import subprocess
                result = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
                if "Apple" in result:
                    info["apple_chip"] = result
            except:
                pass
    
    # Memory information
    try:
        vm = psutil.virtual_memory()
        info["total_memory_gb"] = f"{vm.total / (1024**3):.2f}"
        info["available_memory_gb"] = f"{vm.available / (1024**3):.2f}"
        info["memory_percent_used"] = f"{vm.percent:.1f}"
    except ImportError:
        # psutil not installed, provide basic info
        info["memory_info_available"] = False
    
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
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    
    # Ensure directories exist
    for directory in [run_dir, model_dir, log_dir, viz_dir, checkpoint_dir]:
        os.makedirs(directory, exist_ok=True)
    
    paths = {
        "run": run_dir,
        "models": model_dir,
        "logs": log_dir,
        "viz": viz_dir,
        "checkpoints": checkpoint_dir
    }
    
    return paths

def clean_memory():
    """
    Clean up memory to prevent memory leaks during long training runs.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def memory_stats():
    """
    Get current memory usage statistics.
    
    Returns:
        dict: Dictionary with memory statistics
    """
    stats = {}
    
    try:
        import psutil
        # System memory
        vm = psutil.virtual_memory()
        stats["system_total_gb"] = vm.total / (1024**3)
        stats["system_available_gb"] = vm.available / (1024**3)
        stats["system_used_percent"] = vm.percent
        
        # Process memory
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        stats["process_rss_mb"] = mem_info.rss / (1024**2)  # Resident Set Size
        stats["process_vms_mb"] = mem_info.vms / (1024**2)  # Virtual Memory Size
        
    except ImportError:
        stats["error"] = "psutil not available"
    
    # GPU memory if available
    if torch.cuda.is_available():
        try:
            stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
            stats["gpu_cached_mb"] = torch.cuda.memory_reserved() / (1024**2)
            stats["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
        except:
            stats["error_gpu"] = "Could not get GPU memory stats"
            
    return stats

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

