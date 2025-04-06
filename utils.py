import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
import platform  # 用於檢測操作系統
import multiprocessing as mp  # 用於獲取CPU核心數

def setup_device():
    """
    設置並回傳最適合當前環境的計算設備
    
    自動檢測並選擇:
    - CUDA GPU (如果可用)
    - Mac M系列晶片的 Metal Performance Shaders (MPS)
    - CPU (作為後備選項)
    
    Returns:
        torch.device: 計算設備
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 獲取GPU信息
        gpu_name = torch.cuda.get_device_name(0)
        print(f"使用 CUDA GPU: {gpu_name}")
        
        # 設置CUDA參數以優化性能
        torch.backends.cudnn.benchmark = True
        
        return device
    else:
        # 檢查是否為M系列Mac
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and platform.processor() == 'arm':
                device = torch.device("mps")
                print("使用 M系列 Mac GPU (Metal)")
                return device
        except:
            pass
        
        # 使用CPU
        cpu_count = mp.cpu_count()
        print(f"使用 CPU: {cpu_count} 核心")
        
        # 設置線程數以更好地利用CPU
        torch.set_num_threads(cpu_count)
        return torch.device("cpu")

def plot_training_metrics(stats, save_dir="plots"):
    """
    Plot training metrics from stats dictionary and save to files
    
    Args:
        stats (dict): Dictionary containing training statistics
        save_dir (str): Directory to save the plots
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # ===== Plot 1: Rewards =====
    plt.figure(figsize=(12, 6))
    
    # Plot episode rewards
    plt.plot(stats['episode_rewards'], alpha=0.6, label='Episode Rewards')
    
    # Plot smoothed rewards if we have enough data
    if len(stats['episode_rewards']) > 10:
        window_size = min(100, len(stats['episode_rewards']) // 5)
        smoothed_rewards = np.convolve(stats['episode_rewards'], 
                                       np.ones(window_size)/window_size, 
                                       mode='valid')
        plt.plot(range(window_size-1, window_size-1+len(smoothed_rewards)), 
                 smoothed_rewards, 
                 label=f'Smoothed Rewards (window={window_size})')
    
    # Add evaluation rewards if they exist
    if 'eval_rewards' in stats and stats['eval_rewards']:
        eval_episodes, eval_rewards = zip(*stats['eval_rewards'])
        plt.scatter(eval_episodes, eval_rewards, color='red', 
                    marker='o', label='Evaluation Rewards')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training and Evaluation Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'rewards.png'))
    
    # ===== Plot 2: Loss and Q-values =====
    if 'episode_losses' in stats and stats['episode_losses']:
        plt.figure(figsize=(12, 6))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(stats['episode_losses'], label='Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        
        # Plot epsilon decay
        plt.subplot(1, 2, 2)
        plt.plot(stats['epsilons'], label='Epsilon')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'learning_metrics.png'))
    
    plt.close('all')

def load_training_stats(path="models/training_stats.pkl"):
    """
    Load training statistics from file
    
    Args:
        path (str): Path to the statistics file
        
    Returns:
        dict: Dictionary containing training statistics or None if file doesn't exist
    """
    if not os.path.exists(path):
        print(f"No statistics file found at {path}")
        return None
    
    with open(path, 'rb') as f:
        stats = pickle.load(f)
    
    print(f"Statistics loaded from {path}")
    return stats

def save_training_stats(stats, path="models/training_stats.pkl"):
    """
    Save training statistics to file
    
    Args:
        stats (dict): Dictionary containing training statistics
        path (str): Path to save the statistics file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"Statistics saved to {path}")

def print_training_summary(stats):
    """
    Print a summary of training statistics
    
    Args:
        stats (dict): Dictionary containing training statistics
    """
    if not stats:
        print("No statistics available")
        return
    
    # Get number of episodes
    num_episodes = len(stats['episode_rewards'])
    
    # Get average reward of last 100 episodes
    avg_reward_last_100 = np.mean(stats['episode_rewards'][-100:]) if num_episodes >= 100 else np.mean(stats['episode_rewards'])
    
    # Get average loss of last 100 episodes
    avg_loss_last_100 = np.mean(stats['episode_losses'][-100:]) if 'episode_losses' in stats and len(stats['episode_losses']) >= 100 else "N/A"
    
    # Get latest epsilon
    latest_epsilon = stats['epsilons'][-1] if 'epsilons' in stats and stats['epsilons'] else "N/A"
    
    # Get evaluation results
    latest_eval = stats['eval_rewards'][-1][1] if 'eval_rewards' in stats and stats['eval_rewards'] else "N/A"
    
    print("\n===== Training Summary =====")
    print(f"Number of episodes: {num_episodes}")
    print(f"Average reward (last 100 episodes): {avg_reward_last_100:.2f}")
    print(f"Average loss (last 100 episodes): {avg_loss_last_100}")
    print(f"Current epsilon: {latest_epsilon}")
    print(f"Latest evaluation reward: {latest_eval}")
    print("============================\n")

def visualize_model_action(state, agent, device):
    """
    Visualize the model's action preference for a given state
    
    Args:
        state: Environment state
        agent: DQN agent
        device: Device to use for computation
    """
    # Convert state to tensor
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    # Get Q-values
    with torch.no_grad():
        agent.q_network.eval()
        q_values = agent.q_network(state_tensor)
        agent.q_network.train()
    
    # Convert to numpy for plotting
    q_values = q_values.cpu().numpy()[0]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(q_values)), q_values)
    plt.xlabel('Action')
    plt.ylabel('Q-value')
    plt.title('Q-values for each action')
    plt.xticks(range(len(q_values)))
    plt.grid(True, alpha=0.3)
    plt.show()

# Test code for this module
if __name__ == "__main__":
    # Create sample stats for testing
    sample_stats = {
        'episode_rewards': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
        'episode_losses': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        'epsilons': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        'eval_rewards': [(5, 30), (10, 60)]
    }
    
    # Test plotting
    plot_training_metrics(sample_stats, "test_plots")
    
    # Test saving and loading
    save_training_stats(sample_stats, "test_stats.pkl")
    loaded_stats = load_training_stats("test_stats.pkl")
    
    # Test summary printing
    print_training_summary(loaded_stats)
    
    # Clean up test files
    os.remove("test_stats.pkl")
    print("Utils test completed successfully!")
