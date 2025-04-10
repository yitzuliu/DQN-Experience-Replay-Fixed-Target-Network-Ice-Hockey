"""
DQN Agent Implementation

This module implements the Deep Q-Network (DQN) agent that learns to play Atari Ice Hockey.
The agent follows the algorithm introduced by Mnih et al. (2015) in "Human-level control
through deep reinforcement learning" with the following key components:

1. Deep Q-Network and Target Network
2. Experience Replay Memory
3. ε-greedy Action Selection
4. Bellman Equation-based Q-learning

Each component is carefully implemented with educational comments to help understand
the reinforcement learning process.
"""

import numpy as np          # For numerical computations
import random               # For random action selection (ε-greedy)
import torch                # PyTorch deep learning framework
import torch.nn as nn       # Neural network modules
import torch.optim as optim # Optimization algorithms
import os                   # For file operations
import math                 # For mathematical operations

# Import project modules
from q_network import create_q_network  # Function to create Q-network
import config               # Configuration parameters
import utils                # Utility functions


class DQNAgent:
    """
    Deep Q-Network (DQN) Agent
    
    The DQN Agent combines deep learning and reinforcement learning to learn an optimal
    policy for playing Atari games. It uses a deep neural network to approximate the
    Q-function, which estimates the expected future rewards for actions in each state.
    
    Key methods:
    - select_action: Choose actions based on ε-greedy policy
    - store_transition: Store experiences in replay memory
    - learn: Update Q-network based on sampled experiences
    - update_target_network: Periodically update target network with Q-network weights
    """
    
    def __init__(self, state_shape, n_actions, memory, device=None):
        """
        Initialize the DQN Agent with networks, replay memory, and parameters.
        
        Args:
            state_shape (tuple): Shape of state observations (C, H, W)
            n_actions (int): Number of possible actions
            memory: Experience replay memory instance
            device (torch.device, optional): Device to run on (auto-detected if None)
        """
        # Detect device (CPU/GPU) if not provided
        self.device = device if device is not None else utils.get_device()
        
        # Store action space size
        self.n_actions = n_actions
        
        # Create networks
        # Initialize action-value network Q (θ₁) with random weights - PSEUDOCODE LINE 2
        self.q_network = create_q_network(state_shape, n_actions, self.device)
        
        # Initialize target network Q_target (θ₂) ← θ₁ - PSEUDOCODE LINE 3
        self.target_network = create_q_network(state_shape, n_actions, self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Copy weights from Q-network
        self.target_network.eval()  # Set to evaluation mode (no gradient updates)
        
        # Setup optimizer - Adam with learning rate from config
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=config.LEARNING_RATE
        )
        
        # Setup loss function - Huber Loss (more robust than MSE)
        # Huber loss combines MSE for small errors and MAE for large errors
        # This makes it less sensitive to outliers in the value estimates
        self.loss_fn = nn.SmoothL1Loss()  
        
        # Store replay memory
        self.memory = memory
        
        # Initialize epsilon for exploration-exploitation tradeoff
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        
        # Initialize step counter for updating target network
        self.steps_done = 0
        
        # Track training metrics
        self.losses = []
        self.avg_q_values = []
        
        # Enable cuDNN benchmarking for faster convolutions if using CUDA
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
    
    def select_action(self, state, evaluate=None):
        """
        Select action using ε-greedy policy.
        
        This is a core component of the DQN algorithm that balances exploration and exploitation:
        - With probability ε, select a random action for exploration
        - With probability 1-ε, select the action with highest Q-value for exploitation
        
        Args:
            state (torch.Tensor): Current state observation
            evaluate (bool, optional): Special mode for evaluation only. When True, forces pure exploitation
                                       by setting ε=0. If None, uses the value from config.DEFAULT_EVALUATE_MODE.
                                       Default is None.
            
        Returns:
            int: Selected action
        """
        # Use config default if evaluate is None
        if evaluate is None:
            evaluate = config.DEFAULT_EVALUATE_MODE
        
        # Decay epsilon value based on steps
        # Modified decay strategy: Using cosine decay instead of linear decay
        if not evaluate:
            self.steps_done += 1
            # Exponential decay formula: Maintains high exploration rate early, gradually decreases
            # self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            #               math.exp(-1.0 * self.steps_done / self.epsilon_decay)
            
            # Alternative: Using cosine decay
            progress = min(1.0, self.steps_done / self.epsilon_decay)
            self.epsilon = self.epsilon_end + 0.5 * (self.epsilon_start - self.epsilon_end) * \
                         (1 + math.cos(progress * math.pi))
        
        # Use greedy policy for evaluation
        if evaluate:
            epsilon = 0.0
        else:
            epsilon = self.epsilon
        
        # With probability ε, select a random action (exploration) - PSEUDOCODE LINE 7
        if random.random() < epsilon:
            action = random.randrange(self.n_actions)
            return action
        
        # Otherwise, select action with highest Q-value (exploitation) - PSEUDOCODE LINE 8
        else:
            # Convert state to proper format and move to device
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state = state.unsqueeze(0).to(self.device)
            
            # Set network to evaluation mode and disable gradients for inference
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state)
                action = q_values.max(1)[1].item()  # argmax of Q-values
                
                # Track average Q-value for monitoring
                if not evaluate and len(self.avg_q_values) < 10000:
                    self.avg_q_values.append(q_values.mean().item())
            
            # Set network back to training mode
            self.q_network.train()
            return action
        
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay memory.
        
        This implements PSEUDOCODE LINE 10:
        "Store transition (Sₜ, Aₜ, Rₜ₊₁, Sₜ₊₁) into replay buffer D"
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether the episode ended (True/False)
        """
        # Convert boolean done to integer (1 for done, 0 for not done)
        done_int = 1 if done else 0
        
        # Store transition in replay memory
        self.memory.add(state, action, reward, next_state, done_int)
    
    def learn(self):
        """
        Update Q-network using sampled batch from replay memory.
        
        This implements PSEUDOCODE LINES 11-13:
        - Sample a random minibatch of transitions
        - Calculate target Q-values using Bellman equation
        - Perform gradient descent to minimize the loss
        
        Returns:
            float: Loss value for monitoring
            
        Raises:
            KeyboardInterrupt: Re-raises KeyboardInterrupt for proper training termination
        """
        # Check if enough samples in memory for learning
        if not self.memory.can_sample(config.BATCH_SIZE):
            return None
        
        try:
            # Sample random minibatch of transitions from replay memory - PSEUDOCODE LINE 11
            if self.device.type == 'cuda':
                # Use pinned memory for faster CPU to GPU transfer
                states, actions, rewards, next_states, dones = self.memory.sample_pinned(config.BATCH_SIZE)
            else:
                # Regular sampling for CPU
                states, actions, rewards, next_states, dones = self.memory.sample(config.BATCH_SIZE)
            
            # Check if sampling was interrupted (this should not happen with new KeyboardInterrupt handling)
            if states is None:
                return None
                
            # Move tensors to the correct device
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            
            # Compute current Q-values
            q_values = self.q_network(states)
            q_values = q_values.gather(1, actions)
            
            # Compute next state Q-values using target network: max_a' Q_target(s', a')
            with torch.no_grad():  # No need for gradients for target values
                # Use target network to compute next state Q-values
                # This stabilizes learning by separating prediction from the target
                next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
                
                # Compute target Q values using Bellman equation - PSEUDOCODE LINE 12
                # If next state is terminal (done): yⱼ ← Rⱼ₊₁
                # Else: yⱼ ← Rⱼ₊₁ + γ * maxₐ' Q_target(Sⱼ₊₁, a'; θ₂)
                target_q_values = rewards + (1 - dones) * config.GAMMA * next_q_values
            
            # Compute loss - how far current Q-values are from target Q-values
            # PSEUDOCODE LINE 13: L = (yⱼ - Q(Sⱼ, Aⱼ; θ₁))²
            loss = self.loss_fn(q_values, target_q_values)
            
            # Perform gradient descent step - PSEUDOCODE LINE 13 (continued)
            self.optimizer.zero_grad()  # Clear previous gradients
            
            # Compute gradients
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
            
            # Update network weights
            self.optimizer.step()
            
            # Track loss for monitoring
            loss_value = loss.item()
            self.losses.append(loss_value)
            
            return loss_value
        
        except KeyboardInterrupt:
            # Re-raise the KeyboardInterrupt to propagate it properly to the training loop
            raise
        except Exception as e:
            print(f"\nError during learning: {e}")
            return None
    
    def update_target_network(self):
        """
        Update target network with Q-network weights.
        
        This implements PSEUDOCODE LINE 14:
        "Every C steps: Update target network: θ₂ ← θ₁"
        
        This is done periodically to stabilize training, as it prevents the target from
        constantly changing while the Q-network is being updated.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
        return True
    
    def save_model(self, filepath):
        """
        Save trained model weights to disk.
        
        Args:
            filepath (str): Path to save the model
            
        Returns:
            bool: True if saving was successful
        """
        # Fix the directory issue - ensure filepath has directory component
        if os.path.dirname(filepath) == '':
            # No directory in path, save in current directory
            directory = './models'
            if not os.path.exists(directory):
                os.makedirs(directory)
            filepath = os.path.join(directory, os.path.basename(filepath))
        else:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # When statistics become large, limit the number saved
        if len(self.avg_q_values) > 10000:
            self.avg_q_values = self.avg_q_values[-10000:]
        if len(self.losses) > 10000:
            self.losses = self.losses[-10000:]
        
        # Save model with useful information
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'avg_q_values': self.avg_q_values,
            'losses': self.losses
        }
        
        # Use PyTorch's recommended way to save models
        torch.save(checkpoint, filepath)
        return True
    
    def load_model(self, filepath):
        """
        Load trained model weights from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            bool: True if loading was successful
        """
        if not os.path.exists(filepath):
            return False
        
        # Load checkpoint to appropriate device
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load parameters and states
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training statistics
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.steps_done = checkpoint.get('steps_done', 0)
        self.avg_q_values = checkpoint.get('avg_q_values', [])
        self.losses = checkpoint.get('losses', [])
        
        return True
    
    def get_statistics(self):
        """
        Get agent's training statistics.
        
        Returns:
            dict: Dictionary with training statistics
        """
        stats = {
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'avg_q_value': np.mean(self.avg_q_values[-1000:]) if self.avg_q_values else 0,
            'avg_loss': np.mean(self.losses[-1000:]) if self.losses else 0,
        }
        return stats
    
    @property
    def epsilon_start(self):
        """Return the starting epsilon value from config"""
        return config.EPSILON_START


# Testing code
if __name__ == "__main__":
    import time
    from replay_memory import OptimizedArrayReplayMemory
    
    # Display system information
    system_info = utils.get_system_info()
    print("System Information:")
    print(f"OS: {system_info['os']} with PyTorch {system_info['torch_version']}")
    
    if system_info.get('cuda_available', False):
        print(f"GPU: {system_info.get('gpu_name', 'Unknown')} "
              f"({system_info.get('gpu_memory_gb', 'Unknown')} GB)")
    elif system_info.get('mps_available', False):
        print("GPU: Apple Silicon (Metal)")
    else:
        print("No GPU detected, using CPU only")
    
    # Create sample state and replay memory
    state_shape = (config.FRAME_STACK, config.FRAME_HEIGHT, config.FRAME_WIDTH)
    memory = OptimizedArrayReplayMemory(
        capacity=10000,
        state_shape=state_shape
    )
    
    # Create DQN agent
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=config.ACTION_SPACE_SIZE,
        memory=memory
    )
    
    # Test action selection
    print("\nTesting action selection...")
    test_state = np.random.rand(*state_shape).astype(np.float32)
    
    # Test exploratory action
    agent.epsilon = 1.0  # Force exploration
    action = agent.select_action(test_state)
    print(f"Exploratory action (ε=1.0): {action}")
    
    # Test exploitative action
    agent.epsilon = 0.0  # Force exploitation
    action = agent.select_action(test_state)
    print(f"Exploitative action (ε=0.0): {action}")
    
    # Test experience storage and learning
    print("\nTesting experience storage and learning...")
    
    # Fill memory with some random transitions
    for i in range(config.BATCH_SIZE * 2):
        state = np.random.rand(*state_shape).astype(np.float32)
        action = random.randint(0, config.ACTION_SPACE_SIZE - 1)
        reward = random.uniform(-1, 1)
        next_state = np.random.rand(*state_shape).astype(np.float32)
        done = random.random() > 0.8
        
        agent.store_transition(state, action, reward, next_state, done)
    
    # Test learning step
    start_time = time.time()
    loss = agent.learn()
    end_time = time.time()
    
    print(f"Learning step completed in {(end_time - start_time)*1000:.2f}ms with loss: {loss:.6f}")
    
    # Test target network update
    start_time = time.time()
    agent.update_target_network()
    end_time = time.time()
    
    print(f"Target network updated in {(end_time - start_time)*1000:.2f}ms")
    
    # Test model saving and loading
    test_save_path = "models/test_model.pth"
    os.makedirs(os.path.dirname(test_save_path), exist_ok=True)
    agent.save_model(test_save_path)
    agent.load_model(test_save_path)
    
    # Clean up test file
    if os.path.exists(test_save_path):
        os.remove(test_save_path)
    
    print("\nAll DQN agent tests completed successfully!")
