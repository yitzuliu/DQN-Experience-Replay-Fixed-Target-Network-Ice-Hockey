# In this file, implement the DQN agent class that will:
# - Initialize Q-network and target network
# - Implement epsilon-greedy action selection
# - Implement learning functionality according to original DQN algorithm:
#   - Sample batches from replay memory
#   - Calculate Q-values and target Q-values using target network
#   - Update network weights using gradient descent
#   - Update target network periodically
# - Provide save/load functionality for the trained model
#
# This implementation follows the original DQN algorithm as described in the pseudocode:
# Initialize replay memory D with capacity N
# Initialize action-value network Q (θ₁) with random weights
# Initialize target network Q_target (θ₂) ← θ₁
#
# For each episode = 1 to M:
#     Initialize initial state S₁
#
#     For t = 1 to T:
#         With probability ε, select a random action Aₜ (exploration)
#         Otherwise, select Aₜ = argmaxₐ Q(Sₜ, a; θ₁) (exploitation)
#
#         Execute action Aₜ, observe reward Rₜ₊₁ and next state Sₜ₊₁
#
#         Store transition (Sₜ, Aₜ, Rₜ₊₁, Sₜ₊₁) into replay buffer D
#
#         Sample a random minibatch of transitions (Sⱼ, Aⱼ, Rⱼ₊₁, Sⱼ₊₁) from D
#
#         For each sample j in the minibatch:
#             If Sⱼ₊₁ is terminal:
#                 yⱼ ← Rⱼ₊₁
#             Else:
#                 yⱼ ← Rⱼ₊₁ + γ * maxₐ' Q_target(Sⱼ₊₁, a'; θ₂)
#
#         Perform gradient descent step to minimize:
#             L = (yⱼ - Q(Sⱼ, Aⱼ; θ₁))²
#
#         Every C steps:
#             Update target network: θ₂ ← θ₁

import numpy as np          # For numerical computations
import random               # For random action selection (ε-greedy)
import torch                # PyTorch deep learning framework
import torch.nn as nn       # Neural network modules
import torch.optim as optim # Optimization algorithms
import os                   # For file operations

from q_network import create_q_network  # Function to create Q-network
from replay_memory import ReplayMemory  # Experience replay memory
import config                          # Configuration parameters

class DQNAgent:
    """
    Deep Q-Network agent for playing Atari Ice Hockey
    Following the original DQN algorithm
    """
    def __init__(self, env, device=None):
        """
        Initialize the DQN agent
        
        Args:
            env: Gymnasium environment
            device: Device to run the model on ("cuda", "mps", or "cpu")
        """
        # Environment and device settings
        self.env = env                        
        
        # Auto-select device if not specified
        if device is None:
            import utils
            self.device = utils.setup_device()
        else:
            self.device = device
            
        self.action_size = env.action_space.n 
        
        # Initialize Q-Network (policy network)
        # This network is used to select actions and is updated through learning
        self.q_network = create_q_network(env).to(self.device)
        
        # Initialize Target Network with same weights
        self.target_network = create_q_network(env).to(self.device)
        self.update_target_network()  # Copy initial weights from Q-Network
        self.target_network.eval()    # Set to evaluation mode (no gradients needed)
        
        # Initialize optimizer using Adam instead of SGD
        # Adam (Adaptive Moment Estimation) is often more efficient for training neural networks
        # It adaptively adjusts the learning rates and incorporates momentum
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LEARNING_RATE)
        
        # Initialize experience replay memory
        if config.MEMORY_IMPLEMENTATION == 'numpy' or config.MEMORY_IMPLEMENTATION == 'optimized':
            # Support for optimized memory implementation
            state_shape = env.observation_space.shape
            self.memory = ReplayMemory(capacity=config.MEMORY_CAPACITY, state_shape=state_shape)
        else:
            self.memory = ReplayMemory(capacity=config.MEMORY_CAPACITY)
        
        # Exploration parameters (ε-greedy)
        self.epsilon = config.EPSILON_START     # Start with high exploration
        self.epsilon_end = config.EPSILON_END   # End with low exploration
        self.epsilon_decay = config.EPSILON_DECAY  # How quickly to decay
        
        # Training parameters
        self.gamma = config.GAMMA                # Discount factor for future rewards
        self.batch_size = config.BATCH_SIZE      # Number of experiences per batch
        self.target_update_frequency = config.TARGET_UPDATE_FREQUENCY  # How often to update target
        self.steps_done = 0                      # Counter for total steps
        
        # Metrics tracking
        self.losses = []
        self.q_values = []
        
        # Gradient accumulation steps counter
        self.accumulation_steps = 0
    
    def select_action(self, state, evaluate=False):
        """
        Select action using epsilon-greedy policy:
        - With probability ε, select random action (exploration)
        - With probability 1-ε, select action with highest Q-value (exploitation)
        
        Args:
            state: Current state observation
            evaluate: If True, use greedy policy (ε=0)
            
        Returns:
            int: Selected action
        """
        # Convert state to PyTorch tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # In evaluation mode, always choose the best action
        epsilon = 0 if evaluate else self.epsilon
        
        # Decide whether to explore or exploit
        if random.random() > epsilon:
            # Exploit: choose action with highest Q-value
            with torch.no_grad():  # No gradients needed for action selection
                self.q_network.eval()  # Set to evaluation mode
                q_values = self.q_network(state)  # Get Q-values for all actions
                self.q_network.train()  # Set back to training mode
                
                # Record Q-values for monitoring
                if not evaluate:
                    self.q_values.append(q_values.max().item())
                
                # Return action with highest Q-value
                return torch.argmax(q_values).item()
        else:
            # Explore: choose random action
            return random.randrange(self.action_size)
    
    def update_epsilon(self):
        """
        Update epsilon value according to linear decay schedule
        """
        # Linear decay from EPSILON_START to EPSILON_END over EPSILON_DECAY steps
        self.epsilon = max(
            self.epsilon_end, 
            self.epsilon - (self.epsilon - self.epsilon_end) / self.epsilon_decay
        )
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store transition in replay memory
        
        Args:
            state: Current state
            action: Chosen action
            reward: Received reward
            next_state: Next state
            done: Whether episode terminated
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def learn(self, reset_grads=True):
        """
        Learn from a batch of experiences using Q-learning with target network
        
        This method implements:
        - Sampling from replay memory [Pseudocode step 11]
        - Computing target Q-values [Pseudocode step 12]
        - Updating network via gradient descent [Pseudocode step 13]
        - Updating target network periodically [Pseudocode step 14]
        Args:
            reset_grads: Whether to reset gradients (True for first accumulation step)
        Returns:
            float: Loss value or None if not enough samples
        """
        # Check if enough samples in memory
        if not self.memory.can_sample(self.batch_size):
            return None
        
        # 1. Sample random minibatch from replay memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch tensors and move to device
        if self.device.type == 'cuda':
            states = states.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            rewards = rewards.to(self.device, non_blocking=True)
            next_states = next_states.to(self.device, non_blocking=True)
            dones = dones.to(self.device, non_blocking=True)
        else:
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
        
        # 2. Compute current Q-values: Q(Sⱼ, Aⱼ; θ₁)
        current_q_values = self.q_network(states).gather(1, actions)
        
        # 3. Compute target Q-values
        with torch.no_grad():  # Target computation doesn't need gradients
            # Get maximum Q-values for next states using target network
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            # If next state is terminal (done=1): target = reward
            # Else: target = reward + γ * max_a' Q_target(next_state, a')
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # 4. Compute loss: L = (yⱼ - Q(Sⱼ, Aⱼ; θ₁))²
        loss = nn.functional.mse_loss(current_q_values, target_q_values)
        
        # 5. Perform gradient descent with accumulation
        # Only reset gradients on the first accumulation step
        if reset_grads:
            self.optimizer.zero_grad()
            
        # Scale loss to maintain mathematical equivalence
        # If using N batches, each batch's gradient should be scaled to 1/N
        scaled_loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        
        # Backpropagate to compute gradients (but do not update yet)
        scaled_loss.backward()
        
        # Record accumulation steps
        self.accumulation_steps += 1
        
        # 6. Update weights only when accumulation steps are reached
        if self.accumulation_steps >= config.GRADIENT_ACCUMULATION_STEPS:
            # Gradient clipping (if not None)
            for param in self.q_network.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            
            # Update parameters
            self.optimizer.step()
            self.optimizer.zero_grad()  # Reset gradients
            self.accumulation_steps = 0  # Reset counter
            
            # Increment total steps counter
            self.steps_done += 1
            
            # 7. Every C steps, update target network: θ₂ ← θ₁
            if self.steps_done % self.target_update_frequency == 0:
                self.update_target_network()
        
        # Record loss for monitoring (used unscaled loss)
        self.losses.append(loss.item())
        
        return loss.item()
    
    def update_target_network(self):
        """
        Update target network by copying parameters from Q-network: θ₂ ← θ₁
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, path="models/dqn_icehockey.pth"):
        """
        Save model weights to disk
        
        Args:
            path: Path where to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path="models/dqn_icehockey.pth"):
        """
        Load model weights from disk
        
        Args:
            path: Path to the saved model
        """
        if not os.path.exists(path):
            print(f"No model found at {path}")
            return False
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        
        # Set target network to evaluation mode
        self.target_network.eval()
        
        print(f"Model loaded from {path}")
        return True
    
    def get_statistics(self):
        """
        Get training statistics
        
        Returns:
            dict: Dictionary with training statistics
        """
        return {
            'steps': self.steps_done,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_q_value': np.mean(self.q_values[-100:]) if self.q_values else 0
        }


# Test code for this module
if __name__ == "__main__":
    import env_wrappers
    
    # Create environment
    env = env_wrappers.make_env()
    print(f"Environment: {config.ENV_NAME}")
    
    # Create agent
    agent = DQNAgent(env)
    print(f"DQN Agent created on device: {agent.device}")
    
    # Test action selection
    state, _ = env.reset()
    action = agent.select_action(state)
    print(f"Selected action: {action}")
    
    # Test storing experience
    next_state, reward, terminated, truncated, _ = env.step(action)
    agent.store_experience(state, action, reward, next_state, terminated)
    print(f"Experience stored, memory size: {len(agent.memory)}")
    
    # Add more experiences for testing learning
    for _ in range(agent.batch_size - 1):
        action = agent.select_action(next_state)
        state = next_state
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.store_experience(state, action, reward, next_state, terminated or truncated)
        if terminated or truncated:
            state, _ = env.reset()
    
    # Test learning
    loss = agent.learn()
    print(f"Learning step performed, loss: {loss}")
    
    # Test saving and loading
    agent.save_model("models/test_model.pth")
    agent.load_model("models/test_model.pth")
    
    # Clean up
    env.close()
    
    print("DQN Agent test completed successfully!")
