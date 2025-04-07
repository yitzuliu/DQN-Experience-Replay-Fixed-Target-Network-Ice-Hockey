# In this file, implement the experience replay memory for DQN.

# ===== Key Steps of Experience Replay in DQN =====
# 1. Initialization:
#    - Create a memory buffer with fixed capacity to store experiences
#    - Each experience is a (state, action, reward, next_state, done) tuple
#
# 2. Collecting Experiences:
#    - As the agent interacts with the environment, it records its experiences
#    - New experiences are added to the buffer, replacing oldest ones when full
#
# 3. Batch Sampling:
#    - Randomly sample a batch of experiences from the buffer for learning
#    - Randomness breaks correlation between sequential experiences
#    - Convert sampled experiences to tensor format for neural network processing
#
# 4. Integration with Training Loop:
#    - Add experiences to the buffer after each environment interaction
#    - Once buffer contains sufficient experiences, begin sampling for training
#    - Reuse past experiences multiple times, improving data efficiency
#    - Random sampling prevents the agent from overfitting to recent experiences

# Comparison of different data structures for implementing experience replay:
#
# 1. list:
#    - Pros: Simple to use, fast random access, built-in in Python
#    - Cons: Slow insertion/deletion at the beginning O(n)
#    - Suitable: When sample size is not too large and operations are mainly random sampling
#
# 2. numpy array:
#    - Pros: Memory efficient, fast vectorized operations
#    - Cons: Fixed size, requires manual implementation of circular buffer
#    - Suitable: When memory efficiency and computation speed are critical

# This implementation provides two options: standard Python list and NumPy arrays

import random      # For random sampling
import numpy as np  # For array operations
import torch       # PyTorch library for tensor operations
import config      # Configuration file for hyperparameters

class ListReplayMemory:
    """
    Experience Replay Memory using a simple Python list.
    """
    def __init__(self, capacity, state_shape=None, action_dim=1):
        """
        Initialize the replay memory with a fixed capacity.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.memory = []  # Use standard Python list
        self.capacity = capacity  # Maximum capacity
        self.position = 0  # Current position for circular buffer implementation
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory or replace old one if full.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether the episode ended (True/False or 1/0)
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from memory.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.memory, batch_size)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for experience in batch:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Return the current size of internal memory.
        
        Returns:
            int: Current number of experiences in memory
        """
        return len(self.memory)
    
    def can_sample(self, batch_size):
        """
        Check if enough samples exist to sample a batch of given size.
        
        Args:
            batch_size (int): Desired batch size
        Returns:
            bool: True if enough samples exist, False otherwise
        """
        return len(self) >= batch_size


class ArrayReplayMemory:
    """
    Experience Replay Memory using NumPy arrays for efficiency.
    """
    def __init__(self, capacity, state_shape, action_dim=1):
        """
        Initialize the array-based replay memory.
        
        Args:
            capacity (int): Maximum number of transitions to store
            state_shape (tuple): Shape of state observations
            action_dim (int): Dimension of action space (usually 1)
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.counter = 0  # Total number of experiences added
        self.position = 0  # Current position in the circular buffer
        
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether the episode ended (True/False or 1/0)
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.counter = min(self.counter + 1, self.capacity)
        
    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        indices = np.random.choice(self.counter, batch_size, replace=False)
        
        states = torch.FloatTensor(self.states[indices])
        actions = torch.LongTensor(self.actions[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        dones = torch.FloatTensor(self.dones[indices])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Return current size of memory.
        
        Returns:
            int: Current number of experiences in memory
        """
        return self.counter
    
    def can_sample(self, batch_size):
        """
        Check if enough samples exist.
        
        Args:
            batch_size (int): Desired batch size
            
        Returns:
            bool: True if enough samples exist, False otherwise
        """
        return len(self) >= batch_size


# Add a switch variable at the bottom to decide which implementation to use

# You can modify this variable to choose which implementation to use:
# - 'list': Simple list-based implementation (good for understanding)
# - 'numpy': NumPy array-based implementation (more efficient for large-scale training)
MEMORY_IMPLEMENTATION = config.MEMORY_IMPLEMENTATION  # Using NumPy for better performance

# Set ReplayMemory class based on the chosen implementation
if MEMORY_IMPLEMENTATION == 'list':
    ReplayMemory = ListReplayMemory
elif MEMORY_IMPLEMENTATION == 'numpy':
    ReplayMemory = ArrayReplayMemory
else:
    raise ValueError(f"Unknown implementation: {MEMORY_IMPLEMENTATION}. Use 'list' or 'numpy'.")


# Test code for this module - will only run when this file is executed directly
if __name__ == "__main__":
    # Create a replay memory with capacity 1,000,000
    if MEMORY_IMPLEMENTATION == 'list':
        memory = ReplayMemory(capacity=1000000)
    else:  # numpy implementation needs state shape
        STATE_SHAPE = (210, 160, 3)  # Example Atari frame size
        memory = ReplayMemory(capacity=100, state_shape=STATE_SHAPE)
    
    # Add some dummy experiences to the memory for testing
    for i in range(10):
        state = np.random.rand(210, 160, 3)  # Example: raw Atari frame size
        action = np.random.randint(0, 18)    # Example: action space for Ice Hockey
        reward = np.random.uniform(-1, 1)    # Example: random reward between -1 and 1
        next_state = np.random.rand(210, 160, 3)  # Next state has same shape as state
        done = np.random.choice([0, 1], p=[0.9, 0.1])  # Episode end probability 10%
        
        memory.add(state, action, reward, next_state, done)
    
    # Print the current size of the memory
    print(f"Memory size: {len(memory)}")
    
    # Print which implementation is being used
    print(f"Memory implementation used: {MEMORY_IMPLEMENTATION}")
    
    # Test the sampling functionality
    if memory.can_sample(5):
        states, actions, rewards, next_states, dones = memory.sample(5)
        
        print(f"Sampled batch sizes:")
        print(f"  states shape: {states.shape}")
        print(f"  actions shape: {actions.shape}")
        print(f"  rewards shape: {rewards.shape}")
        print(f"  next_states shape: {next_states.shape}")
        print(f"  dones shape: {dones.shape}")
        print(f"First action: {actions[0].item()}")
        print(f"First reward: {rewards[0].item():.4f}")
    else:
        print("Not enough samples to create a batch")
