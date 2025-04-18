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
import gc          # Garbage collection
import utils       # Utility functions, including clean_memory

class ListReplayMemory:
    """
    Experience Replay Memory using a simple Python list.
    """
    def __init__(self, capacity, state_shape=None, action_dim=1):
        """
        Initialize the replay memory with a fixed capacity.
        
        Args:
            capacity (int): Maximum number of transitions to store
            state_shape (tuple, optional): Shape of state observations
            action_dim (int, optional): Dimension of action space
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
    
    def sample_pinned(self, batch_size):
        """
        Randomly sample a batch of experiences with pinned memory.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones) with pinned memory
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
        
        # Convert to tensors with pinned memory
        states = torch.FloatTensor(np.array(states)).pin_memory()
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).pin_memory()
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).pin_memory()
        next_states = torch.FloatTensor(np.array(next_states)).pin_memory()
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).pin_memory()
        
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
    
    def clear(self):
        """
        Clear the memory buffer and free resources.
        """
        self.memory.clear()
        self.position = 0
        utils.clean_memory()


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
    
    def sample_pinned(self, batch_size):
        """
        Randomly sample a batch of experiences with pinned memory.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones) with pinned memory
        """
        indices = np.random.choice(self.counter, batch_size, replace=False)
        
        # Use pinned memory for faster CPU to GPU transfer
        states = torch.FloatTensor(self.states[indices]).pin_memory()
        actions = torch.LongTensor(self.actions[indices]).pin_memory()
        rewards = torch.FloatTensor(self.rewards[indices]).pin_memory()
        next_states = torch.FloatTensor(self.next_states[indices]).pin_memory()
        dones = torch.FloatTensor(self.dones[indices]).pin_memory()
        
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
    
    def clear(self):
        """
        Clear the replay buffer and free resources.
        """
        self.counter = 0
        self.position = 0
        utils.clean_memory()


class OptimizedArrayReplayMemory:
    """
    Memory-optimized experience replay implementation.
    """
    def __init__(self, capacity, state_shape, action_dim=1):
        """
        Initialize optimized replay memory.
        
        Args:
            capacity (int): Maximum number of transitions to store
            state_shape (tuple): Shape of state observations
            action_dim (int): Dimension of action space (usually 1)
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.counter = 0  # Total number of experiences added
        self.position = 0  # Current position in circular buffer
        
        # Use uint8 for states (save 75% memory compared to float32)
        # Store as single array instead of two separate arrays
        self.states = np.zeros((capacity + 1, *state_shape), dtype=np.uint8)
        
        # Use smaller data types where possible
        self.actions = np.zeros(capacity, dtype=np.uint8)  # Actions are typically < 256
        self.rewards = np.zeros(capacity, dtype=np.float16)  # Half precision for rewards
        self.indices = np.zeros(capacity, dtype=np.uint32)   # Indices to next states
        self.dones = np.zeros(capacity, dtype=np.bool_)     # Boolean for done flag
        
    def add(self, state, action, reward, next_state, done):
        # Convert float32 state to uint8 for storage (multiply by 255)
        state_uint8 = np.clip(state * 255, 0, 255).astype(np.uint8)
        next_state_uint8 = np.clip(next_state * 255, 0, 255).astype(np.uint8)
        
        # Store current state
        self.states[self.position] = state_uint8
        
        # Store next state in the next position (cyclically)
        next_position = (self.position + 1) % self.capacity
        self.states[next_position] = next_state_uint8
        
        # Store transition data
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.indices[self.position] = next_position  # Link to next_state
        self.dones[self.position] = done
        
        # Update position and counter
        self.position = next_position
        self.counter = min(self.counter + 1, self.capacity)
        
    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from memory.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        try:
            indices = np.random.choice(min(self.counter, self.capacity-1), batch_size, replace=False)
            
            # Get corresponding next state indices
            next_indices = self.indices[indices]
            
            # Convert uint8 back to float32 normalized [0,1] range when sampling
            states = self.states[indices].astype(np.float32) / 255.0
            next_states = self.states[next_indices].astype(np.float32) / 255.0
            
            # Convert to PyTorch tensors
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(self.actions[indices]).unsqueeze(1)
            rewards = torch.FloatTensor(self.rewards[indices].astype(np.float32)).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(self.dones[indices].astype(np.float32)).unsqueeze(1)
            
            return states, actions, rewards, next_states, dones
            
        except Exception as e:
            print(f"Warning: Memory sampling error: {e}, returning empty batch")
            # Return empty batch instead of failing
            empty_shape = (0, *self.state_shape)
            return (torch.FloatTensor(np.zeros(empty_shape)),
                    torch.LongTensor(np.zeros((0, 1))),
                    torch.FloatTensor(np.zeros((0, 1))),
                    torch.FloatTensor(np.zeros(empty_shape)),
                    torch.FloatTensor(np.zeros((0, 1))))
    
    def sample_pinned(self, batch_size):
        """
        Randomly sample a batch of experiences from memory with pinned memory.
        Pinned memory allows for faster CPU to GPU transfers.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones) with pinned memory
            
        Raises:
            KeyboardInterrupt: Re-raises KeyboardInterrupt for proper training termination
        """
        try:
            # Make sure we don't select the last element as an index for a next state
            indices = np.random.choice(min(self.counter, self.capacity-1), batch_size, replace=False)
            
            # Get corresponding next state indices
            next_indices = self.indices[indices]
            
            # Pre-merge batches to reduce PyTorch operations
            states_and_next = np.vstack([self.states[indices], self.states[next_indices]])
            states_and_next_gpu = torch.from_numpy(states_and_next.astype(np.float32) / 255.0).pin_memory()
            
            # Use slicing rather than creating separate tensors
            states = states_and_next_gpu[:batch_size]
            next_states = states_and_next_gpu[batch_size:]
            
            # Convert to PyTorch tensors with pinned memory for faster GPU transfer
            actions = torch.LongTensor(self.actions[indices]).unsqueeze(1).pin_memory()
            rewards = torch.FloatTensor(self.rewards[indices].astype(np.float32)).unsqueeze(1).pin_memory()
            dones = torch.FloatTensor(self.dones[indices].astype(np.float32)).unsqueeze(1).pin_memory()
            
            return states, actions, rewards, next_states, dones
            
        except KeyboardInterrupt:
            # Re-raise the KeyboardInterrupt for proper handling in the training loop
            print("\nKeyboard interrupt detected during sampling. Preparing for clean shutdown...")
            raise  # Re-raise the keyboard interrupt rather than returning None
            
        except Exception as e:
            # More descriptive error message
            print(f"Warning: Error during memory sampling: {e}")
            print(f"Memory state: capacity={self.capacity}, counter={self.counter}, position={self.position}")
            
            # Try to return a valid but smaller batch
            try:
                reduced_batch = max(1, batch_size // 4)
                print(f"Attempting to return a reduced batch of size {reduced_batch}")
                return self.sample(reduced_batch)
            except:
                # If that fails too, return None
                print("Failed to create recovery batch, returning None")
                return None, None, None, None, None
    
    def __len__(self):
        return min(self.counter, self.capacity)
    
    def can_sample(self, batch_size):
        return len(self) >= batch_size
    
    def clear(self):
        """Clear the replay buffer and free resources."""
        self.counter = 0
        self.position = 0
        utils.clean_memory()
        
    def memory_usage(self):
        """
        Calculate approximate memory usage.
        
        Returns:
            float: Approximate memory usage in MB
        """
        states_size = self.states.nbytes / (1024 * 1024)
        actions_size = self.actions.nbytes / (1024 * 1024)
        rewards_size = self.rewards.nbytes / (1024 * 1024)
        indices_size = self.indices.nbytes / (1024 * 1024)
        dones_size = self.dones.nbytes / (1024 * 1024)
        
        total = states_size + actions_size + rewards_size + indices_size + dones_size
        
        return {
            "states_mb": states_size,
            "actions_mb": actions_size,
            "rewards_mb": rewards_size,
            "indices_mb": indices_size,
            "dones_mb": dones_size,
            "total_mb": total
        }

