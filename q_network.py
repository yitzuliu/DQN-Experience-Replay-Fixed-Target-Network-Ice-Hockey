# In this file, implement the Q-network architecture using PyTorch:
# - Define a neural network class inheriting from nn.Module
# - Set up appropriate layers for Atari input processing:
#   - Convolutional layers for image processing
#   - Fully connected layers
# - Implement forward pass
# - Consider appropriate activation functions (ReLU, etc.)
#
# The network should take preprocessed game frames as input and
# output Q-values for each possible action in Ice Hockey.

# Q-Network Design Notes:
# 1. Input Structure:
#    - Input will be raw game frames (RGB images) from the Ice Hockey environment
#    - Shape will be something like (batch_size, frames_stacked, height, width, channels)
#      where frames_stacked is often 4 consecutive frames to capture motion
#
# 2. Convolutional Layers:
#    - Purpose: Extract visual features from game frames
#    - Typically 3 conv layers with increasing filter counts
#    - Common pattern: 32 filters → 64 filters → 64 filters
#    - Use stride of 2 to reduce spatial dimensions (downsampling)
#    - Include ReLU activation after each conv layer for non-linearity
#
# 3. Flatten Operation:
#    - Convert the 3D feature maps to 1D vector for fully connected layers
#
# 4. Fully Connected Layers:
#    - Process extracted features to estimate Q-values
#    - Usually 1-2 hidden layers (e.g., 512 neurons)
#    - Final layer should have num_actions neurons (18 for Ice Hockey)
#    - Use ReLU activation for hidden layers but not for output layer
#
# 5. Output:
#    - Q-values for each possible action in the environment
#    - No activation on final layer (raw Q-values)
#
# 6. Architecture Variants:
#    - Nature DQN: The original architecture from the Nature paper
#    - Dueling DQN: Separate streams for state value and action advantages
#    - Deeper networks: More layers for potentially better feature extraction
#
# 7. Initialization:
#    - Use proper weight initialization (e.g., Kaiming/He initialization)
#    - Initialize bias terms to small values or zeros
#
# 8. Additional Considerations:
#    - Batch normalization: Can improve training stability but adds complexity
#    - Dropout: Can help with overfitting but may slow down training
#    - Residual connections: Can help with gradient flow in deeper networks

# This file should implement:
# 1. A QNetwork class that inherits from nn.Module
# 2. The forward method to compute Q-values given states
# 3. Any helper methods needed for the network architecture
# 4. Optionally, different network architectures for experimentation

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import config

class QNetwork(nn.Module):
    """
    CNN-based Q-Network for processing Atari game frames and predicting Q-values
    """
    def __init__(self, input_shape, num_actions):
        """
        Initialize the Q-Network
        
        Args:
            input_shape (tuple): Shape of input observations (frames, height, width, channels)
            num_actions (int): Number of possible actions in the environment
        """
        super(QNetwork, self).__init__()
        
        # Extract dimensions from input shape
        self.frames, self.height, self.width, self.channels = input_shape
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(self.frames * self.channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size of flattened features after conv layers
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.width, 8, 4), 4, 2), 3, 1)
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.height, 8, 4), 4, 2), 3, 1)
        linear_input_size = conv_width * conv_height * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, frames, height, width, channels)
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        # Reshape input for convolutional layers
        batch_size = x.size(0)
        x = x.view(batch_size, self.frames * self.channels, self.height, self.width)
        
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values

class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture that separates state value and action advantages
    """
    def __init__(self, input_shape, num_actions):
        """
        Initialize the Dueling Q-Network
        
        Args:
            input_shape (tuple): Shape of input observations (frames, height, width, channels)
            num_actions (int): Number of possible actions in the environment
        """
        super(DuelingQNetwork, self).__init__()
        
        # Extract dimensions from input shape
        self.frames, self.height, self.width, self.channels = input_shape
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(self.frames * self.channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size of flattened features after conv layers
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.width, 8, 4), 4, 2), 3, 1)
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.height, 8, 4), 4, 2), 3, 1)
        linear_input_size = conv_width * conv_height * 64
        
        # Value stream
        self.value_fc1 = nn.Linear(linear_input_size, 512)
        self.value_fc2 = nn.Linear(512, 1)
        
        # Advantage stream
        self.advantage_fc1 = nn.Linear(linear_input_size, 512)
        self.advantage_fc2 = nn.Linear(512, num_actions)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, frames, height, width, channels)
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        # Reshape input for convolutional layers
        batch_size = x.size(0)
        x = x.view(batch_size, self.frames * self.channels, self.height, self.width)
        
        # Apply shared convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten features
        x = x.view(x.size(0), -1)
        
        # Value stream
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


def create_q_network(env, network_type='standard'):
    """
    Factory function to create a Q-Network based on the environment and specified type
    
    Args:
        env (gym.Env): Gymnasium environment
        network_type (str): Type of network architecture ('standard' or 'dueling')
        
    Returns:
        nn.Module: Q-Network instance
    """
    # Get observation shape and action count from environment
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    # Determine input shape based on observation space
    # For environments using wrappers, adjust accordingly
    input_shape = (1,) + obs_shape  # Assuming single frame as default
    
    # Create the network based on specified type
    if network_type.lower() == 'dueling':
        return DuelingQNetwork(input_shape, num_actions)
    else:
        return QNetwork(input_shape, num_actions)


# Test code for this module
if __name__ == "__main__":
    import env_wrappers
    
    # Create environment
    env = env_wrappers.make_env()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create Q-Network
    q_net = create_q_network(env)
    print(f"Q-Network architecture:\n{q_net}")
    
    # Test with a sample observation
    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension
    q_values = q_net(obs_tensor)
    
    print(f"Input shape: {obs_tensor.shape}")
    print(f"Output Q-values: {q_values}")
    print(f"Best action: {torch.argmax(q_values).item()}")
    
    env.close()
