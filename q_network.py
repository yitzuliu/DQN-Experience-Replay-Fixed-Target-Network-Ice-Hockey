"""
Q-Network Architecture for DQN

This module defines the neural network architecture used to approximate the Q-function
in Deep Q-Network (DQN) reinforcement learning. The architecture follows the design
introduced in the original DQN paper by Mnih et al. (2015), with modifications to
allow for different network depths.

Key components:
1. Convolutional layers for visual feature extraction
2. Fully connected layers for Q-value estimation
3. Configurable network depth via hyperparameters

Hardware optimizations:
1. Efficient data representation for GPU processing
2. Memory-optimized parameter initialization
3. Batch normalization for faster training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class DQN(nn.Module):
    """
    Deep Q-Network for approximating Q-values from raw game frames.
    
    The network takes preprocessed game frames as input and outputs Q-values
    for each possible action. Architecture includes convolutional layers for
    feature extraction followed by fully connected layers for Q-value prediction.
    
    Network depth can be configured from 1-3 convolutional layers, allowing for
    trade-offs between model complexity and computational efficiency.
    """
    
    def __init__(self, input_shape, n_actions, device, use_two_layers=False, use_three_layers=True):
        """
        Initialize Q-Network with configurable depth.
        
        Args:
            input_shape (tuple): Shape of input observations (C, H, W)
            n_actions (int): Number of possible actions (output size)
            device (torch.device): Device to run the model on (CPU/GPU/MPS)
            use_two_layers (bool): Whether to use 2 convolutional layers 
            use_three_layers (bool): Whether to use 3 convolutional layers
                                     (overrides use_two_layers if True)
        """
        super(DQN, self).__init__()
        
        self.device = device
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # Input channels (usually 4 for frame stacking)
        c, h, w = input_shape
        
        # --- Feature Extraction: Convolutional Layers ---
        
        # First convolutional layer (always used)
        self.conv1 = nn.Conv2d(
            in_channels=c, 
            out_channels=config.CONV1_CHANNELS, 
            kernel_size=config.CONV1_KERNEL_SIZE, 
            stride=config.CONV1_STRIDE
        )
        self.bn1 = nn.BatchNorm2d(config.CONV1_CHANNELS)  # Batch normalization for faster training
        
        # Second convolutional layer (conditional)
        if use_two_layers or use_three_layers:
            self.conv2 = nn.Conv2d(
                in_channels=config.CONV1_CHANNELS, 
                out_channels=config.CONV2_CHANNELS, 
                kernel_size=config.CONV2_KERNEL_SIZE, 
                stride=config.CONV2_STRIDE
            )
            self.bn2 = nn.BatchNorm2d(config.CONV2_CHANNELS)
        
        # Third convolutional layer (conditional)
        if use_three_layers:
            self.conv3 = nn.Conv2d(
                in_channels=config.CONV2_CHANNELS, 
                out_channels=config.CONV3_CHANNELS, 
                kernel_size=config.CONV3_KERNEL_SIZE, 
                stride=config.CONV3_STRIDE
            )
            self.bn3 = nn.BatchNorm2d(config.CONV3_CHANNELS)
        
        # Calculate the size of the feature maps after convolutions
        # This calculation prevents hardcoding the size and adapts to input dimensions
        conv_output_size = self._calculate_conv_output_size(h, w, use_two_layers, use_three_layers)
        
        # --- Q-Value Estimation: Fully Connected Layers ---
        
        # First fully connected layer
        self.fc1 = nn.Linear(in_features=conv_output_size, out_features=config.FC_SIZE)
        
        # Output layer: one value per action
        self.fc2 = nn.Linear(in_features=config.FC_SIZE, out_features=n_actions)
        
        # Initialize weights using Kaiming/He initialization for better gradient flow
        self._initialize_weights()
        
        # Move model to the specified device (GPU/CPU)
        self.to(device)
    
    def _calculate_conv_output_size(self, h, w, use_two_layers, use_three_layers):
        """
        Calculate the output size of the convolutional layers.
        
        This eliminates the need for hardcoding the size and makes the model
        adaptable to different input dimensions.
        
        Formula: ((W-K+2P)/S)+1, where:
        - W: input size
        - K: kernel size
        - P: padding
        - S: stride
        
        Args:
            h (int): Height of input
            w (int): Width of input
            use_two_layers (bool): Whether using 2 conv layers
            use_three_layers (bool): Whether using 3 conv layers
            
        Returns:
            int: Flattened size of the convolutional output
        """
        # First conv layer
        h = (h - config.CONV1_KERNEL_SIZE) // config.CONV1_STRIDE + 1
        w = (w - config.CONV1_KERNEL_SIZE) // config.CONV1_STRIDE + 1
        
        if use_two_layers or use_three_layers:
            # Second conv layer
            h = (h - config.CONV2_KERNEL_SIZE) // config.CONV2_STRIDE + 1
            w = (w - config.CONV2_KERNEL_SIZE) // config.CONV2_STRIDE + 1
        
        if use_three_layers:
            # Third conv layer
            h = (h - config.CONV3_KERNEL_SIZE) // config.CONV3_STRIDE + 1
            w = (w - config.CONV3_KERNEL_SIZE) // config.CONV3_STRIDE + 1
        
        # If using 3 layers, output has CONV3_CHANNELS channels
        # If using 2 layers, output has CONV2_CHANNELS channels
        # If using 1 layer, output has CONV1_CHANNELS channels
        channels = config.CONV1_CHANNELS
        if use_two_layers or use_three_layers:
            channels = config.CONV2_CHANNELS
        if use_three_layers:
            channels = config.CONV3_CHANNELS
            
        return channels * h * w
    
    def _initialize_weights(self):
        """
        Initialize network weights for better training performance.
        
        Uses Kaiming/He initialization for convolutional and linear layers,
        which is particularly effective for networks with ReLU activations.
        """
        # Initialize convolutional layers with ReLU-specific initialization
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        
        if hasattr(self, 'conv2'):
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
            nn.init.constant_(self.conv2.bias, 0)
        
        if hasattr(self, 'conv3'):
            nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
            nn.init.constant_(self.conv3.bias, 0)
        
        # Initialize fully connected layers with ReLU-specific initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        
        # Xavier/Glorot initialization for the output layer (no ReLU after)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Processes input game frames through convolutional layers for feature extraction,
        then through fully connected layers for Q-value prediction.
        
        Args:
            x (torch.Tensor): Batch of preprocessed game frames [B, C, H, W]
            
        Returns:
            torch.Tensor: Q-values for each action [B, n_actions]
        """
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # --- Feature Extraction ---
        
        # First convolutional layer (always used) with ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second convolutional layer (conditional) with ReLU
        if hasattr(self, 'conv2'):
            x = F.relu(self.bn2(self.conv2(x)))
        
        # Third convolutional layer (conditional) with ReLU
        if hasattr(self, 'conv3'):
            x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten the output of convolutional layers
        x = x.view(x.size(0), -1)
        
        # --- Q-Value Estimation ---
        
        # First fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        
        # Output layer (no activation - raw Q-values)
        q_values = self.fc2(x)
        
        return q_values


def create_q_network(input_shape, n_actions, device=None):
    """
    Factory function to create a Q-network based on configuration.
    
    Args:
        input_shape (tuple): Shape of input observations (C, H, W)
        n_actions (int): Number of possible actions
        device (torch.device, optional): Device to run on, auto-detected if None
        
    Returns:
        DQN: Initialized Q-network
    """
    # Auto-detect device if not provided
    if device is None:
        device = utils.get_device()
    
    # Determine network depth based on config
    use_one_layer = config.USE_ONE_CONV_LAYER
    use_two_layers = config.USE_TWO_CONV_LAYERS
    use_three_layers = config.USE_THREE_CONV_LAYERS
    
    # Apply priority rules for conflicting settings
    if use_one_layer:
        use_two_layers = False
        use_three_layers = False
    elif use_two_layers:
        use_three_layers = False
    # If none specified, default to three layers
    elif not use_three_layers:
        use_three_layers = True
    
    # Create network with specified depth
    network = DQN(
        input_shape=input_shape,
        n_actions=n_actions,
        device=device,
        use_two_layers=use_two_layers,
        use_three_layers=use_three_layers
    )
    
    # Enable cuDNN benchmarking for faster convolutions if using CUDA
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        
    return network

