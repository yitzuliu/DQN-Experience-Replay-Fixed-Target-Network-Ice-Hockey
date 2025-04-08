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
import numpy as np
import config
import utils


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
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for faster training
        
        # Second convolutional layer (conditional)
        if use_two_layers or use_three_layers:
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
            self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional layer (conditional)
        if use_three_layers:
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
            self.bn3 = nn.BatchNorm2d(64)
        
        # Calculate the size of the feature maps after convolutions
        # This calculation prevents hardcoding the size and adapts to input dimensions
        conv_output_size = self._calculate_conv_output_size(h, w, use_two_layers, use_three_layers)
        
        # --- Q-Value Estimation: Fully Connected Layers ---
        
        # First fully connected layer
        self.fc1 = nn.Linear(in_features=conv_output_size, out_features=512)
        
        # Output layer: one value per action
        self.fc2 = nn.Linear(in_features=512, out_features=n_actions)
        
        # Initialize weights using Kaiming/He initialization for better gradient flow
        self._initialize_weights()
        
        # Move model to the specified device (GPU/CPU)
        self.to(device)
        
        # Log model architecture decisions
        layers_used = 1
        if use_two_layers:
            layers_used = 2
        if use_three_layers:
            layers_used = 3
        print(f"Created DQN with {layers_used} convolutional layer(s), output size: {n_actions}")
    
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
        # First conv layer: 8x8 kernel, stride 4, no padding
        h = (h - 8) // 4 + 1
        w = (w - 8) // 4 + 1
        
        if use_two_layers or use_three_layers:
            # Second conv layer: 4x4 kernel, stride 2, no padding
            h = (h - 4) // 2 + 1
            w = (w - 4) // 2 + 1
        
        if use_three_layers:
            # Third conv layer: 3x3 kernel, stride 1, no padding
            h = (h - 3) // 1 + 1
            w = (w - 3) // 1 + 1
        
        # If using 3 layers, output has 64 channels
        # If using 2 layers, output has 64 channels
        # If using 1 layer, output has 32 channels
        channels = 32
        if use_two_layers or use_three_layers:
            channels = 64
            
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
        
    # Log device to be used
    print(f"Creating Q-network on device: {device}")
    
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
    
    # Log network architecture
    if use_three_layers:
        print("Created 3-layer convolutional Q-network (deep)")
    elif use_two_layers:
        print("Created 2-layer convolutional Q-network (medium)")
    else:
        print("Created 1-layer convolutional Q-network (shallow)")
    
    # Enable cuDNN benchmarking for faster convolutions if using CUDA
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        
    return network


# Testing code for quick verification
if __name__ == "__main__":
    # Display system information
    system_info = utils.get_system_info()
    print(f"Testing Q-Network on {system_info['os']} with PyTorch {system_info['torch_version']}")
    
    # Get device (CPU/GPU)
    device = utils.get_device()
    print(f"Using device: {device}")
    
    # Parameters from typical Atari preprocessing
    input_channels = config.FRAME_STACK  # Typically 4 stacked frames
    input_height = config.FRAME_HEIGHT   # Typically 84
    input_width = config.FRAME_WIDTH     # Typically 84
    n_actions = config.ACTION_SPACE_SIZE # 18 for Ice Hockey
    
    # Test network creation
    print("\nTesting network creation:")
    
    # Test with 1 convolutional layer
    net1 = DQN(
        input_shape=(input_channels, input_height, input_width),
        n_actions=n_actions,
        device=device,
        use_two_layers=False,
        use_three_layers=False
    )
    print("1-layer network created successfully")
    
    # Test with 2 convolutional layers
    net2 = DQN(
        input_shape=(input_channels, input_height, input_width),
        n_actions=n_actions,
        device=device,
        use_two_layers=True,
        use_three_layers=False
    )
    print("2-layer network created successfully")
    
    # Test with 3 convolutional layers
    net3 = DQN(
        input_shape=(input_channels, input_height, input_width),
        n_actions=n_actions,
        device=device,
        use_two_layers=False,
        use_three_layers=True
    )
    print("3-layer network created successfully")
    
    # Test default network creation through factory function
    default_net = create_q_network(
        input_shape=(input_channels, input_height, input_width),
        n_actions=n_actions
    )
    print("Default network from factory function created successfully")
    
    # Test forward pass
    print("\nTesting forward pass:")
    batch_size = 32
    test_input = torch.randn(batch_size, input_channels, input_height, input_width)
    
    # Test on CPU first to avoid potential GPU memory issues
    with torch.no_grad():
        test_input = test_input.to(device)
        output = default_net(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output norm: {output.norm().item():.4f}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Measure single-batch inference time
    import time
    n_trials = 100
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_trials):
            _ = default_net(test_input)
    
    # Calculate metrics
    elapsed = time.time() - start_time
    avg_time_ms = (elapsed / n_trials) * 1000
    
    # Memory usage info
    if hasattr(torch.cuda, 'memory_allocated') and torch.cuda.is_available():
        memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"\nGPU memory used: {memory_mb:.2f} MB")
    
    print(f"Avg. inference time: {avg_time_ms:.2f} ms per batch")
    print(f"Total parameters: {sum(p.numel() for p in default_net.parameters()):,}")
    print("\nQ-Network test completed successfully")

