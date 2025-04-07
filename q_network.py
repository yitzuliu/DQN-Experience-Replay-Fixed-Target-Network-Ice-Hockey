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
#    - We'll use 3 conv layers with increasing filter counts
#    - Pattern: 32 filters → 64 filters → 64 filters
#    - Stride parameters reduce spatial dimensions (downsampling)
#    - ReLU activation after each conv layer adds non-linearity
#
# 3. Flatten Operation:
#    - Converts the 3D feature maps to 1D vector for fully connected layers
#
# 4. Fully Connected Layers:
#    - Process extracted features to estimate Q-values
#    - We'll use 1 hidden layer with 512 neurons
#    - Final layer will have num_actions neurons (output size depends on action space)
#    - ReLU activation for hidden layer but not for output layer
#
# 5. Output:
#    - Q-values for each possible action in the environment
#    - No activation on final layer (raw Q-values)
#
# 6. Initialization:
#    - We'll use Kaiming/He initialization for better performance with ReLU
#    - Initialize bias terms to zeros
#
# 7. Utility Functions:
#    - Factory function to create networks based on environment parameters
#    - Testing code to verify network structure and forward pass

# 導入必要的庫
import torch                     # PyTorch主要庫，用於構建和訓練神經網絡
import torch.nn as nn            # 神經網絡模組庫，提供網絡層和組件
import torch.nn.functional as F  # 函數庫，提供激活函數等功能
import gymnasium as gym          # Gymnasium庫，用於環境交互
import config                    # 配置文件，包含環境和訓練參數

class QNetwork(nn.Module):
    """
    CNN-based Q-Network for processing Atari game frames and predicting Q-values
    這是一個基於卷積神經網絡(CNN)的Q網絡，用於處理Atari遊戲畫面並預測每個動作的Q值。
    """
    def __init__(self, input_shape, num_actions):
        """
        Initialize the Q-Network
        Args:
            input_shape (tuple): Shape of input observations (frames, height, width, channels)
            num_actions (int): Number of possible actions in the environment
        參數:
            input_shape (tuple): 輸入觀察的形狀 (幀數, 高度, 寬度, 通道數)
            num_actions (int): 環境中可能的動作數量
        """

        # Call the parent class constructor
        # 調用父類構造函數
        super(QNetwork, self).__init__()
        
        # Extract dimensions from input shape
        # 從輸入形狀中提取維度
        self.frames, self.height, self.width, self.channels = input_shape
        
        # Convolutional layers - used to extract features from images 
        # First convolutional layer: takes stacked frames as input
        # The input shape is (batch_size, frames * channels, height, width)
        # The number of input channels is frames * channels
        # The number of output channels is 32
        # The kernel size is 8x8 and the stride is 4
        self.conv1 = nn.Conv2d(self.frames * self.channels, 32, kernel_size=8, stride=4)
        # Second convolutional layer: takes the output of the first layer as input
        # The number of input channels is 32
        # The number of output channels is 64
        # The kernel size is 4x4 and the stride is 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # Third convolutional layer: takes the output of the second layer as input
        # The number of input channels is 64
        # The number of output channels is 64
        # The kernel size is 3x3 and the stride is 1
        # This layer is only used if self.use_two_conv_layers is False
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Store the number of actions
        self.use_two_conv_layers = config.USE_TWO_CONV_LAYERS

        # Calculate the input size for the fully connected layer
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        # according to the number of convolutional layers, calculate the input size for the fully connected layer
        if self.use_two_conv_layers:
            # calculate the output size after two convolutional layers
            conv_width = conv2d_size_out(conv2d_size_out(self.width, 8, 4), 4, 2)
            conv_height = conv2d_size_out(conv2d_size_out(self.height, 8, 4), 4, 2)
            linear_input_size = conv_width * conv_height * 64
        else:
            # calculate the output size after three convolutional layers
            conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.width, 8, 4), 4, 2), 3, 1)
            conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.height, 8, 4), 4, 2), 3, 1)
            linear_input_size = conv_width * conv_height * 64
        
        # Fully connected layers - used to process the features extracted by the convolutional layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        # Initialize weights
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization
        使用Kaiming初始化方法初始化網絡權重。
        這種初始化方法特別適合ReLU激活函數，有助於防止梯度消失問題。
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Kaiming initialization for convolutional and linear layers
                # 對卷積層和線性層使用kaiming_normal_初始化權重
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                # Initialize bias to zero
                # 將偏置初始化為0
                # 這是為了確保在訓練開始時，所有神經元的輸出都是0，這樣可以避免初始階段的偏差。
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, frames, height, width, channels)

        Returns:
            torch.Tensor: Q-values for each action
        
        網絡的前向傳播方法，處理輸入數據並生成輸出。
        參數:
            x (torch.Tensor): 形狀為 (batch_size, frames, height, width, channels) 的輸入張量
   
        返回:
            torch.Tensor: 每個動作的Q值
        """
        # Reshape input for convolutional layers
        # 將輸入數據重塑為卷積層所需的形狀
        # batch_size是當前批次的大小，x.size(0)獲取批次大小
        batch_size = x.size(0)
        # 將輸入數據重塑為 (batch_size, frames * channels, height, width) 的形狀
        x = x.view(batch_size, self.frames * self.channels, self.height, self.width)
        
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # 條件執行第三層卷積
        if not self.use_two_conv_layers:
            x = F.relu(self.conv3(x))
        
        # Flatten for fully connected layers
        # 將卷積層的輸出展平為一維向量，以便輸入到全連接層
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        # 應用第一個全連接層，然後使用ReLU激活函數
        x = F.relu(self.fc1(x))
        # 應用輸出層，不使用激活函數，直接輸出Q值
        q_values = self.fc2(x)
        
        return q_values


def create_q_network(env):
    """
    Factory function to create a Q-Network based on the environment
    Args:
        env (gym.Env): Gymnasium environment
    Returns:
        nn.Module: Q-Network instance
    
    創建Q網絡的工廠函數，根據環境參數自動配置網絡結構
    參數:
        env (gym.Env): Gymnasium環境
    返回:
        nn.Module: Q網絡實例
    """
    # Get observation shape and action count from environment
    # 從環境中獲取觀察形狀和動作數量
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    # Determine input shape based on observation space
    # 根據觀察空間確定輸入形狀
    input_shape = (1,) + obs_shape  # Assuming single frame as default / 假設默認為單一幀
    
    # Create and return the Q-Network
    # 創建並返回Q網絡
    return QNetwork(input_shape, num_actions)


# Test code for this module - will only run when this file is executed directly
# 測試代碼 - 只有當直接運行此文件時才會執行
if __name__ == "__main__":
    import env_wrappers
    
    # Create environment
    # 創建環境
    env = env_wrappers.make_env()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create Q-Network
    # 創建Q網絡
    q_net = create_q_network(env)
    print(f"Q-Network architecture:\n{q_net}")
    
    # Test with a sample observation
    # 使用樣本觀察進行測試
    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension / 添加批次維度
    q_values = q_net(obs_tensor)
    
    print(f"Input shape: {obs_tensor.shape}")
    print(f"Output Q-values: {q_values}")
    print(f"Best action: {torch.argmax(q_values).item()}")
    
    env.close()
