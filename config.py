"""
Configuration file for DQN training on Atari Ice Hockey.

This file contains all hyperparameters and settings used throughout the project,
organized by category for better readability and management.

DQN 在 Atari 冰球遊戲上訓練的配置文件。

本文件包含項目中使用的所有超參數和設置，按類別組織以提高可讀性和可管理性。
"""

###############################
# GAME ENVIRONMENT SETTINGS
# 遊戲環境設置
###############################

# Basic environment settings
# 基本環境設置
ENV_NAME = 'ALE/IceHockey-v5'  # 遊戲環境名稱
ACTION_SPACE_SIZE = 18  # IceHockey has 18 possible actions (冰球遊戲有18個可能的動作)
OBS_TYPE = "rgb"  # Default observation type for ALE/IceHockey-v5 (默認觀察類型)

# Game difficulty settings
# 遊戲難度設置
DIFFICULTY = 0  # 0=Easy, 1=Normal, 2=Hard, 3=Expert (0=簡單, 1=普通, 2=困難, 3=專家)
MODE = 0  # 0=Default mode, values vary by game (0=默認模式，具體影響因遊戲而異)

# Frame processing settings
# 幀處理設置
FRAME_WIDTH = 84  # Downscaled from native 160 (從原始160縮小到84)
FRAME_HEIGHT = 84  # Downscaled from native 210 (從原始210縮小到84)
FRAME_STACK = 4  # Number of frames to stack together (堆疊幀數量)
FRAME_SKIP = 4  # Matches the environment's built-in frameskip for v5 (與環境內建的v5版本跳幀設置匹配)
NOOP_MAX = 30  # Maximum number of no-op actions at the start of an episode (開始時最大無操作動作數)

# Visualization settings
# 視覺化設置
RENDER_MODE = None  # No rendering during training for maximum speed (訓練期間不渲染以提高速度)
TRAINING_MODE = True  # Ensure render_mode is None during training (確保訓練模式下不渲染)

###############################
# DEEP Q-LEARNING PARAMETERS
# 深度Q學習參數
###############################

# Core DQN parameters
# 核心DQN參數
LEARNING_RATE = 0.0001  # Standard learning rate for Adam optimizer (Adam優化器的標準學習率)
GAMMA = 0.99  # Standard discount factor (標準折扣因子)
BATCH_SIZE = 128  # Standard batch size (標準批次大小)
MEMORY_CAPACITY = 200000  # Increased to 200K for better experience diversity (增加到20萬，提供更多樣化的經驗)
TARGET_UPDATE_FREQUENCY = 10000  # Update target network every 10K steps (每1萬步更新一次目標網絡)
TRAINING_EPISODES = 10000  # Total number of training episodes (訓練總回合數)

# Exploration parameters
# 探索參數
EPSILON_START = 1.0  # Initial exploration rate (初始探索率：完全隨機)
EPSILON_END = 0.01  # Lower final exploration rate for better policy (最終較低的探索率，有利於更好的策略)
EPSILON_DECAY = 500000  # Slower decay over more steps for better exploration (較慢的衰減速率，分佈更多步數以改善探索)
DEFAULT_EVALUATE_MODE = False  # Default evaluation mode (默認評估模式)

# Training control parameters
# 訓練控制參數
LEARNING_STARTS = 50000  # Wait for more experiences before starting learning (開始學習前等待的經驗數量)
UPDATE_FREQUENCY = 4  # Standard update frequency (標準更新頻率)
SAVE_FREQUENCY = 100  # Save more frequently to prevent data loss (頻繁保存以防止數據丟失)

# Neural network settings
# 神經網絡設置
USE_ONE_CONV_LAYER = True  # Use 1 convolutional layer (使用1個卷積層)
USE_TWO_CONV_LAYERS = False  # Use 2 convolutional layers (使用2個卷積層)
USE_THREE_CONV_LAYERS = False  # Using full 3-layer architecture (hardware can handle it) (使用完整的3層架構，由硬件支持)

# Gradient accumulation
# 梯度累積
GRADIENT_ACCUMULATION_STEPS = 1  # Default is 1 (no accumulation) (默認為1，不進行累積)

# Evaluation settings
# 評估設置
EVAL_EPISODES = 30  # More evaluation episodes for better statistics (更多評估回合以獲得更好的統計數據)
EVAL_FREQUENCY = 500  # Evaluate every 500 episodes (每500回合進行一次評估)

# 添加定期評估設置
EVAL_DURING_TRAINING = True  # 訓練過程中進行評估
EVAL_EPISODES_TRAINING = 10  # 每次評估的回合數
EVAL_FREQUENCY_TRAINING = 500  # 每訓練500回合進行一次評估

###############################
# SYSTEM AND OPTIMIZATION
# 系統與優化
###############################

# Memory optimization
# 記憶體優化
MEMORY_IMPLEMENTATION = "optimized"  # Using memory-efficient implementation (使用記憶體高效的實現)

# GPU settings
# GPU設置
USE_GPU_PREPROCESSING = True  # Use GPU for frame preprocessing (使用GPU進行幀預處理)

# Advanced features (disabled until implemented)
# 高級功能（暫未實現）
PRIORITIZED_REPLAY = False  # Disabled until PrioritizedReplayMemory is implemented (優先經驗回放，尚未實現)
PRIORITIZED_REPLAY_ALPHA = 0.6  # Standard PER alpha (not used currently) (標準PER alpha值，目前未使用)
PRIORITIZED_REPLAY_BETA_START = 0.4  # Starting beta value (not used currently) (起始beta值，目前未使用)
PRIORITIZED_REPLAY_BETA_FRAMES = 100000  # Frames over which to anneal beta (not used currently) (beta退火的幀數，目前未使用)
DUELING_NETWORK = False  # Disabled until Dueling Network architecture is implemented (對決網絡架構，尚未實現)
DOUBLE_DQN = False  # Disabled until Double DQN logic is implemented (雙DQN邏輯，尚未實現)

