"""
Configuration file for DQN training on Atari Ice Hockey.

This file contains all hyperparameters and settings used throughout the project,
organized by category for better readability and management.
"""

###############################
# GAME ENVIRONMENT SETTINGS
###############################

# Basic environment settings
ENV_NAME = 'ALE/IceHockey-v5'
ACTION_SPACE_SIZE = 18  # IceHockey has 18 possible actions
OBS_TYPE = "rgb"  # Default observation type for ALE/IceHockey-v5

# Game difficulty settings
DIFFICULTY = 0  # 0=Easy, 1=Normal, 2=Hard, 3=Expert
MODE = 0  # 0=Default mode, values vary by game

# Frame processing settings
FRAME_WIDTH = 84  # Downscaled from native 160
FRAME_HEIGHT = 84  # Downscaled from native 210
FRAME_STACK = 4  # Number of frames to stack together
FRAME_SKIP = 4  # Matches the environment's built-in frameskip for v5
NOOP_MAX = 30  # Maximum number of no-op actions at the start of an episode

# Visualization settings
RENDER_MODE = None  # No rendering during training for maximum speed
TRAINING_MODE = True  # Ensure render_mode is None during training

###############################
# DEEP Q-LEARNING PARAMETERS
###############################

# Core DQN parameters - scaled up for powerful hardware
LEARNING_RATE = 0.0001  # Standard learning rate for Adam optimizer
GAMMA = 0.99  # Standard discount factor
BATCH_SIZE = 4096  # Increased batch size for RTX 3090 (compared to standard 32-128)
MEMORY_CAPACITY = 1000000  # 2M transitions (increased from typical 1M for more experience)
TARGET_UPDATE_FREQUENCY = 10000  # Update target network every 10K steps
TRAINING_EPISODES = 10000  # Increased for more thorough training

# Exploration parameters
EPSILON_START = 1.0
EPSILON_END = 0.01  # Lower final exploration rate for better policy
EPSILON_DECAY = 500000  # Slower decay over more steps for better exploration
DEFAULT_EVALUATE_MODE = False

# Training control parameters
LEARNING_STARTS = 50000  # Wait for more experiences before starting learning
UPDATE_FREQUENCY = 4  # Standard update frequency
SAVE_FREQUENCY = 200  # Save model every 200 episodes
TARGET_UPDATE_PRINT_FREQUENCY = 50  # Print target update info every 10 updates
DETAILED_LOG_FREQUENCY = 100  # Detailed logs every 50 episodes

# Neural network settings
USE_ONE_CONV_LAYER = False
USE_TWO_CONV_LAYERS = False
USE_THREE_CONV_LAYERS = True  # Using full 3-layer architecture (hardware can handle it)
GRADIENT_ACCUMULATION_STEPS = 1  # No need for gradient accumulation with large GPU

# Evaluation settings
EVAL_EPISODES = 30  # More evaluation episodes for better statistics
EVAL_FREQUENCY = 500  # Evaluate every 500 episodes

###############################
# SYSTEM AND OPTIMIZATION
###############################

# Parallelization - utilizing the 64 cores and RTX 3090
NUM_PARALLEL_ENVS = 32  # Utilize multiple cores without overwhelming the system
BATCH_PREFETCH_NUM_WORKERS = 16  # Multiple workers for data loading

# Memory optimization - system has 125GB RAM, can use optimized implementation
MEMORY_IMPLEMENTATION = "optimized"  # Using memory-efficient implementation

# GPU settings - leveraging the RTX 3090
USE_GPU_PREPROCESSING = True  # Use GPU for frame preprocessing
USE_MULTI_GPU = False  # Only one GPU available

# Advanced features (disabled until implemented)
PRIORITIZED_REPLAY = False  # Disabled until PrioritizedReplayMemory is implemented
PRIORITIZED_REPLAY_ALPHA = 0.6  # Standard PER alpha (not used currently)
PRIORITIZED_REPLAY_BETA_START = 0.4  # Starting beta value (not used currently)
PRIORITIZED_REPLAY_BETA_FRAMES = 100000  # Frames over which to anneal beta (not used currently)
DUELING_NETWORK = False  # Disabled until Dueling Network architecture is implemented
DOUBLE_DQN = False  # Disabled until Double DQN logic is implemented

# Reporting and monitoring
TENSORBOARD_UPDATE_FREQUENCY = 100  # Update tensorboard every 100 steps
WANDB_ENABLED = False  # Set to True if using Weights & Biases

