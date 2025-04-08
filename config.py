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
DIFFICULTY = 1  # 0=Easy, 1=Normal, 2=Hard, 3=Expert
MODE = 0  # 0=Default mode, values vary by game

# Frame processing settings
FRAME_WIDTH = 84  # Downscaled from native 160
FRAME_HEIGHT = 84  # Downscaled from native 210
FRAME_STACK = 4  # Number of frames to stack together
FRAME_SKIP = 4  # Matches the environment's built-in frameskip for v5
NOOP_MAX = 30  # Maximum number of no-op actions at the start of an episode

# Visualization settings
RENDER_MODE = None  # Set to 'human' to see visualization, None for training
TRAINING_MODE = False  # When True, forces render_mode to None regardless of RENDER_MODE setting

###############################
# DEEP Q-LEARNING PARAMETERS
###############################

# Core DQN parameters
LEARNING_RATE = 0.0001  # Learning rate for optimizer
GAMMA = 0.99  # Discount factor for future rewards
BATCH_SIZE = 2048  # Number of transitions to sample for each update
MEMORY_CAPACITY = 1000000  # Maximum size of replay memory
TARGET_UPDATE_FREQUENCY = 1000  # Update target network every N steps
TRAINING_EPISODES = 3000  # Total number of training episodes

# Exploration parameters
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.1  # Final exploration rate
EPSILON_DECAY = 10000  # Number of steps for epsilon to decay from start to end
DEFAULT_EVALUATE_MODE = False  # Whether to use pure greedy policy (ε=0) by default

# Training control parameters
LEARNING_STARTS = 10000  # Number of steps before starting to train
UPDATE_FREQUENCY = 1  # Learn after every N steps
SAVE_FREQUENCY = 200  # Save model every N episodes

# Neural network settings
USE_ONE_CONV_LAYER = True  # Use only one convolutional layer (simplest architecture)
USE_TWO_CONV_LAYERS = False  # Use two convolutional layers (medium complexity)
USE_THREE_CONV_LAYERS = False  # Use three convolutional layers (most complex, default)
GRADIENT_ACCUMULATION_STEPS = 1  # Steps to accumulate gradients before update

# Evaluation settings
EVAL_EPISODES = 10  # Number of episodes to evaluate on
EVAL_FREQUENCY = 1000  # Evaluate every N episodes

###############################
# SYSTEM AND OPTIMIZATION
###############################

# Memory optimization
MEMORY_IMPLEMENTATION = "optimized"  # Options: "list", "array", "optimized"

# Hardware utilization
NUM_PARALLEL_ENVS = 32  # Number of parallel environments to run
BATCH_PREFETCH_NUM_WORKERS = 8  # Number of workers for batch prefetching
USE_MULTI_GPU = False  # Whether to use multiple GPUs if available

# GPU settings
USE_GPU_PREPROCESSING = True  # Use GPU for frame preprocessing if available

