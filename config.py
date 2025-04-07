# In this file, define all the hyperparameters needed for DQN training:
# - Learning rate (e.g., 0.0001)
# - Discount factor (gamma, e.g., 0.99)
# - Epsilon values (start, end, decay rate for exploration)
# - Memory capacity (e.g., 10000)
# - Batch size (e.g., 32 or 64)
# - Target network update frequency
# - Number of training episodes
# - Frame preprocessing parameters (size, grayscale, etc.)
# - Other environment-specific settings
#
# These parameters will be imported by other modules to ensure consistency
# throughout the project.

# Environment settings
ENV_NAME = 'ALE/IceHockey-v5'
RENDER_MODE = "human"  # Enable visualization of the environment
ACTION_SPACE_SIZE = 18  # IceHockey has 18 possible actions
OBS_TYPE = "rgb"  # Default observation type for ALE/IceHockey-v5

# Frame processing
FRAME_WIDTH = 84  # Downscaled from native 160
FRAME_HEIGHT = 84  # Downscaled from native 210
FRAME_STACK = 4  # Number of frames to stack together
FRAME_SKIP = 4   # Matches the environment's built-in frameskip for v5

# DQN hyperparameters
LEARNING_RATE = 0.0001  # Learning rate for optimizer
GAMMA = 0.99            # Discount factor
BATCH_SIZE = 32         # Size of minibatch sampled from replay memory
MEMORY_CAPACITY = 10000 # Capacity of replay memory
TARGET_UPDATE_FREQUENCY = 1000  # Update target network every N steps
TRAINING_EPISODES = 10000        # Total number of training episodes

# Exploration parameters
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.1    # Final exploration rate
EPSILON_DECAY = 1000000  # Number of steps for epsilon to decay from start to end

# Training settings
LEARNING_STARTS = 10000   # Number of steps before starting to train
UPDATE_FREQUENCY = 1      # Learn after every step (matches the pseudocode)
SAVE_FREQUENCY = 1000     # Save model every N episodes

# Evaluation settings
EVAL_EPISODES = 10       # Number of episodes to evaluate on
EVAL_FREQUENCY = 1000    # Evaluate every N episodes

