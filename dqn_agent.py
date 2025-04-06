# In this file, implement the DQN agent class that will:
# - Initialize Q-network and target network
# - Implement epsilon-greedy action selection
# - Implement learning functionality:
#   - Sample batches from replay memory
#   - Calculate Q-values and target Q-values
#   - Update network weights
#   - Update target network periodically
# - Provide save/load functionality for the trained model
#
# This is the core component that brings together the neural network,
# replay memory, and training logic to implement the DQN algorithm.
