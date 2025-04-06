# DQN Training Steps Checklist

## Basic Setup
- [✓] Import necessary libraries (PyTorch, NumPy, Gymnasium, etc.)
- [✓] Set hyperparameters (learning rate, discount factor γ, ε-greedy parameters, etc.)
- [✓] Set up Atari Ice Hockey environment
  - [✓] Install Atari environments (`pip install ale-py` and `pip install gymnasium`)
  - [✓] Create environment using `gymnasium.make('ALE/IceHockey-v5')`
  - [✓] Apply necessary wrappers for Atari (frame stacking, frame skipping, etc.)
- [✓] Understand observation space (game frames) and action space (joystick movements)

## Network Architecture
- [✓] Define Q-network (input layer, hidden layers, output layer)
- [✓] Initialize target network with the same weights as the Q-network
- [✓] Set up optimizer (Adam) and loss function (MSE)

## Experience Replay
- [✓] Create experience replay memory
- [✓] Implement experience storage functionality (state, action, reward, next_state, done)
- [✓] Implement random batch sampling functionality

## Training Loop
- [ ] Initialize environment
- [ ] Execute actions and collect experiences
- [ ] Store experiences in memory
- [ ] When memory is sufficiently large, begin training:
  - [ ] Sample batch data from memory
  - [ ] Calculate current Q-values
  - [ ] Calculate target Q-values (using target network)
  - [ ] Calculate loss
  - [ ] Backpropagate to update Q-network
- [ ] Periodically update target network (e.g., every N steps)

## ε-greedy Exploration
- [✓] Implement ε-greedy strategy (random action with probability ε, best action with probability 1-ε)
- [✓] Decay ε value as training progresses

## Evaluation and Saving
- [ ] Periodically evaluate model performance
- [✓] Save model weights
- [✓] Record training metrics (average reward, loss, etc.)

## Debugging Tips
- [ ] Ensure rewards are correctly set
- [ ] Monitor if training loss decreases normally
- [ ] Check if Q-values increase reasonably
- [ ] Ensure target network is updated periodically

## Ice Hockey Specific Considerations
- [✓] Understand the game dynamics (controlling players, scoring goals)
- [ ] Consider preprocessing game frames (grayscale conversion, resizing, normalization)
- [ ] Account for sparse rewards (goals are infrequent)
- [ ] Consider using reward shaping if training is slow

## Suggested Project Files
- [✓] `main.py` - Main script to run the training process
- [✓] `dqn_agent.py` - DQN agent implementation
- [✓] `replay_memory.py` - Experience replay buffer implementation
- [✓] `q_network.py` - Neural network model definition
- [✓] `env_wrappers.py` - Environment wrappers for Atari preprocessing
- [✓] `train.py` - Training loop implementation
- [✓] `evaluate.py` - Evaluation script for trained models
- [✓] `utils.py` - Utility functions for data processing and visualization
- [✓] `config.py` - Configuration parameters and hyperparameters
- [✓] `logger.py` - Logging and metrics tracking
