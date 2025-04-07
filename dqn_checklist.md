# DQN Training Steps Checklist

## Basic Setup
- [✓] Import necessary libraries (PyTorch, NumPy, Gymnasium, etc.)
- [✓] Set hyperparameters (learning rate, discount factor γ, ε-greedy parameters, etc.)
- [✓] Set up Atari Ice Hockey environment
  - [✓] Install Atari environments (`pip install ale-py` and `pip install gymnasium`)
  - [✓] Create environment using `gymnasium.make('ALE/IceHockey-v5')`
  - [✓] Apply necessary wrappers for Atari (frame stacking, frame skipping, etc.)
- [✓] Understand observation space (game frames) and action space (joystick movements)

## Environment Preprocessing Steps
- [✓] Frame Preprocessing
  - [✓] **Grayscale Conversion**: Convert RGB frames to grayscale to reduce dimensionality
  - [✓] **Image Resizing**: Resize frames from native resolution (210x160) to 84x84 for efficiency
  - [✓] **Pixel Normalization**: Normalize pixel values (typically to range [0,1])
  - [✓] **Frame Stacking**: Stack multiple frames (typically 4) to capture temporal information
  
- [✓] Action Space Handling
  - [✓] **Frameskip**: Skip frames to speed up training (typically 4 frames)
  - [✓] **Max-pooling**: Take maximum pixel value over skipped frames to capture fast movements

- [✓] Reward and Signal Processing
  - [✓] **Reward Clipping**: Clip rewards to {-1, 0, 1} to stabilize training
  - [✓] **Episode Termination**: Handle episode termination signals properly
  - [✓] **Life Loss Detection**: Treat loss of life as episode termination (optional)

- [✓] Performance Optimizations
  - [✓] **Fire on Reset**: Automatically press FIRE at the start of episodes when needed
  - [✓] **No-op Starts**: Begin episodes with random number of no-operations for exploration
  - [✓] **Memory-efficient Frame Processing**: Avoid unnecessary memory allocation

- [✓] Wrapper Implementation
  - [✓] Create custom wrapper classes inheriting from `gymnasium.Wrapper`
  - [✓] Chain wrappers in appropriate order
  - [✓] Test wrapped environment to verify correct implementation

## Network Architecture
- [✓] Define Q-network (input layer, hidden layers, output layer)
- [✓] Initialize target network with the same weights as the Q-network
- [✓] Set up optimizer (Adam) and loss function (MSE)

## Experience Replay
- [✓] Create experience replay memory
- [✓] Implement experience storage functionality (state, action, reward, next_state, done)
- [✓] Implement random batch sampling functionality

## Training Loop
- [✓] Initialize environment
- [✓] Execute actions and collect experiences
- [✓] Store experiences in memory
- [✓] When memory is sufficiently large, begin training:
  - [✓] Sample batch data from memory
  - [✓] Calculate current Q-values
  - [✓] Calculate target Q-values (using target network)
  - [✓] Calculate loss
  - [✓] Backpropagate to update Q-network
- [✓] Periodically update target network (e.g., every N steps)

## ε-greedy Exploration
- [✓] Implement ε-greedy strategy (random action with probability ε, best action with probability 1-ε)
- [✓] Decay ε value as training progresses

## Evaluation and Saving
- [✓] Periodically evaluate model performance (currently implemented but temporarily commented out)
- [✓] Save model weights
- [✓] Record training metrics (average reward, loss, etc.)

## Debugging Tips
- [✓] Ensure rewards are correctly set
- [✓] Monitor if training loss decreases normally
- [✓] Check if Q-values increase reasonably
- [✓] Ensure target network is updated periodically

## Ice Hockey Specific Considerations
- [✓] Understand the game dynamics (controlling players, scoring goals)
- [✓] Consider preprocessing game frames (grayscale conversion, resizing, normalization)
- [✓] Account for sparse rewards (goals are infrequent)
- [ ] Consider using reward shaping if training is slow (optional, can be implemented if needed)

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

## Pseudocode
1. Initialize replay memory D with capacity N
2. Initialize action-value network Q (θ₁) with random weights
3. Initialize target network Q_target (θ₂) ← θ₁

4. For each episode = 1 to M:
    5. Initialize initial state S₁

    6. For t = 1 to T:
        7. With probability ε, select a random action Aₜ (exploration)
        8. Otherwise, select Aₜ = argmaxₐ Q(Sₜ, a; θ₁) (exploitation)

        9. Execute action Aₜ, observe reward Rₜ₊₁ and next state Sₜ₊₁

        10. Store transition (Sₜ, Aₜ, Rₜ₊₁, Sₜ₊₁) into replay buffer D

        11. Sample a random minibatch of transitions (Sⱼ, Aⱼ, Rⱼ₊₁, Sⱼ₊₁) from D

        12. For each sample j in the minibatch:
            If Sⱼ₊₁ is terminal:
                yⱼ ← Rⱼ₊₁
            Else:
                yⱼ ← Rⱼ₊₁ + γ * maxₐ' Q_target(Sⱼ₊₁, a'; θ₂)

        13. Perform gradient descent step to minimize:
            L = (yⱼ - Q(Sⱼ, Aⱼ; θ₁))²

        14. Every C steps:
            Update target network: θ₂ ← θ₁

    End For
End For

