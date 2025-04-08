# Deep Q-Network (DQN) for Atari Ice Hockey

## Overview 概述

This project implements a Deep Q-Network (DQN) to play the Atari game "Ice Hockey" using reinforcement learning. The implementation follows the algorithm described in the paper "Human-level control through deep reinforcement learning" by Mnih et al. (2015), including key features such as experience replay, target networks, and ε-greedy exploration.

本專案實現了一個深度 Q 網絡 (DQN)，用於通過強化學習玩 Atari 遊戲 "Ice Hockey"。實現基於 Mnih 等人 (2015) 發表的論文 "Human-level control through deep reinforcement learning"，包括經驗回放、目標網絡和 ε-greedy 探索等關鍵功能。

---

## Features 功能特點

- **Complete DQN Implementation**: Full implementation of the DQN algorithm with all key components.
- **Optimized Performance**: GPU acceleration, memory-efficient experience replay, and batch processing.
- **Flexible Architecture**: Configurable network depth (1-3 convolutional layers) for different performance needs.
- **Comprehensive Evaluation**: Tools for model evaluation, comparison, and visualization.
- **Educational Design**: Well-documented code with clear comments explaining the RL concepts.

- **完整 DQN 實現**：實現了具有所有關鍵組件的完整 DQN 算法。
- **優化性能**：GPU 加速、記憶體高效的經驗回放和批處理。
- **靈活架構**：可配置的網絡深度（1-3 個卷積層）以滿足不同的性能需求。
- **全面評估**：用於模型評估、比較和可視化的工具。
- **教育設計**：代碼文檔完善，清晰註解解釋強化學習概念。

---

## Project Structure 專案結構

- `main.py`: Unified entry point with command-line interface for all functionality
- `train.py`: Core training loop implementing the DQN algorithm
- `evaluate.py`: Evaluation and comparison of trained models
- `dqn_agent.py`: DQN agent implementation with Q-network and target network
- `q_network.py`: Neural network architecture for Q-function approximation
- `replay_memory.py`: Experience replay buffer implementations (list, array, and optimized)
- `env_wrappers.py`: Environment preprocessing and wrappers for Atari games
- `config.py`: Hyperparameters and configuration settings
- `utils.py`: Utility functions for device detection, visualization, etc.
- `logger.py`: Metrics tracking and visualization

---

## Installation 安裝

1. Clone the repository 克隆倉庫：
   ```bash
   git clone https://github.com/yourusername/dqn-ice-hockey.git
   cd dqn-ice-hockey
   ```

2. Create a virtual environment (recommended) 創建虛擬環境（推薦）：
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```

4. Install Atari ROMs (if needed) 安裝 Atari ROM（如需要）：
   ```bash
   python -m ale_py.roms --install-dir ./roms
   ```

---

## Usage 使用方法

### Training a Model 訓練模型

```bash
# Basic training with default parameters
python main.py train

# Training with custom parameters
python main.py train --episodes 5000 --learning_starts 5000 --gpu

# Training with visualization
python main.py train --render
```

### Evaluating a Model 評估模型

```bash
# Evaluate a specific model with rendering
python main.py evaluate models/best_model.pth --render --episodes 5

# Record a video of the agent playing
python main.py evaluate models/best_model.pth --video
```

### Comparing Models 比較模型

```bash
# Compare multiple models
python main.py compare models/model_ep200.pth models/model_ep400.pth models/model_ep600.pth
```

### Visualizing Results 可視化結果

```bash
# Visualize evaluation results
python main.py visualize results/eval_best_model.pkl
```

---

## Configuration 配置

Key hyperparameters can be modified in `config.py`:

- **Network Architecture**: Change `USE_ONE_CONV_LAYER`, `USE_TWO_CONV_LAYERS`, or `USE_THREE_CONV_LAYERS`
- **Learning Parameters**: Adjust `LEARNING_RATE`, `GAMMA`, `BATCH_SIZE`
- **Exploration**: Modify `EPSILON_START`, `EPSILON_END`, `EPSILON_DECAY`
- **Memory**: Change `MEMORY_CAPACITY` or `MEMORY_IMPLEMENTATION`

For quick changes, many parameters can be overridden via command-line arguments.

---

## System Requirements 系統需求

- **Python**: 3.8+
- **PyTorch**: 1.8+
- **CUDA**: Optional but recommended for GPU acceleration
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: Optional but recommended for faster training (NVIDIA GPU or Apple Silicon)

### Required Libraries 所需庫
- torch
- numpy
- gymnasium (with atari environments)
- ale-py
- matplotlib
- tqdm
- opencv-python

---

## Performance Tips 性能提示

- **GPU Acceleration**: Use `--gpu` flag to force GPU usage (if available)
- **Memory Optimization**: Reduce `MEMORY_CAPACITY` on systems with limited RAM
- **Training Speed**: Lower `BATCH_SIZE` for faster iterations, higher for better learning
- **Network Size**: Use `USE_ONE_CONV_LAYER=True` for faster training on weaker hardware

---

## Implementation Details 實現細節

The implementation follows the DQN algorithm pseudocode:

1. Initialize replay memory D with capacity N
2. Initialize action-value network Q with random weights
3. Initialize target network Q̂ with weights from Q
4. For each episode:
   - Initialize state
   - For each time step:
     - Select action using ε-greedy policy
     - Execute action and observe reward and next state
     - Store transition in replay memory D
     - Sample random mini-batch of transitions from D
     - Compute target Q-values using target network Q̂
     - Perform gradient descent step on the loss
     - Periodically update target network Q̂ with weights from Q

---

## Acknowledgments 致謝

- This implementation is based on the DQN algorithm described in [Mnih et al. (2015)](https://www.nature.com/articles/nature14236)
- Environment handling uses [Gymnasium](https://gymnasium.farama.org/)
- Special thanks to the PyTorch and AI research communities for their valuable resources

---

## License 許可證

This project is licensed under the MIT License - see the LICENSE file for details.