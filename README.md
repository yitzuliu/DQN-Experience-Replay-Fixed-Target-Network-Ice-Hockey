# Deep Q-Network (DQN) for Atari Ice Hockey

## Overview 概述

This project implements a Deep Q-Network (DQN) to play the Atari game "Ice Hockey" using reinforcement learning. The implementation includes key features such as experience replay, target networks, and ε-greedy exploration.

本專案實現了一個深度 Q 網絡 (DQN)，用於通過強化學習玩 Atari 遊戲 "Ice Hockey"。實現包括經驗回放、目標網絡和 ε-greedy 探索等關鍵功能。

---

## Features 功能特點

- **Experience Replay**: Efficiently reuses past experiences to improve learning stability.
- **Target Network**: Stabilizes training by using a separate target network.
- **ε-greedy Exploration**: Balances exploration and exploitation during training.
- **Logging and Visualization**: Tracks training progress and visualizes metrics.
- **Evaluation**: Periodically evaluates the trained agent's performance.

- **經驗回放**：高效重用過去的經驗以提高學習穩定性。
- **目標網絡**：通過使用單獨的目標網絡穩定訓練。
- **ε-greedy 探索**：在訓練期間平衡探索與利用。
- **日誌記錄與可視化**：跟蹤訓練進度並可視化指標。
- **評估**：定期評估訓練代理的性能。

---

## Installation 安裝

1. Clone the repository 克隆倉庫：
   ```bash
   git clone https://github.com/your-repo/dqn-ice-hockey.git
   cd dqn-ice-hockey
   ```

2. Install dependencies 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```

3. Install Atari ROMs 安裝 Atari ROM：
   ```bash
   python -m ale_py.roms --install-dir ./roms
   ```

---

## Usage 使用方法

### Training 訓練

To train the agent, run the following command:
運行以下命令訓練代理：
```bash
python main.py --mode train --episodes 10000
```

### Evaluation 評估

To evaluate a trained agent, run:
運行以下命令評估訓練代理：
```bash
python main.py --mode evaluate --model models/dqn_final.pth
```

### Visualization 可視化

To visualize training results, run:
運行以下命令可視化訓練結果：
```bash
python main.py --mode visualize
```

---

## File Structure 文件結構

- `main.py`: Main entry point for training, evaluation, and visualization.
- `train.py`: Implements the DQN training loop.
- `evaluate.py`: Evaluates the trained agent.
- `dqn_agent.py`: Defines the DQN agent.
- `q_network.py`: Implements the Q-network architecture.
- `replay_memory.py`: Implements experience replay memory.
- `env_wrappers.py`: Handles environment creation and preprocessing.
- `utils.py`: Utility functions for device setup, plotting, and statistics.
- `logger.py`: Tracks and visualizes training progress.
- `config.py`: Defines hyperparameters and environment settings.

- `main.py`：訓練、評估和可視化的主入口。
- `train.py`：實現 DQN 訓練循環。
- `evaluate.py`：評估訓練代理。
- `dqn_agent.py`：定義 DQN 代理。
- `q_network.py`：實現 Q 網絡架構。
- `replay_memory.py`：實現經驗回放記憶。
- `env_wrappers.py`：處理環境創建和預處理。
- `utils.py`：設備設置、繪圖和統計的工具函數。
- `logger.py`：跟蹤和可視化訓練進度。
- `config.py`：定義超參數和環境設置。

---

## Requirements 系統需求

- Python 3.8+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- tqdm
- ale-py

---

## Acknowledgments 致謝

This project is based on the original DQN algorithm introduced by DeepMind in the paper "Playing Atari with Deep Reinforcement Learning".

本專案基於 DeepMind 在論文《Playing Atari with Deep Reinforcement Learning》中提出的原始 DQN 算法。