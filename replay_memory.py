# In this file, implement the experience replay memory:
# - Create a data structure to store experiences (state, action, reward, next_state, done)
# - Implement adding new experiences to memory
# - Implement random batch sampling
# - Handle memory capacity limits (replacing old experiences when full)
# - Consider efficient storage and retrieval for large observation spaces
#
# Replay memory is crucial for DQN as it breaks correlation between consecutive 
# samples and enables more efficient learning from past experiences.
