"""Module 2: Network Model and Rollout Buffer for Multi-Objective PPO (MO-PPO).

Exports:
- ActorCriticResNet: IMPALA-Lite Residual CNN with Policy Head and 5-Head Vector Critic.
- MultiObjectiveRolloutBuffer: Replay buffer storing uint8 observations and computing 5D GAE.
- RolloutBatch: Data structure for mini-batch delivery during PPO updates.
"""

from src.model.impala_cnn import ActorCriticResNet, ConvSequence, ResidualBlock, layer_init
from src.model.rollout_buffer import MultiObjectiveRolloutBuffer, RolloutBatch

__all__ = [
    "ActorCriticResNet",
    "MultiObjectiveRolloutBuffer",
    "RolloutBatch",
    "ResidualBlock",
    "ConvSequence",
    "layer_init",
]
