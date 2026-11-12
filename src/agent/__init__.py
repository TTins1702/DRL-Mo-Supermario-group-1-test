"""MO-PPO Agent module for Multi-Objective Super Mario Bros.

Phân hệ 3 (Module 3):
- Multi-Objective Proximal Policy Optimization (MO-PPO) agent.
- 5D Per-Objective Advantage Normalization.
- Preference vector scalarization.
- 5-Head Vector Critic loss with Smooth L1 (Huber) loss.
"""

from src.agent.mo_ppo_agent import MOPPOAgent

__all__ = ["MOPPOAgent"]
