"""Reward wrappers for Super Mario Bros Multi-Objective PPO.

Tracks raw 5D rewards, applies component scaling, and provides optional milestone / anti-stuck shaping.
"""

from typing import Optional, Sequence, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Default component scaling factors from Section 6.1 of the design spec
# Brings [dx, dt, death, coin, enemy] to roughly [-2.0, 2.0]
DEFAULT_SCALING = np.array([0.05, 0.05, 0.08, 0.01, 0.02], dtype=np.float32)


class MultiObjectiveRewardWrapper(gym.Wrapper):
    """Multi-objective reward processing wrapper.
    
    1. Preserves raw 5D rewards in `info['raw_rewards']` and running episode sum in `info['episode_raw_rewards']`.
    2. Scales the 5 components by `scaling` factors.
    3. Optionally shapes the x_pos reward with a bonus for new maximum x coordinates (+beta_milestone)
       and a penalty for staying stuck without advancing for `stuck_steps` (-beta_stuck).
    """

    def __init__(
        self,
        env: gym.Env,
        scaling: Optional[Sequence[float]] = None,
        beta_milestone: float = 0.0,
        beta_stuck: float = 0.0,
        stuck_steps: int = 60,
    ):
        super().__init__(env)
        if scaling is None:
            self.scaling = DEFAULT_SCALING.copy()
        else:
            self.scaling = np.asarray(scaling, dtype=np.float32)
            if self.scaling.shape != (5,):
                raise ValueError(f"scaling must be a 5-element array, got {self.scaling.shape}")

        self.beta_milestone = float(beta_milestone)
        self.beta_stuck = float(beta_stuck)
        self.stuck_steps = int(stuck_steps)

        self.reward_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32,
        )

        self.episode_raw_rewards = np.zeros(5, dtype=np.float32)
        self.max_x_pos = 0.0
        self.stuck_counter = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self.episode_raw_rewards = np.zeros(5, dtype=np.float32)
        self.max_x_pos = float(info.get("x_pos", 0.0))
        self.stuck_counter = 0

        info["raw_rewards"] = np.zeros(5, dtype=np.float32)
        info["episode_raw_rewards"] = self.episode_raw_rewards.copy()
        info.setdefault("flag_get", False)
        info.setdefault("x_pos", int(self.max_x_pos))
        info.setdefault("time", 0)
        info.setdefault("coins", 0)
        info.setdefault("score", 0)

        return obs, info

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. Parse raw 5-element reward vector
        raw_r = np.reshape(np.asarray(reward, dtype=np.float32), (5,)).copy()
        self.episode_raw_rewards += raw_r

        # Populate metrics into info dict
        info["raw_rewards"] = raw_r
        info["episode_raw_rewards"] = self.episode_raw_rewards.copy()
        info.setdefault("flag_get", False)
        info.setdefault("time", 0)
        info.setdefault("coins", 0)
        info.setdefault("score", 0)

        # 2. Component scaling
        scaled_r = (raw_r * self.scaling).astype(np.float32)

        # 3. Milestone and Anti-Stuck Shaping (applied to dimension 0: x_pos)
        curr_x = float(info.get("x_pos", self.max_x_pos))
        if self.beta_milestone > 0.0 or self.beta_stuck > 0.0:
            if curr_x > self.max_x_pos:
                if self.beta_milestone > 0.0:
                    scaled_r[0] += float(self.beta_milestone)
                self.max_x_pos = curr_x
                self.stuck_counter = 0
            else:
                self.stuck_counter += 1
                if self.beta_stuck > 0.0 and self.stuck_counter >= self.stuck_steps:
                    scaled_r[0] -= float(self.beta_stuck)
        else:
            if curr_x > self.max_x_pos:
                self.max_x_pos = curr_x

        info["x_pos"] = int(curr_x)

        return obs, scaled_r, terminated, truncated, info
