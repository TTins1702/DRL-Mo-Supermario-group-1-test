"""Observation wrappers for Super Mario Bros speedrun RL.

Includes SkipFrame, GrayScaleObservation, ResizeObservation, and FrameStack.
"""

from collections import deque
from typing import Sequence, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None


class SkipFrame(gym.Wrapper):
    """Repeat action across `skip` NES frames and sum the 5D multi-objective reward vector.
    
    If an episode terminates or is truncated before reaching `skip` frames, loop breaks
    immediately.
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        if skip < 1:
            raise ValueError(f"skip must be >= 1, got {skip}")
        self.skip = skip

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        total_reward = None
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.skip):
            obs, reward, term, trunc, info = self.env.step(action)
            r_vec = np.asarray(reward, dtype=np.float32)

            if total_reward is None:
                total_reward = r_vec.copy()
            else:
                total_reward += r_vec

            terminated = bool(term)
            truncated = bool(trunc)
            if terminated or truncated:
                break

        if total_reward is None:
            total_reward = np.zeros(5, dtype=np.float32)

        return obs, total_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert RGB frame (240, 256, 3) to single-channel 2D grayscale (240, 256)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_shape = self.env.observation_space.shape
        if len(obs_shape) >= 2:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=obs_shape[:2],
                dtype=np.uint8,
            )
        else:
            self.observation_space = self.env.observation_space

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 3 and obs.shape[2] == 3:
            if cv2 is not None:
                return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            # Fallback if cv2 is not installed
            return np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return obs.astype(np.uint8)


class ResizeObservation(gym.ObservationWrapper):
    """Resize 2D observation to target shape (default 84x84) using INTER_AREA interpolation."""

    def __init__(self, env: gym.Env, shape: Union[int, Sequence[int]] = (84, 84)):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape: Tuple[int, int] = (shape, shape)
        else:
            self.shape = (int(shape[0]), int(shape[1]))

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.shape,
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        target_w, target_h = self.shape[1], self.shape[0]
        if cv2 is not None:
            resized = cv2.resize(obs, (target_w, target_h), interpolation=cv2.INTER_AREA)
            return resized.astype(np.uint8)
        elif Image is not None:
            pil_img = Image.fromarray(obs)
            resized = pil_img.resize((target_w, target_h), resample=Image.BILINEAR)
            return np.array(resized, dtype=np.uint8)
        else:
            raise RuntimeError("Either cv2 or PIL is required for ResizeObservation")


class FrameStack(gym.Wrapper):
    """Stack `num_stack` consecutive frames along the channel axis (shape (4, 84, 84)).
    
    Stores frames in uint8 format (0..255) to conserve RAM memory.
    """

    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        if num_stack < 1:
            raise ValueError(f"num_stack must be >= 1, got {num_stack}")
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        low = np.repeat(
            self.env.observation_space.low[np.newaxis, ...],
            num_stack,
            axis=0,
        )
        high = np.repeat(
            self.env.observation_space.high[np.newaxis, ...],
            num_stack,
            axis=0,
        )
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(num_stack, *self.env.observation_space.shape),
            dtype=np.uint8,
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        return np.stack(self.frames, axis=0).astype(np.uint8)
