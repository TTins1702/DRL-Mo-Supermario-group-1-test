"""Unit tests for Super Mario Bros environment wrappers and factories."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest

from src.env.action_wrappers import (
    NES_BUTTON_MAP,
    SPEEDRUN_ACTIONS,
    SpeedrunActionWrapper,
)
from src.env.make_env import (
    MOSuperMarioBrosEnv,
    MOSyncVectorEnv,
    VectorInfos,
    apply_mario_compatibility_shim,
    make_mario_env,
    make_mario_vector_env,
)
from src.env.obs_wrappers import (
    FrameStack,
    GrayScaleObservation,
    ResizeObservation,
    SkipFrame,
)
from src.env.reward_wrappers import (
    DEFAULT_SCALING,
    MultiObjectiveRewardWrapper,
)


class DummyMarioEnv(gym.Env):
    """Mock base Mario environment for fast, deterministic unit testing."""

    def __init__(self, terminate_at: int = 100):
        super().__init__()
        self.action_space = spaces.Discrete(256)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(240, 256, 3), dtype=np.uint8
        )
        self.reward_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        self.reward_dim = 5
        self.terminate_at = terminate_at
        self.step_count = 0
        self.last_action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.last_action = None
        obs = np.full((240, 256, 3), 42, dtype=np.uint8)
        info = {
            "x_pos": 40,
            "time": 400,
            "coins": 0,
            "score": 0,
            "flag_get": False,
            "life": 2,
        }
        return obs, info

    def step(self, action):
        self.step_count += 1
        self.last_action = action
        obs = np.full((240, 256, 3), (42 + self.step_count) % 256, dtype=np.uint8)

        # 5D reward: [dx, dt, death, coin, enemy]
        reward = np.array([2.0, -0.1, 0.0, 0.0, 0.0], dtype=np.float32)
        terminated = self.step_count >= self.terminate_at
        truncated = False
        info = {
            "x_pos": 40 + self.step_count * 2,
            "time": 400 - self.step_count,
            "coins": 0,
            "score": self.step_count * 10,
            "flag_get": bool(terminated),
            "life": 2,
        }
        return obs, reward, terminated, truncated, info


# =============================================================================
# Action Wrapper Tests
# =============================================================================


def test_speedrun_action_wrapper():
    dummy = DummyMarioEnv()
    wrapped = SpeedrunActionWrapper(dummy)

    assert isinstance(wrapped.action_space, spaces.Discrete)
    assert wrapped.action_space.n == 7

    expected_bytes = {
        0: 0,                           # NOOP
        1: NES_BUTTON_MAP["right"],     # 128
        2: NES_BUTTON_MAP["right"] | NES_BUTTON_MAP["A"],  # 129
        3: NES_BUTTON_MAP["right"] | NES_BUTTON_MAP["B"],  # 130
        4: NES_BUTTON_MAP["right"] | NES_BUTTON_MAP["A"] | NES_BUTTON_MAP["B"],  # 131
        5: NES_BUTTON_MAP["A"],         # 1
        6: NES_BUTTON_MAP["left"],      # 64
    }

    for action_idx, expected_byte in expected_bytes.items():
        mapped = wrapped.action(action_idx)
        assert mapped == expected_byte, f"Action {action_idx} mapped to {mapped}, expected {expected_byte}"

    meanings = wrapped.get_action_meanings()
    assert len(meanings) == 7
    assert meanings[0] == "NOOP"
    assert meanings[4] == "right A B"


# =============================================================================
# Observation Wrapper Tests
# =============================================================================


def test_skip_frame():
    dummy = DummyMarioEnv()
    wrapped = SkipFrame(dummy, skip=4)
    wrapped.reset()

    # Step once: should advance dummy 4 times and sum rewards
    obs, reward, term, trunc, info = wrapped.step(1)
    assert dummy.step_count == 4
    # Each step gives [2.0, -0.1, 0, 0, 0], sum over 4 frames is [8.0, -0.4, 0, 0, 0]
    np.testing.assert_allclose(reward, [8.0, -0.4, 0.0, 0.0, 0.0], rtol=1e-5)
    assert term is False
    assert trunc is False


def test_skip_frame_early_break():
    dummy = DummyMarioEnv(terminate_at=2)
    wrapped = SkipFrame(dummy, skip=4)
    wrapped.reset()

    obs, reward, term, trunc, info = wrapped.step(1)
    assert term is True
    # Should have broken after frame 2
    assert dummy.step_count == 2
    np.testing.assert_allclose(reward, [4.0, -0.2, 0.0, 0.0, 0.0], rtol=1e-5)


def test_grayscale_observation():
    dummy = DummyMarioEnv()
    wrapped = GrayScaleObservation(dummy)
    assert wrapped.observation_space.shape == (240, 256)
    assert wrapped.observation_space.dtype == np.uint8

    obs, _ = wrapped.reset()
    assert obs.shape == (240, 256)
    assert obs.dtype == np.uint8


def test_resize_observation():
    dummy = DummyMarioEnv()
    gray = GrayScaleObservation(dummy)
    resized = ResizeObservation(gray, shape=(84, 84))

    assert resized.observation_space.shape == (84, 84)
    assert resized.observation_space.dtype == np.uint8

    obs, _ = resized.reset()
    assert obs.shape == (84, 84)
    assert obs.dtype == np.uint8


def test_frame_stack():
    dummy = DummyMarioEnv()
    gray = GrayScaleObservation(dummy)
    resized = ResizeObservation(gray, shape=(84, 84))
    stacked = FrameStack(resized, num_stack=4)

    assert stacked.observation_space.shape == (4, 84, 84)
    assert stacked.observation_space.dtype == np.uint8

    obs, _ = stacked.reset()
    assert obs.shape == (4, 84, 84)
    assert obs.dtype == np.uint8

    # All 4 stacked frames on reset should be identical
    np.testing.assert_array_equal(obs[0], obs[1])
    np.testing.assert_array_equal(obs[1], obs[2])
    np.testing.assert_array_equal(obs[2], obs[3])

    obs_step, _, _, _, _ = stacked.step(0)
    assert obs_step.shape == (4, 84, 84)
    assert obs_step.dtype == np.uint8


# =============================================================================
# Reward Wrapper Tests
# =============================================================================


def test_multi_objective_reward_wrapper():
    dummy = DummyMarioEnv()
    scaling = [0.05, 0.05, 0.08, 0.01, 0.02]
    wrapped = MultiObjectiveRewardWrapper(
        dummy,
        scaling=scaling,
        beta_milestone=0.5,
        beta_stuck=0.1,
        stuck_steps=2,
    )

    obs, info = wrapped.reset()
    assert "raw_rewards" in info
    assert "episode_raw_rewards" in info
    np.testing.assert_allclose(info["episode_raw_rewards"], [0, 0, 0, 0, 0])

    # Step 1: dx = 2.0 (new max x = 42 > 40), so milestone applied: 2.0 * 0.05 + 0.5 = 0.6
    obs, reward, term, trunc, info = wrapped.step(0)
    np.testing.assert_allclose(info["raw_rewards"], [2.0, -0.1, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(info["episode_raw_rewards"], [2.0, -0.1, 0.0, 0.0, 0.0])
    assert info["x_pos"] == 42
    assert reward.shape == (5,)
    assert reward.dtype == np.float32
    assert pytest.approx(reward[0], rel=1e-4) == 0.6  # 2.0 * 0.05 + 0.5 milestone
    assert pytest.approx(reward[1], rel=1e-4) == -0.005  # -0.1 * 0.05


def test_anti_stuck_penalty():
    class StuckEnv(gym.Env):
        def __init__(self):
            self.action_space = spaces.Discrete(7)
            self.observation_space = spaces.Box(0, 255, shape=(84, 84), dtype=np.uint8)
            self.reward_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)

        def reset(self, **kwargs):
            return np.zeros((84, 84), dtype=np.uint8), {"x_pos": 100}

        def step(self, action):
            # Not advancing x_pos
            return np.zeros((84, 84), dtype=np.uint8), np.zeros(5, dtype=np.float32), False, False, {"x_pos": 100}

    stuck_env = StuckEnv()
    wrapper = MultiObjectiveRewardWrapper(
        stuck_env,
        scaling=[1.0, 1.0, 1.0, 1.0, 1.0],
        beta_milestone=0.0,
        beta_stuck=0.25,
        stuck_steps=2,
    )

    wrapper.reset()
    # Step 1: stuck counter = 1 (< 2), no penalty
    _, r1, _, _, _ = wrapper.step(0)
    assert r1[0] == 0.0

    # Step 2: stuck counter = 2 (>= 2), penalty applied
    _, r2, _, _, _ = wrapper.step(0)
    assert pytest.approx(r2[0]) == -0.25


# =============================================================================
# Vector Environment Tests
# =============================================================================


def test_mo_sync_vector_env():
    num_envs = 3
    vec_env = MOSyncVectorEnv([lambda: DummyMarioEnv(terminate_at=3) for _ in range(num_envs)])

    assert vec_env.num_envs == num_envs
    obs_batch, infos = vec_env.reset()
    assert obs_batch.shape == (num_envs, 240, 256, 3)
    assert obs_batch.dtype == np.uint8
    assert len(infos) == num_envs

    actions = [1, 2, 3]
    obs_b, r_b, term_b, trunc_b, infos = vec_env.step(actions)
    assert obs_b.shape == (num_envs, 240, 256, 3)
    assert obs_b.dtype == np.uint8
    assert r_b.shape == (num_envs, 5)
    assert r_b.dtype == np.float32
    assert term_b.shape == (num_envs,)
    assert trunc_b.shape == (num_envs,)

    # Test auto-reset and final_observation on episode termination
    # Sub-envs terminate at step 3
    vec_env.step(actions)
    obs_b, r_b, term_b, trunc_b, infos = vec_env.step(actions)
    assert all(term_b)
    # Check that final_info was captured
    for i in range(num_envs):
        assert "final_observation" in infos[i]
        assert "final_info" in infos[i]

    # Check VectorInfos dict-like access
    final_infos = infos.get("final_info")
    assert len(final_infos) == num_envs

    vec_env.close()


# =============================================================================
# Live Mario Environment Smoke Tests
# =============================================================================


def test_make_mario_env_live():
    """Verify live Super Mario Bros integration with all 5 wrappers attached."""
    env = make_mario_env(world=1, stage=1)
    try:
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 7

        obs, info = env.reset()
        assert obs.shape == (4, 84, 84)
        assert obs.dtype == np.uint8
        assert "raw_rewards" in info
        assert "episode_raw_rewards" in info
        assert "x_pos" in info
        assert "time" in info
        assert "coins" in info
        assert "score" in info

        # Step with action 4 (sprint + jump right)
        obs, reward, terminated, truncated, info = env.step(4)
        assert obs.shape == (4, 84, 84)
        assert obs.dtype == np.uint8
        assert reward.shape == (5,)
        assert reward.dtype == np.float32
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "episode_raw_rewards" in info
    finally:
        env.close()


def test_make_mario_vector_env_live():
    """Verify vectorized Super Mario Bros integration with 2 parallel envs."""
    vec_env = make_mario_vector_env(num_envs=2, world=1, stage=1)
    try:
        assert vec_env.num_envs == 2
        obs_batch, infos = vec_env.reset()
        assert obs_batch.shape == (2, 4, 84, 84)
        assert obs_batch.dtype == np.uint8
        assert len(infos) == 2

        obs_b, r_b, term_b, trunc_b, infos = vec_env.step([1, 4])
        assert obs_b.shape == (2, 4, 84, 84)
        assert obs_b.dtype == np.uint8
        assert r_b.shape == (2, 5)
        assert r_b.dtype == np.float32
        assert term_b.shape == (2,)
        assert trunc_b.shape == (2,)
    finally:
        vec_env.close()
