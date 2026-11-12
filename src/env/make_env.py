"""Super Mario Bros Multi-Objective environment factory and compatibility shims.

Provides:
- apply_mario_compatibility_shim: shims mo-gymnasium and gym-super-mario-bros across versions.
- MOSuperMarioBrosEnv: robust base gymnasium environment wrapper for any world/stage.
- make_mario_env: single-environment factory with all wrappers applied.
- make_mario_vector_env: vectorized environment factory producing (N, 4, 84, 84) obs and (N, 5) rewards.
"""

import inspect
from typing import Callable, List, Optional, Sequence, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.env.action_wrappers import SpeedrunActionWrapper
from src.env.obs_wrappers import (
    FrameStack,
    GrayScaleObservation,
    ResizeObservation,
    SkipFrame,
)
from src.env.reward_wrappers import MultiObjectiveRewardWrapper


def apply_mario_compatibility_shim() -> bool:
    """Apply compatibility patches to mo_gymnasium and gym_super_mario_bros if present.
    
    Resolves discrepancies between gym-super-mario-bros (7.4.0 vs 9.x) and mo-gymnasium:
    - __init__ parameter signatures (rom_mode vs lost_levels/target/render_mode)
    - reset() return types (obs vs (obs, info))
    - step() return signatures (4-tuple vs 5-tuple)
    """
    try:
        import mo_gymnasium.envs.mario.mario as mm
        from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

        sig = inspect.signature(SuperMarioBrosEnv.__init__)
        has_rom_mode = "rom_mode" in sig.parameters

        def patched_init(
            self,
            rom_mode="pixel",
            lost_levels=False,
            target=None,
            objectives=["x_pos", "time", "death", "coin", "enemy"],
            death_as_penalty=False,
            render_mode=None,
        ):
            from gymnasium.utils import EzPickle

            EzPickle.__init__(self, rom_mode, lost_levels, target, objectives, death_as_penalty, render_mode)
            self.render_mode = render_mode
            if has_rom_mode:
                SuperMarioBrosEnv.__init__(self, rom_mode=rom_mode, lost_levels=lost_levels, target=target)
            else:
                SuperMarioBrosEnv.__init__(self, lost_levels=lost_levels, target=target, render_mode=render_mode)

            self.objectives = set(objectives)
            self.death_as_penalty = death_as_penalty
            if self.death_as_penalty:
                self.objectives.discard("death")
            self.reward_dim = len(self.objectives)

            low = np.empty(self.reward_dim, dtype=np.float32)
            high = np.empty(self.reward_dim, dtype=np.float32)
            obj_idx = 0
            if "x_pos" in self.objectives:
                low[obj_idx] = -np.inf
                high[obj_idx] = np.inf
                obj_idx += 1
            if "time" in self.objectives:
                low[obj_idx] = -np.inf
                high[obj_idx] = 0.0
                obj_idx += 1
            if "death" in self.objectives:
                low[obj_idx] = -25.0
                high[obj_idx] = 0.0
                obj_idx += 1
            if "coin" in self.objectives:
                low[obj_idx] = 0.0
                high[obj_idx] = 100.0
                obj_idx += 1
            if "enemy" in self.objectives:
                low[obj_idx] = 0.0
                high[obj_idx] = np.inf

            self.reward_space = spaces.Box(low=low, high=high, shape=(len(self.objectives),), dtype=np.float32)
            from nes_py.nes_env import SCREEN_SHAPE_24_BIT

            self.observation_space = spaces.Box(low=0, high=255, shape=SCREEN_SHAPE_24_BIT, dtype=np.uint8)
            self.action_space = spaces.Discrete(256)
            self.single_stage = True
            self.done_when_dead = True

        def patched_reset(self, seed=None, **kwargs):
            from gymnasium.utils import seeding

            self._np_random, seed = seeding.np_random(seed)
            self.coin = 0
            self.x_pos = 0
            self.time = 0
            self.score = 0
            self.stage_bonus = 0
            self.lives = 2
            res = SuperMarioBrosEnv.reset(self, **kwargs)
            if isinstance(res, tuple):
                obs, info = res
            else:
                obs, info = res, {}
            if self.render_mode == "human":
                self.render()
            return obs, info

        def patched_step(self, action):
            step_ret = SuperMarioBrosEnv.step(self, action)
            if len(step_ret) == 5:
                obs, reward, terminated, truncated, info = step_ret
                done = terminated or truncated
            else:
                obs, reward, done, info = step_ret
                terminated = done
                truncated = False

            if self.single_stage and info.get("flag_get", False):
                self.stage_bonus = 10000
                done = True
                terminated = True

            vec_reward = np.zeros(self.reward_dim, dtype=np.float32)
            obj_idx = 0

            # 1. x position
            if "x_pos" in self.objectives:
                xpos_r = float(info["x_pos"] - self.x_pos)
                self.x_pos = info["x_pos"]
                if xpos_r < -5:
                    xpos_r = 0.0
                vec_reward[obj_idx] = xpos_r
                obj_idx += 1

            # 2. time penalty
            if "time" in self.objectives:
                time_r = float(info["time"] - self.time)
                self.time = info["time"]
                if time_r > 0:
                    time_r = 0.0
                vec_reward[obj_idx] = time_r
                obj_idx += 1

            # 3. death
            current_life = info.get("life", self.lives)
            if self.lives > current_life or info.get("is_dead", False):
                death_r = -25.0
            else:
                death_r = 0.0
            if "death" in self.objectives:
                vec_reward[obj_idx] = death_r
                obj_idx += 1

            # 4. coin
            coin_r = 0.0
            if "coin" in self.objectives:
                coin_r = float((info["coins"] - self.coin) * 100)
                self.coin = info["coins"]
                vec_reward[obj_idx] = coin_r
                obj_idx += 1

            # 5. enemy
            if "enemy" in self.objectives:
                enemy_r = float(info["score"] - self.score)
                if coin_r > 0 or done:
                    enemy_r = 0.0
                self.score = info["score"]
                vec_reward[obj_idx] = enemy_r
                obj_idx += 1

            if self.death_as_penalty:
                vec_reward += death_r

            if self.done_when_dead:
                if self.lives > current_life and current_life > 0:
                    done = True
                    terminated = True

            self.lives = current_life
            info["score"] = info.get("score", 0) + self.stage_bonus

            if self.render_mode == "human":
                self.render()

            return obs, vec_reward, bool(terminated), bool(truncated), info

        mm.MOSuperMarioBros.__init__ = patched_init
        mm.MOSuperMarioBros.reset = patched_reset
        mm.MOSuperMarioBros.step = patched_step
        return True
    except Exception:
        return False


class MOSuperMarioBrosEnv(gym.Env):
    """Standalone, robust multi-objective Super Mario Bros environment.
    
    Directly wraps `gym_super_mario_bros.SuperMarioBrosEnv`, handling any (world, stage)
    and computing the 5D reward vector: [x_pos, time, death, coin, enemy].
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        world: int = 1,
        stage: int = 1,
        render_mode: Optional[str] = None,
        death_as_penalty: bool = False,
    ):
        super().__init__()
        self.world = world
        self.stage = stage
        self.render_mode = render_mode
        self.death_as_penalty = death_as_penalty

        from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
        from nes_py.nes_env import SCREEN_SHAPE_24_BIT

        sig = inspect.signature(SuperMarioBrosEnv.__init__)
        if "rom_mode" in sig.parameters:
            self._env = SuperMarioBrosEnv(rom_mode="pixel", lost_levels=False, target=(world, stage))
        else:
            self._env = SuperMarioBrosEnv(lost_levels=False, target=(world, stage), render_mode=render_mode)

        self.action_space = spaces.Discrete(256)
        self.observation_space = spaces.Box(low=0, high=255, shape=SCREEN_SHAPE_24_BIT, dtype=np.uint8)
        self.reward_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -25.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, 0.0, 0.0, 100.0, np.inf], dtype=np.float32),
            shape=(5,),
            dtype=np.float32,
        )
        self.reward_dim = 5
        self.single_stage = True
        self.done_when_dead = True

        self.coin = 0
        self.x_pos = None
        self.time = None
        self.score = 0
        self.lives = 2
        self.stage_bonus = 0

    @property
    def screen(self):
        return getattr(self._env, "screen", None)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        res = self._env.reset()
        if isinstance(res, tuple):
            obs, info = res
        else:
            obs, info = res, {}

        self.coin = info.get("coins", 0)
        self.x_pos = info.get("x_pos", None)
        self.time = info.get("time", None)
        self.score = info.get("score", 0)
        self.lives = info.get("life", 2)
        self.stage_bonus = 0

        info.setdefault("flag_get", False)
        info.setdefault("x_pos", 40 if self.x_pos is None else self.x_pos)
        info.setdefault("time", 400 if self.time is None else self.time)
        info.setdefault("coins", self.coin)
        info.setdefault("score", self.score)

        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        step_ret = self._env.step(action)
        if len(step_ret) == 5:
            obs, reward, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_ret
            terminated = bool(done)
            truncated = False

        if self.single_stage and info.get("flag_get", False):
            self.stage_bonus = 10000
            done = True
            terminated = True

        vec_reward = np.zeros(5, dtype=np.float32)

        # 1. x_pos delta
        curr_x = info.get("x_pos", 0)
        if self.x_pos is None:
            self.x_pos = curr_x
            xpos_r = 0.0
        else:
            xpos_r = float(curr_x - self.x_pos)
            self.x_pos = curr_x
            if xpos_r < -5:
                xpos_r = 0.0
        vec_reward[0] = xpos_r

        # 2. time penalty (<= 0)
        curr_time = info.get("time", 0)
        if self.time is None:
            self.time = curr_time
            time_r = 0.0
        else:
            time_r = float(curr_time - self.time)
            self.time = curr_time
            if time_r > 0:
                time_r = 0.0
        vec_reward[1] = time_r

        # 3. death penalty (-25.0)
        curr_life = info.get("life", self.lives)
        if self.lives > curr_life or info.get("is_dead", False):
            death_r = -25.0
        else:
            death_r = 0.0
        vec_reward[2] = death_r

        # 4. coin reward (+100.0)
        curr_coin = info.get("coins", 0)
        coin_r = float((curr_coin - self.coin) * 100.0)
        self.coin = curr_coin
        vec_reward[3] = coin_r

        # 5. enemy reward
        curr_score = info.get("score", 0)
        enemy_r = float(curr_score - self.score)
        if coin_r > 0 or done:
            enemy_r = 0.0
        self.score = curr_score
        vec_reward[4] = enemy_r

        if self.death_as_penalty:
            vec_reward += death_r

        if self.done_when_dead:
            if self.lives > curr_life and curr_life > 0:
                done = True
                terminated = True

        self.lives = curr_life
        info["score"] = info.get("score", 0) + self.stage_bonus

        if self.render_mode == "human":
            self.render()

        return obs, vec_reward, bool(terminated), bool(truncated), info

    @property
    def screen(self):
        return getattr(self._env, "screen", None)

    @property
    def unwrapped(self):
        return getattr(self._env, "unwrapped", self)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()


class VectorInfos(list):
    """List subclass that also allows key-based lookup for Gymnasium/CleanRL vector infos."""

    def __getitem__(self, key):
        if isinstance(key, str):
            return [info.get(key) for info in self]
        return super().__getitem__(key)

    def get(self, key, default=None):
        if any(isinstance(info, dict) and key in info for info in self):
            return [info.get(key) if isinstance(info, dict) else None for info in self]
        return default


class MOSyncVectorEnv(gym.vector.VectorEnv):
    """Synchronous vector environment supporting 5-dimensional multi-objective reward vectors.
    
    Batches observations to (N, 4, 84, 84) uint8 and rewards to (N, 5) float32.
    Automatically resets sub-environments on termination/truncation and preserves
    `final_observation` and `final_info` in `infos`.
    """

    def __init__(self, env_fns: Sequence[Callable[[], gym.Env]]):
        self.env_fns = env_fns
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

        first_env = self.envs[0]
        self.single_observation_space = first_env.observation_space
        self.single_action_space = first_env.action_space
        self.reward_dim = getattr(first_env, "reward_dim", 5)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.num_envs, *self.single_observation_space.shape),
            dtype=self.single_observation_space.dtype,
        )
        self.action_space = spaces.MultiDiscrete([self.single_action_space.n] * self.num_envs)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, VectorInfos]:
        obs_list = []
        info_list = VectorInfos()
        for i, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + i
            obs, info = env.reset(seed=env_seed, options=options)
            obs_list.append(obs)
            info_list.append(info)
        return np.stack(obs_list), info_list

    def step(
        self, actions: Union[np.ndarray, Sequence[int]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, VectorInfos]:
        obs_list = []
        reward_list = []
        term_list = []
        trunc_list = []
        info_list = VectorInfos()

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                info["final_observation"] = obs
                info["final_info"] = info.copy()
                obs, _ = env.reset()

            obs_list.append(obs)
            reward_list.append(reward)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)

        return (
            np.stack(obs_list),
            np.asarray(reward_list, dtype=np.float32),
            np.asarray(term_list, dtype=bool),
            np.asarray(trunc_list, dtype=bool),
            info_list,
        )

    def close(self):
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass


def make_mario_env(
    world: int = 1,
    stage: int = 1,
    render_mode: Optional[str] = None,
    skip: int = 4,
    num_stack: int = 4,
    obs_shape: Tuple[int, int] = (84, 84),
    scaling: Optional[Sequence[float]] = None,
    beta_milestone: float = 0.0,
    beta_stuck: float = 0.0,
    stuck_steps: int = 60,
    death_as_penalty: bool = False,
) -> gym.Env:
    """Create a fully wrapped Super Mario Bros environment for Multi-Objective RL.
    
    Wrappers applied:
    1. Base MOSuperMarioBrosEnv (5D reward vector)
    2. SpeedrunActionWrapper (Discrete(7))
    3. SkipFrame(skip=4)
    4. GrayScaleObservation
    5. ResizeObservation(shape=(84, 84))
    6. FrameStack(num_stack=4) -> (4, 84, 84) uint8
    7. MultiObjectiveRewardWrapper (component scaling + shaping + raw metrics)
    """
    apply_mario_compatibility_shim()

    env = MOSuperMarioBrosEnv(
        world=world,
        stage=stage,
        render_mode=render_mode,
        death_as_penalty=death_as_penalty,
    )
    env = SpeedrunActionWrapper(env)
    env = SkipFrame(env, skip=skip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=obs_shape)
    env = FrameStack(env, num_stack=num_stack)
    env = MultiObjectiveRewardWrapper(
        env,
        scaling=scaling,
        beta_milestone=beta_milestone,
        beta_stuck=beta_stuck,
        stuck_steps=stuck_steps,
    )
    return env


def make_mario_vector_env(
    num_envs: int = 8,
    world: int = 1,
    stage: int = 1,
    render_mode: Optional[str] = None,
    skip: int = 4,
    num_stack: int = 4,
    obs_shape: Tuple[int, int] = (84, 84),
    scaling: Optional[Sequence[float]] = None,
    beta_milestone: float = 0.0,
    beta_stuck: float = 0.0,
    stuck_steps: int = 60,
    death_as_penalty: bool = False,
) -> MOSyncVectorEnv:
    """Create a synchronous vectorized environment running `num_envs` Mario instances.
    
    Returns:
        MOSyncVectorEnv yielding batched observations (N, 4, 84, 84) uint8
        and batched rewards (N, 5) float32.
    """
    def make_thunk():
        return make_mario_env(
            world=world,
            stage=stage,
            render_mode=render_mode,
            skip=skip,
            num_stack=num_stack,
            obs_shape=obs_shape,
            scaling=scaling,
            beta_milestone=beta_milestone,
            beta_stuck=beta_stuck,
            stuck_steps=stuck_steps,
            death_as_penalty=death_as_penalty,
        )

    return MOSyncVectorEnv([make_thunk for _ in range(num_envs)])
