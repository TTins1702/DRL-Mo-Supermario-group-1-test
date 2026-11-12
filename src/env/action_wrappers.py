"""Action wrappers for Super Mario Bros speedrun reinforcement learning.

Maps discrete speedrun actions to underlying NES controller bitmasks.
"""

from typing import List, Sequence, Union
import gymnasium as gym
from gymnasium import spaces

# Speedrun action set: 7 discrete actions tailored for speedrunning and jumping
SPEEDRUN_ACTIONS: List[List[str]] = [
    ["NOOP"],              # 0: Do nothing
    ["right"],             # 1: Walk right
    ["right", "A"],        # 2: Jump right (normal)
    ["right", "B"],        # 3: Sprint right
    ["right", "A", "B"],   # 4: Sprint + jump right (primary speedrun action)
    ["A"],                 # 5: Vertical jump (hit blocks / coins / escape tight corners)
    ["left"],              # 6: Reposition left
]

# NES joypad button bitmask mapping (standard nes-py button encoding)
NES_BUTTON_MAP = {
    "right":  0b10000000,
    "left":   0b01000000,
    "down":   0b00100000,
    "up":     0b00010000,
    "start":  0b00001000,
    "select": 0b00000100,
    "B":      0b00000010,
    "A":      0b00000001,
    "NOOP":   0b00000000,
}


class SpeedrunActionWrapper(gym.ActionWrapper):
    """Wrapper that reduces the action space to 7 speedrun-optimized discrete actions.
    
    Transforms Discrete(7) integer actions into the corresponding NES joypad button byte
    (when wrapping a raw NES/gym-super-mario-bros env with Discrete(256)), or passes the
    mapped action appropriately.
    """

    def __init__(self, env: gym.Env, actions: Sequence[Sequence[str]] = SPEEDRUN_ACTIONS):
        super().__init__(env)
        self.actions = [list(a) for a in actions]
        self.action_space = spaces.Discrete(len(self.actions))

        # Build bitmap mapping for each discrete action
        self._action_map = {}
        self._action_meanings = {}
        for idx, button_list in enumerate(self.actions):
            byte_action = 0
            for button in button_list:
                if button in NES_BUTTON_MAP:
                    byte_action |= NES_BUTTON_MAP[button]
            self._action_map[idx] = byte_action
            self._action_meanings[idx] = " ".join(button_list)

    def action(self, act: Union[int, Sequence[int]]) -> int:
        """Convert a discrete action (0..6) to the underlying environment action."""
        if hasattr(act, "item"):
            act = int(act.item())
        else:
            act = int(act)

        # Check if underlying action space is Discrete(256) (NES byte)
        underlying_space = self.env.action_space
        if isinstance(underlying_space, spaces.Discrete) and underlying_space.n == 256:
            return self._action_map.get(act, 0)

        # If underlying env already expects 0..N discrete index
        if isinstance(underlying_space, spaces.Discrete) and underlying_space.n == len(self.actions):
            return act

        # Default fallback to byte mapping
        return self._action_map.get(act, act)

    def get_action_meanings(self) -> List[str]:
        """Return list of human-readable meanings for each action."""
        return [self._action_meanings[i] for i in sorted(self._action_meanings.keys())]
