"""Configuration loader for MO-PPO Super Mario training and evaluation.

Provides YAML loading, deep merging, dot-notation key updates,
and CLI override parsing.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional, Union
import yaml


class ConfigDict(dict):
    """Dictionary supporting dot-attribute access alongside dict key access."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, ConfigDict):
                val = ConfigDict(val)
                self[key] = val
            return val
        except KeyError as err:
            raise AttributeError(f"Config has no attribute '{key}'") from err

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError as err:
            raise AttributeError(f"Config has no attribute '{key}'") from err


def _parse_value(val_str: str) -> Any:
    """Parse string representation of value into proper Python type."""
    val_stripped = val_str.strip()
    # Check booleans
    if val_stripped.lower() == "true":
        return True
    if val_stripped.lower() == "false":
        return False
    if val_stripped.lower() in ("none", "null"):
        return None

    # Check JSON structures (lists, dicts)
    if (val_stripped.startswith("[") and val_stripped.endswith("]")) or (
        val_stripped.startswith("{") and val_stripped.endswith("}")
    ):
        try:
            return json.loads(val_stripped)
        except Exception:
            pass

    # Try integer
    try:
        return int(val_stripped)
    except ValueError:
        pass

    # Try float
    try:
        return float(val_stripped)
    except ValueError:
        pass

    return val_str


def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update nested dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def _apply_dot_key(cfg: Dict[str, Any], dot_key: str, value: Any) -> None:
    """Apply a dot-notation key (e.g. 'env.num_envs') to a nested dictionary."""
    tokens = dot_key.split(".")
    curr = cfg
    for token in tokens[:-1]:
        if token not in curr or not isinstance(curr[token], dict):
            curr[token] = {}
        curr = curr[token]
    curr[tokens[-1]] = value


def load_config(
    path: str,
    overrides: Optional[Union[Dict[str, Any], List[str]]] = None,
) -> ConfigDict:
    """Load configuration from a YAML file with optional overrides.

    Args:
        path: Path to the YAML configuration file.
        overrides: Dictionary of overrides or list of 'key=value' strings.
                   Supports dot-notation (e.g. 'agent.learning_rate=1e-4').

    Returns:
        ConfigDict containing merged configurations.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    config = copy.deepcopy(raw_cfg)

    if overrides:
        if isinstance(overrides, list):
            # Parse 'key=value' list
            for item in overrides:
                if "=" in item:
                    k, v = item.split("=", 1)
                    parsed_val = _parse_value(v)
                    _apply_dot_key(config, k.strip(), parsed_val)
        elif isinstance(overrides, dict):
            for k, v in overrides.items():
                if "." in k:
                    _apply_dot_key(config, k, v)
                elif isinstance(v, dict) and k in config and isinstance(config[k], dict):
                    _deep_update(config[k], v)
                else:
                    config[k] = v

    return ConfigDict(config)


def save_config(config: Union[Dict[str, Any], ConfigDict], path: str) -> None:
    """Save configuration dictionary to a YAML file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Convert ConfigDict back to standard dict
    clean_dict = json.loads(json.dumps(config))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(clean_dict, f, default_flow_style=False, sort_keys=False)
