"""Checkpoint manager for MO-PPO Super Mario training.

Handles automatic environment detection (Google Colab Drive, Kaggle, Local),
atomic checkpoint writing, tracking best models (best_winrate, best_speedrun),
and seamless resuming.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def detect_default_checkpoint_dir(exp_name: str = "mo_ppo_default") -> str:
    """Detect appropriate checkpoint directory based on environment."""
    colab_drive_path = "/content/drive/MyDrive"
    kaggle_path = "/kaggle/working"

    if os.path.exists(colab_drive_path) and os.path.isdir(colab_drive_path):
        target = os.path.join(colab_drive_path, "DRL_Mario_Checkpoints", exp_name)
        logger.info("Detected Google Colab environment with mounted Drive. Target: %s", target)
        return target

    if os.path.exists(kaggle_path) and os.path.isdir(kaggle_path):
        target = os.path.join(kaggle_path, "checkpoints", exp_name)
        logger.info("Detected Kaggle environment. Target: %s", target)
        return target

    target = os.path.join("./checkpoints", exp_name)
    logger.info("Using local checkpoint directory: %s", target)
    return target


class CheckpointManager:
    """Manages saving, loading, atomic writing, and resuming model checkpoints."""

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        exp_name: str = "mo_ppo_mario",
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.exp_name = exp_name
        self.device = torch.device(device) if isinstance(device, str) else device

        if checkpoint_dir is not None:
            self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        else:
            self.checkpoint_dir = detect_default_checkpoint_dir(exp_name)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_win_rate = -1.0
        self.best_speedrun_time = float("inf")

    def _atomic_torch_save(self, data: Dict[str, Any], filepath: str) -> None:
        """Atomically save PyTorch dictionary via tempfile and os.replace."""
        dir_name = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(dir_name, exist_ok=True)

        temp_path = filepath + ".tmp"
        try:
            with open(temp_path, "wb") as f:
                torch.save(data, f)
            os.replace(temp_path, filepath)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

    def save_checkpoint(
        self,
        network: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        update_idx: int,
        global_step: int,
        best_metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        is_best_winrate: bool = False,
        is_best_speedrun: bool = False,
        filename: str = "latest.pt",
    ) -> str:
        """Save a training checkpoint with atomic replacement.

        Args:
            network: Policy/Value network.
            optimizer: Optimizer instance.
            update_idx: Index of current training update.
            global_step: Total environment interaction steps so far.
            best_metrics: Dictionary tracking best records.
            config: Full run configuration dictionary.
            is_best_winrate: Flag if this checkpoint sets a new win rate record.
            is_best_speedrun: Flag if this checkpoint sets a new speedrun time record.
            filename: Target filename for current save.

        Returns:
            Absolute path to saved latest checkpoint file.
        """
        metrics = best_metrics.copy() if best_metrics else {}
        if "best_win_rate" not in metrics:
            metrics["best_win_rate"] = self.best_win_rate
        if "best_speedrun_time" not in metrics:
            metrics["best_speedrun_time"] = self.best_speedrun_time

        checkpoint_data = {
            "update_idx": update_idx,
            "global_step": global_step,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "best_metrics": metrics,
            "config": config or {},
        }

        save_path = os.path.join(self.checkpoint_dir, filename)
        self._atomic_torch_save(checkpoint_data, save_path)
        logger.info("Saved checkpoint: %s (update: %d, step: %d)", save_path, update_idx, global_step)

        if is_best_winrate:
            winrate_path = os.path.join(self.checkpoint_dir, "best_winrate.pt")
            shutil.copyfile(save_path, winrate_path)
            if "win_rate" in metrics:
                self.best_win_rate = metrics["win_rate"]
            logger.info("Updated best winrate checkpoint: %s", winrate_path)

        if is_best_speedrun:
            speedrun_path = os.path.join(self.checkpoint_dir, "best_speedrun.pt")
            shutil.copyfile(save_path, speedrun_path)
            if "completion_time" in metrics:
                self.best_speedrun_time = metrics["completion_time"]
            logger.info("Updated best speedrun checkpoint: %s", speedrun_path)

        return save_path

    def load(self, filepath: str) -> Dict[str, Any]:
        """Load checkpoint dictionary from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file does not exist: {filepath}")
        return torch.load(filepath, map_location=self.device)

    def resume(
        self,
        filepath: str,
        network: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        load_optimizer: bool = True,
    ) -> Tuple[int, int, Dict[str, Any], Dict[str, Any]]:
        """Resume training state from a checkpoint file.

        Args:
            filepath: Path to the .pt file.
            network: Neural network to load weights into.
            optimizer: Optional optimizer to restore state into.
            load_optimizer: If False, skip loading optimizer state (e.g. for fine-tuning).

        Returns:
            Tuple of (update_idx, global_step, best_metrics, config).
        """
        ckpt = self.load(filepath)
        network.load_state_dict(ckpt["model_state_dict"])

        if load_optimizer and optimizer is not None and ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        update_idx = ckpt.get("update_idx", 0)
        global_step = ckpt.get("global_step", 0)
        best_metrics = ckpt.get("best_metrics", {})
        config = ckpt.get("config", {})

        self.best_win_rate = best_metrics.get("best_win_rate", -1.0)
        self.best_speedrun_time = best_metrics.get("best_speedrun_time", float("inf"))

        logger.info(
            "Resumed from %s: update=%d, step=%d, best_win_rate=%.2f%%",
            filepath,
            update_idx,
            global_step,
            self.best_win_rate * 100.0,
        )
        return update_idx, global_step, best_metrics, config

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find latest.pt or newest checkpoint file in the directory if available."""
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        if os.path.exists(latest_path):
            return latest_path

        # Search for any .pt files
        pt_files = [
            os.path.join(self.checkpoint_dir, f)
            for f in os.listdir(self.checkpoint_dir)
            if f.endswith(".pt")
        ]
        if not pt_files:
            return None

        # Return file with latest modification time
        pt_files.sort(key=os.path.getmtime, reverse=True)
        return pt_files[0]
