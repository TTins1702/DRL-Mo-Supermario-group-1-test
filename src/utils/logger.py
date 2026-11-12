"""Unified logging system for MO-PPO Super Mario training.

Provides synchronized logging to Console (Rich or standard formatting),
local CSV files, TensorBoard SummaryWriter, and Weights & Biases (WandB).
"""

from __future__ import annotations

import csv
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Logger:
    """Unified logger supporting Console, CSV, TensorBoard, and WandB."""

    def __init__(
        self,
        log_dir: str = "./logs/mo_ppo_mario",
        exp_name: str = "mo_ppo_mario",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "mo-supermario-group-1",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log_dir = os.path.abspath(log_dir)
        self.exp_name = exp_name
        self.use_tensorboard = use_tensorboard and HAS_TENSORBOARD
        self.use_wandb = use_wandb and HAS_WANDB
        self.config = config or {}

        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        self._csv_headers_written = os.path.exists(self.csv_path)

        # TensorBoard
        self.writer: Optional[SummaryWriter] = None
        if self.use_tensorboard:
            tb_dir = os.path.join(self.log_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_dir)
            logger.info("TensorBoard logging enabled at: %s", tb_dir)

        # Rich console
        self.console = Console() if HAS_RICH else None

        # WandB
        self.wandb_run = None
        if self.use_wandb:
            try:
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_run_name or exp_name,
                    config=self.config,
                    dir=self.log_dir,
                    reinit=True,
                )
                logger.info("WandB logging initialized for run: %s", wandb_run_name or exp_name)
            except Exception as e:
                logger.warning("Failed to initialize WandB: %s. Continuing without WandB.", e)
                self.use_wandb = False

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log a generic dictionary of scalar metrics to all enabled targets."""
        # TensorBoard
        if self.writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, float(v), global_step=step)

        # WandB
        if self.use_wandb and self.wandb_run:
            wandb.log(metrics, step=step)

    def log_train(
        self,
        update_idx: int,
        global_step: int,
        loss_dict: Dict[str, float],
        fps: float,
        reward_dict: Optional[Dict[str, float]] = None,
        elapsed_sec: Optional[float] = None,
    ) -> None:
        """Log training update metrics to console, CSV, TensorBoard, and WandB."""
        combined: Dict[str, Any] = {
            "time": time.time(),
            "update_idx": update_idx,
            "global_step": global_step,
            "perf/fps": fps,
        }
        if elapsed_sec is not None:
            combined["perf/elapsed_sec"] = elapsed_sec

        # Loss metrics
        for k, v in loss_dict.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                for idx, item in enumerate(v):
                    combined[f"train/{k}_{idx}"] = float(item)
            elif isinstance(v, (int, float, np.floating, np.integer)):
                combined[f"train/{k}"] = float(v)

        # Rollout reward metrics
        if reward_dict:
            for k, v in reward_dict.items():
                if isinstance(v, (list, tuple, np.ndarray)):
                    for idx, item in enumerate(v):
                        combined[f"rollout/{k}_{idx}"] = float(item)
                elif isinstance(v, (int, float, np.floating, np.integer)):
                    combined[f"rollout/{k}"] = float(v)

        # Append to CSV
        self._write_csv_row(combined)

        # TensorBoard & WandB
        self.log_metrics(combined, step=global_step)

        # Console printout
        if self.console and HAS_RICH:
            table = Table(title=f"Update {update_idx} | Step {global_step:,} | FPS {fps:.1f}", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            for k, v in loss_dict.items():
                if isinstance(v, (list, tuple, np.ndarray)):
                    v_str = "[" + ", ".join(f"{float(x):.2f}" for x in v) + "]"
                    table.add_row(k, v_str)
                else:
                    table.add_row(k, f"{float(v):.4f}")
            if reward_dict:
                for k, v in reward_dict.items():
                    if isinstance(v, (list, tuple, np.ndarray)):
                        v_str = "[" + ", ".join(f"{float(x):.2f}" for x in v) + "]"
                        table.add_row(k, v_str)
                    else:
                        table.add_row(k, f"{float(v):.4f}")
            self.console.print(table)
        else:
            loss_summary = " | ".join(
                f"{k}: [{','.join(f'{float(x):.2f}' for x in v)}]" if isinstance(v, (list, tuple, np.ndarray))
                else f"{k}: {float(v):.4f}"
                for k, v in loss_dict.items()
            )
            logger.info("[Update %4d | Step %8d | FPS %5.1f] %s", update_idx, global_step, fps, loss_summary)

    def log_eval(
        self,
        global_step: int,
        eval_metrics: Dict[str, Any],
    ) -> None:
        """Log evaluation metrics (win rate, raw rewards, completion times)."""
        eval_data: Dict[str, Any] = {
            "time": time.time(),
            "global_step": global_step,
        }
        for k, v in eval_metrics.items():
            if isinstance(v, (int, float)):
                eval_data[f"eval/{k}"] = float(v)

        self._write_csv_row(eval_data)
        self.log_metrics(eval_data, step=global_step)

        # Console summary
        win_rate = eval_metrics.get("win_rate", 0.0)
        avg_x = eval_metrics.get("avg_x_pos", 0.0)
        avg_time = eval_metrics.get("avg_completion_time", 0.0)
        logger.info(
            "=== EVALUATION @ Step %8d === Win Rate: %5.1f%% | Avg X: %6.1f | Avg Speedrun Time: %5.1fs",
            global_step,
            win_rate * 100.0,
            avg_x,
            avg_time,
        )

    def log_video(self, video_path: str, step: int, caption: str = "Super Mario MO-PPO Gameplay") -> None:
        """Log a gameplay MP4 video to WandB."""
        if self.use_wandb and self.wandb_run and os.path.exists(video_path):
            try:
                wandb.log(
                    {"gameplay_video": wandb.Video(video_path, fps=30, format="mp4", caption=caption)},
                    step=step,
                )
                logger.info("Uploaded video to WandB: %s", video_path)
            except Exception as e:
                logger.warning("Could not upload video to WandB: %s", e)

    def _write_csv_row(self, row_data: Dict[str, Any], filepath: Optional[str] = None) -> None:
        """Append row to CSV file with dynamic header handling."""
        target_path = filepath or self.csv_path
        file_exists = os.path.exists(target_path)
        existing_headers: List[str] = []
        existing_rows: List[Dict[str, Any]] = []

        if file_exists:
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        existing_headers = list(reader.fieldnames)
                        existing_rows = list(reader)
            except Exception:
                pass

        current_headers = list(row_data.keys())
        if not file_exists or not existing_headers:
            with open(target_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=current_headers)
                writer.writeheader()
                writer.writerow(row_data)
        else:
            new_headers = [h for h in current_headers if h not in existing_headers]
            if new_headers:
                merged_headers = existing_headers + new_headers
                with open(target_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=merged_headers)
                    writer.writeheader()
                    for r in existing_rows:
                        writer.writerow(r)
                    writer.writerow(row_data)
            else:
                with open(target_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=existing_headers)
                    writer.writerow(row_data)

    def close(self) -> None:
        """Close TensorBoard writer and WandB run cleanly."""
        if self.writer:
            self.writer.flush()
            self.writer.close()
        if self.use_wandb and self.wandb_run:
            wandb.finish()
        logger.info("Logger closed cleanly.")
