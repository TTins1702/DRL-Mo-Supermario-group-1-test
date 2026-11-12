"""Evaluation script for MO-PPO Super Mario models.

Loads trained checkpoints, evaluates performance over N episodes (Win Rate,
Speedrun Times, 5D Raw Rewards, Score), renders annotated gameplay MP4 videos,
and outputs a comprehensive performance report.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.env import make_mario_env
from src.model import ActorCriticResNet
from src.utils import load_config, save_episode_video

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def _extract_screen_frame(env: Any) -> Optional[np.ndarray]:
    """Helper to traverse env wrappers and extract RGB screen frame."""
    curr = env
    while curr is not None:
        if hasattr(curr, "screen") and curr.screen is not None:
            return curr.screen
        if hasattr(curr, "_screen") and curr._screen is not None:
            return curr._screen
        if hasattr(curr, "_env"):
            curr = curr._env
        elif hasattr(curr, "env"):
            curr = curr.env
        else:
            break
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained MO-PPO Mario Agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--config", type=str, default=None, help="Optional config path (defaults to checkpoint config)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes (default: 20)")
    parser.add_argument("--world", type=int, default=1, help="World number (default: 1)")
    parser.add_argument("--stage", type=int, default=1, help="Stage number (default: 1)")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions stochastically instead of argmax")
    parser.add_argument("--no-video", action="store_true", help="Disable recording mp4 videos")
    parser.add_argument("--video-dir", type=str, default="./eval_videos", help="Directory to save mp4 videos")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    return parser.parse_args()


def run_evaluation(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    num_episodes: int = 20,
    world: int = 1,
    stage: int = 1,
    deterministic: bool = True,
    record_video: bool = True,
    video_dir: str = "./eval_videos",
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute evaluation protocol and return dictionary of aggregated metrics."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Evaluating on device: %s", device)

    # 1. Load Checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_cfg = ckpt.get("config", {})

    if config_path and os.path.exists(config_path):
        cfg = load_config(config_path)
    elif saved_cfg:
        cfg = saved_cfg
    else:
        cfg = {}

    model_cfg = cfg.get("model", {})
    in_channels = int(model_cfg.get("in_channels", 4))
    action_dim = int(model_cfg.get("action_dim", 7))
    features_dim = int(model_cfg.get("feature_dim", 512))
    critic_dim = int(model_cfg.get("critic_dim", 5))

    network = ActorCriticResNet(
        in_channels=in_channels,
        num_actions=action_dim,
        num_objectives=critic_dim,
        features_dim=features_dim,
    ).to(device)

    network.load_state_dict(ckpt["model_state_dict"])
    network.eval()
    logger.info("Loaded network weights from %s (Update: %s)", checkpoint_path, ckpt.get("update_idx", "N/A"))

    # 2. Setup Single Evaluation Environment
    env = make_mario_env(world=world, stage=stage)

    if record_video:
        os.makedirs(video_dir, exist_ok=True)

    episode_results: List[Dict[str, Any]] = []
    wins = 0
    clear_times: List[float] = []

    logger.info("Starting %d evaluation episodes on World %d-%d...", num_episodes, world, stage)

    try:
        for ep in range(1, num_episodes + 1):
            obs, info = env.reset()
            done = False
            ep_frames: List[np.ndarray] = []
            ep_hud: List[Dict[str, Any]] = []
            max_x = float(info.get("x_pos", 0))

            while not done:
                if record_video:
                    screen = _extract_screen_frame(env)
                    if screen is not None:
                        ep_frames.append(screen.copy())
                        ep_hud.append(info.copy())

                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, val_5d = network(obs_t)
                    if deterministic:
                        action = torch.argmax(logits, dim=-1).item()
                    else:
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample().item()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                curr_x = float(info.get("x_pos", max_x))
                if curr_x > max_x:
                    max_x = curr_x

            flag_get = bool(info.get("flag_get", False))
            time_left = float(info.get("time", 0))
            elapsed_time = max(0.0, 400.0 - time_left) if flag_get else 0.0

            if flag_get:
                wins += 1
                clear_times.append(elapsed_time)

            raw_rewards = info.get("episode_raw_rewards", np.zeros(5, dtype=np.float32))

            ep_summary = {
                "episode": ep,
                "flag_get": flag_get,
                "max_x": max_x,
                "time_left": time_left,
                "clear_time": elapsed_time,
                "score": float(info.get("score", 0)),
                "coins": float(info.get("coins", 0)),
                "raw_x_pos": float(raw_rewards[0]),
                "raw_time": float(raw_rewards[1]),
                "raw_death": float(raw_rewards[2]),
                "raw_coin": float(raw_rewards[3]),
                "raw_enemy": float(raw_rewards[4]),
            }
            episode_results.append(ep_summary)

            status_str = "CLEARED! [FLAG REACHED]" if flag_get else f"Died/Timeout @ X={max_x:.0f}"
            logger.info("Episode %2d/%2d: %s | Time: %5.1fs | Pts: %6.0f", ep, num_episodes, status_str, elapsed_time, ep_summary["score"])

            # Save video for this episode if enabled
            if record_video and ep_frames:
                v_filename = f"ep_{ep:02d}_{'WIN' if flag_get else 'FAIL'}_x{int(max_x)}.mp4"
                v_path = os.path.join(video_dir, v_filename)
                save_episode_video(ep_frames, v_path, fps=30, hud_info=ep_hud)

    finally:
        env.close()

    # 3. Aggregate Performance Metrics
    win_rate = (wins / max(1, num_episodes)) * 100.0
    avg_max_x = float(np.mean([e["max_x"] for e in episode_results]))
    avg_clear_time = float(np.mean(clear_times)) if clear_times else 0.0
    min_clear_time = float(np.min(clear_times)) if clear_times else 0.0
    max_clear_time = float(np.max(clear_times)) if clear_times else 0.0
    avg_score = float(np.mean([e["score"] for e in episode_results]))
    avg_coins = float(np.mean([e["coins"] for e in episode_results]))

    avg_raw_x = float(np.mean([e["raw_x_pos"] for e in episode_results]))
    avg_raw_time = float(np.mean([e["raw_time"] for e in episode_results]))
    avg_raw_death = float(np.mean([e["raw_death"] for e in episode_results]))
    avg_raw_coin = float(np.mean([e["raw_coin"] for e in episode_results]))
    avg_raw_enemy = float(np.mean([e["raw_enemy"] for e in episode_results]))

    summary = {
        "num_episodes": num_episodes,
        "wins": wins,
        "win_rate_pct": win_rate,
        "avg_max_x": avg_max_x,
        "avg_clear_time_sec": avg_clear_time,
        "min_clear_time_sec": min_clear_time,
        "max_clear_time_sec": max_clear_time,
        "avg_score": avg_score,
        "avg_coins": avg_coins,
        "avg_raw_x_pos": avg_raw_x,
        "avg_raw_time": avg_raw_time,
        "avg_raw_death": avg_raw_death,
        "avg_raw_coin": avg_raw_coin,
        "avg_raw_enemy": avg_raw_enemy,
    }

    # 4. Display Report
    if HAS_RICH:
        console = Console()
        table = Table(title="=== MO-PPO SUPER MARIO EVALUATION REPORT ===", show_header=True)
        table.add_column("Evaluation Metric", style="cyan", justify="left")
        table.add_column("Result", style="green", justify="right")

        table.add_row("Total Episodes", str(num_episodes))
        table.add_row("Level Cleared (Wins)", f"{wins} / {num_episodes}")
        table.add_row("Win Rate (Flag Reached)", f"{win_rate:.1f}%")
        table.add_row("Avg Max Distance (X pos)", f"{avg_max_x:.1f} / 3266")
        table.add_row("Avg Speedrun Time (Clears)", f"{avg_clear_time:.2f}s" if wins > 0 else "N/A")
        table.add_row("Best Speedrun Time", f"{min_clear_time:.2f}s" if wins > 0 else "N/A")
        table.add_row("Avg Game Score", f"{avg_score:.1f}")
        table.add_row("Avg Coins Collected", f"{avg_coins:.2f}")
        table.add_row("Raw Component 0 (Distance Δx)", f"{avg_raw_x:+.1f}")
        table.add_row("Raw Component 1 (Time Penalty)", f"{avg_raw_time:+.1f}")
        table.add_row("Raw Component 2 (Death Penalty)", f"{avg_raw_death:+.1f}")
        table.add_row("Raw Component 3 (Coin Bonus)", f"{avg_raw_coin:+.1f}")
        table.add_row("Raw Component 4 (Enemy Bonus)", f"{avg_raw_enemy:+.1f}")
        console.print(table)
    else:
        logger.info("==================================================")
        logger.info("       MO-PPO SUPER MARIO EVALUATION REPORT       ")
        logger.info("==================================================")
        logger.info("Total Episodes               : %d", num_episodes)
        logger.info("Win Rate (Flag Reached)      : %.1f%% (%d/%d)", win_rate, wins, num_episodes)
        logger.info("Avg Max Distance (X pos)     : %.1f", avg_max_x)
        logger.info("Avg Speedrun Time (Clears)   : %.2fs", avg_clear_time)
        logger.info("Avg Game Score / Coins       : %.1f / %.1f", avg_score, avg_coins)
        logger.info("5D Raw Rewards [x,t,d,c,e]   : [%.1f, %.1f, %.1f, %.1f, %.1f]",
                    avg_raw_x, avg_raw_time, avg_raw_death, avg_raw_coin, avg_raw_enemy)
        logger.info("==================================================")

    # 5. Save Results CSV
    if record_video and video_dir:
        csv_path = os.path.join(video_dir, "eval_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_results[0].keys()))
            writer.writeheader()
            for r in episode_results:
                writer.writerow(r)
        logger.info("Per-episode details saved to: %s", csv_path)

    return summary


def main() -> None:
    args = parse_args()
    run_evaluation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_episodes=args.episodes,
        world=args.world,
        stage=args.stage,
        deterministic=not args.stochastic,
        record_video=not args.no_video,
        video_dir=args.video_dir,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
