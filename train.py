"""Main training script for Multi-Objective PPO (MO-PPO) on Super Mario Bros.

Coordinates parallel environments, ActorCriticResNet network, 5D RolloutBuffer,
MOPPOAgent, CheckpointManager, and unified Logger across Kaggle, Colab, and local GPU/CPU.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agent import MOPPOAgent
from src.env import make_mario_env, make_mario_vector_env
from src.model import ActorCriticResNet, MultiObjectiveRolloutBuffer
from src.utils import CheckpointManager, Logger, load_config, save_episode_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def set_seed(seed: int, torch_deterministic: bool = True) -> None:
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def evaluate_policy(
    network: nn.Module,
    world: int = 1,
    stage: int = 1,
    num_episodes: int = 5,
    device: Union[str, torch.device] = "cpu",
    record_best_video: bool = False,
    video_dir: Optional[str] = None,
    step_num: int = 0,
) -> Dict[str, Any]:
    """Run standalone evaluation on a single environment instance."""
    eval_env = make_mario_env(world=world, stage=stage)
    wins = 0
    max_x_list: List[float] = []
    completion_times: List[float] = []
    scores: List[float] = []
    coins_list: List[float] = []
    raw_5d_list: List[np.ndarray] = []

    best_episode_frames: List[np.ndarray] = []
    best_episode_hud: List[Dict[str, Any]] = []
    best_x = -1.0

    device = torch.device(device)
    network.eval()

    try:
        for ep in range(num_episodes):
            obs, info = eval_env.reset()
            done = False
            ep_frames = []
            ep_hud = []
            ep_x = float(info.get("x_pos", 0))

            while not done:
                if record_best_video:
                    frame = _extract_screen_frame(eval_env)
                    if frame is not None:
                        ep_frames.append(frame.copy())
                        ep_hud.append(info.copy())

                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = network(obs_t)
                    action = torch.argmax(logits, dim=-1).item()

                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                curr_x = float(info.get("x_pos", ep_x))
                if curr_x > ep_x:
                    ep_x = curr_x

            flag_get = bool(info.get("flag_get", False))
            if flag_get:
                wins += 1
                time_left = float(info.get("time", 0))
                elapsed_time = max(0.0, 400.0 - time_left)
                completion_times.append(elapsed_time)

            max_x_list.append(ep_x)
            scores.append(float(info.get("score", 0)))
            coins_list.append(float(info.get("coins", 0)))
            raw_r = info.get("episode_raw_rewards", np.zeros(5, dtype=np.float32))
            raw_5d_list.append(raw_r)

            if ep_x > best_x and record_best_video:
                best_x = ep_x
                best_episode_frames = ep_frames
                best_episode_hud = ep_hud
    finally:
        eval_env.close()
        network.train()

    win_rate = wins / max(1, num_episodes)
    avg_x = float(np.mean(max_x_list)) if max_x_list else 0.0
    avg_speedrun_time = float(np.mean(completion_times)) if completion_times else 0.0
    avg_score = float(np.mean(scores)) if scores else 0.0
    avg_coins = float(np.mean(coins_list)) if coins_list else 0.0
    avg_raw_5d = np.mean(raw_5d_list, axis=0) if raw_5d_list else np.zeros(5)

    video_path = None
    if record_best_video and video_dir and best_episode_frames:
        os.makedirs(video_dir, exist_ok=True)
        v_name = f"eval_step_{step_num:08d}_win_{wins}of{num_episodes}.mp4"
        video_path = os.path.join(video_dir, v_name)
        save_episode_video(best_episode_frames, video_path, fps=30, hud_info=best_episode_hud)

    metrics = {
        "win_rate": win_rate,
        "avg_x_pos": avg_x,
        "avg_completion_time": avg_speedrun_time,
        "avg_score": avg_score,
        "avg_coins": avg_coins,
        "raw_x_pos": float(avg_raw_5d[0]),
        "raw_time": float(avg_raw_5d[1]),
        "raw_death": float(avg_raw_5d[2]),
        "raw_coin": float(avg_raw_5d[3]),
        "raw_enemy": float(avg_raw_5d[4]),
    }
    if video_path:
        metrics["video_path"] = video_path

    return metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with YAML config support."""
    parser = argparse.ArgumentParser(description="Train MO-PPO Agent on Super Mario Bros")
    parser.add_argument("--config", type=str, default="configs/base_8pt_clear.yaml", help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path or 'auto' to resume")
    parser.add_argument("--no-resume", action="store_true", help="Force new training run without resuming")
    parser.add_argument("--reset-lr", action="store_true", help="Reset learning rate & optimizer on resume (fine-tuning)")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total training timesteps")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=None, help="Override rollout horizon T")
    parser.add_argument("--lr", type=float, default=None, help="Override initial learning rate")
    parser.add_argument("--weights", type=str, default=None, help="Override 5D weights, e.g. '0.4,0.4,0.15,0.02,0.03'")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB team or user entity")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--save-freq", type=int, default=None, help="Updates between checkpoint saves")
    parser.add_argument("--eval-freq", type=int, default=None, help="Updates between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Number of eval episodes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load Configuration
    overrides: Dict[str, Any] = {}
    if args.total_timesteps is not None:
        overrides["agent.total_timesteps"] = args.total_timesteps
    if args.num_envs is not None:
        overrides["env.num_envs"] = args.num_envs
    if args.num_steps is not None:
        overrides["agent.num_steps"] = args.num_steps
    if args.lr is not None:
        overrides["agent.learning_rate"] = args.lr
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.exp_name is not None:
        overrides["exp_name"] = args.exp_name
    if args.wandb:
        overrides["logging.use_wandb"] = True
    if args.wandb_project is not None:
        overrides["logging.wandb_project"] = args.wandb_project
    if args.wandb_entity is not None:
        overrides["logging.wandb_entity"] = args.wandb_entity
    if args.save_freq is not None:
        overrides["logging.save_freq_updates"] = args.save_freq
    if args.eval_freq is not None:
        overrides["logging.eval_freq_updates"] = args.eval_freq
    if args.eval_episodes is not None:
        overrides["logging.eval_episodes"] = args.eval_episodes

    if args.weights is not None:
        parsed_w = [float(x.strip()) for x in args.weights.split(",")]
        if len(parsed_w) != 5:
            raise ValueError(f"--weights must have 5 comma-separated floats, got {len(parsed_w)}")
        overrides["agent.weights"] = parsed_w

    cfg = load_config(args.config, overrides=overrides)
    exp_name = cfg.get("exp_name", "mo_ppo_mario")

    # 2. Setup Device & Seed
    if args.device is not None:
        device_str = args.device
    elif cfg.get("cuda", True) and torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info("Operating on device: %s", device)

    seed = int(cfg.get("seed", 42))
    set_seed(seed, torch_deterministic=bool(cfg.get("torch_deterministic", True)))

    # 3. Environment Parameters
    env_cfg = cfg.get("env", {})
    world = int(env_cfg.get("world", 1))
    stage = int(env_cfg.get("stage", 1))
    num_envs = int(env_cfg.get("num_envs", 8))
    skip_frames = int(env_cfg.get("skip_frames", 4))
    stack_frames = int(env_cfg.get("stack_frames", 4))
    scale_components = env_cfg.get("scale_components", [0.05, 0.05, 0.08, 0.01, 0.02])
    beta_milestone = float(env_cfg.get("beta_milestone", 0.1 if env_cfg.get("use_milestone", True) else 0.0))
    beta_stuck = float(env_cfg.get("beta_stuck", 0.1 if env_cfg.get("use_anti_stuck", True) else 0.0))
    stuck_steps = int(env_cfg.get("stuck_steps", 60))

    logger.info("Initializing %d parallel environments for World %d-%d...", num_envs, world, stage)
    envs = make_mario_vector_env(
        num_envs=num_envs,
        world=world,
        stage=stage,
        skip=skip_frames,
        num_stack=stack_frames,
        scaling=scale_components,
        beta_milestone=beta_milestone,
        beta_stuck=beta_stuck,
        stuck_steps=stuck_steps,
    )

    # 4. Model Architecture & Optimizer
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

    agent_cfg = cfg.get("agent", {})
    lr_start = float(agent_cfg.get("learning_rate", 2.5e-4))
    lr_end = float(agent_cfg.get("lr_end", 1.0e-5))
    anneal_lr = bool(agent_cfg.get("anneal_lr", True))
    weights = agent_cfg.get("weights", [0.65, 0.10, 0.20, 0.02, 0.03])
    num_steps = int(agent_cfg.get("num_steps", 512))
    total_timesteps = int(agent_cfg.get("total_timesteps", 2000000))
    batch_size = num_envs * num_steps
    num_updates = max(1, (total_timesteps + batch_size - 1) // batch_size)
    gamma = float(agent_cfg.get("gamma", 0.99))
    gae_lambda = float(agent_cfg.get("gae_lambda", 0.95))

    optimizer = optim.Adam(network.parameters(), lr=lr_start, eps=1e-5)

    # 5. Rollout Buffer & MO-PPO Agent
    buffer = MultiObjectiveRolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        obs_shape=(in_channels, 84, 84),
        num_actions=action_dim,
        num_objectives=critic_dim,
        device=device,
    )

    agent = MOPPOAgent(
        network=network,
        optimizer=optimizer,
        config=agent_cfg,
        device=device,
    )

    # 6. CheckpointManager & Logger
    ckpt_manager = CheckpointManager(exp_name=exp_name, device=device)

    log_cfg = cfg.get("logging", {})
    save_freq = int(log_cfg.get("save_freq_updates", 25))
    eval_freq = int(log_cfg.get("eval_freq_updates", 25))
    eval_episodes = int(log_cfg.get("eval_episodes", 10))

    logger_inst = Logger(
        log_dir=os.path.join("./logs", exp_name),
        exp_name=exp_name,
        use_tensorboard=bool(log_cfg.get("use_tensorboard", True)),
        use_wandb=bool(log_cfg.get("use_wandb", False)),
        wandb_project=str(log_cfg.get("wandb_project", "mo-supermario-group-1")),
        wandb_entity=log_cfg.get("wandb_entity", None),
        wandb_run_name=args.wandb_run_name,
        config=dict(cfg),
    )

    # 7. Checkpoint Resuming
    start_update = 1
    global_step = 0
    resume_target = None

    if not args.no_resume:
        if args.resume:
            resume_target = args.resume if args.resume != "auto" else ckpt_manager.find_latest_checkpoint()
        else:
            resume_target = ckpt_manager.find_latest_checkpoint()

    if resume_target and os.path.exists(resume_target):
        logger.info("Found existing checkpoint: %s. Loading...", resume_target)
        load_opt = not args.reset_lr
        up_idx, g_step, best_metrics, _ = ckpt_manager.resume(
            resume_target, network, optimizer=optimizer if load_opt else None, load_optimizer=load_opt
        )
        if not args.reset_lr:
            start_update = up_idx + 1
            global_step = g_step
            logger.info("Resuming training from Update %d | Global Step %d", start_update, global_step)
        else:
            logger.info("Weights transferred from %s with RESET learning rate and optimizer.", resume_target)

    # 8. Training Loop
    logger.info("Starting MO-PPO Training: %d updates (~%d total steps)", num_updates, total_timesteps)
    obs, _ = envs.reset()

    # Track rolling rollout episode statistics
    rollout_wins = deque(maxlen=50)
    rollout_x_pos = deque(maxlen=50)
    rollout_raw_rewards = deque(maxlen=50)

    start_time = time.time()

    try:
        for update in range(start_update, num_updates + 1):
            # Anneal learning rate linearly
            if anneal_lr:
                frac = 1.0 - (update - 1.0) / max(1.0, float(num_updates))
                curr_lr = lr_end + frac * (lr_start - lr_end)
                agent.set_learning_rate(curr_lr)
            else:
                curr_lr = lr_start

            update_start_time = time.time()

            # --- ROLLOUT PHASE ---
            for step in range(num_steps):
                global_step += num_envs
                obs_tensor = torch.as_tensor(obs, device=device)

                with torch.no_grad():
                    action, logprob, _, value_5d = network.get_action_and_value(obs_tensor)

                next_obs, reward, term, trunc, infos = envs.step(action.cpu().numpy())
                done = np.logical_or(term, trunc)

                # Store into rollout buffer
                buffer.add(
                    obs=obs,
                    action=action,
                    logprob=logprob,
                    reward=reward,
                    done=done,
                    value=value_5d,
                )

                # Process terminal infos for rolling stats
                for i in range(num_envs):
                    if done[i]:
                        info_dict = infos[i].get("final_info", infos[i])
                        rollout_wins.append(1.0 if info_dict.get("flag_get", False) else 0.0)
                        rollout_x_pos.append(float(info_dict.get("x_pos", 0)))
                        if "episode_raw_rewards" in info_dict:
                            rollout_raw_rewards.append(info_dict["episode_raw_rewards"])

                obs = next_obs

            # --- GAE ADVANTAGE ESTIMATION ---
            obs_tensor = torch.as_tensor(obs, device=device)
            with torch.no_grad():
                next_val_5d = network.get_value(obs_tensor)

            buffer.compute_returns_and_advantages(
                next_obs=obs,
                next_done=done,
                next_value=next_val_5d,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )

            # --- MO-PPO AGENT UPDATE ---
            train_metrics = agent.train_step(buffer, weights=weights)
            buffer.reset()

            # --- METRICS & LOGGING ---
            step_duration = time.time() - update_start_time
            fps = float(batch_size) / max(1e-6, step_duration)

            reward_metrics: Dict[str, float] = {}
            if rollout_wins:
                reward_metrics["rollout_win_rate"] = float(np.mean(rollout_wins))
                reward_metrics["rollout_avg_x"] = float(np.mean(rollout_x_pos))
            if rollout_raw_rewards:
                mean_raw = np.mean(rollout_raw_rewards, axis=0)
                reward_metrics["raw_x_pos"] = float(mean_raw[0])
                reward_metrics["raw_time"] = float(mean_raw[1])
                reward_metrics["raw_death"] = float(mean_raw[2])
                reward_metrics["raw_coin"] = float(mean_raw[3])
                reward_metrics["raw_enemy"] = float(mean_raw[4])

            logger_inst.log_train(
                update_idx=update,
                global_step=global_step,
                loss_dict=train_metrics,
                fps=fps,
                reward_dict=reward_metrics,
                elapsed_sec=time.time() - start_time,
            )

            # --- PERIODIC EVALUATION ---
            if update % eval_freq == 0 or update == num_updates:
                logger.info("Evaluating agent policy at update %d (step %d)...", update, global_step)
                video_out_dir = os.path.join(ckpt_manager.checkpoint_dir, "eval_videos")
                eval_stats = evaluate_policy(
                    network=network,
                    world=world,
                    stage=stage,
                    num_episodes=eval_episodes,
                    device=device,
                    record_best_video=True,
                    video_dir=video_out_dir,
                    step_num=global_step,
                )
                logger_inst.log_eval(global_step=global_step, eval_metrics=eval_stats)

                if "video_path" in eval_stats:
                    logger_inst.log_video(eval_stats["video_path"], step=global_step)

                # Check if best win rate or best speedrun
                is_best_win = eval_stats["win_rate"] > ckpt_manager.best_win_rate
                is_best_speed = (
                    eval_stats["win_rate"] > 0.0
                    and eval_stats["avg_completion_time"] < ckpt_manager.best_speedrun_time
                )

                ckpt_manager.save_checkpoint(
                    network=network,
                    optimizer=optimizer,
                    update_idx=update,
                    global_step=global_step,
                    best_metrics=eval_stats,
                    config=dict(cfg),
                    is_best_winrate=is_best_win,
                    is_best_speedrun=is_best_speed,
                )

            # --- PERIODIC SAVE (LATEST.PT) ---
            elif update % save_freq == 0:
                current_metrics = {
                    "win_rate": np.mean(rollout_wins) if rollout_wins else 0.0,
                    "avg_x_pos": np.mean(rollout_x_pos) if rollout_x_pos else 0.0,
                }
                ckpt_manager.save_checkpoint(
                    network=network,
                    optimizer=optimizer,
                    update_idx=update,
                    global_step=global_step,
                    best_metrics=current_metrics,
                    config=dict(cfg),
                    filename="latest.pt",
                )

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving emergency checkpoint...")
    finally:
        # Final checkpoint save (ensures both latest.pt and final.pt exist)
        ckpt_manager.save_checkpoint(
            network=network,
            optimizer=optimizer,
            update_idx=update if "update" in locals() else 0,
            global_step=global_step,
            config=dict(cfg),
            filename="latest.pt",
        )
        ckpt_manager.save_checkpoint(
            network=network,
            optimizer=optimizer,
            update_idx=update if "update" in locals() else 0,
            global_step=global_step,
            config=dict(cfg),
            filename="final.pt",
        )
        envs.close()
        logger_inst.close()
        logger.info("Training session terminated cleanly.")


if __name__ == "__main__":
    main()
