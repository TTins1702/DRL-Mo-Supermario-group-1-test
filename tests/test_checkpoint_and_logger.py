"""Unit tests for config_loader, CheckpointManager, Logger, and video_recorder."""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.utils.config_loader import ConfigDict, load_config, save_config
from src.utils.checkpoint import CheckpointManager, detect_default_checkpoint_dir
from src.utils.logger import Logger
from src.utils.video_recorder import save_episode_video


class SimpleToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="test_mo_ppo_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_config_loader(tmp_dir):
    config_file = os.path.join(tmp_dir, "test_cfg.yaml")
    initial_data = {
        "exp_name": "test_exp",
        "num_envs": 4,
        "agent": {
            "learning_rate": 0.00025,
            "weights": [0.65, 0.10, 0.20, 0.02, 0.03],
            "clip_coef": 0.2,
        },
        "use_cuda": True,
    }
    save_config(initial_data, config_file)
    assert os.path.exists(config_file)

    # Test load without override
    cfg = load_config(config_file)
    assert cfg.exp_name == "test_exp"
    assert cfg["num_envs"] == 4
    assert cfg.agent.learning_rate == 0.00025
    assert cfg.agent.weights[0] == 0.65

    # Test dictionary overrides
    cfg_override = load_config(
        config_file,
        overrides={
            "num_envs": 8,
            "agent.learning_rate": 0.0005,
            "agent.clip_coef": 0.1,
            "agent.weights": [0.4, 0.4, 0.15, 0.02, 0.03],
        },
    )
    assert cfg_override.num_envs == 8
    assert cfg_override.agent.learning_rate == 0.0005
    assert cfg_override.agent.clip_coef == 0.1
    assert cfg_override.agent.weights == [0.4, 0.4, 0.15, 0.02, 0.03]

    # Test list string overrides
    str_overrides = [
        "num_envs=16",
        "agent.learning_rate=0.001",
        "use_cuda=false",
        "agent.weights=[0.3, 0.05, 0.15, 0.25, 0.25]",
    ]
    cfg_str_override = load_config(config_file, overrides=str_overrides)
    assert cfg_str_override.num_envs == 16
    assert cfg_str_override.agent.learning_rate == 0.001
    assert cfg_str_override.use_cuda is False
    assert cfg_str_override.agent.weights == [0.3, 0.05, 0.15, 0.25, 0.25]


def test_checkpoint_manager(tmp_dir):
    ckpt_dir = os.path.join(tmp_dir, "checkpoints")
    manager = CheckpointManager(checkpoint_dir=ckpt_dir, exp_name="test_run")
    assert os.path.exists(ckpt_dir)

    net = SimpleToyNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Save regular latest checkpoint
    save_path = manager.save_checkpoint(
        network=net,
        optimizer=optimizer,
        update_idx=10,
        global_step=5120,
        best_metrics={"win_rate": 0.5, "completion_time": 45.2},
        config={"lr": 1e-3},
        is_best_winrate=True,
        is_best_speedrun=True,
    )
    assert os.path.exists(save_path)
    assert os.path.exists(os.path.join(ckpt_dir, "best_winrate.pt"))
    assert os.path.exists(os.path.join(ckpt_dir, "best_speedrun.pt"))

    # Test resume
    new_net = SimpleToyNet()
    new_opt = torch.optim.Adam(new_net.parameters(), lr=1e-3)
    update_idx, global_step, best_metrics, config = manager.resume(
        save_path, new_net, new_opt
    )
    assert update_idx == 10
    assert global_step == 5120
    assert best_metrics["win_rate"] == 0.5

    # Check that weights match
    for p1, p2 in zip(net.parameters(), new_net.parameters()):
        assert torch.allclose(p1, p2)

    # Check find_latest_checkpoint
    latest = manager.find_latest_checkpoint()
    assert latest is not None
    assert "latest.pt" in latest


def test_logger(tmp_dir):
    log_dir = os.path.join(tmp_dir, "logs")
    log = Logger(
        log_dir=log_dir,
        exp_name="test_logger",
        use_tensorboard=True,
        use_wandb=False,
    )

    # Log train
    log.log_train(
        update_idx=1,
        global_step=512,
        loss_dict={"loss_total": 0.45, "loss_actor": 0.20, "loss_critic": 0.15},
        fps=120.5,
        reward_dict={"raw_x_pos": 1500.0, "raw_coin": 2.0},
    )

    # Log eval
    log.log_eval(
        global_step=512,
        eval_metrics={
            "win_rate": 0.8,
            "avg_x_pos": 2400.0,
            "avg_completion_time": 42.5,
        },
    )

    # Verify CSV file exists and has content
    assert os.path.exists(log.csv_path)
    with open(log.csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) >= 3  # Header + 2 data rows

    log.close()


def test_video_recorder(tmp_dir):
    video_path = os.path.join(tmp_dir, "videos", "test_mario.mp4")
    # Generate 10 dummy RGB frames (84x84x3)
    frames = [
        np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
        for _ in range(10)
    ]
    hud_info = [
        {"x_pos": 100 * i, "time": 400 - i, "coins": i, "score": 200 * i, "flag_get": (i == 9)}
        for i in range(10)
    ]

    out_file = save_episode_video(frames, video_path, fps=15, hud_info=hud_info)
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 0
