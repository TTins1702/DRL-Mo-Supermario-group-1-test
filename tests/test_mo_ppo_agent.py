"""Unit tests for Module 3 (Phân hệ 3): MO-PPO Agent (src/agent/mo_ppo_agent.py).

Verifies:
1. Advantage normalization per objective (mean=0, std=1 on each of 5 dimensions).
2. Scalarization with preference weight vector w in R^5.
3. Explained variance 5D computation.
4. Loss computation, clipping, and backward pass gradient flow.
5. train_step execution across update epochs with dummy rollout buffer.
6. Learning rate adjustment (set_learning_rate).
7. Agent save and load checkpoint roundtrip.
8. get_action and get_value inference interfaces.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Generator, List, Tuple
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.agent.mo_ppo_agent import MOPPOAgent, _unpack_batch


class MockActorCriticNetwork(nn.Module):
    """Mock ActorCritic network adhering to ActorCriticResNet interface."""

    def __init__(self, in_channels: int = 4, num_actions: int = 7, num_objectives: int = 5):
        super().__init__()
        self.num_actions = num_actions
        self.num_objectives = num_objectives

        # Feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(),
        )

        # Actor and 5D Critic heads
        self.actor_head = nn.Linear(64, num_actions)
        self.critic_head = nn.Linear(64, num_objectives)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        h = self.fc(self.conv(obs))
        logits = self.actor_head(h)
        values_5d = self.critic_head(h)
        return logits, values_5d

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        h = self.fc(self.conv(obs))
        return self.critic_head(h)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        h = self.fc(self.conv(obs))
        logits = self.actor_head(h)
        values_5d = self.critic_head(h)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), values_5d


class DummyRolloutBufferDict:
    """Dummy rollout buffer yielding batches as dictionaries."""

    def __init__(self, num_samples: int = 64, batch_size: int = 16):
        self.num_samples = num_samples
        self.batch_size = batch_size

        torch.manual_seed(42)
        self.obs = torch.randint(0, 256, (num_samples, 4, 84, 84), dtype=torch.uint8)
        self.actions = torch.randint(0, 7, (num_samples,), dtype=torch.long)
        self.logprobs = torch.randn(num_samples, dtype=torch.float32)
        self.advantages = torch.randn(num_samples, 5, dtype=torch.float32) * 2.0 + 1.0
        self.returns = torch.randn(num_samples, 5, dtype=torch.float32)
        self.values = torch.randn(num_samples, 5, dtype=torch.float32)

    def get_batches(self, batch_size: int = None) -> Generator[Dict[str, torch.Tensor], None, None]:
        bs = batch_size or self.batch_size
        indices = torch.randperm(self.num_samples)
        for start in range(0, self.num_samples, bs):
            end = min(start + bs, self.num_samples)
            batch_idx = indices[start:end]
            yield {
                "obs": self.obs[batch_idx],
                "actions": self.actions[batch_idx],
                "logprobs": self.logprobs[batch_idx],
                "advantages": self.advantages[batch_idx],
                "returns": self.returns[batch_idx],
                "values": self.values[batch_idx],
            }


class DummyRolloutBufferTuple:
    """Dummy rollout buffer yielding batches as tuples."""

    def __init__(self, num_samples: int = 32, batch_size: int = 16):
        self.num_samples = num_samples
        self.batch_size = batch_size

        torch.manual_seed(42)
        self.obs = torch.randint(0, 256, (num_samples, 4, 84, 84), dtype=torch.uint8)
        self.actions = torch.randint(0, 7, (num_samples,), dtype=torch.long)
        self.logprobs = torch.randn(num_samples, dtype=torch.float32)
        self.advantages = torch.randn(num_samples, 5, dtype=torch.float32)
        self.returns = torch.randn(num_samples, 5, dtype=torch.float32)
        self.values = torch.randn(num_samples, 5, dtype=torch.float32)

    def get_batches(self, batch_size: int = None) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        bs = batch_size or self.batch_size
        for start in range(0, self.num_samples, bs):
            end = min(start + bs, self.num_samples)
            yield (
                self.obs[start:end],
                self.actions[start:end],
                self.logprobs[start:end],
                self.advantages[start:end],
                self.returns[start:end],
                self.values[start:end],
            )


# ==============================================================================
# Unit Tests
# ==============================================================================

def test_agent_initialization():
    """Verify default initialization, config override, and optimizer setup."""
    network = MockActorCriticNetwork()
    config = {
        "lr": 1e-4,
        "clip_coef": 0.15,
        "ent_coef": 0.02,
        "vf_coef": 0.6,
        "max_grad_norm": 0.4,
        "update_epochs": 2,
        "batch_size": 128,
    }
    agent = MOPPOAgent(network=network, config=config)

    assert agent.lr == 1e-4
    assert agent.clip_coef == 0.15
    assert agent.ent_coef == 0.02
    assert agent.vf_coef == 0.6
    assert agent.max_grad_norm == 0.4
    assert agent.update_epochs == 2
    assert agent.batch_size == 128
    assert agent.optimizer is not None
    assert agent.optimizer.param_groups[0]["lr"] == 1e-4


def test_normalize_advantages_per_objective_unmasked():
    """Verify each of the 5 objective dimensions is independently normalized to mean=0, std=1."""
    network = MockActorCriticNetwork()
    agent = MOPPOAgent(network=network)

    torch.manual_seed(123)
    # Generate batch with widely different means and scales across 5 objectives
    scales = torch.tensor([1.0, 5.0, 10.0, 0.1, 20.0])
    offsets = torch.tensor([10.0, -25.0, 50.0, 0.5, -100.0])
    raw_adv = torch.randn(200, 5) * scales + offsets

    norm_adv = agent.normalize_advantages_per_objective(raw_adv)

    assert norm_adv.shape == (200, 5)

    # Check each dimension independently
    for dim in range(5):
        dim_mean = norm_adv[:, dim].mean().item()
        dim_std = norm_adv[:, dim].std(unbiased=False).item()
        assert abs(dim_mean) < 1e-4, f"Dim {dim} mean {dim_mean} is not ~ 0"
        assert abs(dim_std - 1.0) < 1e-2, f"Dim {dim} std {dim_std} is not ~ 1"


def test_normalize_advantages_per_objective_constant_safe():
    """Verify normalizing constant advantages does not produce NaN or error."""
    network = MockActorCriticNetwork()
    agent = MOPPOAgent(network=network)

    constant_adv = torch.ones(50, 5) * 5.0
    norm_adv = agent.normalize_advantages_per_objective(constant_adv)

    assert not torch.isnan(norm_adv).any()
    assert not torch.isinf(norm_adv).any()
    # When all values are identical, (x - mean) == 0, so normalized values are 0
    assert torch.allclose(norm_adv, torch.zeros_like(norm_adv), atol=1e-5)


def test_normalize_advantages_per_objective_masked():
    """Verify masked advantage normalization computes statistics only over active entries."""
    network = MockActorCriticNetwork()
    agent = MOPPOAgent(network=network)

    torch.manual_seed(99)
    adv = torch.randn(100, 5)
    # Mask where first 60 items are active (1) and last 40 are inactive (0)
    mask = torch.zeros(100, dtype=torch.bool)
    mask[:60] = True

    norm_adv = agent.normalize_advantages_per_objective(adv, mask=mask)

    assert norm_adv.shape == (100, 5)
    # Inactive entries must be exactly 0
    assert torch.all(norm_adv[60:] == 0.0)

    # Active entries should have mean ~ 0 and std ~ 1
    active_norm = norm_adv[:60]
    for dim in range(5):
        m = active_norm[:, dim].mean().item()
        s = active_norm[:, dim].std(unbiased=False).item()
        assert abs(m) < 1e-4
        assert abs(s - 1.0) < 1e-2


def test_scalarize_advantages():
    """Verify dot product scalarization across the 5 dimensions with weight vector w."""
    network = MockActorCriticNetwork()
    agent = MOPPOAgent(network=network)

    norm_adv = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, -1.0, 0.0, 1.0, 3.0],
    ], dtype=torch.float32)

    weights = [0.65, 0.10, 0.20, 0.02, 0.03]

    scalar_adv = agent.scalarize_advantages(norm_adv, weights)

    assert scalar_adv.shape == (2,)

    # Expected values
    # Item 0: 1*0.65 + 2*0.10 + 3*0.20 + 4*0.02 + 5*0.03 = 0.65 + 0.20 + 0.60 + 0.08 + 0.15 = 1.68
    # Item 1: 2*0.65 - 1*0.10 + 0*0.20 + 1*0.02 + 3*0.03 = 1.30 - 0.10 + 0.00 + 0.02 + 0.09 = 1.31
    assert math.isclose(scalar_adv[0].item(), 1.68, abs_tol=1e-5)
    assert math.isclose(scalar_adv[1].item(), 1.31, abs_tol=1e-5)


def test_scalarize_advantages_normalization():
    """Verify weight normalization when normalize_weights=True."""
    network = MockActorCriticNetwork()
    agent = MOPPOAgent(network=network)

    norm_adv = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    # Unnormalized weights summing to 10
    unnormalized_w = [2.0, 2.0, 2.0, 2.0, 2.0]

    scalar_adv = agent.scalarize_advantages(norm_adv, unnormalized_w, normalize_weights=True)
    assert math.isclose(scalar_adv[0].item(), 1.0, abs_tol=1e-5)


def test_scalarize_advantages_invalid_weights_length():
    """Verify ValueError is raised if weights vector does not have 5 elements."""
    network = MockActorCriticNetwork()
    agent = MOPPOAgent(network=network)
    adv = torch.randn(10, 5)

    with pytest.raises(ValueError, match="Expected 5 weights"):
        agent.scalarize_advantages(adv, [0.5, 0.5])


def test_compute_explained_variance_5d():
    """Verify 5D explained variance computation for perfect, zero, and noisy predictions."""
    y_true = torch.randn(100, 5)

    # 1. Perfect prediction (y_pred == y_true) -> EV should be 1.0 for all 5 dims
    ev_perfect = MOPPOAgent.compute_explained_variance_5d(y_true, y_true)
    assert len(ev_perfect) == 5
    for val in ev_perfect:
        assert math.isclose(val, 1.0, abs_tol=1e-4)

    # 2. Zero-variance constant target
    constant_true = torch.zeros(50, 5)
    ev_zero = MOPPOAgent.compute_explained_variance_5d(constant_true, constant_true)
    for val in ev_zero:
        assert math.isclose(val, 1.0, abs_tol=1e-4)

    # 3. Known variance reduction
    torch.manual_seed(42)
    target = torch.randn(1000, 5)
    noise = torch.randn(1000, 5) * 0.1
    pred = target + noise
    ev_noisy = MOPPOAgent.compute_explained_variance_5d(pred, target)
    for val in ev_noisy:
        assert val > 0.95  # Signal dominated by target


def test_backward_pass_gradients():
    """Verify loss calculation propagates non-zero gradients to actor and critic heads."""
    network = MockActorCriticNetwork()
    agent = MOPPOAgent(network=network)

    obs = torch.randint(0, 256, (8, 4, 84, 84), dtype=torch.uint8)
    actions = torch.randint(0, 7, (8,), dtype=torch.long)
    old_logprobs = torch.zeros(8)
    advantages = torch.randn(8, 5)
    returns = torch.randn(8, 5)
    weights = [0.65, 0.10, 0.20, 0.02, 0.03]

    batch = {
        "obs": obs,
        "actions": actions,
        "logprobs": old_logprobs,
        "advantages": advantages,
        "returns": returns,
    }

    class SingleBatchBuffer:
        def get_batches(self, batch_size=8):
            yield batch

    agent.optimizer.zero_grad()
    metrics = agent.train_step(SingleBatchBuffer(), weights=weights, update_epochs=1, batch_size=8)

    # Verify gradients exist and are non-zero on both actor and critic heads
    assert network.actor_head.weight.grad is not None
    assert torch.sum(torch.abs(network.actor_head.weight.grad)) > 0.0
    assert network.critic_head.weight.grad is not None
    assert torch.sum(torch.abs(network.critic_head.weight.grad)) > 0.0

    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert "total_loss" in metrics


def test_train_step_with_dummy_buffer_dict():
    """Verify complete train_step execution with dictionary-based rollout buffer."""
    network = MockActorCriticNetwork()
    agent = MOPPOAgent(network=network, update_epochs=2, batch_size=16)
    buffer = DummyRolloutBufferDict(num_samples=48, batch_size=16)

    weights = [0.65, 0.10, 0.20, 0.02, 0.03]
    metrics = agent.train_step(buffer, weights=weights)

    required_keys = [
        "actor_loss",
        "critic_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance_5d",
        "total_loss",
        "learning_rate",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing metric key: {key}"
        assert not math.isnan(metrics[key]) if not isinstance(metrics[key], list) else True

    assert isinstance(metrics["explained_variance_5d"], list)
    assert len(metrics["explained_variance_5d"]) == 5


def test_train_step_with_dummy_buffer_tuple():
    """Verify train_step handles tuple-based batches correctly."""
    network = MockActorCriticNetwork()
    agent = MOPPOAgent(network=network, update_epochs=1, batch_size=16)
    buffer = DummyRolloutBufferTuple(num_samples=32, batch_size=16)

    weights = [0.40, 0.40, 0.15, 0.02, 0.03]
    metrics = agent.train_step(buffer, weights=weights)

    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert "explained_variance_5d" in metrics
    assert len(metrics["explained_variance_5d"]) == 5


def test_set_learning_rate():
    """Verify set_learning_rate updates optimizer parameter groups."""
    network = MockActorCriticNetwork()
    agent = MOPPOAgent(network=network, lr=2.5e-4)

    assert agent.optimizer.param_groups[0]["lr"] == 2.5e-4
    agent.set_learning_rate(1.0e-5)
    assert agent.lr == 1.0e-5
    assert agent.optimizer.param_groups[0]["lr"] == 1.0e-5


def test_save_and_load(tmp_path):
    """Verify checkpoint save and load restores network and optimizer states."""
    network1 = MockActorCriticNetwork()
    agent1 = MOPPOAgent(network=network1, lr=3e-4)

    # Modify a parameter so weights differ from default initialization
    with torch.no_grad():
        network1.actor_head.weight.add_(1.5)

    save_path = tmp_path / "checkpoints" / "test_model.pt"
    saved_file = agent1.save(save_path)
    assert save_path.exists()
    assert str(save_path) == saved_file

    network2 = MockActorCriticNetwork()
    agent2 = MOPPOAgent(network=network2, lr=1e-4)

    # Before loading, weights should differ
    assert not torch.allclose(network1.actor_head.weight, network2.actor_head.weight)

    # After loading, weights should match exactly
    loaded_checkpoint = agent2.load(saved_file)
    assert loaded_checkpoint is not None
    assert torch.allclose(network1.actor_head.weight, network2.actor_head.weight)
    assert agent2.lr == 3e-4


def test_inference_methods():
    """Verify get_action, get_value, and get_action_and_value."""
    network = MockActorCriticNetwork()
    agent = MOPPOAgent(network=network)

    obs_single = torch.randint(0, 256, (4, 84, 84), dtype=torch.uint8)
    obs_batch = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8)

    # 1. get_action single
    act_single = agent.get_action(obs_single, deterministic=False)
    assert act_single.numel() == 1
    assert 0 <= act_single.item() < 7

    # 2. get_action deterministic
    act_det = agent.get_action(obs_batch, deterministic=True)
    assert act_det.shape == (3,)

    # 3. get_value
    val_single = agent.get_value(obs_single)
    assert val_single.shape == (1, 5)
    val_batch = agent.get_value(obs_batch)
    assert val_batch.shape == (3, 5)

    # 4. get_action_and_value
    act, logprob, ent, val = agent.get_action_and_value(obs_batch)
    assert act.shape == (3,)
    assert logprob.shape == (3,)
    assert ent.shape == (3,)
    assert val.shape == (3, 5)


def test_unpack_batch_invalid():
    """Verify unpack_batch error handling on unsupported types."""
    with pytest.raises(TypeError, match="Unsupported batch type"):
        _unpack_batch("invalid_batch_string")

    with pytest.raises(ValueError, match="at least 5 elements"):
        _unpack_batch((1, 2, 3))


def test_agent_with_real_model_and_buffer():
    """Integration test: MOPPOAgent trained with ActorCriticResNet and MultiObjectiveRolloutBuffer."""
    from src.model.impala_cnn import ActorCriticResNet
    from src.model.rollout_buffer import MultiObjectiveRolloutBuffer

    num_steps = 8
    num_envs = 2
    network = ActorCriticResNet(in_channels=4, num_actions=7, num_objectives=5)
    agent = MOPPOAgent(network=network, lr=2.5e-4, update_epochs=2, batch_size=8)
    buffer = MultiObjectiveRolloutBuffer(num_steps=num_steps, num_envs=num_envs)

    # Populate buffer with dummy rollout
    obs = torch.randint(0, 256, (num_envs, 4, 84, 84), dtype=torch.uint8)
    with torch.no_grad():
        for _ in range(num_steps):
            actions, logprobs, _, values = network.get_action_and_value(obs)
            rewards = torch.randn(num_envs, 5)
            dones = torch.zeros(num_envs, dtype=torch.bool)
            buffer.add(obs, actions, logprobs, rewards, dones, values)
            obs = torch.randint(0, 256, (num_envs, 4, 84, 84), dtype=torch.uint8)

        # Compute returns & advantages
        next_obs = obs
        next_done = torch.zeros(num_envs, dtype=torch.bool)
        buffer.compute_returns_and_advantages(next_obs=next_obs, next_done=next_done, model=network)

    # Run train step
    weights = [0.65, 0.10, 0.20, 0.02, 0.03]
    metrics = agent.train_step(buffer, weights=weights)

    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert "explained_variance_5d" in metrics
    assert len(metrics["explained_variance_5d"]) == 5
    assert not math.isnan(metrics["total_loss"])

