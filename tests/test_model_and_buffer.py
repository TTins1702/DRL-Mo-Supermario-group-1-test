"""Unit tests for Module 2: ActorCriticResNet and MultiObjectiveRolloutBuffer.

Kiểm thử toàn diện:
1. Khởi tạo trực giao (Orthogonal Initialization) cho Actor (0.01), Critic (1.0), Backbone.
2. Hình dạng tensor đầu ra (Forward Pass Shapes) cho uint8, float32, ndarray.
3. Feature Extractor dimensions (flatten 7744 -> 512).
4. Khả năng lưu trữ và nén bộ nhớ của MultiObjectiveRolloutBuffer (obs uint8).
5. Tính đúng đắn toán học của 5D GAE (so sánh từng bước đệ quy với tính toán giải tích).
6. Cơ chế ngắt lan truyền lợi thế khi gặp trạng thái kết thúc (terminal mask m_t = 1 - done_t).
7. Phân phối mini-batches (RolloutBatch) với cả 3 kiểu truy cập: attribute, dict key, tuple unpack.
8. Luồng tích hợp Rollout -> GAE -> Mini-batch -> Forward pass.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.model import (
    ActorCriticResNet,
    MultiObjectiveRolloutBuffer,
    RolloutBatch,
    ResidualBlock,
    ConvSequence,
)


class TestActorCriticResNet:
    """Kiểm thử kiến trúc mạng ActorCriticResNet."""

    def test_orthogonal_initialization(self):
        """Kiểm tra khởi tạo trực giao cho Actor (gain 0.01) và Critic (gain 1.0)."""
        model = ActorCriticResNet(
            in_channels=4,
            num_actions=7,
            num_objectives=5,
            features_dim=512,
        )

        # Actor head gain=0.01: phương sai trọng số phải nhỏ
        actor_w = model.actor.weight.data
        actor_bias = model.actor.bias.data
        assert torch.all(actor_bias == 0.0), "Actor bias phải khởi tạo bằng 0.0"
        # Với orthogonal init gain 0.01 trên ma trận (7, 512), norm của từng hàng xấp xỉ gain = 0.01
        row_norms = torch.norm(actor_w, dim=1)
        assert torch.allclose(row_norms, torch.full_like(row_norms, 0.01), atol=1e-3), (
            f"Norm hàng của Actor weight phải xấp xỉ 0.01, nhận: {row_norms}"
        )

        # Critic head gain=1.0: norm hàng xấp xỉ 1.0
        critic_w = model.critic.weight.data
        critic_bias = model.critic.bias.data
        assert torch.all(critic_bias == 0.0), "Critic bias phải khởi tạo bằng 0.0"
        row_norms_c = torch.norm(critic_w, dim=1)
        assert torch.allclose(row_norms_c, torch.full_like(row_norms_c, 1.0), atol=1e-3), (
            f"Norm hàng của Critic weight phải xấp xỉ 1.0, nhận: {row_norms_c}"
        )

        # FC layer bias = 0.0
        assert torch.all(model.fc.bias.data == 0.0)

    def test_forward_shapes_uint8(self):
        """Kiểm tra forward pass với đầu vào uint8 (tự động chia 255.0)."""
        model = ActorCriticResNet()
        obs_uint8 = torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.uint8)

        logits, value_5d = model(obs_uint8)
        assert logits.shape == (4, 7), f"Logits shape phải là (4, 7), nhận {logits.shape}"
        assert value_5d.shape == (4, 5), f"Value shape phải là (4, 5), nhận {value_5d.shape}"

    def test_forward_shapes_float32(self):
        """Kiểm tra forward pass với đầu vào float32 trong [0, 1]."""
        model = ActorCriticResNet()
        obs_float = torch.rand((2, 4, 84, 84), dtype=torch.float32)

        logits, value_5d = model(obs_float)
        assert logits.shape == (2, 7)
        assert value_5d.shape == (2, 5)

    def test_single_obs_3d(self):
        """Kiểm tra truyền vào 1 observation 3D (4, 84, 84) đơn lẻ."""
        model = ActorCriticResNet()
        obs_single = torch.randint(0, 256, (4, 84, 84), dtype=torch.uint8)

        logits, value_5d = model(obs_single)
        assert logits.shape == (1, 7)
        assert value_5d.shape == (1, 5)

    def test_numpy_input(self):
        """Kiểm tra truyền vào mảng NumPy ndarray."""
        model = ActorCriticResNet()
        obs_np = np.random.randint(0, 256, size=(3, 4, 84, 84), dtype=np.uint8)

        logits, value_5d = model(obs_np)
        assert logits.shape == (3, 7)
        assert value_5d.shape == (3, 5)

    def test_get_value(self):
        """Kiểm tra phương thức get_value trả về tensor (B, 5)."""
        model = ActorCriticResNet()
        obs = torch.randint(0, 256, (5, 4, 84, 84), dtype=torch.uint8)

        values = model.get_value(obs)
        assert values.shape == (5, 5)
        assert values.dtype == torch.float32

    def test_get_action_and_value_stochastic(self):
        """Kiểm tra phương thức get_action_and_value lấy mẫu ngẫu nhiên."""
        model = ActorCriticResNet()
        obs = torch.randint(0, 256, (8, 4, 84, 84), dtype=torch.uint8)

        action, log_prob, entropy, value_5d = model.get_action_and_value(obs)
        assert action.shape == (8,)
        assert action.dtype == torch.long
        assert torch.all(action >= 0) and torch.all(action < 7)

        assert log_prob.shape == (8,)
        assert log_prob.dtype == torch.float32

        assert entropy.shape == (8,)
        assert torch.all(entropy >= 0.0)

        assert value_5d.shape == (8, 5)

    def test_get_action_and_value_deterministic(self):
        """Kiểm tra chế độ deterministic trả về argmax logits."""
        model = ActorCriticResNet()
        obs = torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.uint8)

        logits, _ = model(obs)
        expected_action = torch.argmax(logits, dim=-1)

        action, _, _, _ = model.get_action_and_value(obs, deterministic=True)
        assert torch.equal(action, expected_action)

    def test_get_action_and_value_with_given_action(self):
        """Kiểm tra tính log_prob và entropy khi truyền action có sẵn (vòng update PPO)."""
        model = ActorCriticResNet()
        obs = torch.randint(0, 256, (6, 4, 84, 84), dtype=torch.uint8)
        given_actions = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)

        action, log_prob, entropy, value_5d = model.get_action_and_value(
            obs, action=given_actions
        )
        assert torch.equal(action, given_actions)
        assert log_prob.shape == (6,)
        assert entropy.shape == (6,)
        assert value_5d.shape == (6, 5)

    def test_feature_extractor_dimension(self):
        """Kiểm tra kích thước vector đặc trưng h in R^{B x 512} và gradient backward."""
        model = ActorCriticResNet()
        obs = torch.rand((2, 4, 84, 84), dtype=torch.float32, requires_grad=True)

        h = model.extract_features(obs)
        assert h.shape == (2, 512)

        # Backward test
        loss = h.sum()
        loss.backward()
        assert obs.grad is not None
        assert not torch.isnan(obs.grad).any()


class TestMultiObjectiveRolloutBuffer:
    """Kiểm thử bộ đệm Rollout đa mục tiêu và 5D GAE."""

    def test_buffer_initialization(self):
        """Kiểm tra khởi tạo bộ đệm với dung lượng và kiểu dữ liệu chuẩn (uint8)."""
        num_steps = 16
        num_envs = 4
        buffer = MultiObjectiveRolloutBuffer(
            num_steps=num_steps,
            num_envs=num_envs,
            obs_shape=(4, 84, 84),
            num_actions=7,
            num_objectives=5,
        )

        assert buffer.obs.shape == (16, 4, 4, 84, 84)
        assert buffer.obs.dtype == torch.uint8, "obs phải lưu uint8 để tiết kiệm RAM"
        assert buffer.actions.shape == (16, 4)
        assert buffer.rewards.shape == (16, 4, 5)
        assert buffer.dones.shape == (16, 4)
        assert buffer.values.shape == (16, 4, 5)
        assert buffer.step == 0
        assert not buffer.full

    def test_buffer_add_and_overflow(self):
        """Kiểm tra thêm dữ liệu vào buffer, cờ full, và ngoại lệ khi tràn buffer."""
        buffer = MultiObjectiveRolloutBuffer(num_steps=3, num_envs=2)

        for step in range(3):
            obs = np.random.randint(0, 256, (2, 4, 84, 84), dtype=np.uint8)
            action = np.array([1, 2], dtype=np.int64)
            logprob = np.array([-0.5, -0.8], dtype=np.float32)
            reward = np.ones((2, 5), dtype=np.float32) * (step + 1)
            done = np.array([0.0, 0.0], dtype=np.float32)
            value = np.zeros((2, 5), dtype=np.float32)

            buffer.add(obs, action, logprob, reward, done, value)

        assert buffer.step == 3
        assert buffer.full is True

        # Thêm bước thứ 4 phải ném lỗi IndexError
        with pytest.raises(IndexError):
            buffer.add(obs, action, logprob, reward, done, value)

        # Reset
        buffer.reset()
        assert buffer.step == 0
        assert buffer.full is False

    def test_gae_5d_exact_math(self):
        """Kiểm chứng toán học chính xác của 5-Dimensional GAE đệ quy ngược.

        Thiết lập bài toán nhỏ kiểm chứng giải tích:
        T = 3, N = 1, D = 5, gamma = 0.9, lambda = 0.8
        """
        buffer = MultiObjectiveRolloutBuffer(
            num_steps=3,
            num_envs=1,
            obs_shape=(4, 84, 84),
            num_objectives=5,
        )

        r0 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        r1 = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        r2 = np.array([[2.0, 1.0, 0.0, 3.0, -1.0]], dtype=np.float32)

        v0 = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        v1 = np.array([[2.0, 2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
        v2 = np.array([[1.5, 1.5, 1.5, 1.5, 1.5]], dtype=np.float32)

        dummy_obs = np.zeros((1, 4, 84, 84), dtype=np.uint8)
        dummy_act = np.array([0], dtype=np.int64)
        dummy_lp = np.array([0.0], dtype=np.float32)

        # Thêm 3 bước vào buffer
        buffer.add(dummy_obs, dummy_act, dummy_lp, r0, np.array([0.0]), v0)
        buffer.add(dummy_obs, dummy_act, dummy_lp, r1, np.array([0.0]), v1)
        buffer.add(dummy_obs, dummy_act, dummy_lp, r2, np.array([0.0]), v2)

        next_val = np.array([[3.0, 3.0, 3.0, 3.0, 3.0]], dtype=np.float32)
        next_done = np.array([0.0], dtype=np.float32)

        buffer.compute_returns_and_advantages(
            next_value=next_val,
            next_done=next_done,
            gamma=0.9,
            gae_lambda=0.8,
        )

        # Tính toán giải tích từng bước:
        # t = 2:
        # delta_2 = r2 + 0.9 * 3.0 - 1.5 = r2 + 1.2
        # delta_2 = [3.2, 2.2, 1.2, 4.2, 0.2]
        # A_hat_2 = delta_2
        expected_a2 = np.array([3.2, 2.2, 1.2, 4.2, 0.2], dtype=np.float32)
        expected_r2 = expected_a2 + v2[0]

        # t = 1:
        # delta_1 = r1 + 0.9 * 1.5 - 2.0 = 0.5 + 1.35 - 2.0 = -0.15
        # A_hat_1 = -0.15 + (0.9 * 0.8) * A_hat_2 = -0.15 + 0.72 * A_hat_2
        expected_a1 = -0.15 + 0.72 * expected_a2
        expected_r1 = expected_a1 + v1[0]

        # t = 0:
        # delta_0 = r0 + 0.9 * 2.0 - 1.0 = r0 + 0.8
        delta_0 = r0[0] + 0.8
        expected_a0 = delta_0 + 0.72 * expected_a1
        expected_r0 = expected_a0 + v0[0]

        # So khớp kết quả buffer tính toán với kết quả giải tích
        assert np.allclose(buffer.advantages[2, 0].cpu().numpy(), expected_a2, atol=1e-5)
        assert np.allclose(buffer.returns[2, 0].cpu().numpy(), expected_r2, atol=1e-5)

        assert np.allclose(buffer.advantages[1, 0].cpu().numpy(), expected_a1, atol=1e-5)
        assert np.allclose(buffer.returns[1, 0].cpu().numpy(), expected_r1, atol=1e-5)

        assert np.allclose(buffer.advantages[0, 0].cpu().numpy(), expected_a0, atol=1e-5)
        assert np.allclose(buffer.returns[0, 0].cpu().numpy(), expected_r0, atol=1e-5)

    def test_gae_5d_terminal_masking(self):
        """Kiểm tra khi Mario chết tại bước t (done_t = 1):

        Lợi thế tương lai không được rò rỉ ngược qua ranh giới kết thúc episode.
        """
        buffer = MultiObjectiveRolloutBuffer(
            num_steps=3,
            num_envs=1,
            obs_shape=(4, 84, 84),
            num_objectives=5,
        )

        dummy_obs = np.zeros((1, 4, 84, 84), dtype=np.uint8)
        dummy_act = np.array([0], dtype=np.int64)
        dummy_lp = np.array([0.0], dtype=np.float32)

        r = np.ones((1, 5), dtype=np.float32)
        v = np.full((1, 5), 2.0, dtype=np.float32)

        # Bước 1 bị kết thúc: done = 1.0
        buffer.add(dummy_obs, dummy_act, dummy_lp, r, np.array([0.0]), v)
        buffer.add(dummy_obs, dummy_act, dummy_lp, r, np.array([1.0]), v)
        buffer.add(dummy_obs, dummy_act, dummy_lp, r, np.array([0.0]), v)

        next_val = np.full((1, 5), 2.0, dtype=np.float32)
        buffer.compute_returns_and_advantages(
            next_value=next_val,
            next_done=np.array([0.0]),
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Tại bước t = 1, done = 1 -> non_terminal = 0:
        # delta_1 = r_1 + gamma * 0 * V(s_2) - V(s_1) = 1.0 - 2.0 = -1.0
        # A_hat_1 = delta_1 + gamma * lambda * 0 * A_hat_2 = delta_1 = -1.0
        # R_1 = A_hat_1 + V(s_1) = -1.0 + 2.0 = 1.0 = r_1
        adv_1 = buffer.advantages[1, 0].cpu().numpy()
        ret_1 = buffer.returns[1, 0].cpu().numpy()

        assert np.allclose(adv_1, np.full((5,), -1.0), atol=1e-5)
        assert np.allclose(ret_1, np.full((5,), 1.0), atol=1e-5)

    def test_get_batches_and_rollout_batch_access(self):
        """Kiểm tra hàm get_batches và 3 cơ chế truy cập của RolloutBatch."""
        num_steps = 4
        num_envs = 2
        buffer = MultiObjectiveRolloutBuffer(num_steps=num_steps, num_envs=num_envs)

        for step in range(num_steps):
            obs = np.full((num_envs, 4, 84, 84), step, dtype=np.uint8)
            action = np.array([step % 7, (step + 1) % 7], dtype=np.int64)
            logprob = np.array([-0.1 * step, -0.2 * step], dtype=np.float32)
            reward = np.ones((num_envs, 5), dtype=np.float32) * step
            done = np.array([0.0, 0.0], dtype=np.float32)
            value = np.ones((num_envs, 5), dtype=np.float32) * step
            buffer.add(obs, action, logprob, reward, done, value)

        buffer.compute_returns_and_advantages(
            next_value=np.zeros((num_envs, 5), dtype=np.float32)
        )

        # Tổng số mẫu là 4 * 2 = 8
        batches = list(buffer.get_batches(batch_size=3, shuffle=False))
        assert len(batches) == 3
        assert batches[0].obs.shape[0] == 3
        assert batches[1].obs.shape[0] == 3
        assert batches[2].obs.shape[0] == 2

        first_batch: RolloutBatch = batches[0]

        # 1. Truy cập theo Attribute
        assert first_batch.obs.shape == (3, 4, 84, 84)
        assert first_batch.actions.shape == (3,)
        assert first_batch.logprobs.shape == (3,)
        assert first_batch.advantages.shape == (3, 5)
        assert first_batch.returns.shape == (3, 5)
        assert first_batch.values.shape == (3, 5)

        # 2. Truy cập theo Dictionary Key
        assert torch.equal(first_batch["obs"], first_batch.obs)
        assert torch.equal(first_batch["advantages"], first_batch.advantages)

        # 3. Tuple Unpacking
        obs, act, lp, adv, ret, val = first_batch
        assert torch.equal(obs, first_batch.obs)
        assert torch.equal(act, first_batch.actions)
        assert torch.equal(lp, first_batch.logprobs)
        assert torch.equal(adv, first_batch.advantages)
        assert torch.equal(ret, first_batch.returns)
        assert torch.equal(val, first_batch.values)

    def test_get_all(self):
        """Kiểm tra get_all trả về toàn bộ buffer đã trải phẳng."""
        buffer = MultiObjectiveRolloutBuffer(num_steps=3, num_envs=2)
        for _ in range(3):
            buffer.add(
                np.zeros((2, 4, 84, 84), dtype=np.uint8),
                np.array([0, 1]),
                np.array([0.0, 0.0]),
                np.zeros((2, 5)),
                np.array([0.0, 0.0]),
                np.zeros((2, 5)),
            )
        buffer.compute_returns_and_advantages(next_value=np.zeros((2, 5)))

        all_batch = buffer.get_all()
        assert all_batch.obs.shape == (6, 4, 84, 84)
        assert all_batch.advantages.shape == (6, 5)


class TestModelAndBufferIntegration:
    """Kiểm tra tích hợp toàn diện Model và Buffer qua 1 chu kỳ Rollout."""

    def test_end_to_end_rollout_cycle(self):
        """Mô phỏng 1 chu trình thu thập mẫu và tính toán cập nhật mạng."""
        num_steps = 4
        num_envs = 2
        model = ActorCriticResNet()
        buffer = MultiObjectiveRolloutBuffer(num_steps=num_steps, num_envs=num_envs)

        obs = np.random.randint(0, 256, (num_envs, 4, 84, 84), dtype=np.uint8)

        for step in range(num_steps):
            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(obs)

            # Mô phỏng env step
            next_obs = np.random.randint(0, 256, (num_envs, 4, 84, 84), dtype=np.uint8)
            reward = np.random.randn(num_envs, 5).astype(np.float32)
            done = np.zeros(num_envs, dtype=np.float32)

            buffer.add(obs, action, logprob, reward, done, value)
            obs = next_obs

        # Tính toán GAE sử dụng model và next_obs
        buffer.compute_returns_and_advantages(
            next_obs=obs,
            next_done=done,
            model=model,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert buffer.full is True
        assert not torch.isnan(buffer.advantages).any()
        assert not torch.isnan(buffer.returns).any()

        # Thử lấy batches và forward lại qua model
        for batch in buffer.get_batches(batch_size=4):
            new_action, new_logprob, entropy, new_value = model.get_action_and_value(
                batch.obs, action=batch.actions
            )
            assert new_action.shape == (4,)
            assert new_logprob.shape == (4,)
            assert entropy.shape == (4,)
            assert new_value.shape == (4, 5)
