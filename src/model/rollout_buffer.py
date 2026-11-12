"""Multi-Objective Rollout Buffer with 5D GAE (Generalized Advantage Estimation).

Phân hệ 2 (Module 2): Bộ đệm thu thập mẫu đa mục tiêu MultiObjectiveRolloutBuffer.
- Tối ưu hóa bộ nhớ: Lưu trữ trạng thái (obs) dưới dạng torch.uint8 (T, N, 4, 84, 84)
  giúp tiết kiệm 4 lần dung lượng RAM so với float32 trên Colab/Kaggle.
- Lưu trữ đầy đủ: actions (T, N), logprobs (T, N), rewards (T, N, 5), dones (T, N), values (T, N, 5).
- Ước lượng lợi thế 5 chiều (5-Dimensional GAE):
  delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t) in R^{N x 5}
  A_hat_t = delta_t + gamma * lambda * (1 - done_t) * A_hat_{t+1} in R^{N x 5}
  R_t = A_hat_t + V(s_t) in R^{N x 5}
- Phân phối mini-batches (get_batches) linh hoạt cho vòng cập nhật PPO.
"""

from typing import Generator, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class RolloutBatch:
    """Mini-batch dữ liệu được trích xuất từ MultiObjectiveRolloutBuffer.

    Hỗ trợ 3 phong cách truy cập linh hoạt:
    1. Truy cập thuộc tính (Named attributes): batch.obs, batch.actions, batch.logprobs,
       batch.advantages, batch.returns, batch.values
    2. Truy cập kiểu từ điển (Dict keys): batch["obs"], batch["advantages"], ...
    3. Phân rã kiểu Tuple (Unpacking): obs, act, logp, adv, ret, val = batch
    """

    __slots__ = ("obs", "actions", "logprobs", "advantages", "returns", "values")

    def __init__(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
    ):
        self.obs = obs
        self.actions = actions
        self.logprobs = logprobs
        self.advantages = advantages
        self.returns = returns
        self.values = values

    def __iter__(self):
        yield self.obs
        yield self.actions
        yield self.logprobs
        yield self.advantages
        yield self.returns
        yield self.values

    def __getitem__(self, item: Union[str, int]) -> torch.Tensor:
        if isinstance(item, str):
            if hasattr(self, item):
                return getattr(self, item)
            raise KeyError(f"Key '{item}' does not exist in RolloutBatch.")
        elif isinstance(item, int):
            fields = (
                self.obs,
                self.actions,
                self.logprobs,
                self.advantages,
                self.returns,
                self.values,
            )
            return fields[item]
        raise TypeError(f"Invalid index type: {type(item)}")

    def get(self, key: str, default: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return (
            f"RolloutBatch(\n"
            f"  obs={self.obs.shape}, dtype={self.obs.dtype},\n"
            f"  actions={self.actions.shape}, dtype={self.actions.dtype},\n"
            f"  logprobs={self.logprobs.shape},\n"
            f"  advantages={self.advantages.shape},\n"
            f"  returns={self.returns.shape},\n"
            f"  values={self.values.shape}\n"
            f")"
        )


class MultiObjectiveRolloutBuffer:
    """Bộ đệm Rollout đa mục tiêu hỗ trợ 5D GAE cho PPO.

    Lưu trữ các bước di chuyển (rollout steps) từ N môi trường song song:
    - num_steps (T): Số bước thu thập mỗi môi trường trước khi update (ví dụ 512).
    - num_envs (N): Số môi trường chạy song song (ví dụ 8).
    - Tổng dung lượng mỗi lần cập nhật B = T * N (ví dụ 4096 mẫu).
    """

    def __init__(
        self,
        num_steps: int = 512,
        num_envs: int = 8,
        obs_shape: Tuple[int, ...] = (4, 84, 84),
        num_actions: int = 7,
        num_objectives: int = 5,
        device: Union[str, torch.device] = "cpu",
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.device = torch.device(device)

        self.step: int = 0
        self.full: bool = False

        # Khởi tạo bộ nhớ đệm
        # obs lưu uint8: (T, N, 4, 84, 84) -> Tiết kiệm 4x RAM
        self.obs = torch.zeros(
            (num_steps, num_envs) + obs_shape,
            dtype=torch.uint8,
            device=self.device,
        )
        self.actions = torch.zeros(
            (num_steps, num_envs),
            dtype=torch.long,
            device=self.device,
        )
        self.logprobs = torch.zeros(
            (num_steps, num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        # Vector reward 5 chiều: (T, N, 5)
        self.rewards = torch.zeros(
            (num_steps, num_envs, num_objectives),
            dtype=torch.float32,
            device=self.device,
        )
        self.dones = torch.zeros(
            (num_steps, num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        # Vector giá trị 5 chiều: (T, N, 5)
        self.values = torch.zeros(
            (num_steps, num_envs, num_objectives),
            dtype=torch.float32,
            device=self.device,
        )

        # Ước lượng lợi thế và kỳ vọng phần thưởng 5 chiều
        self.advantages = torch.zeros(
            (num_steps, num_envs, num_objectives),
            dtype=torch.float32,
            device=self.device,
        )
        self.returns = torch.zeros(
            (num_steps, num_envs, num_objectives),
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self) -> None:
        """Đặt lại con trỏ bộ đệm về 0 để chuẩn bị cho chu kỳ rollout tiếp theo."""
        self.step = 0
        self.full = False

    def add(
        self,
        obs: Union[torch.Tensor, np.ndarray],
        action: Union[torch.Tensor, np.ndarray],
        logprob: Union[torch.Tensor, np.ndarray],
        reward: Union[torch.Tensor, np.ndarray],
        done: Union[torch.Tensor, np.ndarray],
        value: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Ghi nhận 1 bước chuyển tiếp của N môi trường vào buffer.

        Args:
            obs: Observation tại bước t dạng (N, 4, 84, 84) kiểu uint8 hoặc float32.
            action: Hành động đã chọn dạng (N,).
            logprob: Log xác suất của hành động dạng (N,).
            reward: Vector phần thưởng 5 chiều dạng (N, 5).
            done: Cờ kết thúc episode dạng (N,).
            value: Vector giá trị 5 chiều dự đoán tại bước t dạng (N, 5).
        """
        if self.step >= self.num_steps:
            raise IndexError(
                f"MultiObjectiveRolloutBuffer đã đầy (step={self.step}, num_steps={self.num_steps}). "
                "Cần gọi compute_returns_and_advantages() và reset() trước khi thêm bước mới."
            )

        # Xử lý obs: đảm bảo lưu uint8 (0..255)
        if isinstance(obs, np.ndarray):
            if obs.dtype != np.uint8:
                if obs.max() <= 1.01 and obs.min() >= 0.0:
                    obs = (obs * 255.0).astype(np.uint8)
                else:
                    obs = obs.astype(np.uint8)
            obs_tensor = torch.from_numpy(obs)
        else:
            if obs.dtype != torch.uint8:
                if obs.max() <= 1.01 and obs.min() >= 0.0:
                    obs_tensor = (obs * 255.0).clamp(0, 255).to(torch.uint8)
                else:
                    obs_tensor = obs.clamp(0, 255).to(torch.uint8)
            else:
                obs_tensor = obs

        self.obs[self.step] = obs_tensor.to(device=self.device)

        # Action: int64 / long
        if isinstance(action, np.ndarray):
            action_tensor = torch.from_numpy(action)
        else:
            action_tensor = action
        self.actions[self.step] = action_tensor.to(
            device=self.device, dtype=torch.long
        ).reshape(self.num_envs)

        # Logprob: float32
        if isinstance(logprob, np.ndarray):
            logprob_tensor = torch.from_numpy(logprob)
        else:
            logprob_tensor = logprob
        self.logprobs[self.step] = logprob_tensor.to(
            device=self.device, dtype=torch.float32
        ).reshape(self.num_envs)

        # Reward: float32 (N, 5)
        if isinstance(reward, np.ndarray):
            reward_tensor = torch.from_numpy(reward)
        else:
            reward_tensor = reward
        reward_tensor = reward_tensor.to(
            device=self.device, dtype=torch.float32
        )
        if reward_tensor.ndim == 1 and self.num_objectives == 1:
            reward_tensor = reward_tensor.unsqueeze(-1)
        self.rewards[self.step] = reward_tensor.reshape(self.num_envs, self.num_objectives)

        # Done: float32 (N,)
        if isinstance(done, np.ndarray):
            done_tensor = torch.from_numpy(done)
        else:
            done_tensor = done
        self.dones[self.step] = done_tensor.to(
            device=self.device, dtype=torch.float32
        ).reshape(self.num_envs)

        # Value: float32 (N, 5)
        if isinstance(value, np.ndarray):
            value_tensor = torch.from_numpy(value)
        else:
            value_tensor = value
        value_tensor = value_tensor.to(
            device=self.device, dtype=torch.float32
        )
        if value_tensor.ndim == 1 and self.num_objectives == 1:
            value_tensor = value_tensor.unsqueeze(-1)
        self.values[self.step] = value_tensor.reshape(self.num_envs, self.num_objectives)

        self.step += 1
        if self.step == self.num_steps:
            self.full = True

    def compute_returns_and_advantages(
        self,
        next_obs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        next_done: Optional[Union[torch.Tensor, np.ndarray]] = None,
        gamma: Union[float, Sequence[float], torch.Tensor] = 0.99,
        gae_lambda: Union[float, Sequence[float], torch.Tensor] = 0.95,
        next_value: Optional[Union[torch.Tensor, np.ndarray]] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        """Tính toán ước lượng lợi thế 5 chiều (5D GAE) và Target Returns R_t bằng đệ quy ngược:

        Toán học:
            delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t) in R^{N x 5}
            A_hat_t = delta_t + gamma * lambda * (1 - done_t) * A_hat_{t+1} in R^{N x 5}
            returns_t = A_hat_t + V(s_t) in R^{N x 5}

        Hỗ trợ truyền next_value trực tiếp hoặc truyền (next_obs + model). Nếu next_obs có kích thước
        (N, num_objectives), nó sẽ tự động được nhận diện là next_value.
        """
        # Xác định next_value
        if next_value is not None:
            if isinstance(next_value, np.ndarray):
                next_value_tensor = torch.from_numpy(next_value)
            else:
                next_value_tensor = next_value
            next_value_tensor = next_value_tensor.to(
                device=self.device, dtype=torch.float32
            )
        elif next_obs is not None:
            if isinstance(next_obs, np.ndarray):
                next_obs_tensor = torch.from_numpy(next_obs)
            else:
                next_obs_tensor = next_obs

            # Nếu next_obs truyền vào đã là vector giá trị (N, 5)
            if (
                next_obs_tensor.ndim == 2
                and next_obs_tensor.shape[-1] == self.num_objectives
            ):
                next_value_tensor = next_obs_tensor.to(
                    device=self.device, dtype=torch.float32
                )
            elif model is not None:
                with torch.no_grad():
                    val = model.get_value(next_obs_tensor)
                    next_value_tensor = val.to(
                        device=self.device, dtype=torch.float32
                    )
            else:
                raise ValueError(
                    "Khi truyền next_obs dạng ảnh (N, C, H, W), cần truyền kèm tham số model "
                    "hoặc truyền trực tiếp next_value."
                )
        else:
            raise ValueError(
                "Cần cung cấp ít nhất một trong hai tham số: next_value hoặc next_obs."
            )

        # Xác định next_done
        if next_done is not None:
            if isinstance(next_done, np.ndarray):
                next_done_tensor = torch.from_numpy(next_done)
            else:
                next_done_tensor = next_done
            next_done_tensor = next_done_tensor.to(
                device=self.device, dtype=torch.float32
            ).reshape(self.num_envs)
        else:
            next_done_tensor = self.dones[-1]

        # Chuyển gamma và gae_lambda thành tensor 1D (5,) để nhân vector hóa
        if isinstance(gamma, (int, float)):
            gamma_tensor = torch.full(
                (self.num_objectives,),
                float(gamma),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            gamma_tensor = torch.as_tensor(
                gamma, device=self.device, dtype=torch.float32
            )

        if isinstance(gae_lambda, (int, float)):
            lambda_tensor = torch.full(
                (self.num_objectives,),
                float(gae_lambda),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            lambda_tensor = torch.as_tensor(
                gae_lambda, device=self.device, dtype=torch.float32
            )

        # Khởi tạo A_hat_{T} = 0 in R^{N x 5}
        last_gae_lam = torch.zeros(
            (self.num_envs, self.num_objectives),
            dtype=torch.float32,
            device=self.device,
        )

        T = self.num_steps
        for t in reversed(range(T)):
            if t == T - 1:
                # Bước cuối cùng trong rollout liên kết với next_obs / next_value
                next_non_terminal = (1.0 - next_done_tensor).unsqueeze(-1)  # (N, 1)
                next_values = next_value_tensor  # (N, 5)
            else:
                # Các bước trước đó liên kết với bước t+1
                next_non_terminal = (1.0 - self.dones[t]).unsqueeze(-1)  # (N, 1)
                next_values = self.values[t + 1]  # (N, 5)

            # delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t) in R^{N x 5}
            delta = (
                self.rewards[t]
                + gamma_tensor * next_values * next_non_terminal
                - self.values[t]
            )

            # A_hat_t = delta_t + gamma * lambda * (1 - done_t) * A_hat_{t+1} in R^{N x 5}
            last_gae_lam = (
                delta
                + gamma_tensor * lambda_tensor * next_non_terminal * last_gae_lam
            )
            self.advantages[t] = last_gae_lam

        # R_t = A_hat_t + V(s_t) in R^{N x 5}
        self.returns = self.advantages + self.values

    def get_batches(
        self,
        batch_size: int = 256,
        shuffle: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Generator[RolloutBatch, None, None]:
        """Tạo Generator phân phối các mini-batches đã trải phẳng (B_mini) cho cập nhật PPO.

        Args:
            batch_size: Kích thước mỗi mini-batch (ví dụ 256).
            shuffle: Có xáo trộn thứ tự các mẫu (True khi huấn luyện PPO).
            device: Thiết bị (CPU hoặc CUDA) để đưa các tensor trong batch đến.
                    Mặc định None (giữ nguyên self.device).

        Yields:
            RolloutBatch chứa:
            - obs: (B_mini, 4, 84, 84) kiểu uint8
            - actions: (B_mini,) kiểu long
            - logprobs: (B_mini,) kiểu float32
            - advantages: (B_mini, 5) kiểu float32
            - returns: (B_mini, 5) kiểu float32
            - values: (B_mini, 5) kiểu float32
        """
        total_samples = self.num_steps * self.num_envs
        target_device = torch.device(device) if device is not None else self.device

        # Trải phẳng (T, N, ...) thành (T * N, ...)
        b_obs = self.obs.reshape((-1,) + self.obs_shape)
        b_actions = self.actions.reshape(-1)
        b_logprobs = self.logprobs.reshape(-1)
        b_advantages = self.advantages.reshape(-1, self.num_objectives)
        b_returns = self.returns.reshape(-1, self.num_objectives)
        b_values = self.values.reshape(-1, self.num_objectives)

        if shuffle:
            indices = torch.randperm(total_samples, device=self.device)
        else:
            indices = torch.arange(total_samples, device=self.device)

        for start_idx in range(0, total_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]

            yield RolloutBatch(
                obs=b_obs[batch_indices].to(device=target_device),
                actions=b_actions[batch_indices].to(device=target_device),
                logprobs=b_logprobs[batch_indices].to(device=target_device),
                advantages=b_advantages[batch_indices].to(device=target_device),
                returns=b_returns[batch_indices].to(device=target_device),
                values=b_values[batch_indices].to(device=target_device),
            )

    def get_all(
        self, device: Optional[Union[str, torch.device]] = None
    ) -> RolloutBatch:
        """Trả về toàn bộ dữ liệu trong buffer dưới dạng một RolloutBatch duy nhất đã trải phẳng."""
        target_device = torch.device(device) if device is not None else self.device
        return RolloutBatch(
            obs=self.obs.reshape((-1,) + self.obs_shape).to(device=target_device),
            actions=self.actions.reshape(-1).to(device=target_device),
            logprobs=self.logprobs.reshape(-1).to(device=target_device),
            advantages=self.advantages.reshape(-1, self.num_objectives).to(
                device=target_device
            ),
            returns=self.returns.reshape(-1, self.num_objectives).to(
                device=target_device
            ),
            values=self.values.reshape(-1, self.num_objectives).to(
                device=target_device
            ),
        )

    def __len__(self) -> int:
        return self.step * self.num_envs
