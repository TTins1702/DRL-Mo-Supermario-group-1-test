"""IMPALA-Lite Residual CNN for Multi-Objective PPO (MO-PPO) in mo-supermario-v0.

Phân hệ 2 (Module 2): Kiến trúc mạng ActorCriticResNet kết hợp Feature Extractor
dạng IMPALA-Lite Residual CNN, Actor Head phân loại hành động (Discrete 7),
và Multi-Head Vector Critic Head dự đoán 5 chiều giá trị kỳ vọng:
V(s) = [V_xpos, V_time, V_death, V_coin, V_enemy] in R^5.
"""

from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def layer_init(
    layer: nn.Module,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Module:
    """Khởi tạo trực giao (Orthogonal Initialization) cho weights và hằng số cho bias.

    Args:
        layer: Lớp nn.Conv2d hoặc nn.Linear cần khởi tạo.
        std: Hệ số tỷ lệ chuẩn hóa trực giao (gain). Mặc định sqrt(2) cho activation ReLU.
        bias_const: Giá trị khởi tạo bias (mặc định 0.0).

    Returns:
        Lớp nn.Module đã được khởi tạo tham số.
    """
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.orthogonal_(layer.weight, gain=std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    """Residual Block đơn tầng của IMPALA-Lite:

    Cấu trúc:
        x -> ReLU -> Conv2d(3x3) -> ReLU -> Conv2d(3x3) + x -> out
    Identity skip connection giúp gradient truyền trực tiếp qua các tầng sâu,
    ngăn chặn triệt để hiện tượng triệt tiêu gradient (vanishing gradient)
    khi xử lý chuỗi frame xếp chồng 84x84.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = layer_init(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            std=np.sqrt(2),
        )
        self.conv2 = layer_init(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            std=np.sqrt(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return x + out


class ConvSequence(nn.Module):
    """Một tầng ConvSequence trong kiến trúc IMPALA-Lite:

    Cấu trúc:
        x -> Conv2d(in_c, out_c, 3, pad=1)
          -> MaxPool2d(3, stride=2, pad=1)
          -> ResidualBlock(out_c)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = layer_init(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            std=np.sqrt(2),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.res_block(x)
        return x


class ActorCriticResNet(nn.Module):
    """Mạng Actor-Critic đa mục tiêu (MO-PPO) sử dụng Backbone IMPALA Residual CNN.

    - Đầu vào: Tensor trạng thái s in R^{B x 4 x 84 x 84} (uint8 hoặc float32).
      Tự động chia 255.0 nếu là uint8 để tiết kiệm băng thông và tối ưu bộ nhớ.
    - Trích xuất đặc trưng (Backbone):
      3 tầng ConvSequence với channel widths [32, 64, 64]:
      84x84 -> 42x42 -> 21x21 -> 11x11.
      Trải phẳng (64 * 11 * 11 = 7744) -> ReLU -> Linear(7744 -> 512) -> ReLU -> h in R^512.
    - Policy Head (Actor): Linear(512 -> 7) với orthogonal init gain=0.01.
    - Vector Value Head (Critic): Linear(512 -> 5) với orthogonal init gain=1.0,
      ước lượng vector giá trị kỳ vọng 5 chiều: V(s) in R^5.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_actions: int = 7,
        num_objectives: int = 5,
        features_dim: int = 512,
        channels: Sequence[int] = (32, 64, 64),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.features_dim = features_dim

        # 3 khối ConvSequence
        c0 = channels[0]  # 32
        c1 = channels[1]  # 64
        c2 = channels[2]  # 64

        self.conv_seq1 = ConvSequence(in_channels, c0)
        self.conv_seq2 = ConvSequence(c0, c1)
        self.conv_seq3 = ConvSequence(c1, c2)

        # Trải phẳng: kích thước sau 3 lần MaxPool stride 2: 84 -> 42 -> 21 -> 11
        # c2 * 11 * 11 = 64 * 121 = 7744
        self.flatten_dim = c2 * 11 * 11

        self.fc = layer_init(
            nn.Linear(self.flatten_dim, features_dim),
            std=np.sqrt(2),
        )

        # Actor head (gain=0.01 cho policy head để phân phối xác suất ban đầu gần đều)
        self.actor = layer_init(
            nn.Linear(features_dim, num_actions),
            std=0.01,
            bias_const=0.0,
        )

        # Vector Critic head (gain=1.0 dự đoán kỳ vọng 5 chiều phần thưởng)
        self.critic = layer_init(
            nn.Linear(features_dim, num_objectives),
            std=1.0,
            bias_const=0.0,
        )

    def preprocess_obs(
        self, obs: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """Tiền xử lý observation: chuyển sang float tensor, chuyển device, và chia 255 nếu là uint8."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)

        # Đưa về device của model
        device = next(self.parameters()).device
        if obs.device != device:
            obs = obs.to(device)

        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        elif obs.dtype != torch.float32:
            obs = obs.to(torch.float32)

        return obs

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Trích xuất vector đặc trưng h in R^{B x 512} từ ảnh trạng thái."""
        x = self.conv_seq1(obs)
        x = self.conv_seq2(x)
        x = self.conv_seq3(x)
        x = F.relu(x)
        x = x.flatten(start_dim=1)
        h = F.relu(self.fc(x))
        return h

    def forward(
        self, obs: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lan truyền xuôi tính toán cả Actor Logits và Vector Critic Value:

        Args:
            obs: Observation tensor hoặc ndarray dạng (B, 4, 84, 84).

        Returns:
            logits: Logits phân loại hành động dạng (B, num_actions).
            value_5d: Vector giá trị kỳ vọng 5 chiều dạng (B, num_objectives).
        """
        obs_tensor = self.preprocess_obs(obs)
        h = self.extract_features(obs_tensor)
        logits = self.actor(h)
        value_5d = self.critic(h)
        return logits, value_5d

    def get_value(
        self, obs: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """Dự đoán vector giá trị kỳ vọng 5 chiều V(s) in R^{B x 5}.

        Args:
            obs: Observation tensor hoặc ndarray dạng (B, 4, 84, 84) hoặc (4, 84, 84).

        Returns:
            value_5d: Vector giá trị 5 chiều dạng (B, 5).
        """
        obs_tensor = self.preprocess_obs(obs)
        h = self.extract_features(obs_tensor)
        value_5d = self.critic(h)
        return value_5d

    def get_action_and_value(
        self,
        obs: Union[torch.Tensor, np.ndarray],
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lấy hành động, log_prob, entropy và vector giá trị 5 chiều cho PPO.

        Args:
            obs: Observation tensor hoặc ndarray dạng (B, 4, 84, 84) hoặc (4, 84, 84).
            action: (Tùy chọn) Tensor hành động đã chọn trước đó (được dùng trong vòng cập nhật PPO).
            deterministic: Nếu True, chọn hành động bằng argmax logits (cho chế độ đánh giá / evaluation).

        Returns:
            action: Hành động được lấy mẫu hoặc argmax dạng (B,).
            log_prob: Log xác suất của hành động đã chọn dạng (B,).
            entropy: Độ đo bất định (Entropy) của phân phối chính sách dạng (B,).
            value_5d: Vector giá trị 5 chiều dạng (B, 5).
        """
        obs_tensor = self.preprocess_obs(obs)
        h = self.extract_features(obs_tensor)
        logits = self.actor(h)
        value_5d = self.critic(h)

        dist = Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value_5d
