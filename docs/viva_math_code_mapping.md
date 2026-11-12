# Sổ Tay Vấn Đáp Đồ Án: Đối Chiếu Toàn Diện Công Thức Toán Học và Mã Nguồn (Viva Math-to-Code Mapping)

- **Nhóm thực hiện:** Group 1 (4 thành viên)
- **Học phần:** Học tăng cường (Reinforcement Learning)
- **Đề tài:** Multi-Objective Speedrun PPO (MO-PPO) trên môi trường `mo-supermario-v0`
- **Phiên bản:** 1.0 (Production Release)

---

## 1. Kiến Trúc Tổng Thể & Phân Công 4 Thành Viên

Hệ thống được thiết kế theo nguyên tắc phân hệ hoá mạch lạc, đảm bảo mỗi thành viên phụ trách một khối nghiệp vụ chuyên biệt nhưng cả 4 thành viên đều nắm vững mối liên kết luồng dữ liệu (Data Flow) và các phương trình toán học nền tảng:

```text
               +-------------------------------------------------------+
               |              Phân hệ 1: Môi trường (Member 1)         |
               |  mo-supermario-v0 -> SpeedrunAction(7) -> SkipFrame(4)|
               |  -> GrayScale -> Resize(84x84) -> FrameStack(4)       |
               |  -> Component Scaling s in R^5 -> Anti-Stuck Shaping  |
               +-------------------------------------------------------+
                                           |  s_t in R^{8 x 4 x 84 x 84} (uint8)
                                           |  r_t in R^{8 x 5} (float32)
                                           v
               +-------------------------------------------------------+
               |             Phân hệ 2: Mô Hình & Bộ Đệm (Member 2)     |
               |  IMPALA Residual CNN -> h_t in R^{512}                |
               |  Actor Head -> pi_theta(a|s) (Discrete 7)             |
               |  5-Head Vector Critic -> V_phi(s) in R^5              |
               |  RolloutBuffer (T=512, N=8) -> Vectorized 5D GAE      |
               +-------------------------------------------------------+
                                           |  Batch (s, a, logp, A_5d, R_5d, V_5d)
                                           v
               +-------------------------------------------------------+
               |             Phân hệ 3: Thuật Toán MO-PPO (Member 3)   |
               |  1. Per-Objective Advantage Normalization (mu=0, s=1) |
               |  2. Scalarization: A_scalar = w . A_norm              |
               |  3. Actor Loss: Clipped Surrogate L^CLIP(theta)       |
               |  4. Critic Loss: 5-Head Smooth L1 L^VF(phi)           |
               |  5. Entropy Regularization + Linear LR Decay          |
               +-------------------------------------------------------+
                                           |  Gradient Step & Checkpoints
                                           v
               +-------------------------------------------------------+
               |        Phân hệ 4: Hạ Tầng, Checkpoint & Đánh Giá (Member 4)
               |  - ConfigLoader: YAML & CLI overrides                 |
               |  - CheckpointManager: Colab Drive / Kaggle / Local    |
               |  - Logger: Console Rich + CSV + TensorBoard + WandB   |
               |  - Evaluator & Video Recorder: MP4 + HUD overlay      |
               +-------------------------------------------------------+
```

---

## 2. Bảng Đối Chiếu Nhanh Toán Học $\leftrightarrow$ Mã Nguồn (Core Mapping Table)

Dành cho cả 4 thành viên tra cứu nhanh trong lúc giảng viên kiểm tra vấn đáp từng dòng code:

| STT | Khái niệm & Ký hiệu toán học | Công thức toán học (Formal Equation) | Tệp mã nguồn (Source File) | Dòng code & Hàm / Lớp | Thành viên chính |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **1** | **Khởi tạo trực giao** | $\mathbf{W} \sim \text{Orthogonal}(\text{gain})$ | `src/model/impala_cnn.py` | Dòng 17–36 (`layer_init`) | Thành viên 2 |
| **2** | **Residual Block** | $\mathbf{x}_{l+1} = \mathbf{x}_l + \text{Conv}_2(\text{ReLU}(\text{Conv}_1(\text{ReLU}(\mathbf{x}_l))))$ | `src/model/impala_cnn.py` | Dòng 39–66 (`ResidualBlock`) | Thành viên 2 |
| **3** | **IMPALA Feature Extractor** | $\mathbf{h} = \text{ReLU}(\mathbf{W}_f \cdot \text{Flatten}(\text{ConvSeq}_3(s)) + \mathbf{b}_f)$ | `src/model/impala_cnn.py` | Dòng 174–183 (`extract_features`) | Thành viên 2 |
| **4** | **Policy Head (Actor)** | $\pi_\theta(a \mid s) = \text{Softmax}(\mathbf{W}_\pi \mathbf{h} + \mathbf{b}_\pi), \; \text{gain}=0.01$ | `src/model/impala_cnn.py` | Dòng 140–144, 187–205 (`forward`) | Thành viên 2 |
| **5** | **5-Head Vector Critic** | $\mathbf{V}_\phi(s) = \mathbf{W}_v \mathbf{h} + \mathbf{b}_v \in \mathbb{R}^5, \; \text{gain}=1.0$ | `src/model/impala_cnn.py` | Dòng 147–151, 206–218 (`get_value`) | Thành viên 2 |
| **6** | **Rút gọn Action Speedrun** | $\mathcal{A}_{\text{env}} = \text{Discrete}(256) \longrightarrow \text{Discrete}(7)$ | `src/env/action_wrappers.py` | Dòng 21–64 (`SpeedrunActionWrapper`) | Thành viên 1 |
| **7** | **Co giãn Phần thưởng 5D** | $\mathbf{r}_{\text{scaled}} = \mathbf{r}_{\text{raw}} \odot \mathbf{s}, \; \mathbf{s}=[0.05, 0.05, 0.08, 0.01, 0.02]$ | `src/env/reward_wrappers.py` | Dòng 87–89 (`MultiObjectiveRewardWrapper`) | Thành viên 1 |
| **8** | **Anti-Stuck Shaping** | $r^{(0)} \leftarrow r^{(0)} + \beta_{\text{mile}} \cdot \mathbb{I}_{\Delta x > 0} - \beta_{\text{stuck}} \cdot \mathbb{I}_{k \ge 60}$ | `src/env/reward_wrappers.py` | Dòng 90–105 (`step`) | Thành viên 1 |
| **9** | **5D Temporal Difference** | $\boldsymbol{\delta}_t = \mathbf{r}_t + \gamma (1 - d_t) \mathbf{V}_\phi(s_{t+1}) - \mathbf{V}_\phi(s_t) \in \mathbb{R}^5$ | `src/model/rollout_buffer.py` | Dòng 381–386 (`compute_returns_and_advantages`) | Thành viên 2 |
| **10** | **5D GAE Recursion** | $\hat{\mathbf{A}}_t = \boldsymbol{\delta}_t + (\gamma \lambda) (1 - d_t) \hat{\mathbf{A}}_{t+1} \in \mathbb{R}^5$ | `src/model/rollout_buffer.py` | Dòng 388–394 (`compute_returns_and_advantages`) | Thành viên 2 |
| **11** | **5D Target Returns** | $\mathbf{R}_t = \hat{\mathbf{A}}_t + \mathbf{V}_\phi(s_t) \in \mathbb{R}^5$ | `src/model/rollout_buffer.py` | Dòng 396–397 | Thành viên 2 |
| **12** | **Chuẩn hoá Lợi thế từng chiều** | $\tilde{A}_t^{(i)} = \frac{\hat{A}_t^{(i)} - \mu(\hat{A}^{(i)})}{\sigma(\hat{A}^{(i)}) + 10^{-8}}, \; \forall i \in \{0..4\}$ | `src/agent/mo_ppo_agent.py` | Dòng 170–218 (`normalize_advantages_per_objective`) | Thành viên 3 |
| **13** | **Chiếu Trọng Số (Scalarization)** | $\hat{A}_t^{\mathbf{w}} = \sum_{i=0}^4 w_i \tilde{A}_t^{(i)} = \mathbf{w}^\top \tilde{\mathbf{A}}_t$ | `src/agent/mo_ppo_agent.py` | Dòng 235–265 (`scalarize_advantages`) | Thành viên 3 |
| **14** | **Tỷ số Xác suất (Ratio)** | $r_t(\theta) = \exp(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t))$ | `src/agent/mo_ppo_agent.py` | Dòng 421–423 (`train_step`) | Thành viên 3 |
| **15** | **PPO Clipped Actor Loss** | $\mathcal{L}^{\text{CLIP}}(\theta) = -\frac{1}{M}\sum \min(r_t \hat{A}_t^{\mathbf{w}}, \text{clip}(r_t, 1\pm\epsilon)\hat{A}_t^{\mathbf{w}})$ | `src/agent/mo_ppo_agent.py` | Dòng 440–445 (`train_step`) | Thành viên 3 |
| **16** | **Multi-Head Huber Critic Loss** | $\mathcal{L}^{\text{VF}}(\phi) = \frac{1}{5M}\sum_{t=1}^M \sum_{i=0}^4 \text{SmoothL1}(V_\phi^{(i)}(s_t), R_t^{(i)})$ | `src/agent/mo_ppo_agent.py` | Dòng 446–458 (`train_step`) | Thành viên 3 |
| **17** | **Entropy Regularization** | $\mathcal{L}^{\text{ENT}}(\theta) = \frac{1}{M} \sum_{t=1}^M \mathcal{H}(\pi_\theta(\cdot \mid s_t))$ | `src/agent/mo_ppo_agent.py` | Dòng 459–461 (`train_step`) | Thành viên 3 |
| **18** | **Tổng Loss Tối Ưu Hoá** | $\mathcal{L}^{\text{TOTAL}} = \mathcal{L}^{\text{CLIP}} + c_{\text{vf}} \mathcal{L}^{\text{VF}} - c_{\text{ent}} \mathcal{L}^{\text{ENT}}$ | `src/agent/mo_ppo_agent.py` | Dòng 462–464 (`train_step`) | Thành viên 3 |
| **19** | **Cắt Gradient & Bước Lặp** | $\|\mathbf{g}\|_2 \le g_{\max}=0.5, \; \theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$ | `src/agent/mo_ppo_agent.py` | Dòng 465–471 (`train_step`) | Thành viên 3 |
| **20** | **Giảm Tốc Độ Học Tuyến Tính** | $\alpha_k = \alpha_{\text{end}} + (1 - \frac{k}{K})(\alpha_{\text{start}} - \alpha_{\text{end}})$ | `src/agent/mo_ppo_agent.py` <br> `train.py` | Dòng 503–513 (`set_learning_rate`) <br> Dòng 356–361 | Thành viên 3, 4 |
| **21** | **Ghi Checkpoint Nguyên Tử** | Write to `.tmp` $\rightarrow$ `os.replace` (tránh hỏng file) | `src/utils/checkpoint.py` | Dòng 62–77 (`_atomic_torch_save`) | Thành viên 4 |
| **22** | **Tự Động Nhận Diện Môi Trường** | Google Drive (`/content/drive`) vs Kaggle (`/kaggle/working`) | `src/utils/checkpoint.py` | Dòng 25–45 (`detect_default_checkpoint_dir`) | Thành viên 4 |

---

## 3. Phân Tích Kỹ Thuật Chi Tiết Từng Phân Hệ

### 3.1. Phân hệ 1: Xử lý Môi trường & Vector Phần thưởng Đa Mục tiêu (`src/env/`)

#### 1. Rút gọn không gian hành động (`src/env/action_wrappers.py`)
- **Vấn đề toán học:** Không gian nút bấm NES ban đầu là $\mathcal{A}_{\text{raw}} = \text{Discrete}(256)$ ($2^8$ tổ hợp của 8 nút: Up, Down, Left, Right, A, B, Select, Start). Hầu hết các tổ hợp này là vô nghĩa hoặc phản tác dụng (ví dụ bấm cùng lúc Left và Right, hoặc bấm Start làm tạm dừng game).
- **Giải pháp:** Rút gọn về không gian $\text{Discrete}(7)$ được tuyển chọn tối ưu cho Mario Speedrun:
  1. `0`: `['NOOP']`
  2. `1`: `['right']` (Đi bộ sang phải)
  3. `2`: `['right', 'A']` (Nhảy thường sang phải)
  4. `3`: `['right', 'B']` (Chạy nước rút sang phải)
  5. `4`: `['right', 'A', 'B']` (Chạy nước rút + nhảy xa/cao — hành động chủ lực vượt chướng ngại vật)
  6. `5`: `['A']` (Nhảy thẳng đứng phá gạch, ăn xu trên đầu, hoặc thoát kẹt sát ống cống cao)
  7. `6`: `['left']` (Lùi nhẹ sang trái lấy đà khi Mario bị ép sát chân ống cống)

#### 2. Xử lý khung hình quan sát (`src/env/obs_wrappers.py`)
- **SkipFrame(skip=4):** Lặp lại 1 hành động trên 4 frame NES liên tiếp. Vector phần thưởng 5 chiều được cộng dồn chính xác:
  $$\mathbf{r}_{\text{step}} = \sum_{k=1}^4 \mathbf{r}_{t+k} \in \mathbb{R}^5$$
  Nếu Mario chết hoặc chạm cờ ở frame thứ $k < 4$, ngắt vòng lặp ngay để tránh nhận tín hiệu rác sau khi game đã kết thúc.
- **GrayScaleObservation:** Chuyển ảnh màu RGB $(240, 256, 3)$ sang ảnh xám $(240, 256)$ theo trọng số sinh lý mắt:
  $$Y = 0.299 R + 0.587 G + 0.114 B$$
- **ResizeObservation:** Thu nhỏ về $(84, 84)$ bằng phép nội suy diện tích (`cv2.INTER_AREA`).
- **FrameStack(num_stack=4):** Ghép chồng 4 khung hình xám gần nhất thành tensor $(4, 84, 84)$ kiểu dữ liệu `uint8` ($0 \dots 255$). Việc ghép 4 frame cung cấp đầy đủ thông tin về **vận tốc và gia tốc** (đạo hàm bậc 1 và bậc 2 theo thời gian), giải quyết triệt để tính chất quan sát một phần (POMDP) của bài toán.

#### 3. Co giãn thang đo 5 chiều & Chống kẹt (`src/env/reward_wrappers.py`)
- Vector phần thưởng gốc có sự chênh lệch thang đo cực lớn:
  - Ăn 1 đồng xu: $+100.0$
  - Tử vong: $-25.0$
  - Bước đi: $+1.0 \dots +3.0$
  - Trôi thời gian: $-1.0$
- Nhân với vector hệ số co giãn $\mathbf{s} = [0.05, 0.05, 0.08, 0.01, 0.02]$:
  $$\mathbf{r}_{\text{scaled}} = [0.05 \Delta x, \; 0.05 \Delta t, \; 0.08 r_{\text{death}}, \; 0.01 r_{\text{coin}}, \; 0.02 r_{\text{enemy}}]^\top$$
  Tất cả các chiều đều rơi vào miền giá trị xấp xỉ $[-2.0, +2.0]$, giúp gradient của 5 đầu Critic không bị triệt tiêu lẫn nhau.
- **Anti-Stuck Shaping:** Nếu Mario đứng im trước ống cống quá $K_{\text{stuck}} = 60$ bước (tương đương 240 frame game), phạt nhẹ $-\beta_{\text{stuck}} = -0.1$ mỗi bước trên chiều $x$, ép tác tử phải thử nhảy hoặc lùi lại lấy đà.

---

### 3.2. Phân hệ 2: Mạng Nơ-ron IMPALA-Lite & Bộ Đệm 5D GAE (`src/model/`)

#### 1. Khởi tạo trực giao (`layer_init`)
- Ma trận trọng số $\mathbf{W}$ được khởi tạo trực giao ($\mathbf{W}^\top \mathbf{W} = \mathbf{I}$), giữ nguyên phương sai của tín hiệu kích hoạt khi lan truyền qua các tầng mạng sâu.
- Tham số `gain`:
  - Các lớp Conv2d và Linear ẩn: $\text{gain} = \sqrt{2} \approx 1.414$ (phù hợp với hàm kích hoạt phi tuyến ReLU).
  - Actor Head: $\text{gain} = 0.01$ (để logits ban đầu gần bằng $0$, phân phối xác suất hành động gần như đều $1/7$, kích thích thăm dò tối đa ở giai đoạn đầu).
  - Vector Critic Head: $\text{gain} = 1.0$.

#### 2. Khối trích xuất đặc trưng IMPALA-Lite Residual CNN
- 3 tầng `ConvSequence` với số kênh lần lượt là `[32, 64, 64]`.
- Mỗi tầng gồm: `Conv2d(kernel=3, stride=1, padding=1)` $\rightarrow$ `MaxPool2d(kernel=3, stride=2, padding=1)` $\rightarrow$ `ResidualBlock`.
- Trong `ResidualBlock`, đường tắt đồng nhất (Identity Skip Connection) $\mathbf{x} + \mathcal{F}(\mathbf{x})$ cho phép gradient từ tầng loss truyền thẳng ngược về các tầng tích chập ban đầu mà không bị suy giảm theo cấp số nhân.
- Sau 3 lần giảm chiều không gian: $84 \times 84 \rightarrow 42 \times 42 \rightarrow 21 \times 21 \rightarrow 11 \times 11$.
- Trải phẳng thành vector: $64 \times 11 \times 11 = 7744 \rightarrow \text{Linear}(7744, 512) \rightarrow \text{ReLU} \rightarrow \mathbf{h}_t \in \mathbb{R}^{512}$.

#### 3. Thuật toán 5D GAE trong `MultiObjectiveRolloutBuffer`
- Trạng thái được lưu trong bộ đệm dạng `torch.uint8` $(T=512, N=8, 4, 84, 84)$, chiếm vỏn vẹn ~60 MB RAM thay vì 240 MB nếu dùng `float32`.
- Với $t = T-1, \dots, 0$, đệ quy ngược đồng thời trên cả 5 chiều mục tiêu:
  $$\boldsymbol{\delta}_t = \mathbf{r}_t + \gamma (1 - d_t) \mathbf{V}_\phi(s_{t+1}) - \mathbf{V}_\phi(s_t) \in \mathbb{R}^{N \times 5}$$
  $$\hat{\mathbf{A}}_t = \boldsymbol{\delta}_t + (\gamma \lambda) (1 - d_t) \hat{\mathbf{A}}_{t+1} \in \mathbb{R}^{N \times 5}$$
  $$\mathbf{R}_t = \hat{\mathbf{A}}_t + \mathbf{V}_\phi(s_t) \in \mathbb{R}^{N \times 5}$$

---

### 3.3. Phân hệ 3: Thuật Toán MO-PPO & Hàm Mất Mát (`src/agent/`)

#### 1. Chuẩn hoá Lợi thế Độc lập từng Mục tiêu (Per-Objective Advantage Normalization)
- **Điểm sáng tạo cốt lõi của đề tài:** Không chuẩn hoá sau khi đã cộng gộp, mà chuẩn hoá độc lập từng chiều trên toàn bộ batch $4096$ mẫu:
  $$\mu_i = \frac{1}{B} \sum_{t=1}^B \hat{A}_t^{(i)}, \quad \sigma_i = \sqrt{\frac{1}{B} \sum_{t=1}^B (\hat{A}_t^{(i)} - \mu_i)^2}$$
  $$\tilde{A}_t^{(i)} = \frac{\hat{A}_t^{(i)} - \mu_i}{\sigma_i + 10^{-8}}, \quad \forall i \in \{0, 1, 2, 3, 4\}$$
  Sau phép chuẩn hoá này, cả 5 mục tiêu đều có trung bình bằng $0$ và phương sai bằng $1$.

#### 2. Chiếu Trọng số Ưu tiên (Scalarization)
- Tính tích vô hướng với vector sở thích người dùng $\mathbf{w} = [w_0, w_1, w_2, w_3, w_4]$:
  $$\hat{A}_t^{\mathbf{w}} = \sum_{i=0}^4 w_i \tilde{A}_t^{(i)}$$
- Nhờ chuẩn hoá trước, trọng số $w_i$ phản ánh chính xác 100% mức độ ưu tiên tương đối thực sự của người thiết kế chính sách mà không bị ảnh hưởng bởi đơn vị vật lý của từng chiều phần thưởng.

#### 3. Hàm mất mát Actor (Clipped Surrogate Policy Loss)
- Tỷ số xác suất:
  $$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} = \exp\left( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t) \right)$$
- Hàm mất mát cắt ngưỡng PPO:
  $$\mathcal{L}^{\text{CLIP}}(\theta) = -\frac{1}{M} \sum_{t=1}^M \min\left( r_t(\theta) \hat{A}_t^{\mathbf{w}}, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t^{\mathbf{w}} \right)$$
  với $\epsilon = 0.2$. Cắt ngưỡng giúp chính sách mới $\pi_\theta$ không lệch quá xa chính sách cũ $\pi_{\theta_{\text{old}}}$, bảo đảm bước cập nhật luôn ổn định.

#### 4. Hàm mất mát Vector Critic (Multi-Head Smooth L1 / Huber Loss)
- Dự đoán giá trị của Critic là vector 5 chiều: $\mathbf{V}_\phi(s_t) = [V^{(0)}, V^{(1)}, V^{(2)}, V^{(3)}, V^{(4)}] \in \mathbb{R}^5$.
- Áp dụng Smooth L1 Loss trên từng chiều và lấy trung bình:
  $$\mathcal{L}^{\text{VF}}(\phi) = \frac{1}{5M} \sum_{t=1}^M \sum_{i=0}^4 \text{SmoothL1Loss}\left( V_\phi^{(i)}(s_t), R_t^{(i)} \right)$$
  Smooth L1 hoạt động như hàm bậc hai $0.5 x^2$ khi lỗi nhỏ ($|x| < 1$) và như hàm tuyến tính $|x| - 0.5$ khi lỗi lớn ($|x| \ge 1$), chống lại hiện tượng bùng nổ gradient khi có sự biến động đột ngột (ví dụ Mario bất ngờ rơi xuống vực hoặc chạm cờ).

#### 5. Hàm mất mát Tổng hợp
$$\mathcal{L}^{\text{TOTAL}} = \mathcal{L}^{\text{CLIP}}(\theta) + c_{\text{vf}} \mathcal{L}^{\text{VF}}(\phi) - c_{\text{ent}} \mathcal{L}^{\text{ENT}}(\theta)$$
với $c_{\text{vf}} = 0.5$ và $c_{\text{ent}} = 0.01$.

---

### 3.4. Phân hệ 4: Hạ Tầng, Checkpoint & Đánh Giá (`src/utils/`, `train.py`, `evaluate.py`)

#### 1. Cơ chế Quản lý Checkpoint Nguyên Tử (`CheckpointManager`)
- **Tự động nhận diện môi trường:**
  - Nếu tồn tại `/content/drive/MyDrive` (Google Colab đã mount Drive): Checkpoint được ghi thẳng vào Google Drive `/content/drive/MyDrive/DRL_Mario_Checkpoints/<exp_name>/`.
  - Nếu tồn tại `/kaggle/working`: Checkpoint lưu tại `/kaggle/working/checkpoints/<exp_name>/`.
  - Môi trường cục bộ: lưu tại `./checkpoints/<exp_name>/`.
- **Atomic Save (Ghi nguyên tử):**
  Dữ liệu PyTorch được lưu ra tệp tạm `.tmp` trong cùng thư mục, sau đó dùng lệnh nguyên tử cấp hệ điều hành `os.replace(temp_path, final_path)`. Điều này loại bỏ hoàn toàn nguy cơ file checkpoint bị corrupt nếu phiên huấn luyện Colab/Kaggle bị ngắt kết nối đột ngột ngay giữa lúc đang ghi đĩa.
- **Phân loại Checkpoint:**
  - `latest.pt`: Lưu mỗi 25 vòng cập nhật, dùng để khôi phục (Resume) tự động.
  - `best_winrate.pt`: Lưu khi Win Rate qua màn 1-1 đạt kỷ lục mới.
  - `best_speedrun.pt`: Lưu khi thời gian hoàn thành màn 1-1 ngắn nhất.

#### 2. Chiến lược Chuyển Giao Trọng Số Đua Top 2 Điểm (Transfer Learning)
- Khi giảng viên công bố đề tài 2 điểm mở rộng:
  - **Kịch bản Tốc độ cao nhất (Speedrun):** Dùng file `configs/bonus_speedrun.yaml` với $\mathbf{w} = [0.40, 0.40, 0.15, 0.02, 0.03]$.
  - **Kịch bản Điểm số cao nhất (High Score):** Dùng file `configs/bonus_highscore.yaml` với $\mathbf{w} = [0.30, 0.05, 0.15, 0.25, 0.25]$.
  - **Lệnh thực thi:**
    ```bash
    python train.py --config configs/bonus_speedrun.yaml --resume checkpoints/base_8pt_clear/best_winrate.pt --reset-lr
    ```
    Mạng nơ-ron không phải học lại từ đầu cách nhận biết ống cống và quái vật (Backbone và Critic 5 chiều đã học sẵn giá trị của từng thành phần), chỉ cần vài chục ngàn bước fine-tune với learning rate nhỏ là thích nghi ngay với mục tiêu mới!

---

## 4. Bộ Câu Hỏi Vấn Đáp Chuyên Sâu Của Giảng Viên & Gợi Ý Trả Lời Mẫu

### Câu 1: "Tại sao nhóm không dùng Scalarization ngay từ đầu (cộng gộp 5 reward lại thành 1 số thực) rồi train PPO thông thường, mà phải làm 5-Head Critic phức tạp?"
- **Trả lời mẫu (Thành viên 1 & 3):**
  "Thưa thầy/cô, việc gộp phần thưởng thô trước khi đưa vào Critic (Naive Scalarization) gặp hai vấn đề chí mạng trong Reinforcement Learning:
  1. **Lệch pha thang đo và triệt tiêu gradient:** Đồng xu ăn được $+100$, trong khi bước tiến về phía trước chỉ $+1$ đến $+3$. Nếu cộng gộp, hàm giá trị $V(s)$ sẽ bị thống trị hoàn toàn bởi tín hiệu đồng xu, khiến Mario chỉ chăm chăm tìm chỗ có xu hoặc đứng im đợi xu mà quên nhiệm vụ quan trọng nhất là đi về đích.
  2. **Mất khả năng tái sử dụng mô hình (Transfer Learning):** Với Critic 1 đầu, khi muốn đổi chiến lược từ 'Qua màn an toàn' sang 'Speedrun' hoặc 'Ăn điểm cao', toàn bộ mạng Critic phải bị đập đi học lại từ đầu. Ngược lại, với kiến trúc 5-Head Vector Critic $\mathbf{V}_\phi(s) \in \mathbb{R}^5$ của nhóm, Critic học được kỳ vọng giá trị khách quan của từng mục tiêu riêng biệt. Khi đổi trọng số ưu tiên $\mathbf{w}$, nhóm chỉ cần thay đổi phép chiếu vô hướng trong hàm tính Advantage mà không làm hỏng các biểu diễn đặc trưng đã học."

### Câu 2: "Tại sao lại chuẩn hoá Advantage trên từng mục tiêu rồi mới nhân trọng số $\mathbf{w}$, thay vì nhân trọng số $\mathbf{w}$ trước rồi mới chuẩn hoá Advantage tổng?"
- **Trả lời mẫu (Thành viên 3):**
  "Thưa thầy/cô, đây chính là điểm mấu chốt toán học của phân hệ Agent. Giả sử Advantage của chiều tiền thưởng xu có phương sai là $\sigma_{\text{coin}} = 20.0$, còn Advantage của chiều tiến độ di chuyển có phương sai là $\sigma_{x} = 0.5$.
  - Nếu ta nhân trọng số trước: $A = w_0 A_x + w_3 A_{\text{coin}}$, thành phần đồng xu sẽ áp đảo hoàn toàn tín hiệu di chuyển bất kể ta đặt $w_0$ lớn thế nào.
  - Khi nhóm thực hiện **Per-Objective Advantage Normalization**:
    $$\tilde{A}^{(i)} = \frac{A^{(i)} - \mu_i}{\sigma_i + \epsilon}$$
    tất cả 5 chiều đều có cùng thang đo phương sai bằng $1.0$. Khi đó, biểu thức $\hat{A}^{\mathbf{w}} = \sum w_i \tilde{A}^{(i)}$ đảm bảo trọng số $w_i$ phản ánh chính xác $100\%$ tỷ lệ ưu tiên mong muốn của người lập trình!"

### Câu 3: "SkipFrame(4) có ý nghĩa gì và tại sao trong hàm `step` của `SkipFrame` lại phải cộng dồn vector phần thưởng 5 chiều?"
- **Trả lời mẫu (Thành viên 1):**
  "Thưa thầy/cô:
  1. Trong game NES, tốc độ khung hình là 60 FPS. Một hành động nhảy của Mario cần ít nhất 10–15 frame để có hiệu lực. Nếu agent ra quyết định ở từng frame một, chân trời quyết định (Time Horizon) sẽ dài gấp 4 lần, làm thuật toán RL rất khó gán trách nhiệm tín dụng (Credit Assignment) và tốc độ huấn luyện giảm 4 lần.
  2. Việc cộng dồn vector phần thưởng $\mathbf{r}_{\text{step}} = \sum_{k=1}^4 \mathbf{r}_{t+k}$ trên cả 5 chiều đảm bảo tính nhất quán của phương trình Bellman: tổng chiết khấu phần thưởng tích luỹ qua toàn bộ episode không hề bị mất mát bất kỳ đồng xu hay điểm thưởng nào xảy ra ở các frame trung gian."

### Câu 4: "Tại sao nhóm chọn kiến trúc IMPALA Residual CNN thay vì kiến trúc Nature CNN cổ điển của Mnih et al. (2015)?"
- **Trả lời mẫu (Thành viên 2):**
  "Thưa thầy/cô, mạng Nature CNN chỉ có 3 tầng tích chập không có kết nối tắt (Skip Connection). Khi ta xếp chồng 4 khung hình $(4, 84, 84)$, các chướng ngại vật nhỏ như hạt đạn của cây ăn thịt hay mép vực sâu rất dễ bị nhòe đặc trưng qua các tầng Pooling.
  IMPALA Residual CNN bổ sung khối Residual Block $\mathbf{x} + \mathcal{F}(\mathbf{x})$ cho phép gradient truyền ngược thông suốt, giúp mạng học được các đặc trưng không gian đa tỷ lệ sắc nét hơn nhiều, đồng thời ngăn chặn triệt để hiện tượng suy biến gradient khi tăng độ sâu mạng."

### Câu 5: "Công thức toán học của 5D GAE được cài đặt như thế nào và tại sao nó lại giảm phương sai ước lượng?"
- **Trả lời mẫu (Thành viên 2):**
  "Thưa thầy/cô, GAE (Generalized Advantage Estimator) là một tổ hợp lồi có trọng số mũ của các ước lượng TD k-bước.
  Phương trình đệ quy ngược được cài đặt vector hoá tại dòng 381–394 file `src/model/rollout_buffer.py`:
  $$\boldsymbol{\delta}_t = \mathbf{r}_t + \gamma (1 - d_t) \mathbf{V}(s_{t+1}) - \mathbf{V}(s_t)$$
  $$\hat{\mathbf{A}}_t = \boldsymbol{\delta}_t + (\gamma \lambda) (1 - d_t) \hat{\mathbf{A}}_{t+1}$$
  Khi $\lambda = 0$, $\hat{\mathbf{A}}_t = \boldsymbol{\delta}_t$ (độ lệch TD-1 bước, độ chệch cao, phương sai thấp). Khi $\lambda = 1$, $\hat{\mathbf{A}}_t$ tương đương Monte Carlo (không chệch, nhưng phương sai cực lớn). Với $\lambda = 0.95$, thuật toán cân bằng hoàn hảo giữa độ chệch và phương sai cho cả 5 chiều giá trị!"

### Câu 6: "Tại sao nhóm lại dùng hàm Smooth L1 (Huber Loss) cho Critic thay vì MSE Loss thông thường?"
- **Trả lời mẫu (Thành viên 3):**
  "Thưa thầy/cô, trong Super Mario Bros, phần thưởng tử vong là $-25.0$ và thưởng qua màn là rất lớn. Nếu sử dụng MSE Loss:
  $$\mathcal{L}_{\text{MSE}} = (V(s) - R)^2$$
  khi Mario bất ngờ chết, khoảng cách sai số $|V - R| \approx 20$, hàm bình phương sẽ tạo ra đạo hàm bậc một lên đến $2 \times 20 = 40$, dẫn đến hiện tượng bùng nổ gradient phá hỏng toàn bộ trọng số feature extractor.
  Hàm Smooth L1 (Huber Loss) chuyển sang tuyến tính với độ dốc cố định bằng $1.0$ khi sai số vượt quá $1.0$, giữ cho gradient của Critic luôn ổn định!"

### Câu 7: "Hệ thống làm thế nào để đảm bảo quá trình huấn luyện trên Colab không bị mất khi bị ngắt kết nối đột ngột sau 12 giờ?"
- **Trả lời mẫu (Thành viên 4):**
  "Thưa thầy/cô, Phân hệ 4 cài đặt `CheckpointManager` với 3 cơ chế bảo vệ:
  1. **Tự động gắn kết Google Drive:** Kiểm tra `/content/drive/MyDrive`, tự động tạo thư mục trên Drive của nhóm.
  2. **Ghi đĩa nguyên tử (Atomic Save):** Dùng `torch.save` vào file `.tmp` rồi gọi `os.replace`. Nếu Colab tắt ngang khi đang ghi đĩa, file `latest.pt` cũ vẫn nguyên vẹn 100%, không bao giờ bị dính lỗi corrupt zip file.
  3. **Lưu đầy đủ trạng thái:** Checkpoint chứa cả `model_state_dict`, `optimizer_state_dict`, `update_idx`, `global_step` và `best_metrics`. Khi chạy lệnh `python train.py --resume auto`, mô hình tiếp tục vòng lặp chính xác từ bước trước đó."

---

*Tài liệu này được biên soạn độc quyền cho Group 1. Cả 4 thành viên hãy cùng đọc kỹ và đối chiếu trực tiếp với từng dòng code trong repository trước ngày vấn đáp.*
