"""Multi-Objective Proximal Policy Optimization (MO-PPO) Agent.

This module implements Phân hệ 3 (Module 3) of the Super Mario MO-PPO system:
- 5-Dimensional Advantage Normalization (Per-Objective Normalization).
- Objective Scalarization using weight preference vector w in R^5.
- PPO Clipped Surrogate Actor Loss.
- Multi-Head 5D Vector Critic Loss with SmoothL1Loss (Huber) averaged across 5 dimensions.
- Policy Entropy Regularization.
- Mini-batch policy updates with gradient norm clipping.
- Linear learning rate annealing support.
- Checkpoint persistence (save/load).

Mathematical Specifications (Docs Section 6.3):
1. Per-Objective Advantage Normalization:
   \\tilde{A}_t^{(i)} = \\frac{\\hat{A}_t^{(i)} - \\text{mean}(\\hat{A}^{(i)})}{\\text{std}(\\hat{A}^{(i)}) + 10^{-8}}, \\quad i \\in \\{0,1,2,3,4\\}

2. Scalarized Advantage with Weight Vector \\mathbf{w}:
   \\hat{A}_t^{\\mathbf{w}} = \\sum_{i=0}^4 w_i \\tilde{A}_t^{(i)}

3. Probability Ratio:
   r_t(\\theta) = \\exp(\\log \\pi_\\theta(a_t|s_t) - \\log \\pi_{\\theta_{\\text{old}}}(a_t|s_t))

4. Clipped Surrogate Policy Loss:
   \\mathcal{L}^{\\text{CLIP}}(\\theta) = -\\frac{1}{M} \\sum_{t=1}^M \\min(r_t(\\theta) \\hat{A}_t^{\\mathbf{w}}, \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}_t^{\\mathbf{w}})

5. 5D Vector Critic Loss (Smooth L1 / Huber):
   \\mathcal{L}^{\\text{VF}}(\\phi) = \\frac{1}{5M} \\sum_{t=1}^M \\sum_{i=0}^4 \\text{SmoothL1Loss}(V_\\phi^{(i)}(s_t), R_t^{(i)})

6. Entropy Regularization:
   \\mathcal{L}^{\\text{ENT}}(\\theta) = \\frac{1}{M} \\sum_{t=1}^M \\mathcal{H}(\\pi_\\theta(\\cdot|s_t))

7. Total Loss:
   \\mathcal{L}^{\\text{TOTAL}} = \\mathcal{L}^{\\text{CLIP}}(\\theta) + c_{\\text{vf}} \\mathcal{L}^{\\text{VF}}(\\phi) - c_{\\text{ent}} \\mathcal{L}^{\\text{ENT}}(\\theta)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _unpack_batch(
    batch: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Unpack a training batch yielded by the rollout buffer.

    Supports dict, tuple/list, namedtuple, or objects with attributes.

    Returns:
        (obs, actions, logprobs, advantages, returns, values)
    """
    if isinstance(batch, dict):
        obs = batch.get("obs", batch.get("observations"))
        actions = batch["actions"]
        logprobs = batch.get("logprobs", batch.get("log_probs"))
        advantages = batch["advantages"]
        returns = batch["returns"]
        values = batch.get("values", None)
    elif hasattr(batch, "obs") and hasattr(batch, "actions"):
        obs = batch.obs
        actions = batch.actions
        logprobs = getattr(batch, "logprobs", getattr(batch, "log_probs", None))
        advantages = batch.advantages
        returns = batch.returns
        values = getattr(batch, "values", None)
    elif isinstance(batch, (tuple, list)):
        if len(batch) < 5:
            raise ValueError(f"Batch tuple must contain at least 5 elements, got {len(batch)}")
        obs = batch[0]
        actions = batch[1]
        logprobs = batch[2]
        advantages = batch[3]
        returns = batch[4]
        values = batch[5] if len(batch) > 5 else None
    else:
        raise TypeError(f"Unsupported batch type from rollout buffer: {type(batch)}")

    return obs, actions, logprobs, advantages, returns, values


class MOPPOAgent:
    """Multi-Objective Proximal Policy Optimization (MO-PPO) Agent.

    Coordinates policy updates with:
    - 5-Head Vector Critic learning all 5 objective value functions simultaneously.
    - Per-objective advantage normalization avoiding cross-objective scale distortion.
    - Preference vector scalarization for flexible speedrun/clear/highscore balancing.
    - Orthogonally initialized network optimization with gradient clipping.
    """

    def __init__(
        self,
        network: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[Dict[str, Any]] = None,
        lr: float = 2.5e-4,
        eps: float = 1e-5,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 256,
        norm_adv: bool = True,
        clip_vloss: bool = False,
        target_kl: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Initialize the MO-PPO agent.

        Args:
            network: Actor-Critic network (e.g. ActorCriticResNet) outputting policy logits and 5D value.
            optimizer: Optimizer instance (default: Adam with lr and eps=1e-5).
            config: Optional dictionary containing hyperparameter overrides.
            lr: Base learning rate.
            eps: Epsilon parameter for Adam optimizer.
            clip_coef: PPO surrogate clipping threshold epsilon (default 0.2).
            ent_coef: Entropy bonus coefficient c_ent (default 0.01).
            vf_coef: Value function loss coefficient c_vf (default 0.5).
            max_grad_norm: Maximum gradient norm for clipping (default 0.5).
            update_epochs: Number of gradient epochs per update (default 4).
            batch_size: Mini-batch size for training (default 256).
            norm_adv: Whether to normalize advantages per objective.
            clip_vloss: Whether to use clipped value loss.
            target_kl: Optional target KL divergence threshold for early stopping.
            device: Computing device ('cuda' or 'cpu'). Inferred if None.
        """
        self.config = dict(config) if config is not None else {}

        # Determine computing device
        if device is not None:
            self.device = torch.device(device)
        else:
            try:
                self.device = next(network.parameters()).device
            except (StopIteration, AttributeError):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = network.to(self.device)

        # Hyperparameters (config dict takes precedence over default arguments)
        self.lr = float(self.config.get("lr", self.config.get("learning_rate", lr)))
        self.eps = float(self.config.get("eps", self.config.get("adam_eps", eps)))
        self.clip_coef = float(self.config.get("clip_coef", clip_coef))
        self.ent_coef = float(self.config.get("ent_coef", ent_coef))
        self.vf_coef = float(self.config.get("vf_coef", vf_coef))
        self.max_grad_norm = float(self.config.get("max_grad_norm", max_grad_norm))
        self.update_epochs = int(self.config.get("update_epochs", update_epochs))
        self.batch_size = int(self.config.get("batch_size", batch_size))
        self.norm_adv = bool(self.config.get("norm_adv", norm_adv))
        self.clip_vloss = bool(self.config.get("clip_vloss", clip_vloss))
        self.target_kl = self.config.get("target_kl", target_kl)
        if self.target_kl is not None:
            self.target_kl = float(self.target_kl)

        # Initialize optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.lr,
                eps=self.eps,
            )

    def normalize_advantages_per_objective(
        self,
        advantages_5d: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Normalize advantage estimates independently per objective dimension.

        Formula:
            \\tilde{A}_t^{(i)} = \\frac{\\hat{A}_t^{(i)} - \\mu^{(i)}}{\\sigma^{(i)} + 10^{-8}}, \\quad \\forall i \\in \\{0, 1, 2, 3, 4\\}

        Args:
            advantages_5d: Tensor of shape (..., 5) with advantages for 5 objectives.
            mask: Optional mask tensor of active/valid steps.
            eps: Small epsilon to avoid division by zero (default 1e-8).

        Returns:
            Normalized advantage tensor of identical shape.
        """
        if not isinstance(advantages_5d, torch.Tensor):
            advantages_5d = torch.as_tensor(advantages_5d, device=self.device, dtype=torch.float32)

        orig_shape = advantages_5d.shape
        if advantages_5d.ndim == 1:
            advantages_5d = advantages_5d.unsqueeze(0)

        # Reduce across all leading batch dimensions (all except the last 5D objective dimension)
        reduce_dims = tuple(range(advantages_5d.ndim - 1))

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, device=advantages_5d.device, dtype=torch.float32)
            else:
                mask = mask.to(device=advantages_5d.device, dtype=torch.float32)

            while mask.ndim < advantages_5d.ndim:
                mask = mask.unsqueeze(-1)

            mask_sum = mask.sum(dim=reduce_dims, keepdim=True).clamp(min=1.0)
            mean = (advantages_5d * mask).sum(dim=reduce_dims, keepdim=True) / mask_sum
            var = (((advantages_5d - mean) ** 2) * mask).sum(dim=reduce_dims, keepdim=True) / mask_sum
            std = torch.sqrt(var)
            normalized = (advantages_5d - mean) / (std + eps)
            normalized = normalized * mask
        else:
            mean = advantages_5d.mean(dim=reduce_dims, keepdim=True)
            std = advantages_5d.std(dim=reduce_dims, keepdim=True, unbiased=False)
            normalized = (advantages_5d - mean) / (std + eps)

        if orig_shape != normalized.shape and len(orig_shape) == 1:
            normalized = normalized.squeeze(0)

        return normalized

    def scalarize_advantages(
        self,
        normalized_advantages_5d: torch.Tensor,
        weights: Union[torch.Tensor, Sequence[float]],
        normalize_weights: bool = False,
    ) -> torch.Tensor:
        """Scalarize normalized 5D advantages into a single scalar advantage.

        Formula:
            \\hat{A}_t^{\\mathbf{w}} = \\sum_{i=0}^4 w_i \\tilde{A}_t^{(i)}

        Args:
            normalized_advantages_5d: Tensor of shape (..., 5) with normalized advantages.
            weights: Objective preference weights [w0, w1, w2, w3, w4].
            normalize_weights: If True, normalizes weights so they sum to 1.0.

        Returns:
            Scalarized advantage tensor of shape (...).
        """
        if not isinstance(normalized_advantages_5d, torch.Tensor):
            normalized_advantages_5d = torch.as_tensor(
                normalized_advantages_5d, device=self.device, dtype=torch.float32
            )

        if not isinstance(weights, torch.Tensor):
            w = torch.as_tensor(weights, dtype=torch.float32, device=normalized_advantages_5d.device)
        else:
            w = weights.to(dtype=torch.float32, device=normalized_advantages_5d.device)

        if w.numel() != 5:
            raise ValueError(f"Expected 5 weights for 5 objectives, got {w.numel()}")

        w = w.view(-1)
        if normalize_weights:
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum

        # Ensure w aligns with last dimension of normalized_advantages_5d
        while w.ndim < normalized_advantages_5d.ndim:
            w = w.unsqueeze(0)

        scalar_adv = torch.sum(normalized_advantages_5d * w, dim=-1)
        return scalar_adv

    @staticmethod
    def compute_explained_variance_5d(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> List[float]:
        """Compute explained variance for each of the 5 objective dimensions.

        Formula:
            \\text{EV}^{(i)} = 1 - \\frac{\\text{Var}(y_{\\text{true}}^{(i)} - y_{\\text{pred}}^{(i)})}{\\text{Var}(y_{\\text{true}}^{(i)}) + 10^{-8}}

        Returns:
            List of 5 floats representing explained variance per objective.
        """
        with torch.no_grad():
            if not isinstance(y_pred, torch.Tensor):
                y_pred = torch.as_tensor(y_pred, dtype=torch.float32)
            if not isinstance(y_true, torch.Tensor):
                y_true = torch.as_tensor(y_true, dtype=torch.float32)

            y_pred = y_pred.view(-1, 5)
            y_true = y_true.view(-1, 5)

            var_y = torch.var(y_true, dim=0, unbiased=False)
            var_diff = torch.var(y_true - y_pred, dim=0, unbiased=False)

            ev = torch.zeros(5, dtype=torch.float32, device=y_true.device)
            valid_mask = var_y > 1e-8
            ev[valid_mask] = 1.0 - var_diff[valid_mask] / var_y[valid_mask]

            # When true target has near-zero variance, EV is 1 if diff is also near-zero, else 0
            exact_zero = (~valid_mask) & (var_diff <= 1e-8)
            ev[exact_zero] = 1.0

            return ev.detach().cpu().tolist()

    def train_step(
        self,
        rollout_buffer: Any,
        weights: Union[torch.Tensor, Sequence[float]],
        clip_coef: Optional[float] = None,
        ent_coef: Optional[float] = None,
        vf_coef: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        update_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        target_kl: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute one complete PPO update using mini-batches from rollout_buffer.

        Iterates through update_epochs, updates network parameters using the MO-PPO
        clipped surrogate objective and 5D vector critic Huber loss, and returns
        training diagnostics.

        Args:
            rollout_buffer: MultiObjectiveRolloutBuffer containing collected rollouts.
            weights: Objective preference weights [w0, w1, w2, w3, w4].
            clip_coef: Clipping parameter epsilon (defaults to self.clip_coef = 0.2).
            ent_coef: Entropy coefficient (defaults to self.ent_coef = 0.01).
            vf_coef: Value loss coefficient (defaults to self.vf_coef = 0.5).
            max_grad_norm: Maximum gradient norm for clipping (defaults to self.max_grad_norm = 0.5).
            update_epochs: Number of gradient epochs (defaults to self.update_epochs = 4).
            batch_size: Mini-batch size (defaults to self.batch_size = 256).
            target_kl: Optional KL divergence threshold for early stopping.

        Returns:
            Dictionary with training metrics:
            - actor_loss
            - critic_loss
            - entropy
            - approx_kl
            - clip_fraction
            - explained_variance_5d
            - total_loss
            - learning_rate
        """
        # Resolve hyperparameters
        clip_coef = self.clip_coef if clip_coef is None else float(clip_coef)
        ent_coef = self.ent_coef if ent_coef is None else float(ent_coef)
        vf_coef = self.vf_coef if vf_coef is None else float(vf_coef)
        max_grad_norm = self.max_grad_norm if max_grad_norm is None else float(max_grad_norm)
        update_epochs = self.update_epochs if update_epochs is None else int(update_epochs)
        batch_size = self.batch_size if batch_size is None else int(batch_size)
        target_kl = self.target_kl if target_kl is None else target_kl

        # Metric accumulators
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        total_loss_accum = 0.0
        num_updates = 0
        ev_history: List[List[float]] = []
        early_stopped = False

        self.network.train()

        for epoch in range(update_epochs):
            if early_stopped:
                break

            for batch in rollout_buffer.get_batches(batch_size=batch_size):
                obs, actions, old_logprobs, advantages, returns, old_values = _unpack_batch(batch)

                # Move batch tensors to computing device
                if not isinstance(obs, torch.Tensor):
                    obs = torch.as_tensor(obs, device=self.device)
                else:
                    obs = obs.to(self.device)

                if not isinstance(actions, torch.Tensor):
                    actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
                else:
                    actions = actions.to(self.device, dtype=torch.long)

                if not isinstance(old_logprobs, torch.Tensor):
                    old_logprobs = torch.as_tensor(old_logprobs, device=self.device, dtype=torch.float32)
                else:
                    old_logprobs = old_logprobs.detach().to(self.device, dtype=torch.float32)

                if not isinstance(advantages, torch.Tensor):
                    advantages = torch.as_tensor(advantages, device=self.device, dtype=torch.float32)
                else:
                    advantages = advantages.detach().to(self.device, dtype=torch.float32)

                if not isinstance(returns, torch.Tensor):
                    returns = torch.as_tensor(returns, device=self.device, dtype=torch.float32)
                else:
                    returns = returns.detach().to(self.device, dtype=torch.float32)

                if old_values is not None:
                    if not isinstance(old_values, torch.Tensor):
                        old_values = torch.as_tensor(old_values, device=self.device, dtype=torch.float32)
                    else:
                        old_values = old_values.detach().to(self.device, dtype=torch.float32)

                # 1. Per-Objective Advantage Normalization & Scalarization
                if self.norm_adv:
                    norm_adv_5d = self.normalize_advantages_per_objective(advantages)
                else:
                    norm_adv_5d = advantages

                scalar_adv = self.scalarize_advantages(norm_adv_5d, weights)

                # 2. Network Forward Evaluation
                _, new_logprob, entropy, new_val_5d = self.network.get_action_and_value(obs, actions)

                # 3. Policy Ratio & KL Diagnostics
                logratio = new_logprob - old_logprobs
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    # Schulman approximation: ((ratio - 1) - logratio)
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > clip_coef).float().mean()

                # Check early stopping condition
                if target_kl is not None and approx_kl.item() > target_kl:
                    early_stopped = True
                    logger.info(
                        "Target KL reached (%.4f > %.4f). Early stopping at epoch %d.",
                        approx_kl.item(),
                        target_kl,
                        epoch,
                    )
                    break

                # 4. Clipped Surrogate Policy Loss
                # L_clip = -min(r_t * A_scalar, clip(r_t, 1-eps, 1+eps) * A_scalar).mean()
                surr1 = ratio * scalar_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * scalar_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # 5. Multi-Head 5D Vector Critic Loss (Smooth L1 / Huber)
                # L_vf = (1/5) * sum_{i=0}^4 SmoothL1(new_val_5d[:, i], returns_5d[:, i])
                if self.clip_vloss and old_values is not None:
                    v_loss_unclipped = F.smooth_l1_loss(new_val_5d, returns, reduction="none")
                    v_clipped = old_values + torch.clamp(
                        new_val_5d - old_values, -clip_coef, clip_coef
                    )
                    v_loss_clipped = F.smooth_l1_loss(v_clipped, returns, reduction="none")
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    critic_loss = v_loss_max.mean()
                else:
                    critic_loss = F.smooth_l1_loss(new_val_5d, returns, reduction="mean")

                # 6. Policy Entropy Bonus
                entropy_bonus = entropy.mean()

                # 7. Total MO-PPO Loss
                loss = actor_loss + (vf_coef * critic_loss) - (ent_coef * entropy_bonus)

                # 8. Gradient Step
                self.optimizer.zero_grad()
                loss.backward()
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
                self.optimizer.step()

                # 9. Track Batch Metrics
                num_updates += 1
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy_bonus.item()
                total_approx_kl += approx_kl.item()
                total_clip_fraction += clip_frac.item()
                total_loss_accum += loss.item()

                batch_ev = self.compute_explained_variance_5d(new_val_5d, returns)
                ev_history.append(batch_ev)

        # Average metrics across all mini-batch updates
        denom = max(1, num_updates)
        avg_ev = np.mean(ev_history, axis=0).tolist() if ev_history else [0.0] * 5

        current_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else self.lr

        metrics: Dict[str, Any] = {
            "actor_loss": total_actor_loss / denom,
            "critic_loss": total_critic_loss / denom,
            "entropy": total_entropy / denom,
            "approx_kl": total_approx_kl / denom,
            "clip_fraction": total_clip_fraction / denom,
            "explained_variance_5d": avg_ev,
            "total_loss": total_loss_accum / denom,
            "learning_rate": current_lr,
        }

        return metrics

    def set_learning_rate(self, lr: float) -> None:
        """Update the learning rate for linear annealing.

        Args:
            lr: New learning rate value.
        """
        self.lr = float(lr)
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr

    def get_action_and_value(
        self,
        obs: Any,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass delegating to underlying network.

        Returns:
            Tuple of (action, log_prob, entropy, value_5d).
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=self.device)
        else:
            obs = obs.to(self.device)
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)

        if action is not None and not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=self.device, dtype=torch.long)
        elif action is not None:
            action = action.to(self.device, dtype=torch.long)

        return self.network.get_action_and_value(obs, action)

    def get_action(
        self,
        obs: Any,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample or greedily choose an action from the policy for evaluation.

        Args:
            obs: Observation array or tensor.
            deterministic: If True, choose argmax action over policy logits.

        Returns:
            Action tensor.
        """
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, device=self.device)
            else:
                obs = obs.to(self.device)
            if obs.ndim == 3:
                obs = obs.unsqueeze(0)

            if deterministic and hasattr(self.network, "forward"):
                logits, _ = self.network.forward(obs)
                action = torch.argmax(logits, dim=-1)
                return action

            action, _, _, _ = self.network.get_action_and_value(obs)
            return action

    def get_value(self, obs: Any) -> torch.Tensor:
        """Predict 5D value vector V_phi(s) for the given observation.

        Args:
            obs: Observation array or tensor.

        Returns:
            Value tensor in R^{B x 5}.
        """
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, device=self.device)
            else:
                obs = obs.to(self.device)
            if obs.ndim == 3:
                obs = obs.unsqueeze(0)

            if hasattr(self.network, "get_value"):
                return self.network.get_value(obs)
            _, _, _, val = self.network.get_action_and_value(obs)
            return val

    def save(self, path: Union[str, Path]) -> str:
        """Save network state dict, optimizer state dict, and agent config.

        Args:
            path: Destination file path.

        Returns:
            Saved absolute file path string.
        """
        save_path = Path(path).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "model_state_dict": self.network.state_dict(),  # Compatibility alias
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "config": self.config,
            "lr": self.lr,
        }
        torch.save(checkpoint, save_path)
        logger.info("Saved agent checkpoint to %s", save_path)
        return str(save_path)

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load network weights and optimizer state from checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Loaded checkpoint dictionary.
        """
        load_path = Path(path).resolve()
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        # Restore network weights
        if "network_state_dict" in checkpoint:
            self.network.load_state_dict(checkpoint["network_state_dict"])
        elif "model_state_dict" in checkpoint:
            self.network.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
            # Direct network state_dict
            self.network.load_state_dict(checkpoint)

        # Restore optimizer state
        if (
            self.optimizer is not None
            and "optimizer_state_dict" in checkpoint
            and checkpoint["optimizer_state_dict"] is not None
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore learning rate if stored
        if "lr" in checkpoint:
            self.set_learning_rate(checkpoint["lr"])

        logger.info("Loaded agent checkpoint from %s", load_path)
        return checkpoint
