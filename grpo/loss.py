from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay_buffer import Experience


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Monte-Carlo approximation of KL divergence, k3 estimator, see: http://joschu.net/blog/kl-approx.html
    """

    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    # Добавляем клиппинг для числовой стабильности
    log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)

    return log_ratio.exp() - log_ratio - 1


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


class GRPOLoss(nn.Module):
    """GRPO actor loss"""

    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(
        self,
        logits: torch.Tensor,
        sequences: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """
        Вычисляет лосс GRPO
        
        Args:
            logits: Логиты модели [batch_size, seq_len, vocab_size]
            sequences: Токены последовательности [batch_size, seq_len]
            old_logprobs: Логарифмы вероятностей от старой модели [batch_size, seq_len]
            ref_logprobs: Логарифмы вероятностей от референсной модели [batch_size, seq_len]
            advantages: Преимущества для каждого токена [batch_size, seq_len]
            mask: Маска действий [batch_size, seq_len]
            
        Returns:
            dict: Словарь с компонентами лосса
        """
        # Вычисляем log_probs из логитов
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        # Берем логарифмы вероятностей для токенов из sequences
        sequences_shifted = sequences[:, 1:]
        log_probs = torch.gather(log_probs, 2, sequences_shifted.unsqueeze(-1)).squeeze(-1)
        
        # Вычисляем KL-дивергенцию
        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=ref_logprobs[:, 1:],
            action_mask=mask[:, :-1],
        )

        # Клиппинг kl для стабильности
        kl = torch.clamp(kl, max=100.0)
        kl_loss = self.kl_weight * kl

        # Вычисляем коэффициент отношения вероятностей для PPO
        ratio = (log_probs - old_logprobs[:, 1:]).exp()
        surr1 = ratio * advantages[:, 1:]
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages[:, 1:]
        policy_loss = -torch.min(surr1, surr2)
        
        # Полный лосс
        loss = policy_loss + kl_loss

        # Маскируем и усредняем
        action_mask = mask[:, :-1]
        policy_loss_mean = masked_mean(policy_loss, action_mask, dim=-1).mean()
        kl_loss_mean = masked_mean(kl_loss, action_mask, dim=-1).mean()
        loss_mean = masked_mean(loss, action_mask, dim=-1).mean()
        kl_mean = masked_mean(kl, action_mask, dim=-1).mean()
        
        # Вычисляем энтропию (не используется в лоссе, только для логирования)
        # Это приближенная энтропия, просто для отслеживания
        entropy = -masked_mean(log_probs, action_mask, dim=-1).mean()
        
        return {
            "loss": loss_mean,
            "policy_loss": policy_loss_mean,
            "kl_loss": kl_loss_mean,
            "kl": kl_mean,
            "entropy": entropy
        }
