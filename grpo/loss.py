from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .replay_buffer import Experience

logger = logging.getLogger(__name__)


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Monte-Carlo approximation of KL divergence, k3 estimator, see: http://joschu.net/blog/kl-approx.html
    """
    # Предварительный клиппинг для стабильности
    log_probs = torch.clamp(log_probs, min=-20.0, max=0.0)
    log_probs_ref = torch.clamp(log_probs_ref, min=-20.0, max=0.0)

    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    # Более строгий клиппинг для числовой стабильности
    log_ratio = torch.clamp(log_ratio, min=-5.0, max=5.0)
    
    # Формула KL: exp(log_ratio) - log_ratio - 1
    # Дополнительно клиппируем результат
    kl = log_ratio.exp() - log_ratio - 1
    
    # Финальный клиппинг KL
    kl = torch.clamp(kl, min=0.0, max=5.0)
    
    return kl


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


class GRPOLoss(nn.Module):
    """
    Класс для расчета функции потерь GRPO (Generalized Reinforcement Learning with Policy Optimization).
    Точная копия логики из примера с калькулятором.
    """
    
    def __init__(
        self,
        clip_eps: float = 0.2,
        kl_weight: float = 0.01,  # точно как в калькуляторе
    ):
        """
        Инициализирует GRPOLoss.
        
        Args:
            clip_eps: Эпсилон для клиппирования отношения вероятностей
            kl_weight: Вес для KL дивергенции
        """
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(
        self,
        logits: torch.Tensor,
        sequences: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Вычисляет GRPO loss.
        
        Args:
            logits: Логиты от модели [B, T, V]
            sequences: Токены входной последовательности [B, T]
            advantages: Масштабирующие коэффициенты для policy loss [B, T]
            action_mask: Маска указывающая, какие токены являются действиями [B, T]
            old_logprobs: Логарифмы вероятностей от предыдущей модели [B, T]
            ref_logprobs: Логарифмы вероятностей от референсной модели [B, T]
            
        Returns:
            Словарь с loss и её компонентами
        """
        # Получаем log probs для действий из логитов
        log_probs = F.log_softmax(logits, dim=-1)
        current_logprobs = torch.gather(log_probs, 2, sequences.unsqueeze(-1)).squeeze(-1)
        
        # --- ВАЖНОЕ ИЗМЕНЕНИЕ: Предварительный клиппинг значений для стабильности ---
        # Клиппинг логарифмов вероятностей
        current_logprobs = torch.clamp(current_logprobs, min=-20.0, max=0.0)
        old_logprobs = torch.clamp(old_logprobs, min=-20.0, max=0.0)
        if ref_logprobs is not None:
            ref_logprobs = torch.clamp(ref_logprobs, min=-20.0, max=0.0)
        
        # Клиппинг advantages
        advantages = torch.clamp(advantages, min=-3.0, max=3.0)
        
        # Вычисляем KL-дивергенцию
        kl = approx_kl_divergence(
            log_probs=current_logprobs,
            log_probs_ref=ref_logprobs,
            action_mask=action_mask,
        )
        
        # Клиппинг KL (как в калькуляторе)
        kl = torch.clamp(kl, max=5.0)  # Снижаем макс значение с 100 до 5 для стабильности

        # Вычисляем policy loss
        ratio = (current_logprobs - old_logprobs).exp()
        # Клиппинг ratio для стабильности
        ratio = torch.clamp(ratio, min=0.1, max=10.0)
        
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2)
        
        # --- ВАЖНОЕ ИЗМЕНЕНИЕ: Сначала применяем маску к компонентам, а потом суммируем ---
        # Применяем маску к каждому компоненту по отдельности
        policy_loss_masked = masked_mean(policy_loss, action_mask, dim=-1).mean()
        kl_masked = masked_mean(kl, action_mask, dim=-1).mean()
        
        # Теперь складываем уже усредненные компоненты
        loss = policy_loss_masked + self.kl_weight * kl_masked
        
        # --- ВАЖНОЕ ИЗМЕНЕНИЕ: Убираем финальный клиппинг, который скрывает проблемы ---
        # Вместо жесткого клиппинга на выходе используем стабилизацию на ранних этапах
        
        # Формируем словарь с результатами
        result = {
            "loss": loss,
            "policy_loss": policy_loss_masked,
            "kl": kl_masked,
            "entropy": torch.tensor(0.0, device=policy_loss.device)
        }
        
        return result
