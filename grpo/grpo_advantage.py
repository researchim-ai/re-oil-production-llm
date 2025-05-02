# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from .replay_buffer import Experience, ReplayBuffer, join_experience_batch

# Константы для цветов в консоли
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_CYAN = "\033[96m"

logger = logging.getLogger(__name__)

def calculate_discounted_returns(
    rewards: Union[List[float], torch.Tensor],
    gamma: float,
) -> Union[List[float], torch.Tensor]:
    """
    Вычисляет дисконтированные возвраты.
    
    Args:
        rewards: Список или тензор наград [T]
        gamma: Фактор дисконтирования
        
    Returns:
        Дисконтированные возвраты [T]
    """
    is_tensor = isinstance(rewards, torch.Tensor)
    device = rewards.device if is_tensor else None
    
    if is_tensor:
        rewards = rewards.cpu().tolist()
    
    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    if is_tensor:
        return torch.tensor(returns, device=device)
    return returns

def group_advantages(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Нормализует advantages на основе маски.
    
    Args:
        advantages: Преимущества [B, T]
        mask: Маска действий [B, T]
        normalize: Применить нормализацию если True
        
    Returns:
        Нормализованные преимущества [B, T]
    """
    if not normalize:
        return advantages
    
    # Преобразуем bool маску в float для арифметических операций
    float_mask = mask.float()
    
    # Убедимся, что advantages тоже float
    float_advantages = advantages.float()
    
    # Первичный клиппинг для устранения выбросов
    float_advantages = torch.clamp(float_advantages, min=-10.0, max=10.0)
    
    # Применяем маску
    advantages_masked = float_advantages * float_mask
    mask_sum = float_mask.sum()
    
    if mask_sum > 0:
        # Вычисляем статистики только по маскированным значениям
        mean = advantages_masked.sum() / mask_sum
        
        # Вычисляем стандартное отклонение более стабильным способом
        diff_squared = ((advantages_masked - mean) ** 2) * float_mask
        var = diff_squared.sum() / mask_sum
        std = torch.sqrt(var + 1e-8)  # Увеличиваем эпсилон для большей стабильности
        
        # Нормализуем только маскированные значения
        normalized_advantages = float_mask * (float_advantages - mean) / std
        
        # Дополнительный клиппинг после нормализации, более строгий чем раньше
        normalized_advantages = torch.clamp(normalized_advantages, min=-2.0, max=2.0)
        
        return normalized_advantages
    else:
        return float_advantages * 0.0  # Если маска пустая, возвращаем нули

def get_log_probs(
    logits: torch.Tensor,
    sequences: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Получает log probs для указанных действий из логитов.
    
    Args:
        logits: Логиты модели [B, T, V]
        sequences: Токены последовательности [B, T]
        mask: Маска действий [B, T]
        
    Returns:
        Log probs [B, T]
    """
    # Проверяем размерности входных данных
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)  # [T, V] -> [1, T, V]
    
    if sequences.dim() == 1:
        sequences = sequences.unsqueeze(0)  # [T] -> [1, T]
    
    # Вычисляем log probs из логитов
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Берем log probs для токенов в sequences
    log_probs = torch.gather(log_probs, 2, sequences.unsqueeze(-1)).squeeze(-1)
    
    if mask is not None:
        log_probs = log_probs * mask
    
    return log_probs

def process_episode_batch(
    model_batch_data: Dict[str, torch.Tensor],
    ref_batch_data: Dict[str, torch.Tensor],
    actions_batch: List[Dict],
    device: torch.device,
    gamma: float = 0.99,
    window_size: int = 2048,
    total_steps: int = 10,
    normalize_advantages: bool = True,
) -> Experience:
    """
    Обрабатывает batch эпизодов и создает Experience для ReplayBuffer.
    
    Args:
        model_batch_data: Данные от основной модели
        ref_batch_data: Данные от референсной модели
        actions_batch: Список действий и наград
        device: Устройство для размещения тензоров
        gamma: Фактор дисконтирования
        window_size: Максимальный размер окна для токенизации
        total_steps: Общее число шагов в эпизоде
        normalize_advantages: Нормализовать ли advantages
    
    Returns:
        Experience для добавления в ReplayBuffer
    """
    batch_size = len(actions_batch)
    
    # Получаем данные от моделей
    model_logits = model_batch_data["logits"]
    model_action_mask = model_batch_data["action_masks"]
    model_sequences = model_batch_data["sequences"]
    ref_logits = ref_batch_data["logits"]
    
    # Подготовка списков для хранения данных по каждому эпизоду
    batch_advantages = []
    batch_old_log_probs = []
    batch_ref_log_probs = []
    batch_returns = []
    
    # Обрабатываем каждый эпизод в батче
    for episode_idx in range(batch_size):
        episode_data = actions_batch[episode_idx]
        
        # Извлекаем награды для текущего эпизода
        episode_rewards = [action["reward"] for action in episode_data["actions"]]
        
        # Клиппинг наград для стабильности
        episode_rewards = [min(max(r, -10.0), 10.0) for r in episode_rewards]
        
        # Вычисляем дисконтированные возвраты
        episode_returns = calculate_discounted_returns(
            torch.tensor(episode_rewards, device=device), 
            gamma
        )
        
        # Клиппинг возвратов
        episode_returns = torch.clamp(episode_returns, min=-10.0, max=10.0)
        
        # Получаем маску действий и последовательность токенов для эпизода
        action_mask = model_action_mask[episode_idx]
        sequence = model_sequences[episode_idx]
        
        # Инициализируем массив advantages нулями
        episode_advantages = torch.zeros_like(action_mask, device=device)
        
        # Распределяем advantages по токенам для каждого шага
        token_idx = 0
        for step_idx, action in enumerate(episode_data["actions"]):
            tokens_count = len(action["token_ids"])
            
            # Если есть токены для текущего действия
            if tokens_count > 0:
                # Назначаем соответствующее advantage всем токенам текущего действия
                advantage_value = episode_returns[step_idx]
                
                # Применяем advantage к токенам этого действия
                for token_id in action["token_ids"]:
                    if token_id < len(episode_advantages):
                        episode_advantages[token_id] = advantage_value
                    else:
                        # В случае проблем с индексами (вряд ли должно случиться)
                        print(f"WARNING: token_id {token_id} out of bounds for episode_advantages with shape {episode_advantages.shape}")
        
        # Получаем log probabilities для действий в эпизоде
        # Маскируем логиты до вычисления log probs
        episode_log_probs = get_log_probs(
            model_logits[episode_idx], 
            sequence,
            mask=action_mask,
        )
        
        # Получаем log probs от референсной модели
        episode_ref_log_probs = get_log_probs(
            ref_logits[episode_idx], 
            sequence,
            mask=action_mask,
        )
        
        # Клиппинг log probs для стабильности
        episode_log_probs = torch.clamp(episode_log_probs, min=-20.0, max=0.0)
        episode_ref_log_probs = torch.clamp(episode_ref_log_probs, min=-20.0, max=0.0)
        
        # # Сохраняем данные для эпизода
        # batch_advantages.append(episode_advantages)
        # batch_old_log_probs.append(episode_log_probs)
        # batch_ref_log_probs.append(episode_ref_log_probs)
        # # Для returns создаем тензор той же формы что и advantages, 
        # # но с единым значением для всего эпизода (сумма наград)
        # batch_returns.append(torch.ones_like(episode_advantages) * episode_rewards[-1])
        
        # --- NEW ---
        # Распределяем дисконтированный возврат каждого шага
        token_level_returns = torch.zeros_like(episode_advantages)
        for step_idx, step_data in enumerate(episode_data["actions"]):
            step_return = episode_returns[step_idx]          # γ‑discounted
            for tok_id in step_data["token_ids"]:            # токены, относящиеся к этому действию
                if tok_id < token_level_returns.shape[0]:
                    token_level_returns[tok_id] = step_return
        # --- END NEW ---

        batch_advantages.append(episode_advantages)
        batch_old_log_probs.append(episode_log_probs)
        batch_ref_log_probs.append(episode_ref_log_probs)
        batch_returns.append(token_level_returns)

        # Сохраняем данные для эпизода
    # Паддинг (дополнение) и объединение данных из разных эпизодов
    # Используем torch.nn.utils.rnn.pad_sequence для паддинга
    padded_sequences = pad_sequence(model_batch_data["sequences"], batch_first=True)
    padded_action_masks = pad_sequence(model_batch_data["action_masks"], batch_first=True)
    padded_advantages = pad_sequence(batch_advantages, batch_first=True)
    padded_old_log_probs = pad_sequence(batch_old_log_probs, batch_first=True)
    padded_ref_log_probs = pad_sequence(batch_ref_log_probs, batch_first=True)
    padded_returns = pad_sequence(batch_returns, batch_first=True)
    
    # Создаем маску внимания для паддинга
    attention_mask = (padded_sequences != 0).float()
    
    # Нормализуем advantages, если нужно
    if normalize_advantages:
        padded_advantages = group_advantages(padded_returns.clone(), padded_action_masks)
    
    # Создаем Experience
    experience = Experience(
        sequences=padded_sequences,
        action_log_probs=padded_old_log_probs,
        log_probs_ref=padded_ref_log_probs,
        returns=padded_returns,
        advantages=padded_advantages,
        attention_mask=attention_mask,
        action_mask=padded_action_masks,
        kl=None,  # Не вычисляем KL здесь, это будет сделано в objective
    )
    
    return experience 