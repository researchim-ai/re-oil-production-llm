# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional

from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from .replay_buffer import Experience, ReplayBuffer, join_experience_batch

# Константы для цветов в консоли
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_CYAN = "\033[96m"

def calculate_discounted_returns(rewards, gamma=0.99):
    """
    Вычисляет дисконтированные возвраты для последовательности наград.
    
    Args:
        rewards: список или тензор наград
        gamma: коэффициент дисконтирования
        
    Returns:
        torch.Tensor: дисконтированные возвраты
    """
    if isinstance(rewards, list):
        rewards = torch.tensor(rewards)
    
    returns = torch.zeros_like(rewards, dtype=torch.float32)
    
    # Вычисляем возвраты от конца к началу
    R = 0
    for i in reversed(range(len(rewards))):
        R = rewards[i] + gamma * R
        returns[i] = R
    
    return returns

def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Нормализует преимущества в группе.
    
    Args:
        returns: Тензор возвратов
        eps: Малое число для численной стабильности
        
    Returns:
        torch.Tensor: Нормализованные преимущества
    """
    return (returns - returns.mean()) / (returns.std() + eps)

def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    """
    Вычисляет лог-вероятности для последовательности токенов.
    
    Args:
        logits: Логиты выхода модели [seq_len, vocab_size]
        output_ids: Токены-цели [seq_len]
        
    Returns:
        torch.Tensor: Лог-вероятности [seq_len]
    """
    output_ids = output_ids.to(logits.device) # Убедимся, что на одном устройстве
    log_probs = F.log_softmax(logits, dim=-1)
    action_log_probs = torch.gather(log_probs, -1, output_ids.unsqueeze(-1)).squeeze(-1)
    return action_log_probs

def sequences_log_probs(
    model: AutoModelForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Получает лог-вероятности для всех токенов в батче последовательностей.
    
    Args:
        model: Языковая модель
        sequence_ids: Тензор последовательностей [batch_size, seq_len]
        attention_mask: Маска внимания [batch_size, seq_len]
        
    Returns:
        torch.Tensor: Лог-вероятности [batch_size, seq_len]
    """
    # Переносим на устройство модели, если не там
    model_device = next(model.parameters()).device
    sequence_ids = sequence_ids.to(model_device)
    attention_mask = attention_mask.to(model_device)

    # Убедимся, что модель в режиме eval для вычисления логитов
    is_training = model.training
    model.eval()

    with torch.no_grad(): # Не считаем градиенты здесь
        # Получаем логиты от модели для всей последовательности
        # output.logits shape: [batch_size, seq_len, vocab_size]
        output = model(input_ids=sequence_ids, attention_mask=attention_mask, use_cache=False)
        logits = output.logits

    # Возвращаем модель в исходный режим
    if is_training:
        model.train()

    # Сдвигаем логиты и sequence_ids для вычисления лог-вероятностей P(token_i | token_0...token_{i-1})
    # logits shape: [batch_size, seq_len, vocab_size] -> [batch_size, seq_len-1, vocab_size]
    # sequence_ids shape: [batch_size, seq_len] -> [batch_size, seq_len-1]
    shifted_logits = logits[:, :-1, :]
    shifted_sequence_ids = sequence_ids[:, 1:]

    # Вычисляем лог-вероятности для каждого токена (кроме первого)
    # log_probs shape: [batch_size, seq_len-1]
    log_probs = sequence_log_probs_from_logits(shifted_logits, shifted_sequence_ids)

    # Добавляем 0 в начало для первого токена (его вероятность не определена таким образом)
    # и переносим на CPU, чтобы соответствовать другим данным в Experience
    zero_log_probs = torch.zeros(log_probs.size(0), 1, device=log_probs.device)
    full_log_probs = torch.cat([zero_log_probs, log_probs], dim=1)

    return full_log_probs.to('cpu') # Возвращаем на CPU

def process_episode_batch(
    model: AutoModelForCausalLM,
    reference_model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    episodes_data: Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]],
    max_length: int = 1024,
    gamma: float = 0.99,
    verbose: bool = True,
) -> ReplayBuffer:
    """
    Обрабатывает результаты роллаутов и создает буфер опыта для GRPO.
    
    Args:
        model: Основная языковая модель
        reference_model: Референсная языковая модель
        tokenizer: Токенизатор
        episodes_data: Кортеж (all_episode_tokens, all_action_masks, all_rewards)
        max_length: Максимальная длина последовательности
        gamma: Коэффициент дисконтирования
        verbose: Выводить ли подробную информацию
        
    Returns:
        ReplayBuffer: Буфер опыта для GRPO
    """
    all_episode_tokens, all_action_masks, all_rewards = episodes_data
    replay_buffer = ReplayBuffer()
    
    # Обрабатываем данные каждого эпизода
    processed_count = 0
    
    for i in range(len(all_episode_tokens)):
        # Распаковываем данные эпизода
        sequences = all_episode_tokens[i]
        action_mask = all_action_masks[i]
        rewards_per_step = all_rewards[i]
        
        # Проверяем длину последовательности
        if sequences.shape[0] > max_length:
            if verbose:
                print(f"{COLOR_YELLOW}Пропуск эпизода {i+1}: длина {sequences.shape[0]} > max_length {max_length}{COLOR_RESET}")
            continue
        
        # 1. Вычисляем returns (дисконтированные награды)
        returns = calculate_discounted_returns(rewards_per_step, gamma=gamma)
        
        # 2. Вычисляем advantages (нормализованные returns)
        advantages = group_advantages(returns)
        
        # 3. Распределяем advantages по токенам действия (advantages_per_token)
        advantages_per_token = torch.zeros_like(sequences, dtype=torch.float32)
        action_token_indices = torch.where(action_mask)[0]
        
        if action_token_indices.numel() > 0:
            # Среднее значение преимущества за эпизод
            avg_advantage = advantages.mean().item() if advantages.numel() > 0 else 0.0
            advantages_per_token[action_mask] = avg_advantage
        
        # 4. Вычисляем логарифмы вероятностей действий для текущей модели
        pad_token_id = tokenizer.pad_token_id
        ref_attention_mask = sequences != pad_token_id
        
        action_log_probs = sequences_log_probs(
            model=model,
            sequence_ids=sequences.unsqueeze(0).to(next(model.parameters()).device),
            attention_mask=ref_attention_mask.unsqueeze(0).to(next(model.parameters()).device)
        ).squeeze(0).cpu()
        
        # 5. Вычисляем log_probs_ref (с референсной моделью)
        ref_model_device = next(reference_model.parameters()).device
        log_probs_ref = sequences_log_probs(
            model=reference_model,
            sequence_ids=sequences.unsqueeze(0).to(ref_model_device),
            attention_mask=ref_attention_mask.unsqueeze(0).to(ref_model_device)
        ).squeeze(0).cpu()
        
        # 6. Собираем Experience
        exp = Experience(
            sequences=sequences.unsqueeze(0),
            action_log_probs=action_log_probs.unsqueeze(0),
            log_probs_ref=log_probs_ref.unsqueeze(0),
            returns=returns.unsqueeze(0),
            advantages=advantages_per_token.unsqueeze(0),
            attention_mask=ref_attention_mask.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
            kl=None  # KL будет вычислен в лоссе
        )
        
        replay_buffer.append(exp)
        processed_count += 1
        
        if verbose and processed_count % 10 == 0:
            print(f"{COLOR_BLUE}Обработано {processed_count} эпизодов{COLOR_RESET}")
    
    if verbose:
        print(f"{COLOR_GREEN}Обработано всего {processed_count} эпизодов{COLOR_RESET}")
    
    return replay_buffer 