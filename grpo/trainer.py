# -*- coding: utf-8 -*-
import os
# так можно выбирать устройство для запуска LLM
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from collections.abc import Callable
import json
from pathlib import Path
import random
import re
import numpy as np  # Добавляем numpy для расчетов
from typing import Any, Iterator, Optional, Dict, Union, Tuple, List
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
)
from bitsandbytes.optim import AdamW32bit, AdamW8bit
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .loss import approx_kl_divergence, GRPOLoss
from .replay_buffer import ReplayBuffer, Experience, join_experience_batch, zero_pad_sequences

# Импортируем новые модули
from .parallel_simulator import ParallelSimulator, parallel_rollout
from .grpo_advantage import process_episode_batch, calculate_discounted_returns, group_advantages
from .utils import DISCRETE_ACTIONS, parse_llm_action, format_state, COLOR_RESET, COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_BLUE, COLOR_CYAN

import argparse
from datetime import datetime
from simulators.single_well.simulator import SingleWellSimulator
import time
import pickle
from tqdm import tqdm

from simulators.multi_well.simulator import MultiWellSimulator
from grpo.prompts import get_single_well_prompt, get_subsequent_step_prompt, get_first_step_prompt, BASE_PROMPT_TEMPLATE

# --- Добавляем константы для цветов --- (больше не нужны, импортированы из utils)
# COLOR_RESET = "\033[0m"
# COLOR_GREEN = "\033[92m"
# COLOR_RED = "\033[91m"
# COLOR_YELLOW = "\033[93m"
# COLOR_BLUE = "\033[94m"
# COLOR_CYAN = "\033[96m"
# --- Конец констант для цветов ---

# --- Добавляем константу для дискретных действий --- (больше не нужна, импортирована из utils)
# DISCRETE_ACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# --- Конец константы для дискретных действий ---


class Logger:
    """
    Универсальный логгер с поддержкой wandb и tensorboard.
    Автоматически создает имя запуска/директории на основе имени скрипта и времени.
    """
    def __init__(self, script_name: str, use_wandb: bool = False, log_root_dir: str = "logs", wandb_project: str = "tiny_grpo", config: Optional[Dict] = None):
        self.use_wandb = use_wandb
        self.writer = None
        self.run_name = f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    name=self.run_name,
                    config=config if config else {},
                )
                print(f"{COLOR_BLUE}WandB run initialized: {self.run_name}{COLOR_RESET}")
            except Exception as e:
                print(f"{COLOR_RED}Failed to initialize WandB: {e}{COLOR_RESET}")
                self.use_wandb = False # Отключаем wandb если инициализация не удалась

        if not self.use_wandb:
            # Создаем директорию для TensorBoard
            tb_log_dir = Path(log_root_dir) / self.run_name
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tb_log_dir))
            print(f"{COLOR_BLUE}TensorBoard log directory: {tb_log_dir}{COLOR_RESET}")

    def add_scalar(self, tag: str, scalar_value: float, global_step: Optional[int] = None):
        """
        Добавляет скалярное значение для логирования.
        
        Args:
            tag: Название метрики
            scalar_value: Числовое значение
            global_step: Шаг, на котором записывается значение
        """
        if self.use_wandb and wandb.run:
            try:
                wandb.log({tag: scalar_value}, step=global_step)
            except Exception as e:
                print(f"{COLOR_YELLOW}WandB add_scalar warning: {e}{COLOR_RESET}")
        elif self.writer:
            self.writer.add_scalar(tag, scalar_value, global_step=global_step)
        
    def log(self, metrics: Dict[str, Union[float, str, wandb.Table]], step: Optional[int] = None):
        """Логирует метрики в выбранный бэкенд."""
        if self.use_wandb and wandb.run:
            try:
                 # wandb.log ожидает числовые значения или wandb.* объекты
                 wandb_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, torch.Tensor))}
                 # Отдельно логируем таблицы или другие объекты wandb
                 wandb_objects = {k: v for k, v in metrics.items() if isinstance(v, wandb.Table)}

                 if wandb_metrics:
                     wandb.log(wandb_metrics, step=step)
                 if wandb_objects:
                      wandb.log(wandb_objects, step=step) # Логируем объекты отдельно

            except Exception as e:
                 print(f"{COLOR_YELLOW}WandB log warning: {e}{COLOR_RESET}")
        elif self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, global_step=step)
                # Tensorboard не поддерживает таблицы или строки напрямую в add_scalar

    def log_text(self, tag: str, text: str, step: Optional[int] = None):
         """Логирует текстовые данные."""
         if self.use_wandb and wandb.run:
             # WandB может логировать текст через wandb.log с произвольным ключом
             # или можно использовать wandb.Html для форматирования
             try:
                 wandb.log({tag: text}, step=step)
             except Exception as e:
                 print(f"{COLOR_YELLOW}WandB log_text warning: {e}{COLOR_RESET}")
         elif self.writer:
             # TensorBoard логирует текст через add_text
             self.writer.add_text(tag, text, global_step=step)

    def log_table(self, key: str, columns: list, data: list, step: Optional[int] = None):
         """Логирует таблицу (в основном для WandB)."""
         if self.use_wandb and wandb.run:
             try:
                 table = wandb.Table(columns=columns, data=data)
                 self.log({key: table}, step=step)
             except Exception as e:
                 print(f"{COLOR_YELLOW}WandB log_table warning: {e}{COLOR_RESET}")
         else:
             # TensorBoard не имеет прямого аналога таблиц, можно логировать как текст
             table_str = "| " + " | ".join(columns) + " |\n"
             table_str += "|-" + "-|".join(['-' * len(c) for c in columns]) + "-|\n"
             for row in data:
                 table_str += "| " + " | ".join(map(str, row)) + " |\n"
             self.log_text(f"{key}_table", f"<pre>{table_str}</pre>", step=step)


    def close(self):
        """Закрывает логгер."""
        if self.use_wandb and wandb.run:
            # Добавляем проверку wandb.run перед finish()
            if wandb.run:
                wandb.finish()
                print(f"{COLOR_BLUE}WandB run finished.{COLOR_RESET}")
        if self.writer:
            self.writer.close()
            print(f"{COLOR_BLUE}TensorBoard writer closed.{COLOR_RESET}")


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
    use_4bit: bool = True,
    use_lora: bool = True,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if use_4bit:
        print("Используется 4-битная квантизация")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
    else:
        print("Используется 8-битная квантизация")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["lm_head"],
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        attn_implementation="flash_attention_2",
        device_map=device_map,
        quantization_config=quantization_config,
    )

    print("\nПроверка типов параметров модели:")
    total_size = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            param_size = param.numel() * param.element_size() / 1024**2
            total_size += param_size
            # print(f"{name}: {param.dtype}, размер: {param_size:.2f} MB") # Убрал для краткости логов

    print(f"\nОбщий размер модели: {total_size:.2f} MB")

    if use_lora:
        print("\nПрименяется LoRA")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer

# Шаблон системного промпта (без подстановки значений)
SYSTEM_PROMPT_TEMPLATE = """Ты - система управления нефтяной скважиной. 
Твоя задача - максимизировать добычу нефти за весь период разработки месторождения.
Ты будешь получать состояние скважины и должен выбрать степень открытия штуцера, выбрав ОДИН из 10 вариантов.

ДОСТУПНЫЕ ВАРИАНТЫ:
1 - открытие штуцера на 0.1 (10%)
2 - открытие штуцера на 0.2 (20%)
3 - открытие штуцера на 0.3 (30%)
4 - открытие штуцера на 0.4 (40%)
5 - открытие штуцера на 0.5 (50%)
6 - открытие штуцера на 0.6 (60%)
7 - открытие штуцера на 0.7 (70%)
8 - открытие штуцера на 0.8 (80%)
9 - открытие штуцера на 0.9 (90%)
10 - открытие штуцера на 1.0 (100%)

Состояние скважины включает:
- Давление в пласте (атм)
- Текущий дебит (м³/сут)
- Накопленная добыча (м³)
- Прошедшее время (дни)

Чем больше открыт штуцер, тем выше текущий дебит, но тем быстрее падает давление в пласте.
Необходимо найти баланс между быстрой добычей сейчас и сохранением давления для будущей добычи.

ВАЖНО: Твое решение будет применяться на период {forecast_days} дней.
{weekly_note}
{monthly_note}

ПРАВИЛА ОТВЕТА:
1. Ответ должен содержать ТОЛЬКО ОДНО ЧИСЛО от 1 до 10, обозначающее выбранный вариант степени открытия штуцера.
2. НЕ ДОБАВЛЯЙ никаких объяснений, рассуждений или дополнительного текста.
3. ТОЛЬКО ЧИСЛО от 1 до 10 и ничего больше.

Примеры правильных ответов:
1
5
10
7
3

Примеры НЕПРАВИЛЬНЫХ ответов:
"Я выбираю вариант 5"
"Вариант 7, поскольку это оптимальное значение"
"Степень открытия штуцера: 0.8"
"Выбираю значение 6"
"""

@torch.no_grad()
def rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    logger: Logger, # Принимаем объект Logger
    global_step: int,
    max_length: int = 1024,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:

    model.eval()
    all_sequences = []
    all_completions_text = []
    all_rewards_dicts = []
    
    # Создаем системный промпт на основе шаблона
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        forecast_days=1,  # Значение по умолчанию
        weekly_note="",
        monthly_note=""
    )

    # Метрики, агрегированные по группе роллаутов (для одной задачи)
    group_stats = {
        "total_reward_sum": 0.0,
        "answer_format_ok_count": 0,
        "answer_correct_count": 0,
    }

    for rollout_idx in range(num_rollouts):
        rewards = {
            "answer_format": 0.0,
            "answer_content": 0.0,
        }
        rollout_stats = { # Статистика для одного этого роллаута
             "completion": "", "final_answer": None,
             "is_correct_answer": False, "error_type": None
        }

        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        current_messages = chat_messages.copy()
        full_dialog_text_for_log = "" # Текст для логирования примеров
        rollout_tokens = []

        initial_prompt_text = tokenizer.apply_chat_template(
            current_messages, tokenize=False, add_generation_prompt=True
        )
        full_dialog_text_for_log += f"**Prompt:**\n```\n{initial_prompt_text}\n```\n"
        prompt_tokens = tokenizer(
            initial_prompt_text, return_tensors="pt", padding=False
        ).input_ids.to("cuda")
        rollout_tokens.append(prompt_tokens[0])

        # Генерация ответа
        chat_prompt_text = tokenizer.apply_chat_template(
            current_messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer(
            chat_prompt_text, return_tensors="pt", padding=False
        ).to("cuda")

        generation_config = GenerationConfig(
            do_sample=True, top_p=top_p, temperature=temperature,
            max_new_tokens=128, pad_token_id=tokenizer.eos_token_id,
        )
        sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
        new_tokens = sequence_ids[0, model_inputs["input_ids"].shape[1]:]
        rollout_tokens.append(new_tokens)

        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        rollout_stats["completion"] = completion
        full_dialog_text_for_log += f"**Completion:**\n```\n{completion}\n```\n"
        current_messages.append({"role": "assistant", "content": completion})

        # Проверка формата ответа - ожидаем число от 1 до 10
        answer_match = re.match(r"^\s*(\d+)\s*$", completion)
        if answer_match:
            final_answer = answer_match.group(1).strip()
            # Проверяем, что это число от 1 до 10
            if final_answer.isdigit() and 1 <= int(final_answer) <= 10:
                rewards["answer_format"] += 0.3
                group_stats["answer_format_ok_count"] += 1
                rollout_stats["final_answer"] = final_answer
                full_dialog_text_for_log += f"**Final Answer:** `{final_answer}`\n"

                # Сравниваем с oracle_answer
                if final_answer == oracle_answer:
                    rewards["answer_content"] += 1.0
                    rollout_stats["is_correct_answer"] = True
                    group_stats["answer_correct_count"] += 1
                    print(f"Rollout {rollout_idx+1}/{num_rollouts} | {COLOR_GREEN}Answer OK:{COLOR_RESET} {final_answer} (matches oracle: {oracle_answer})")
                else:
                    rewards["answer_content"] -= 0.5
                    rollout_stats["error_type"] = "Answer Content Mismatch"
                    print(f"Rollout {rollout_idx+1}/{num_rollouts} | {COLOR_YELLOW}Answer Content Mismatch:{COLOR_RESET} Got '{final_answer}', Expected '{oracle_answer}'")
            else:
                rewards["answer_format"] -= 0.8
                rollout_stats["error_type"] = "Answer Out of Range"
                full_dialog_text_for_log += "**Final Answer:** Failed (Number Out of Range)\n"
                print(f"Rollout {rollout_idx+1}/{num_rollouts} | {COLOR_RED}Answer Out of Range:{COLOR_RESET} {completion[:50]}...")
        else:
            rewards["answer_format"] -= 0.8
            rollout_stats["error_type"] = "Answer Format Error"
            full_dialog_text_for_log += "**Final Answer:** Failed (Format Error)\n"
            print(f"Rollout {rollout_idx+1}/{num_rollouts} | {COLOR_RED}Answer Format Error:{COLOR_RESET} {completion[:50]}...")

        total_reward = sum(rewards.values())
        group_stats["total_reward_sum"] += total_reward

        # Логируем детальные награды для *каждого* роллаута
        logger.log({
            f"rollout_rewards/total": total_reward,
            f"rollout_rewards/format": rewards["answer_format"],
            f"rollout_rewards/content": rewards["answer_content"],
        }, step=global_step)

        if rollout_tokens:
            full_sequence = torch.cat(rollout_tokens)
            all_sequences.append(full_sequence)
        else:
            all_sequences.append(torch.tensor([], dtype=torch.long, device="cuda"))

        all_completions_text.append(full_dialog_text_for_log) # Сохраняем текст с разметкой
        all_rewards_dicts.append(rewards)

    # --- Расчет и логирование агрегированных метрик для группы ---
    avg_group_reward = group_stats["total_reward_sum"] / num_rollouts if num_rollouts > 0 else 0.0
    answer_format_ok_rate = group_stats["answer_format_ok_count"] / num_rollouts if num_rollouts > 0 else 0.0
    answer_correct_rate = group_stats["answer_correct_count"] / group_stats["answer_format_ok_count"] if group_stats["answer_format_ok_count"] > 0 else 0.0

    logger.log({
        "group_avg/reward": avg_group_reward,
        "group_rates/answer_format_ok": answer_format_ok_rate,
        "group_rates/answer_correct": answer_correct_rate,
    }, step=global_step)

    # Паддинг и создание маски (остается как было в предыдущей версии)
    if not all_sequences:
        print(f"{COLOR_YELLOW}WARNING: No valid sequences generated in this group.{COLOR_RESET}")
        return torch.empty(0, 0, device="cuda"), \
               torch.empty(0, 1, device="cuda"), \
               torch.empty(0, 0, dtype=torch.bool, device="cuda"), \
               []

    non_empty_sequences = [seq for seq in all_sequences if seq.numel() > 0]
    if not non_empty_sequences:
        print(f"{COLOR_YELLOW}WARNING: All sequences in the group are empty.{COLOR_RESET}")
        return torch.empty(0, 0, device="cuda"), \
               torch.empty(0, 1, device="cuda"), \
               torch.empty(0, 0, dtype=torch.bool, device="cuda"), \
               []

    max_seq_length = max(seq.size(0) for seq in non_empty_sequences)

    padded_sequences = []
    original_lengths = []
    for seq in all_sequences:
        seq_len = seq.size(0)
        original_lengths.append(seq_len)
        padding_length = max_seq_length - seq_len
        if padding_length >= 0:
            padded_seq = torch.cat([seq, torch.full((padding_length,), tokenizer.pad_token_id, device=seq.device)])
        else:
            padded_seq = seq[:max_seq_length]
        padded_sequences.append(padded_seq)

    sequence_ids = torch.stack(padded_sequences)

    action_mask = torch.zeros_like(sequence_ids[:, 1:], dtype=torch.bool)

    len_prompt = rollout_tokens[0].size(0) if rollout_tokens else 0 # Длина первого промпта
    len_comp = rollout_tokens[1].size(0) if len(rollout_tokens) > 1 else 0 # Длина ответа

    for i, total_len in enumerate(original_lengths):
        start = len_prompt
        end = start + len_comp
        mask_start = max(0, start - 1)
        mask_end = max(0, end - 1)
        # Исправляем условие, чтобы не выходить за пределы маски
        if mask_end > mask_start and mask_start < action_mask.shape[1]:
             actual_end = min(mask_end, action_mask.shape[1])
             action_mask[i, mask_start : actual_end] = True

        valid_len_mask = total_len - 1
        if valid_len_mask < action_mask.shape[1]:
             action_mask[i, valid_len_mask:] = False

    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, rew_dict in enumerate(all_rewards_dicts):
        returns[i] = sum(rew_dict.values())

    # Возвращаем текст completions для возможного логирования примеров
    return sequence_ids, returns.to(sequence_ids.device), action_mask, all_completions_text

def init_rng(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    # Дополнительно для GPU, если используется
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Для multi-GPU
        # Не всегда нужно для воспроизводимости, может замедлить
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    """Вычисляет лог-вероятности для последовательности токенов."""
    output_ids = output_ids.to(logits.device) # Убедимся, что на одном устройстве
    log_probs = F.log_softmax(logits, dim=-1)
    action_log_probs = torch.gather(log_probs, -1, output_ids.unsqueeze(-1)).squeeze(-1)
    return action_log_probs


def sequences_log_probs(
    model: AutoModelForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Получает лог-вероятности для всех токенов в батче последовательностей."""
    # Создаем position_ids для корректной работы attention
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=0)
    
    # Пропускаем через модель
    with torch.no_grad():
        output = model.forward(
            input_ids=sequence_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
    
    # Получаем логиты и смещаем их на один шаг вперед
    # чтобы предсказывать следующий токен
    logits = output.logits
    
    # Обрезаем последний токен и добавляем игнорируемый первый токен для выравнивания размеров
    shifted_logits = logits[:, :-1, :]
    shifted_ids = sequence_ids[:, 1:]
    
    # Рассчитываем логарифм вероятности для сдвинутых токенов
    log_probs = sequence_log_probs_from_logits(
        shifted_logits, shifted_ids
    )
    
    # Добавляем нулевую вероятность для первого токена
    # чтобы сохранить размерность
    batch_size = sequence_ids.shape[0]
    zeros = torch.zeros(batch_size, 1, device=sequence_ids.device)
    log_probs = torch.cat([zeros, log_probs], dim=1)
    
    # Клиппинг для стабильности
    log_probs = torch.clamp(log_probs, min=-20.0, max=0.0)
    
    return log_probs


def parse_args():
    # Определяем имя скрипта для логгера
    script_name = Path(__file__).stem # Получаем имя файла без .py
    parser = argparse.ArgumentParser(description='Train a model with GRPO for oil well simulation.') # Обновляем описание

    # Аргументы Логирования и Конфигурации
    parser.add_argument('--run_name', type=str, default=None, help='Custom run name for logging (overrides auto-generated)')
    parser.add_argument('--log_dir', type=str, default="logs", help='Root directory for logs')
    parser.add_argument('--wandb', action='store_true', help='Use WandB for logging')
    parser.add_argument('--wandb_project', type=str, default="grpo-oil-sim", help='WandB project name') # Обновляем проект по умолчанию
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device_index', type=int, default=0, help='CUDA device index')

    # Аргументы Модели
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-3B-Instruct", help='Model name or path')
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints_oil_sim", help='Path to save checkpoints') # Обновляем путь по умолчанию
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='Save checkpoint every N global steps') # Интервал в шагах

    # Аргументы Обучения
    parser.add_argument('--total_steps', type=int, default=1000, help='Total number of optimization steps')
    parser.add_argument('--epochs_per_step', type=int, default=1, help='Number of optimization epochs per collected batch')
    parser.add_argument('--rollouts_per_step', type=int, default=32, help='Number of simulation episodes per global step')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Batch size for training phase (experience buffer)') # Размер батча для SGD
    parser.add_argument('--max_buffer_size', type=int, default=0, help='Maximum replay buffer size (0 for unlimited)')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping norm')

    # Аргументы GRPO/PPO
    parser.add_argument('--gamma', type=float, default=0.97, help='Discount factor for returns')
    parser.add_argument('--kl_weight', type=float, default=0.02, help='Weight for KL penalty in GRPOLoss')
    parser.add_argument('--clip_eps', type=float, default=0.2, help='Clipping epsilon for PPO ratio in GRPOLoss')

    # Аргументы Генерации/Симуляции
    parser.add_argument('--max_length', type=int, default=1024, help='Max sequence length (prompt+responses) in Experience')
    parser.add_argument('--max_new_tokens_per_step', type=int, default=10, help='Max new tokens per simulation step (LLM action)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
    parser.add_argument('--top_p', type=float, default=1.0, help='Generation top_p')
    
    # Добавляем аргумент для дискретных действий
    parser.add_argument('--use_discrete_actions', action='store_true', help='Use discrete actions (1-10) instead of continuous (0-1)')

    # Аргументы Симулятора - общие
    parser.add_argument('--multi_well', action='store_true', help='Use multi-well simulator instead of single well')
    
    # Аргументы Симулятора - одиночной скважины
    parser.add_argument('--initial_pressure', type=float, default=200.0, help='Initial reservoir pressure (atm)')
    parser.add_argument('--initial_bhp', type=float, default=50.0, help='Initial bottom hole pressure (atm)')
    parser.add_argument('--productivity_index', type=float, default=0.1, help='Productivity index (m3/day/atm)')
    parser.add_argument('--total_volume', type=float, default=1e6, help='Total reservoir volume (m3)')
    parser.add_argument('--simulation_dt', type=float, default=1.0, help='Simulation step size (days)')
    parser.add_argument('--simulation_max_time', type=float, default=365.0, help='Maximum simulation time (days)')
    
    # Параметр, определяющий прогноз на N дней вперед
    parser.add_argument('--forecast_days', type=int, default=1, 
                      help='Number of days to forecast in each step (1=daily, 7=weekly, 30=monthly)')
    
    # Аргументы Симулятора - нескольких скважин
    parser.add_argument('--n_wells', type=int, default=3, help='Number of wells in multi-well simulator')
    parser.add_argument('--interaction_strength', type=float, default=0.1, help='Strength of interaction between wells (0-1)')
    parser.add_argument('--shared_reservoir', action='store_true', help='Use shared reservoir in multi-well simulator')

    # Аргументы для логирования
    parser.add_argument('--log_completions_interval', type=int, default=10, help='Log example episode rollout every N global steps')
    
    # Аргументы для случайного начального состояния
    parser.add_argument('--use_random_states', action='store_true', help='Use random initial states for training')
    parser.add_argument('--random_state_min_depletion', type=float, default=0.0, help='Minimum depletion ratio for random states (0-1)')
    parser.add_argument('--random_state_max_depletion', type=float, default=0.8, help='Maximum depletion ratio for random states (0-1)')
    parser.add_argument('--random_state_probability', type=float, default=0.7, 
                      help='Probability of using random states in each global step (0-1). 0=never, 1=always')
    parser.add_argument('--use_realistic_ranges', action='store_true', default=True,
                      help='Use realistic parameter constraints for random states')

    args = parser.parse_args()

    # Добавляем имя скрипта для логгера
    args.script_name = script_name

    # Создаем имя запуска по умолчанию, если не задано
    if args.run_name is None:
        # Используем дефисы в именах аргументов для авто-имени
        args.run_name = f"{args.script_name}_lr{args.lr}_kl{args.kl_weight}_rp{args.rollouts_per_step}_bs{args.train_batch_size}"

    return args


def main():
    args = parse_args()
    
    # Устанавливаем случайные зерна для воспроизводимости
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Выводим информацию об использовании дискретных действий
    if args.use_discrete_actions:
        print(f"{COLOR_GREEN}Используем дискретные действия (варианты 1-10){COLOR_RESET}")
        print(f"Значения действий: {DISCRETE_ACTIONS}")
    else:
        print(f"{COLOR_YELLOW}Используем непрерывные действия (значения от 0 до 1){COLOR_RESET}")
    
    # --- Инициализация ---
    seed = args.seed
    device_index = args.device_index
    model_name = args.model_name
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_interval = args.checkpoint_interval
    train_batch_size = args.train_batch_size
    lr = args.lr
    rollouts_per_step = args.rollouts_per_step
    max_length = 4096  # Увеличиваем для предотвращения пропуска эпизодов
    max_new_tokens_per_step = 5  # Ограничиваем до 5 токенов, нам нужно только число
    temperature = args.temperature
    top_p = args.top_p
    log_completions_interval = args.log_completions_interval
    total_steps = args.total_steps
    epochs_per_step = args.epochs_per_step

    use_4bit = True
    use_lora = True
    bf16 = True

    max_norm = 1.0 # Можно сделать параметром

    # Собираем конфиг для логгера
    run_config = vars(args) # Преобразуем Namespace в dict

    # --- Инициализация логгера ---
    logger = Logger(
        script_name=args.script_name if args.run_name is None else Path(args.run_name).stem, # Используем имя скрипта или кастомное
        use_wandb=args.wandb,
        log_root_dir=args.log_dir,
        wandb_project=args.wandb_project,
        config=run_config # Передаем конфиг запуска
    )
    # Если указано кастомное имя, используем его
    if args.run_name:
        logger.run_name = args.run_name


    # Создаем директорию для чекпоинтов, если ее нет
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    print(f"Using device: {device}")
    init_rng(seed)

    print("Загрузка референсной модели...")
    reference_model, _ = load_model(
        model_name, device_map="auto", use_4bit=use_4bit, use_lora=False, bf16=bf16
    )
    print("Загрузка основной модели...")
    model, tokenizer = load_model(
        model_name, device_map="auto", use_4bit=use_4bit, use_lora=use_lora, bf16=bf16
    )

    optimizer = AdamW32bit(model.parameters(), lr=lr, is_paged=True)

    reference_model.eval()
    # Проверяем, есть ли градиент чекпоинтинг, прежде чем включать
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled.")
    else:
        print(f"{COLOR_YELLOW}Warning: Model does not support gradient_checkpointing_enable directly.{COLOR_RESET}")

    pad_token_id = tokenizer.eos_token_id

    # Создаем симулятор
    if args.multi_well:
        # Создаем мультискважинный симулятор
        from simulators.multi_well.simulator import MultiWellSimulator
        
        # Параметры для отдельных скважин
        well_params = {
            'initial_reservoir_pressure': args.initial_pressure,
            'initial_bhp': args.initial_bhp,
            'productivity_index': args.productivity_index,
            'dt': args.simulation_dt * args.forecast_days,  # Учитываем forecast_days
            'max_time': args.simulation_max_time
        }
        
        simulator = MultiWellSimulator(
            n_wells=args.n_wells,
            interaction_strength=args.interaction_strength,
            shared_reservoir=args.shared_reservoir,
            total_volume=args.total_volume,
            **well_params
        )
    else:
        # Создаем симулятор одиночной скважины
        simulator = SingleWellSimulator(
            initial_reservoir_pressure=args.initial_pressure,
            initial_bhp=args.initial_bhp,
            productivity_index=args.productivity_index,
            total_volume=args.total_volume,
            dt=args.simulation_dt * args.forecast_days,  # Учитываем forecast_days
            max_time=args.simulation_max_time
        )
    
    print(f"Симулятор создан: {simulator.__class__.__name__}")
    print(f"Период прогнозирования: {args.forecast_days} дней (dt = {simulator.dt:.1f})")
    
    # Создаем ReplayBuffer для хранения опыта
    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=args.clip_eps, kl_weight=args.kl_weight)

    global_step = 0 # Счетчик шагов оптимизации
    batch_step = 0 # Счетчик обработанных батчей промптов
    best_batch_mean_return = float('-inf')

    print(f"{COLOR_BLUE}Начало цикла обучения для {total_steps} шагов...{COLOR_RESET}")
    while global_step < total_steps:
        print(f"\n--- Global Step {global_step}/{total_steps} --- ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
        step_metrics = {}
        replay_buffer.clear() # Очищаем буфер перед сбором новых данных (on-policy)

        # --- Фаза сбора данных (Rollout) ---
        print(f"  Сбор {rollouts_per_step} эпизодов...")
        
        # Создаем конфигурацию для симуляторов
        simulator_config = {
            'initial_reservoir_pressure': args.initial_pressure,
            'initial_bhp': args.initial_bhp,
            'productivity_index': args.productivity_index,
            'total_volume': args.total_volume,
            'dt': args.simulation_dt * args.forecast_days,
            'max_time': args.simulation_max_time
        }
        
        # Определяем тип симулятора и добавляем специфичные параметры
        simulator_type = "multi_well" if args.multi_well else "single_well"
        if simulator_type == "multi_well":
            simulator_config.update({
                'n_wells': args.n_wells,
                'interaction_strength': args.interaction_strength,
                'shared_reservoir': args.shared_reservoir
            })
        
        # Создаем набор параллельных симуляторов
        parallel_sim = ParallelSimulator(
            n_simulators=rollouts_per_step,
            simulator_type=simulator_type,
            simulator_config=simulator_config,
            device=device.type
        )
        
        # Выполняем параллельные роллауты
        # Определяем, использовать ли случайные состояния на этом шаге, на основе заданной вероятности
        use_random_on_this_step = args.use_random_states and random.random() < args.random_state_probability
        
        if use_random_on_this_step:
            print(f"{COLOR_CYAN}Шаг {global_step}: Используем случайные начальные состояния "
                 f"(истощение {args.random_state_min_depletion:.2f}-{args.random_state_max_depletion:.2f}){COLOR_RESET}")
        else:
            print(f"{COLOR_CYAN}Шаг {global_step}: Используем начальные состояния скважин{COLOR_RESET}")
        
        episode_tokens, action_masks, rewards, episode_stats = parallel_rollout(
            model=model,
            tokenizer=tokenizer,
            parallel_sim=parallel_sim,
            n_steps=int(args.simulation_max_time / args.simulation_dt),  # Максимальное количество шагов
            temperature=temperature,
            top_p=top_p,
            verbose=True,
            use_random_states=use_random_on_this_step,
            random_state_min_depletion=args.random_state_min_depletion,
            random_state_max_depletion=args.random_state_max_depletion,
            use_realistic_ranges=args.use_realistic_ranges
        )
        
        # Обрабатываем результаты и создаем буфер опыта
        # Подготавливаем данные в нужном формате для process_episode_batch
        device = next(model.parameters()).device
        
        # Генерируем логиты для обоих моделей и формируем входные данные
        model_batch_data = {}
        ref_batch_data = {}
        actions_batch = []
        
        with torch.no_grad():
            # Обрабатываем каждый эпизод
            for ep_idx, (tokens, masks, ep_rewards) in enumerate(zip(episode_tokens, action_masks, rewards)):
                # Создаем внимание для последовательности
                attention_mask = torch.ones_like(tokens, dtype=torch.bool)
                
                # Получаем логиты от основной модели
                model_outputs = model(tokens.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
                model_logits = model_outputs.logits.squeeze(0)
                
                # Получаем логиты от референсной модели
                ref_outputs = reference_model(tokens.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
                ref_logits = ref_outputs.logits.squeeze(0)
                
                # Сохраняем данные для текущего эпизода
                if ep_idx == 0:
                    model_batch_data["logits"] = [model_logits]
                    model_batch_data["action_masks"] = [masks]
                    model_batch_data["sequences"] = [tokens]
                    ref_batch_data["logits"] = [ref_logits]
                else:
                    model_batch_data["logits"].append(model_logits)
                    model_batch_data["action_masks"].append(masks)
                    model_batch_data["sequences"].append(tokens)
                    ref_batch_data["logits"].append(ref_logits)
                
                # Формируем информацию о действиях и наградах
                actions_data = {"actions": []}
                
                # В parallel_rollout.py episode_tokens и episode_action_masks формируются по шагам
                # и объединяются с помощью torch.cat. Нам нужно разделить это обратно.
                
                # Получаем статистику по эпизоду (если доступна)
                ep_stats = episode_stats[ep_idx] if ep_idx < len(episode_stats) else None
                num_steps = len(ep_rewards)
                
                if ep_stats and "actions" in ep_stats:
                    # Используем информацию о действиях из статистики эпизода
                    for step_idx, (reward, action) in enumerate(zip(ep_rewards, ep_stats["actions"])):
                        # Создаем примерные token_ids для действия (упрощенно)
                        # Мы не знаем точно, какие токены соответствуют какому шагу
                        # Это приближение, в идеале нужно более точное отслеживание токенов
                        token_indices = [i for i, mask_val in enumerate(masks) if mask_val]
                        step_token_indices = token_indices[:len(token_indices) // num_steps]
                        
                        step_action = {
                            "reward": reward.item(),
                            "token_ids": step_token_indices,
                            "action_value": action,  # Добавляем фактическое значение действия
                        }
                        actions_data["actions"].append(step_action)
                else:
                    # Если статистики нет, равномерно распределяем маски
                    # по количеству шагов (это упрощение, не идеальное решение)
                    token_indices = torch.where(masks)[0].tolist()
                    tokens_per_step = max(1, len(token_indices) // num_steps)
                    
                    for step_idx, reward in enumerate(ep_rewards):
                        start_idx = step_idx * tokens_per_step
                        end_idx = min((step_idx + 1) * tokens_per_step, len(token_indices))
                        
                        step_token_indices = token_indices[start_idx:end_idx]
                        if not step_token_indices and token_indices:  # Если пусто, но есть токены
                            step_token_indices = [token_indices[0]]  # Используем первый токен
                            
                        step_action = {
                            "reward": reward.item(),
                            "token_ids": step_token_indices,
                        }
                        actions_data["actions"].append(step_action)
                
                actions_batch.append(actions_data)
        
        # Проверяем, что были собраны эпизоды
        # Преобразуем списки в тензоры
        if "logits" not in model_batch_data or not model_batch_data["logits"]:
            print(f"{COLOR_RED}Не удалось собрать ни одного эпизода. Пропускаем шаг обучения.{COLOR_RESET}")
            # Увеличиваем счетчик глобальных шагов и продолжаем
            global_step += 1
            continue
        
        # Проверяем и выравниваем размеры тензоров перед стекированием
        # Определяем максимальную длину последовательности в батче
        max_seq_length = max([logits.shape[0] for logits in model_batch_data["logits"]])
        
        # Выравниваем размеры тензоров с помощью паддинга
        for i in range(len(model_batch_data["logits"])):
            current_length = model_batch_data["logits"][i].shape[0]
            if current_length < max_seq_length:
                # Паддинг для логитов модели
                padding_size = max_seq_length - current_length
                padding = torch.zeros(padding_size, model_batch_data["logits"][i].shape[1], 
                                     device=model_batch_data["logits"][i].device)
                model_batch_data["logits"][i] = torch.cat([model_batch_data["logits"][i], padding], dim=0)
                
                # Паддинг для последовательностей
                seq_padding = torch.zeros(padding_size, device=model_batch_data["sequences"][i].device, 
                                        dtype=model_batch_data["sequences"][i].dtype)
                model_batch_data["sequences"][i] = torch.cat([model_batch_data["sequences"][i], seq_padding], dim=0)
                
                # Паддинг для масок действий
                mask_padding = torch.zeros(padding_size, device=model_batch_data["action_masks"][i].device, 
                                         dtype=model_batch_data["action_masks"][i].dtype)
                model_batch_data["action_masks"][i] = torch.cat([model_batch_data["action_masks"][i], mask_padding], dim=0)
                
                # Паддинг для логитов референсной модели
                ref_padding = torch.zeros(padding_size, ref_batch_data["logits"][i].shape[1], 
                                        device=ref_batch_data["logits"][i].device)
                ref_batch_data["logits"][i] = torch.cat([ref_batch_data["logits"][i], ref_padding], dim=0)
        
        # Теперь стекируем выровненные тензоры
        model_batch_data["logits"] = torch.stack(model_batch_data["logits"])
        model_batch_data["sequences"] = torch.stack(model_batch_data["sequences"])
        model_batch_data["action_masks"] = torch.stack(model_batch_data["action_masks"])
        ref_batch_data["logits"] = torch.stack(ref_batch_data["logits"])
        
        # Теперь вызываем функцию с правильными аргументами
        experiences = process_episode_batch(
            model_batch_data=model_batch_data,
            ref_batch_data=ref_batch_data,
            actions_batch=actions_batch,
            device=device,
            gamma=args.gamma,
            window_size=max_length,
            total_steps=len(rewards[0]) if rewards else 0,
            normalize_advantages=True
        )
        
        # Создаем буфер воспроизведения и добавляем опыт
        replay_buffer = ReplayBuffer()
        replay_buffer.append(experiences)
        
        print(f"  Собрано {len(replay_buffer)} эпизодов.")
        
        # Собираем статистику для логирования
        batch_total_production = []
        batch_total_rewards = []
        
        # Извлекаем статистику из результатов роллаутов
        for stats in episode_stats:
            batch_total_production.append(stats["production"])
            batch_total_rewards.append(stats["reward"])
            
        # --- Логируем статистику собранных данных ---
        batch_step += 1
        mean_batch_production = sum(batch_total_production) / len(batch_total_production) if batch_total_production else 0
        mean_batch_reward = sum(batch_total_rewards) / len(batch_total_rewards) if batch_total_rewards else 0
        is_best_batch = mean_batch_reward > best_batch_mean_return
        if is_best_batch:
            best_batch_mean_return = mean_batch_reward
        
        step_metrics.update({
            "batch/mean_production": mean_batch_production,
            "batch/mean_reward": mean_batch_reward,
            "batch/is_best": 1.0 if is_best_batch else 0.0,
            "batch/best_mean_reward": best_batch_mean_return,
            "buffer/size": len(replay_buffer),
        })
        print(f"  Стат. батча: Ср. добыча={mean_batch_production:.2f}, Ср. награда={mean_batch_reward:.2f}, Лучшая награда={best_batch_mean_return:.2f}")

        # --- Фаза оптимизации через GRPO ---
        # Создаем DataLoader для буфера опыта
        data_loader = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=False,
            collate_fn=join_experience_batch, # Объединяет опыт в один батч
        )

        # Переводим модель в режим обучения
        model.train()
        device = next(model.parameters()).device

        # Обучаем несколько эпох на собранных данных
        for epoch in range(epochs_per_step):
            print(f"  Эпоха {epoch+1}/{epochs_per_step}")
            
            running_loss = 0.0
            running_policy_loss = 0.0
            running_kl_loss = 0.0
            running_entropy = 0.0
            
            batch_count = 0
            
            for batch_idx, batch in enumerate(data_loader):
                batch_count += 1
                # Переносим батч на устройство модели
                batch_device = {
                    "sequences": batch.sequences.to(device),
                    "action_log_probs": batch.action_log_probs.to(device),
                    "log_probs_ref": batch.log_probs_ref.to(device),
                    "returns": batch.returns.to(device),
                    "advantages": batch.advantages.to(device),
                    "attention_mask": batch.attention_mask.to(device),
                    "action_mask": batch.action_mask.to(device),
                }
                
                # Очищаем градиенты
                optimizer.zero_grad()
                
                # Получаем выход модели (логиты)
                outputs = model(
                    input_ids=batch_device["sequences"],
                    attention_mask=batch_device["attention_mask"]
                )
                
                # Вычисляем лосс
                loss_dict = objective(
                    logits=outputs.logits,
                    sequences=batch_device["sequences"],
                    advantages=batch_device["advantages"],
                    action_mask=batch_device["action_mask"],
                    old_logprobs=batch_device["action_log_probs"],
                    ref_logprobs=batch_device["log_probs_ref"]
                )
                
                # Распаковываем составляющие лосса
                loss = loss_dict["loss"]
                policy_loss = loss_dict["policy_loss"]
                kl_loss = loss_dict.get("kl", torch.tensor(0.0, device=device))
                entropy = loss_dict.get("entropy", torch.tensor(0.0, device=device))
                
                # Обратное распространение и оптимизация
                loss.backward()
                if batch_idx == 0:  # один раз за эпоху
                    mean_grad = np.mean([p.grad.abs().mean().item()
                                        for p in model.parameters() if p.grad is not None])
                    print(f"Mean |grad| over trainable params: {mean_grad:.2e}")

                # Обрезаем градиенты для стабильности
                clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()
                
                # Накапливаем метрики
                running_loss += loss.item()
                running_policy_loss += policy_loss.item()
                running_kl_loss += kl_loss.item()
                running_entropy += entropy.item()
                
                # Освобождаем память
                del outputs, loss_dict, loss, policy_loss, kl_loss, entropy
                del batch_device
                
                # Принудительно очищаем кеш CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Усредняем метрики за эпоху
            avg_loss = running_loss / batch_count if batch_count > 0 else 0
            avg_policy_loss = running_policy_loss / batch_count if batch_count > 0 else 0
            avg_kl_loss = running_kl_loss / batch_count if batch_count > 0 else 0
            avg_entropy = running_entropy / batch_count if batch_count > 0 else 0
            
            print(f"    Потери: total={avg_loss:.6f}, policy={avg_policy_loss:.6f}, kl={avg_kl_loss:.6f}, entropy={avg_entropy:.6f}")
            
            # Логируем метрики эпохи
            step_metrics.update({
                f"train/epoch_{epoch}_loss": avg_loss,
                f"train/epoch_{epoch}_policy_loss": avg_policy_loss,
                f"train/epoch_{epoch}_kl_loss": avg_kl_loss,
                f"train/epoch_{epoch}_entropy": avg_entropy,
            })
        
        # Усредненные метрики после всех эпох
        step_metrics.update({
            "train/loss": avg_loss,
            "train/policy_loss": avg_policy_loss,
            "train/kl_loss": avg_kl_loss,
            "train/entropy": avg_entropy,
        })
        
        # Логируем все метрики шага
        logger.log(step_metrics, step=global_step)
        
        # Сохраняем чекпоинт модели
        if global_step % checkpoint_interval == 0 or global_step == total_steps - 1:
            checkpoint_file = checkpoint_path / f"checkpoint_{global_step}.pt"
            print(f"  Сохранение чекпоинта в {checkpoint_file}")
            
            state_dict = {
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_batch_mean_return": best_batch_mean_return,
            }
            
            torch.save(state_dict, checkpoint_file)
        
        # Увеличиваем счетчик глобальных шагов
        global_step += 1
    
    print(f"{COLOR_GREEN}Обучение завершено после {global_step} шагов.{COLOR_RESET}")
    logger.close()


if __name__ == "__main__":
    main()
