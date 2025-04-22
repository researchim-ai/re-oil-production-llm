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

import argparse
from datetime import datetime
from simulators.single_well.simulator import SingleWellSimulator
import time
import pickle
from tqdm import tqdm

from simulators.multi_well.simulator import MultiWellSimulator
from grpo.prompts import get_single_well_prompt, get_subsequent_step_prompt, get_first_step_prompt, BASE_PROMPT_TEMPLATE

# --- Добавляем константы для цветов ---
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_CYAN = "\033[96m"
# --- Конец констант для цветов ---


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


###############################################################################
# БЛОК С ОПРЕДЕЛЕНИЕМ "ИНСТРУМЕНТОВ" (tools) И ФУНКЦИИ ВЫЗОВА
###############################################################################
TOOLS = {}

def register_tool(name: str):
    """
    Декоратор для регистрации инструмента по имени.
    """
    def decorator(func):
        TOOLS[name] = func
        return func
    return decorator

@register_tool("calc")
def calc_tool(expression: str) -> str:
    """
    Простой инструмент для вычисления арифметических выражений.
    Используется eval, в реальном продакшене нужно быть осторожным.
    """
    try:
        # Добавим простую очистку, но основная логика форматирования должна быть на LLM
        expression = expression.strip()
        # Убедимся, что строка не пустая после strip
        if not expression:
            return "Calc error: Empty expression"
        result = eval(expression, {'__builtins__': {}}, {}) # Ограничиваем eval
        return str(result)
    except Exception as e:
        # Возвращаем более информативную ошибку
        return f"Calc error: Cannot evaluate '{expression}'. Details: {e}"

# --- Изменяем detect_and_call_tools ---
def detect_and_call_tools(generated_text: str) -> Optional[Tuple[str, str, str]]:
    """
    Находит *первый* вызов инструмента, выполняет его и возвращает кортеж:
    (tool_name, tool_input, tool_result_str).
    Возвращает None, если инструмент не найден или не вызывался.
    """
    pattern = r"<tool:(\w+)>(.*?)</tool>"
    match = re.search(pattern, generated_text, flags=re.DOTALL)

    if match:
        tool_name = match.group(1)
        tool_input = match.group(2).strip()
        tool_func = TOOLS.get(tool_name)
        tool_result_str: Optional[str] = None

        if tool_func:
            try:
                tool_result_str = tool_func(tool_input)
            except Exception as e:
                tool_result_str = f"Error executing tool '{tool_name}': {e}"
        else:
            tool_result_str = f"[Tool '{tool_name}' not found]"

        # Возвращаем имя, ввод и результат
        if tool_result_str is not None:
            return tool_name, tool_input, tool_result_str
        else:
             # Случай, когда tool_func вернул None, хотя не должен
             return tool_name, tool_input, "[Error: Tool function returned None]"
    else:
        return None # Инструмент не вызывался


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
Ты будешь получать состояние скважины и должен выбрать степень открытия штуцера (от 0 до 1).

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
1. Ответ должен содержать ТОЛЬКО ОДНО ЧИСЛО от 0 до 1, обозначающее степень открытия штуцера.
2. НЕ ДОБАВЛЯЙ никаких объяснений, рассуждений или дополнительного текста.
3. ТОЛЬКО ЧИСЛО и ничего больше.

Примеры правильных ответов:
0.75
0.5
0.25
1
0
0.8

Примеры НЕПРАВИЛЬНЫХ ответов:
"Я выбираю степень открытия 0.5"
"0.75, поскольку это оптимальное значение"
"Степень открытия штуцера: 0.8"
"Выбираю значение 0.6"
"""

# Первый системный промпт - только для рассуждения и вызова инструмента
FIRST_STEP_PROMPT = """- Think about the reasoning process and explain it within <reasoning>...</reasoning> tags
- Call the calculation tool using: <tool:calc>user asked question for calculation</tool>

Here is the format example:

Calculate 2 + 2

<reasoning>I need to add these numbers together</reasoning>
<tool:calc>2 + 2</tool>

Your task:
"""

# Второй системный промпт - только для ответа
SECOND_STEP_PROMPT = """A conversation between User and Assistant. Now you need to copy answer from tool to answer tag.

- Your response MUST contain only the answer tag
- After receiving the tool result, provide the final answer within <answer>...</answer> tags

Format Example:

Tool result: 4
<answer>4</answer>

Here is Tool output:
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

    # Метрики, агрегированные по группе роллаутов (для одной задачи)
    group_stats = {
        "total_reward_sum": 0.0,
        "tool_called_count": 0,
        "tool_executed_ok_count": 0,
        "answer_format_ok_count": 0,
        "answer_correct_count": 0,
    }

    for rollout_idx in range(num_rollouts):
        rewards = {
            "step1_tool_call_format": 0.0,
            "step1_tool_execution": 0.0,
            "step2_answer_format": 0.0,
            "step2_answer_content": 0.0,
        }
        rollout_stats = { # Статистика для одного этого роллаута
             "step1_completion": "", "tool_called": False, "tool_input": None,
             "tool_result": None, "step2_completion": "", "final_answer": None,
             "is_correct_answer": False, "error_type": None
        }

        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": FIRST_STEP_PROMPT + task},
        ]

        current_messages = chat_messages.copy()
        full_dialog_text_for_log = "" # Текст для логирования примеров
        steps_count = 0
        max_steps = 2
        rollout_tokens = []
        actual_tool_result: Optional[str] = None
        step1_failed = False

        initial_prompt_text = tokenizer.apply_chat_template(
            current_messages, tokenize=False, add_generation_prompt=True
        )
        full_dialog_text_for_log += f"**Prompt:**\n```\n{initial_prompt_text}\n```\n"
        prompt_tokens = tokenizer(
            initial_prompt_text, return_tensors="pt", padding=False
        ).input_ids.to("cuda")
        rollout_tokens.append(prompt_tokens[0])

        # --- Шаг 1 ---
        steps_count += 1
        chat_prompt_text_step1 = tokenizer.apply_chat_template(
            current_messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs_step1 = tokenizer(
            chat_prompt_text_step1, return_tensors="pt", padding=False
        ).to("cuda")

        generation_config = GenerationConfig(
            do_sample=True, top_p=top_p, temperature=temperature,
            max_new_tokens=128, pad_token_id=tokenizer.eos_token_id,
        )
        sequence_ids_step1 = model.generate(**model_inputs_step1, generation_config=generation_config)
        new_tokens_step1 = sequence_ids_step1[0, model_inputs_step1["input_ids"].shape[1]:]
        rollout_tokens.append(new_tokens_step1)

        completion_step1 = tokenizer.decode(new_tokens_step1, skip_special_tokens=True)
        rollout_stats["step1_completion"] = completion_step1
        full_dialog_text_for_log += f"**Step 1 Completion:**\n```\n{completion_step1}\n```\n"
        current_messages.append({"role": "assistant", "content": completion_step1})

        # Вызов и проверка инструмента
        tool_call_info = detect_and_call_tools(completion_step1)
        if tool_call_info:
            tool_name, tool_input, actual_tool_result = tool_call_info
            rewards["step1_tool_call_format"] += 0.2
            rollout_stats["tool_called"] = True
            group_stats["tool_called_count"] += 1
            rollout_stats["tool_input"] = tool_input
            rollout_stats["tool_result"] = actual_tool_result
            full_dialog_text_for_log += f"**Tool Call:** `{tool_name}({tool_input})` -> `{actual_tool_result}`\n"

            if "error" in actual_tool_result.lower():
                rewards["step1_tool_execution"] -= 1.0
                step1_failed = True
                rollout_stats["error_type"] = "Tool Execution Error"
                print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | {COLOR_RED}Tool Error:{COLOR_RESET} {actual_tool_result}")
            else:
                rewards["step1_tool_execution"] += 0.5
                group_stats["tool_executed_ok_count"] += 1
                print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | {COLOR_GREEN}Tool OK:{COLOR_RESET} {tool_input} -> {actual_tool_result}")
        else:
            rewards["step1_tool_call_format"] -= 0.5
            step1_failed = True
            rollout_stats["error_type"] = "Tool Format Error"
            full_dialog_text_for_log += "**Tool Call:** Failed (Format Error)\n"
            print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | {COLOR_RED}Tool Call Format Error{COLOR_RESET}")

        # --- Шаг 2 ---
        if not step1_failed and actual_tool_result is not None:
            steps_count += 1
            user_message_step2 = f"{SECOND_STEP_PROMPT}\n\nTool result: {actual_tool_result}"
            current_messages.append({"role": "user", "content": user_message_step2})
            full_dialog_text_for_log += f"**Prompt Step 2 (User):**\n```\nTool result: {actual_tool_result}\n```\n"

            chat_prompt_text_step2 = tokenizer.apply_chat_template(
                current_messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs_step2 = tokenizer(
                chat_prompt_text_step2, return_tensors="pt", padding=False
            ).to("cuda")

            sequence_ids_step2 = model.generate(**model_inputs_step2, generation_config=generation_config)
            new_tokens_step2 = sequence_ids_step2[0, model_inputs_step2["input_ids"].shape[1]:]
            rollout_tokens.append(new_tokens_step2)

            completion_step2 = tokenizer.decode(new_tokens_step2, skip_special_tokens=True)
            rollout_stats["step2_completion"] = completion_step2
            full_dialog_text_for_log += f"**Step 2 Completion:**\n```\n{completion_step2}\n```\n"
            current_messages.append({"role": "assistant", "content": completion_step2})

            answer_match = re.match(r"^\s*<answer>(.*?)</answer>\s*$", completion_step2, flags=re.DOTALL)
            if answer_match:
                rewards["step2_answer_format"] += 0.3
                group_stats["answer_format_ok_count"] += 1
                final_answer = answer_match.group(1).strip()
                rollout_stats["final_answer"] = final_answer
                full_dialog_text_for_log += f"**Final Answer:** `{final_answer}`\n"

                # Сравниваем с oracle_answer вместо actual_tool_result
                if final_answer == oracle_answer:
                    rewards["step2_answer_content"] += 1.0
                    rollout_stats["is_correct_answer"] = True
                    group_stats["answer_correct_count"] += 1
                    print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {COLOR_GREEN}Answer OK:{COLOR_RESET} {final_answer} (matches oracle: {oracle_answer})")
                else:
                    rewards["step2_answer_content"] -= 0.5
                    rollout_stats["error_type"] = "Answer Content Mismatch"
                    print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {COLOR_YELLOW}Answer Content Mismatch:{COLOR_RESET} Got '{final_answer}', Expected '{oracle_answer}' (Tool result was: {actual_tool_result})")
            else:
                rewards["step2_answer_format"] -= 0.8
                rollout_stats["error_type"] = "Answer Format Error"
                full_dialog_text_for_log += "**Final Answer:** Failed (Format Error)\n"
                print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {COLOR_RED}Answer Format Error:{COLOR_RESET} {completion_step2[:50]}...") # Показываем начало ошибки
        else:
             full_dialog_text_for_log += "**Step 2:** Skipped\n"
             print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {COLOR_YELLOW}Skipped{COLOR_RESET}")

        total_reward = sum(rewards.values())
        group_stats["total_reward_sum"] += total_reward

        # Логируем детальные награды для *каждого* роллаута (может быть шумно, но полезно для отладки)
        logger.log({
            f"rollout_rewards/total": total_reward,
            f"rollout_rewards/step1_format": rewards["step1_tool_call_format"],
            f"rollout_rewards/step1_exec": rewards["step1_tool_execution"],
            f"rollout_rewards/step2_format": rewards["step2_answer_format"],
            f"rollout_rewards/step2_content": rewards["step2_answer_content"],
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
    tool_called_rate = group_stats["tool_called_count"] / num_rollouts if num_rollouts > 0 else 0.0
    tool_exec_ok_rate = group_stats["tool_executed_ok_count"] / group_stats["tool_called_count"] if group_stats["tool_called_count"] > 0 else 0.0
    answer_format_ok_rate = group_stats["answer_format_ok_count"] / num_rollouts if num_rollouts > 0 else 0.0 # Или от числа успешных шагов 1? Пока от всех
    answer_correct_rate = group_stats["answer_correct_count"] / group_stats["answer_format_ok_count"] if group_stats["answer_format_ok_count"] > 0 else 0.0

    logger.log({
        "group_avg/reward": avg_group_reward,
        "group_rates/tool_called": tool_called_rate,
        "group_rates/tool_exec_ok": tool_exec_ok_rate,
        "group_rates/answer_format_ok": answer_format_ok_rate,
        "group_rates/answer_correct": answer_correct_rate,
    }, step=global_step)

    # --- Конец изменений в rollout ---

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
    len_comp1 = rollout_tokens[1].size(0) if len(rollout_tokens) > 1 else 0 # Длина ответа 1
    len_comp2 = rollout_tokens[2].size(0) if len(rollout_tokens) > 2 else 0 # Длина ответа 2

    for i, total_len in enumerate(original_lengths):
        start1 = len_prompt
        end1 = start1 + len_comp1
        mask_start1 = max(0, start1 - 1)
        mask_end1 = max(0, end1 - 1)
        # Исправляем условие, чтобы не выходить за пределы маски
        if mask_end1 > mask_start1 and mask_start1 < action_mask.shape[1]:
             actual_end1 = min(mask_end1, action_mask.shape[1]) # Убедимся, что не выходим за границу
             action_mask[i, mask_start1 : actual_end1] = True

        start2 = end1
        end2 = start2 + len_comp2
        mask_start2 = max(0, start2 - 1)
        mask_end2 = max(0, end2 - 1)
        # Исправляем условие
        if mask_end2 > mask_start2 and mask_start2 < action_mask.shape[1]:
             actual_end2 = min(mask_end2, action_mask.shape[1])
             action_mask[i, mask_start2 : actual_end2] = True

        valid_len_mask = total_len - 1
        if valid_len_mask < action_mask.shape[1]:
             action_mask[i, valid_len_mask:] = False
        # Дополнительно обрежем маску по максимальной длине (уже не нужно из-за min выше)
        # action_mask[i, max_seq_length-1:] = False # Можно убрать

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

def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Нормализует преимущества в группе."""
    return (returns - returns.mean()) / (returns.std() + eps)


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
    parser.add_argument('--rollouts_per_step', type=int, default=32, help='Number of simulation episodes per global step') # Число эпизодов
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

    parser.add_argument('--log_completions_interval', type=int, default=10, help='Log example episode rollout every N global steps')

    args = parser.parse_args()

    # Добавляем имя скрипта для логгера
    args.script_name = script_name

    # Создаем имя запуска по умолчанию, если не задано
    if args.run_name is None:
        # Используем дефисы в именах аргументов для авто-имени
        args.run_name = f"{args.script_name}_lr{args.lr}_kl{args.kl_weight}_rp{args.rollouts_per_step}_bs{args.train_batch_size}"

    return args

###############################################################################
# БЛОК ФУНКЦИЙ ДЛЯ РАБОТЫ С СИМУЛЯТОРОМ
###############################################################################

def rollout_simulator(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    simulator: Union[SingleWellSimulator, MultiWellSimulator],
    max_steps: int = 10,
    temperature: float = 0.7,
    top_p: float = 0.95,
    do_sample: bool = True,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[
    torch.Tensor,  # episode_tokens,
    torch.Tensor,  # action_masks
    torch.Tensor,  # rewards
    Dict,          # episode_metrics
]:
    """
    Выполняет симуляцию одного эпизода с одной скважиной.
    
    Args:
        model: Языковая модель
        tokenizer: Токенизатор
        simulator: Объект симулятора (SingleWellSimulator или MultiWellSimulator)
        max_steps: Максимальное количество шагов
        temperature: Температура генерации
        top_p: Параметр top_p для генерации
        do_sample: Выполнять ли семплирование при генерации
        seed: Случайное зерно для воспроизводимости
        verbose: Выводить ли информацию в процессе выполнения
        
    Returns:
        Кортеж из:
            episode_tokens: Тензор токенов взаимодействия
            action_masks: Маски для токенов действий
            rewards: Тензор наград
            episode_metrics: Словарь метрик эпизода
    """
    model.eval()
    device = next(model.parameters()).device

    # Инициализируем метрики эпизода
    episode_tokens = []
    episode_actions = []
    episode_rewards = []
    episode_format_rewards = []
    
    # Инициализируем буфер для всего диалога (вход-ответ-вход-ответ...)
    simulator.reset(seed=seed)
    state = simulator.get_state()
    
    # Проверяем, имеем ли мы дело с многоскважинным симулятором
    is_multi_well = hasattr(simulator, 'well_names') and len(getattr(simulator, 'well_names', [])) > 1
    
    # Сохраняем историю диалога
    history = []
    
    episode_production = 0.0
    total_reward = 0.0
    
    # Для одного роллаута не используем заготовленный шаблон PROMPT_TEMPLATE, 
    # а используем функции из модуля prompts.py
    
    if verbose:
        print("Начинаем новый эпизод")
    
    # Выполняем шаги симуляции
    for step in range(max_steps):
        # Форматируем состояние
        state_text = format_state(state, simulator)
        
        # Получаем промпт на основе текущего состояния и истории
        if not history:
            # Первый шаг эпизода
            prompt = get_first_step_prompt(state_text, is_multi_well)
        else:
            # Последующие шаги с историей
            history_text = ' | '.join(history[-2:])  # Ограничиваем 2 последними шагами
            prompt = get_subsequent_step_prompt(state_text, history_text, is_multi_well)
        
        if verbose:
            print(f"\nШаг {step+1}/{max_steps}")
            print(f"Промпт:\n{prompt}")

        # Токенизируем запрос
        tokenized_prompt = tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Сохраняем длину запроса для создания маски действий
        prompt_length = tokenized_prompt.input_ids.shape[1]
        
        # Генерируем ответ
        with torch.no_grad():
            output = model.generate(
                **tokenized_prompt,
                generation_config=GenerationConfig(
                    max_new_tokens=max_new_tokens_per_step,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                ),
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Декодируем ответ
        response_ids = output.sequences[0, prompt_length:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Парсим действие и получаем награды за форматирование
        action_value, format_rewards = parse_llm_action(response_text)
        
        print(f"Симулятор: Ответ модели: '{response_text}'")
        if action_value is not None:
            print(f"Симулятор: Действие: {action_value:.4f}")
        else:
            print(f"Симулятор: Действие: ПРОПУЩЕНО из-за неправильного формата")
        print(f"Симулятор: Награды за форматирование: {format_rewards}")
        
        # Сохраняем токены, действия, награды
        full_sequence = torch.cat([tokenized_prompt.input_ids[0], response_ids])
        episode_tokens.append(full_sequence)
        
        # Проверяем, правильный ли формат
        if action_value is not None:
            # Применяем действие к симулятору
            simulator.step(action_value)
            
            # Получаем новое состояние и награду
            new_state = simulator.state
            # Награда за шаг = добыча за текущий шаг
            step_reward = new_state[2] - state[2] 
            
            # Добавляем награды за форматирование ответа
            format_reward = sum(format_rewards.values())
            total_format_reward = format_reward
            episode_format_rewards.append(format_rewards)
            
            # Общая награда включает награду за действие и за форматирование
            full_step_reward = step_reward + total_format_reward
            
            # Обновляем общую награду и добычу
            total_reward += full_step_reward
            episode_production = new_state[2]  # Общая добыча - третий элемент в state
            
            episode_rewards.append(torch.tensor(full_step_reward))
            episode_actions.append(action_value)
            
            # Создаем маску для токенов действия
            # Все токены ответа модели (отличные от промта) помечаются как действия
            action_mask = torch.zeros_like(full_sequence, dtype=torch.bool)
            action_mask[prompt_length:] = True
            episode_action_masks.append(action_mask)
            
            # Сохраняем позиции токенов действия для более точного распределения наград
            action_positions = torch.arange(prompt_length, len(full_sequence))
            episode_action_positions.append(action_positions)
            
            print(f"Симулятор: Шаг {step+1}, Награда: {full_step_reward:.4f}, Общая награда: {total_reward:.4f}, Добыча: {episode_production:.2f} м³")
            
            # Обновляем текущее состояние и сохраняем последнее успешное состояние
            history.append(state_text)
            state = new_state
        else:
            # Если формат неправильный, даем отрицательную награду, но не выполняем шаг симулятора
            format_reward = sum(format_rewards.values())
            total_format_reward = format_reward
            episode_format_rewards.append(format_rewards)
            
            # Отрицательная награда только за формат
            full_step_reward = total_format_reward
            total_reward += full_step_reward
            
            episode_rewards.append(torch.tensor(full_step_reward))
            episode_actions.append(0.0)  # Действие не выполнялось, но в списке должно быть значение
            
            # Создаем маску для токенов действия
            action_mask = torch.zeros_like(full_sequence, dtype=torch.bool)
            action_mask[prompt_length:] = True
            episode_action_masks.append(action_mask)
            
            # Сохраняем позиции токенов действия для более точного распределения наград
            action_positions = torch.arange(prompt_length, len(full_sequence))
            episode_action_positions.append(action_positions)
            
            print(f"Симулятор: Шаг {step+1} ПРОПУЩЕН из-за неправильного формата, Штраф: {full_step_reward:.4f}, Общая награда: {total_reward:.4f}")
            skipped_steps += 1
        
        step += 1
        
        # Если слишком много пропущенных шагов подряд, прерываем эпизод
        if skipped_steps > 5:
            print(f"Симулятор: Слишком много пропущенных шагов, прерываем эпизод.")
            break
    
    # Объединяем все токены и маски для эпизода
    if episode_tokens:
        # Объединяем все последовательности в одну
        episode_full_tokens = torch.cat(episode_tokens)
        
        # Создаем полную маску действий
        episode_full_action_mask = torch.zeros(len(episode_full_tokens), dtype=torch.bool, device=device)
        
        # Заполняем маску на основе позиций
        offset = 0
        for action_mask in episode_action_masks:
            length = len(action_mask)
            episode_full_action_mask[offset:offset+length] = action_mask
            offset += length
        
        # Собираем статистику эпизода
        episode_stats = {
            "reward": total_reward,
            "production": episode_production,
            "steps": step - 1,
            "actions": episode_actions,
            "format_rewards": episode_format_rewards,
            "action_positions": episode_action_positions,
            "skipped_steps": skipped_steps
        }
        
        # Добавляем данные эпизода в общие списки
        all_episodes_tokens.append(episode_full_tokens)
        all_episodes_action_masks.append(episode_full_action_mask)
        all_episodes_rewards.append(torch.tensor(episode_rewards))
        all_episodes_stats.append(episode_stats)
    
    return all_episodes_tokens, all_episodes_action_masks, all_episodes_rewards, all_episodes_stats

def format_state(state, simulator):
    """
    Форматирует состояние симулятора для вывода в компактном виде.
    """
    # Проверяем тип симулятора (одна или несколько скважин)
    if hasattr(simulator, 'well_names') and len(simulator.well_names) > 1:
        # Для симулятора с несколькими скважинами
        result = []
        for i, well_name in enumerate(simulator.well_names):
            start_idx = i * simulator.state_dim_per_well
            # Извлекаем параметры для текущей скважины
            reservoir_pressure = state[start_idx]
            bhp = state[start_idx + 1]
            production = state[start_idx + 2]
            time = state[start_idx + 3]
            
            well_info = f"Скв.{well_name}: P={reservoir_pressure:.1f}атм, P_заб={bhp:.1f}атм, V={production:.1f}м³, t={time:.1f}д"
            
            # Добавляем информацию о текущем дебите
            if hasattr(simulator, 'current_rates') and i < len(simulator.current_rates):
                well_info += f", Q={simulator.current_rates[i]:.1f}м³/сут"
            
            result.append(well_info)
        
        return "\n".join(result)
    else:
        # Для симулятора с одной скважиной
        reservoir_pressure = state[0]
        bhp = state[1]
        production = state[2]
        time = state[3]
        
        result = f"P={reservoir_pressure:.1f}атм, P_заб={bhp:.1f}атм, V={production:.1f}м³, t={time:.1f}д"
        
        # Добавляем информацию о текущем дебите
        if hasattr(simulator, 'current_rate'):
            result += f", Q={simulator.current_rate:.1f}м³/сут"
        
        # Добавляем информацию о максимальном времени симуляции
        if hasattr(simulator, 'max_time'):
            remaining_time = simulator.max_time - time
            result += f", ост.время={remaining_time:.1f}д"
        
        return result

def parse_llm_action(response: str) -> Tuple[Optional[float], Dict[str, float]]:
    """
    Extracts action values from the LLM response.
    
    Args:
        response (str): The LLM response text.
        
    Returns:
        Tuple[Optional[float], Dict[str, float]]: A tuple containing the extracted action value (between 0 and 1 or None if format invalid)
        and a dictionary of format rewards.
    """
    try:
        # Clean up the response
        clean_response = response.strip()
        
        # If response is empty, return None to indicate invalid format
        if not clean_response:
            print(f"Пустой ответ: действие не будет выполнено")
            return None, {"empty_response": -1.0}
        
        # Проверяем правильный формат <parameter>число</parameter>
        perfect_pattern = r'<parameter>(.*?)</parameter>'
        perfect_match = re.search(perfect_pattern, clean_response, re.DOTALL)
        
        if perfect_match:
            # Extract the value inside the tags
            value_str = perfect_match.group(1).strip()
            try:
                value = float(value_str)
                # Limit the range
                value = max(0.0, min(1.0, value))
                # Идеальный формат, максимальная награда
                return value, {"parameter_format": 1.0}
            except ValueError:
                # If the content inside tags is not a number, use default and give negative reward
                print(f"Ошибка формата: тег <parameter> содержит не число: '{value_str}'. Действие не будет выполнено.")
                return None, {"parameter_not_number": -1.0}
        
        # Более гибкий паттерн для случаев с незакрытыми тегами
        flexible_pattern = r'<parameter>(.*?)(?:</parameter|</parameter>|$)'
        flexible_match = re.search(flexible_pattern, clean_response, re.DOTALL)
        
        if flexible_match:
            # Extract the value inside the tags
            value_str = flexible_match.group(1).strip()
            try:
                value = float(value_str)
                # Limit the range
                value = max(0.0, min(1.0, value))
                # Почти правильный формат, положительная но не максимальная награда
                print(f"Неполный формат тегов: {clean_response}. Действие будет выполнено, но в следующий раз используйте корректный формат.")
                return value, {"almost_parameter_format": 0.6}
            except ValueError:
                # If the content inside tags is not a number, use default and give negative reward
                print(f"Ошибка формата: неполный тег <parameter> содержит не число: '{value_str}'. Действие не будет выполнено.")
                return None, {"parameter_not_number": -1.0}
        
        # Ищем число в тексте для использования как значение (но даем отрицательную награду)
        number_pattern = r'(?:^|[^\w])(\d+(?:\.\d+)?)(?:[^\w]|$)'
        number_match = re.search(number_pattern, clean_response)
        if number_match:
            print(f"Найдено число без правильного формата: {clean_response}. Действие не будет выполнено, используйте формат <parameter>число</parameter>.")
            return None, {"wrong_format_with_number": -0.8}
        
        # Check for explicit cases of full opening/closing for value extraction
        if any(phrase in clean_response.lower() for phrase in ["fully open", "maximum open", "completely open", "полностью открыть"]):
            print(f"Найдена фраза о полном открытии, но формат неверный. Действие не будет выполнено. Используйте <parameter>1.0</parameter>")
            return None, {"wrong_format_open": -0.8}
        elif any(phrase in clean_response.lower() for phrase in ["fully close", "completely close", "close the choke", "полностью закрыть"]):
            print(f"Найдена фраза о полном закрытии, но формат неверный. Действие не будет выполнено. Используйте <parameter>0.0</parameter>")
            return None, {"wrong_format_close": -0.8}
        elif any(phrase in clean_response.lower() for phrase in ["no change", "maintain", "keep", "без изменений", "сохранить"]):
            print(f"Найдена фраза о сохранении текущего состояния, но формат неверный. Действие не будет выполнено. Используйте <parameter>0.5</parameter>")
            return None, {"wrong_format_maintain": -0.8}
        
        # If unable to extract a value, use the default
        print(f"Не удалось извлечь значение из ответа: '{clean_response}'. Действие не будет выполнено.")
        return None, {"wrong_format_default": -1.0}
    except Exception as e:
        # In case of any error, return None to indicate invalid format
        print(f"Ошибка при обработке ответа: {str(e)}. Действие не будет выполнено.")
        return None, {"error": -1.0}

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

def main():
    args = parse_args()

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
        episode_tokens, action_masks, rewards, episode_stats = parallel_rollout(
            model=model,
            tokenizer=tokenizer,
            parallel_sim=parallel_sim,
            n_steps=int(args.simulation_max_time / args.simulation_dt),  # Максимальное количество шагов
            temperature=temperature,
            verbose=True
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

            print(f"{COLOR_RED}Не удалось собрать ни одного эпизода. Пропускаем шаг обучения.{COLOR_RESET}")
            # Увеличиваем счетчик глобальных шагов и продолжаем
            global_step += 1
            continue
        
        # Преобразуем списки в тензоры
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

def format_short_state(state):
    """
    Форматирует состояние скважины в компактном виде для истории взаимодействий.
    
    Args:
        state: Текущее состояние симулятора
    
    Returns:
        str: Короткое описание состояния
    """
    if len(state) >= 4:
        # Базовое состояние [pressure, flow_rate, production, time]
        return f"P={state[0]:.1f}атм, Q={state[1]:.1f}м³/сут, V={state[2]:.1f}м³, t={state[3]:.1f}д"
    else:
        # Если формат состояния неизвестен, возвращаем просто числа
        return ", ".join([f"{x:.1f}" for x in state])


if __name__ == "__main__":
    main()
