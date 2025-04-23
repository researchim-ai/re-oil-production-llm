#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для обучения модели с использованием промптов с рассуждениями (reasoning).
Демонстрирует, как использовать расширенный формат промптов с тегами <reasoning>.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Добавляем родительский каталог в путь для импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grpo.trainer import main as grpo_main, parse_args as grpo_parse_args

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/train_reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Обучение модели с использованием промптов <reasoning>")
    
    # Параметры модели
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Путь или имя модели для загрузки")
    parser.add_argument("--output_dir", type=str, default="checkpoints_oil_sim/reasoning",
                        help="Директория для сохранения модели")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Максимальная длина последовательности токенов")
    
    # Параметры симулятора
    parser.add_argument("--simulator_type", type=str, choices=["single", "multi"], default="single",
                        help="Тип симулятора: одна скважина или несколько")
    parser.add_argument("--n_wells", type=int, default=3,
                        help="Количество скважин (только для multi)")
    parser.add_argument("--max_time", type=float, default=4.0,
                        help="Максимальное время симуляции (дней)")
    
    # Параметры обучения
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Скорость обучения")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Размер батча для обучения")
    parser.add_argument("--grad_accumulation_steps", type=int, default=4,
                        help="Количество шагов накопления градиента")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Максимальное количество шагов обучения")
    parser.add_argument("--rollout_n_sequences", type=int, default=5,
                        help="Количество эпизодов в роллауте")
    parser.add_argument("--rollout_steps", type=int, default=20,
                        help="Максимальное количество шагов в роллауте")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Частота сохранения модели")
    
    # Параметры для параллельного роллаута
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Температура для генерации при роллауте")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Параметр top_p для генерации при роллауте")
    
    # Параметры GRPO
    parser.add_argument("--kl_weight", type=float, default=0.01,
                        help="Вес для KL дивергенции в функции потерь")
    parser.add_argument("--clip_eps", type=float, default=0.2,
                        help="Эпсилон для отсечения в GRPO")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Коэффициент дисконтирования для расчета возвратов")
    
    # Параметры для wandb
    parser.add_argument("--use_wandb", action="store_true",
                        help="Использовать wandb для логирования")
    parser.add_argument("--wandb_project", type=str, default="oil-sim-reasoning",
                        help="Имя проекта в wandb")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Entity в wandb")
    
    # Дополнительные параметры
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed для воспроизводимости")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Использовать 4-битное квантование")
    parser.add_argument("--verbose", action="store_true",
                        help="Выводить подробную информацию")
                        
    # Параметр для выбора типа промпта
    parser.add_argument("--prompt_type", type=str, choices=["standard", "reasoning"], default="reasoning",
                        help="Тип промпта для обучения")
    
    # Параметр для выбора индекса устройства (GPU)
    parser.add_argument("--device_index", type=int, default=0,
                        help="Индекс GPU для запуска (если доступно несколько)")
    
    return parser.parse_args()

def main():
    """
    Главная функция запуска обучения. Просто конвертирует аргументы из одного формата
    в формат, понятный grpo_main, и запускает основной цикл обучения.
    """
    args = parse_args()
    
    # Отладочная информация о параметрах
    logger.info(f"Максимальное время симуляции: {args.max_time} дней")
    logger.info(f"Модель: {args.model_name_or_path}")
    logger.info(f"Тип симулятора: {args.simulator_type}")
    
    # Создаем директорию для сохранения результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Формируем имя запуска
    run_name = f"reasoning_{args.simulator_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Проверяем, что max_time задано корректно
    if args.max_time <= 0:
        logger.error("Максимальное время симуляции должно быть положительным числом")
        return
    
    # Создаем список аргументов в формате, понятном grpo_parse_args
    grpo_args_list = [
        "--model_name", args.model_name_or_path,
        "--checkpoint_path", args.output_dir,
        "--total_steps", str(args.max_steps),
        "--checkpoint_interval", str(args.save_every),
        "--lr", str(args.lr),
        "--train_batch_size", str(args.batch_size),
        "--epochs_per_step", str(args.grad_accumulation_steps),
        "--max_length", str(args.max_length),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--seed", str(args.seed),
        "--rollouts_per_step", str(args.rollout_n_sequences),
        "--max_new_tokens_per_step", str(args.rollout_steps),
        "--prompt_type", args.prompt_type,
        "--device_index", str(args.device_index),
        
        # Параметры симулятора
        "--initial_pressure", "200.0",
        "--initial_bhp", "50.0",
        "--productivity_index", "0.1",
        "--total_volume", "1000000.0",  # 1e6
        "--simulation_dt", "1.0",
        "--simulation_max_time", str(args.max_time),
        "--forecast_days", "1.0",
        
        # Параметры GRPO
        "--kl_weight", str(args.kl_weight),
        "--clip_eps", str(args.clip_eps),
        "--gamma", str(args.gamma),
        
        # Параметры логирования
        "--log_dir", "logs",
        "--script_name", "train_with_reasoning",
        "--log_completions_interval", "10",
        "--run_name", run_name
    ]
    
    # Добавляем флаги, если они включены
    if args.verbose:
        grpo_args_list.append("--verbose")
    
    if args.simulator_type == "multi":
        grpo_args_list.extend([
            "--multi_well", 
            "--n_wells", str(args.n_wells),
            "--interaction_strength", "0.2",
            "--shared_reservoir"
        ])
    
    if args.use_wandb:
        grpo_args_list.extend(["--wandb", "--wandb_project", args.wandb_project])
        if args.wandb_entity:
            grpo_args_list.extend(["--wandb_entity", args.wandb_entity])
    
    if args.use_4bit:
        grpo_args_list.append("--use_4bit")
    
    # Запускаем обучение
    logger.info("Запускаем обучение с GRPO...")
    logger.info(f"Передаваемые аргументы: {' '.join(grpo_args_list)}")
    
    grpo_args = grpo_parse_args(grpo_args_list)
    grpo_main(grpo_args)
    
    logger.info("Обучение завершено!")

if __name__ == "__main__":
    main() 