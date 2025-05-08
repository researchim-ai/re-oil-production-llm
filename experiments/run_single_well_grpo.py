#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для запуска обучения LLM-модели с использованием GRPO для оптимизации 
добычи нефти из одиночной скважины
"""

import sys
import os

# Добавляем корневую директорию проекта в путь поиска модулей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Запуск обучения для оптимизации нефтяной скважины с GRPO")
    
    # Общие аргументы эксперимента
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Имя модели или путь к модели")
    parser.add_argument('--run_name', type=str, default=None,
                        help="Название запуска (опционально, по умолчанию генерируется)")
    
    # Аргументы обучения
    parser.add_argument('--total_steps', type=int, default=500,
                        help="Общее количество шагов оптимизации")
    parser.add_argument('--rollouts_per_step', type=int, default=16,
                        help="Количество эпизодов симуляции на один шаг обучения")
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help="Размер батча для обучения")
    parser.add_argument('--lr', type=float, default=1e-5,
                        help="Скорость обучения")
    parser.add_argument('--kl_weight', type=float, default=0.1,
                        help="Вес KL-дивергенции в функции потерь")
    parser.add_argument('--clip_eps', type=float, default=0.2,
                        help="Параметр отсечения для PPO")
    parser.add_argument('--use_discrete_actions', action='store_true',
                        help="Использовать дискретные действия (1-10) вместо непрерывных (0-1)")
    parser.add_argument('--forecast_days', type=int, default=30,
                        help="Количество дней для прогнозирования на каждом шаге симуляции")
    
    # Аргументы логирования и чекпоинтов
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/single_well",
                        help="Путь для сохранения чекпоинтов")
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help="Интервал сохранения чекпоинтов (в шагах)")
    parser.add_argument('--log_completions_interval', type=int, default=5,
                        help="Интервал логирования примеров (в шагах)")
    parser.add_argument('--wandb', action='store_true',
                        help="Использовать WandB для логирования")
    parser.add_argument('--wandb_project', type=str, default="tiny_grpo",
                        help="Название проекта в WandB")
    
    # Аргументы симуляции
    parser.add_argument('--initial_pressure', type=float, default=200.0,
                        help="Начальное давление в пласте (атм)")
    parser.add_argument('--initial_bhp', type=float, default=50.0,
                        help="Начальное забойное давление (атм)")
    parser.add_argument('--productivity_index', type=float, default=0.5,
                        help="Индекс продуктивности (м³/сут/атм)")
    parser.add_argument('--total_volume', type=float, default=10000.0,
                        help="Общий объем пласта (м³)")
    parser.add_argument('--simulation_dt', type=float, default=1.0,
                        help="Шаг симуляции (дней)")
    parser.add_argument('--simulation_max_time', type=float, default=365.0,
                        help="Максимальное время симуляции (дней)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Формируем команду для запуска обучения
    cmd = [
        "python", "-m", "grpo.trainer",
        f"--model_name", args.model_name,
        f"--checkpoint_path", args.checkpoint_path,
        f"--total_steps", str(args.total_steps),
        f"--rollouts_per_step", str(args.rollouts_per_step),
        f"--train_batch_size", str(args.train_batch_size),
        f"--lr", str(args.lr),
        f"--kl_weight", str(args.kl_weight),
        f"--clip_eps", str(args.clip_eps),
        f"--checkpoint_interval", str(args.checkpoint_interval),
        f"--log_completions_interval", str(args.log_completions_interval),
        f"--initial_pressure", str(args.initial_pressure),
        f"--initial_bhp", str(args.initial_bhp),
        f"--productivity_index", str(args.productivity_index),
        f"--total_volume", str(args.total_volume),
        f"--simulation_dt", str(args.simulation_dt),
        f"--simulation_max_time", str(args.simulation_max_time),
        f"--forecast_days", str(args.forecast_days),
    ]
    
    # Добавляем опцию дискретных действий, если указана
    if args.use_discrete_actions:
        cmd.append("--use_discrete_actions")
    
    # Добавляем имя запуска, если задано
    if args.run_name:
        cmd.extend(["--run_name", args.run_name])
    else:
        # Формируем имя запуска на основе модели и параметров
        model_short_name = args.model_name.split("/")[-1].lower()
        run_name = f"oil_sim_{model_short_name}_run1"
        cmd.extend(["--run_name", run_name])
    
    # Добавляем WandB, если включено
    if args.wandb:
        cmd.append("--wandb")
        cmd.extend(["--wandb_project", args.wandb_project])
    
    # Запускаем команду
    print("Запуск команды:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
