#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Тестовый скрипт для проверки аргументов командной строки в скриптах запуска
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

# Добавляем корневую директорию проекта в путь поиска модулей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def test_single_well_script():
    """Тестирует скрипт запуска обучения для одной скважины"""
    print("\n--- Тестирование запуска одиночной скважины ---")
    cmd = [
        "python", "experiments/run_single_well_grpo.py",
        "--model_name", "Qwen/Qwen2.5-3B-Instruct",
        "--use_discrete_actions",
        "--total_steps", "1",
        "--rollouts_per_step", "1",
        "--wandb_project", "test_discrete_actions"
    ]
    
    print(f"Выполнение команды: {' '.join(cmd)}")
    
    try:
        # Запускаем команду с флагом check=True для проверки результата
        # Перенаправляем вывод в нулевое устройство, так как нас интересует только код возврата
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ Тест успешно пройден! Скрипт одиночной скважины корректно принимает аргумент --use_discrete_actions")
    except subprocess.CalledProcessError as e:
        print("❌ Тест не пройден! Произошла ошибка при запуске скрипта одиночной скважины")
        print(f"Код ошибки: {e.returncode}")
        print(f"Вывод: {e.stdout.decode('utf-8')}")
        print(f"Ошибка: {e.stderr.decode('utf-8')}")

def test_multi_well_script():
    """Тестирует скрипт запуска обучения для нескольких скважин"""
    print("\n--- Тестирование запуска нескольких скважин ---")
    cmd = [
        "python", "experiments/run_multi_well_grpo.py",
        "--model_name", "Qwen/Qwen2.5-3B-Instruct",
        "--use_discrete_actions",
        "--total_steps", "1",
        "--rollouts_per_step", "1",
        "--n_wells", "2",
        "--wandb_project", "test_discrete_actions"
    ]
    
    print(f"Выполнение команды: {' '.join(cmd)}")
    
    try:
        # Запускаем команду с флагом check=True для проверки результата
        # Перенаправляем вывод в нулевое устройство, так как нас интересует только код возврата
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ Тест успешно пройден! Скрипт для нескольких скважин корректно принимает аргумент --use_discrete_actions")
    except subprocess.CalledProcessError as e:
        print("❌ Тест не пройден! Произошла ошибка при запуске скрипта нескольких скважин")
        print(f"Код ошибки: {e.returncode}")
        print(f"Вывод: {e.stdout.decode('utf-8')}")
        print(f"Ошибка: {e.stderr.decode('utf-8')}")

def main():
    parser = argparse.ArgumentParser(description="Тестирование аргументов командной строки в скриптах запуска")
    parser.add_argument('--single', action='store_true', help='Тестировать только скрипт одиночной скважины')
    parser.add_argument('--multi', action='store_true', help='Тестировать только скрипт нескольких скважин')
    args = parser.parse_args()
    
    print("Тестирование аргумента --use_discrete_actions")
    
    # Если ни один из флагов не указан, тестируем оба скрипта
    if not args.single and not args.multi:
        test_single_well_script()
        test_multi_well_script()
    else:
        if args.single:
            test_single_well_script()
        if args.multi:
            test_multi_well_script()
    
    print("\nТестирование завершено!")

if __name__ == "__main__":
    main() 