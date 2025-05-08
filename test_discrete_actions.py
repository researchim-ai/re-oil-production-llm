#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Тестовый скрипт для проверки системы дискретных действий
"""

import argparse
import torch
import numpy as np

from grpo.utils import parse_llm_action, format_state, DISCRETE_ACTIONS
from simulators.single_well.simulator import SingleWellSimulator

def main():
    print("Тестирование системы дискретных действий")
    print(f"Доступные действия: {DISCRETE_ACTIONS}")
    
    # Создаем симулятор для тестирования
    simulator = SingleWellSimulator(
        initial_reservoir_pressure=200.0,
        initial_bhp=50.0,
        productivity_index=0.5,
        total_volume=10000.0,
        dt=30.0,
        max_time=365.0
    )
    
    # Тестовые ответы моделей
    test_responses = [
        "1",
        "5",
        "10",
        "Выбираю вариант 3",
        "Оптимальным будет вариант 7",
        "Думаю, что 2 будет лучшим выбором",
        "0.5",
        "0.9",
        "1.0",
        "Изменяю штуцер на 0.3",
        "штуцер = 0.7",
        "<parameter>0.4</parameter>",
        "Хочу установить значение дебита 150",
        ""
    ]
    
    print("\nТестирование различных вариантов ответов:")
    for i, response in enumerate(test_responses):
        print(f"\n{i+1}. Ответ модели: '{response}'")
        action, rewards = parse_llm_action(response)
        
        if action is not None:
            # Устанавливаем действие в симулятор
            simulator.last_action = action
            
            # Получаем текущее состояние
            state = simulator.reset()
            
            # Форматируем состояние
            state_str = format_state(state, simulator)
            print(f"Отформатированное состояние: {state_str}")
            
            # Применяем действие
            new_state, reward, done, info = simulator.step(action)
            print(f"Результат после применения {action}: reward={reward:.2f}")
        else:
            print("Действие не распознано или некорректно.")
    
    # Тестируем backward совместимость
    print("\nТестирование backward совместимости:")
    
    # Создаем новый симулятор
    simulator = SingleWellSimulator(
        initial_reservoir_pressure=200.0,
        initial_bhp=50.0,
        productivity_index=0.5,
        total_volume=10000.0,
        dt=30.0,
        max_time=365.0
    )
    
    # Тест на совместимость с старым форматом
    old_format_response = "<parameter>0.65</parameter>"
    print(f"Ответ в старом формате: '{old_format_response}'")
    action, rewards = parse_llm_action(old_format_response)
    
    if action is not None:
        # Устанавливаем действие в симулятор
        simulator.last_action = action
        
        # Получаем текущее состояние
        state = simulator.reset()
        
        # Форматируем состояние
        state_str = format_state(state, simulator)
        print(f"Отформатированное состояние: {state_str}")
        
        # Применяем действие
        new_state, reward, done, info = simulator.step(action)
        print(f"Результат после применения {action}: reward={reward:.2f}")
    else:
        print("Действие не распознано или некорректно.")

if __name__ == "__main__":
    main() 