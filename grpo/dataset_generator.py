#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Импортируем симуляторы
from simulators.single_well.simulator import SingleWellSimulator
from simulators.multi_well.simulator import MultiWellSimulator

# Импортируем утилиты и константы
from grpo.utils import DISCRETE_ACTIONS, COLOR_RESET, COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_BLUE, COLOR_CYAN


def generate_random_state(
    simulator_type: str,
    min_depletion: float = 0.0, 
    max_depletion: float = 0.8,
    use_realistic_ranges: bool = True,
    **simulator_params
) -> Dict:
    """
    Генерирует случайное состояние для симулятора.
    
    Args:
        simulator_type: Тип симулятора ('single_well' или 'multi_well')
        min_depletion: Минимальное истощение резервуара (от 0 до 1)
        max_depletion: Максимальное истощение резервуара (от 0 до 1)
        use_realistic_ranges: Использовать реалистичные диапазоны параметров
        simulator_params: Дополнительные параметры для создания симулятора
        
    Returns:
        Dict: Словарь с параметрами состояния
    """
    # Базовые параметры симулятора (можно переопределить через simulator_params)
    base_params = {
        'initial_reservoir_pressure': 200.0,  # атм
        'initial_bhp': 50.0,  # атм
        'productivity_index': 0.1,  # м3/день/атм
        'total_volume': 1e6,  # м3
        'dt': 1.0,            # дней
        'max_time': 365.0     # дней
    }
    
    # Обновляем базовые параметры переданными аргументами
    params = {**base_params, **simulator_params}
    
    # Генерируем случайный уровень истощения
    depletion_ratio = random.uniform(min_depletion, max_depletion)
    
    if simulator_type == 'single_well':
        # Создаем случайное состояние для одиночной скважины
        if use_realistic_ranges:
            # Реалистичные диапазоны параметров
            params['initial_reservoir_pressure'] = random.uniform(150.0, 300.0)
            params['initial_bhp'] = random.uniform(30.0, 80.0)
            params['productivity_index'] = random.uniform(0.05, 0.2)
        
        # Создаем симулятор с начальными параметрами
        simulator = SingleWellSimulator(**params)
        
        # Используем встроенный метод для создания случайного состояния симулятора
        simulator.reset_to_random_state(
            min_depletion=min_depletion,
            max_depletion=max_depletion,
            use_realistic_ranges=use_realistic_ranges
        )
        
        # Добавляем деплеция для совместимости с нашими функциями
        if 'depletion_ratio' not in locals():
            depletion_ratio = simulator.cumulative_production / simulator.total_volume if simulator.total_volume > 1e-9 else 1.0
        
        # Получаем текущий дебит
        delta_p = max(0.0, simulator.reservoir_pressure - simulator.bhp)
        flow_rate = simulator.pi * delta_p * simulator.current_valve_opening
        
        # Возвращаем состояние и параметры
        return {
            'simulator_params': params,
            'state': {
                'reservoir_pressure': simulator.reservoir_pressure,
                'current_time': simulator.time,
                'cumulative_production': simulator.cumulative_production,
                'flow_rate': flow_rate,
                'depletion_ratio': depletion_ratio
            }
        }
        
    elif simulator_type == 'multi_well':
        # Дополнительные параметры для мультискважинного симулятора
        multi_params = {
            'n_wells': 3,
            'interaction_strength': 0.1,
            'shared_reservoir': True
        }
        
        # Обновляем параметры
        params.update({k: v for k, v in multi_params.items() 
                      if k not in simulator_params})
                      
        if use_realistic_ranges:
            # Реалистичные диапазоны параметров
            params['initial_reservoir_pressure'] = random.uniform(150.0, 300.0)
            params['initial_bhp'] = random.uniform(30.0, 80.0)
            params['productivity_index'] = random.uniform(0.05, 0.2)
            params['interaction_strength'] = random.uniform(0.05, 0.3)
        
        # Создаем симулятор с начальными параметрами
        simulator = MultiWellSimulator(**params)
        
        # Используем встроенный метод reset_to_random_state, если он доступен
        try:
            simulator.reset_to_random_state(
                min_depletion=min_depletion,
                max_depletion=max_depletion,
                use_realistic_ranges=use_realistic_ranges
            )
        except Exception as e:
            print(f"Предупреждение: не удалось использовать reset_to_random_state: {e}")
            # Продолжаем с обычным сбросом
            simulator.reset()
        
        # Извлекаем информацию о состоянии скважин из вектора состояния
        n_wells = params.get('n_wells', 3)
        state_vector = simulator.state
        state_size_per_well = len(state_vector) // n_wells
        
        # Получаем состояния отдельных скважин
        well_states = []
        for i in range(n_wells):
            start_idx = i * state_size_per_well
            well_state = state_vector[start_idx:start_idx+state_size_per_well]
            
            # Распаковываем состояние скважины
            reservoir_pressure = well_state[0]
            flow_rate = well_state[1]
            cumulative_production = well_state[2]
            time = well_state[3]
            
            well_states.append({
                'reservoir_pressure': reservoir_pressure,
                'cumulative_production': cumulative_production,
                'flow_rate': flow_rate
            })
        
        # Общая информация о месторождении
        total_cumulative_production = sum(state_vector[start_idx+2] for start_idx in range(0, len(state_vector), state_size_per_well))
        total_depletion_ratio = total_cumulative_production / params['total_volume'] if params['total_volume'] > 1e-9 else 1.0
        
        # Определяем общее пластовое давление для общего резервуара
        # или None для независимых резервуаров
        shared_reservoir_pressure = None
        if params.get('shared_reservoir', True):
            # В случае общего резервуара берем давление первой скважины
            shared_reservoir_pressure = state_vector[0]  # Первый элемент первой скважины - давление
        
        # Возвращаем состояние и параметры
        return {
            'simulator_params': params,
            'state': {
                'reservoir_pressure': shared_reservoir_pressure,
                'current_time': state_vector[3],  # Время первой скважины (одинаково для всех)
                'cumulative_production': total_cumulative_production,
                'depletion_ratio': total_depletion_ratio,
                'wells': well_states
            }
        }
    else:
        raise ValueError(f"Неизвестный тип симулятора: {simulator_type}")


def find_optimal_action(
    simulator_type: str,
    state: Dict,
    simulator_params: Dict,
    simulation_steps: int = 365,
    action_space: List[float] = DISCRETE_ACTIONS,
    verbose: bool = False
) -> Dict:
    """
    Находит оптимальное действие для заданного состояния симулятора.
    
    Args:
        simulator_type: Тип симулятора ('single_well' или 'multi_well')
        state: Словарь с текущим состоянием симулятора
        simulator_params: Параметры для создания симулятора
        simulation_steps: Количество шагов симуляции для оценки действий
        action_space: Пространство действий (возможные значения штуцера)
        verbose: Выводить подробную информацию
    
    Returns:
        Dict: Словарь с оптимальным действием и результатами
    """
    # Создаем копию параметров
    params = simulator_params.copy()
    
    # Инициализируем переменные для отслеживания лучшего результата
    best_action = None
    best_production = -float('inf')
    best_results = None
    
    # Перебираем все возможные действия
    for action_value in action_space:
        # Создаем симулятор для каждого действия
        if simulator_type == 'single_well':
            simulator = SingleWellSimulator(**params)
            
            # Устанавливаем состояние
            simulator.reservoir_pressure = state['reservoir_pressure']
            simulator.cumulative_production = state['cumulative_production']
            simulator.time = state['current_time']
            
            # Запускаем симуляцию на заданное количество шагов
            start_production = simulator.cumulative_production
            for _ in range(simulation_steps):
                # Передаем действие в метод step
                _, reward, done, _ = simulator.step(action_value)
                # Если скважина истощилась, прерываем симуляцию
                if done:
                    break
            
            # Вычисляем добычу за период
            production = simulator.cumulative_production - start_production
            
            # Проверяем, является ли это действие лучшим
            if production > best_production:
                best_production = production
                best_action = action_value
                best_results = {
                    'final_reservoir_pressure': simulator.reservoir_pressure,
                    'final_cumulative_production': simulator.cumulative_production,
                    'production_delta': production,
                    'steps_simulated': simulation_steps,
                    'is_depleted': simulator.reservoir_pressure <= simulator.bhp
                }
            
            if verbose:
                print(f"Действие {action_value:.1f}: Добыча {production:.2f} м³")
                
        elif simulator_type == 'multi_well':
            simulator = MultiWellSimulator(**params)
            
            # Получаем данные о состоянии симулятора
            n_wells = params.get('n_wells', 3)
            
            # Сначала выполняем reset(), чтобы гарантировать корректное состояние
            simulator.reset()
            
            # Теперь устанавливаем состояние из state для всех скважин
            state_vector = []
            
            if params.get('shared_reservoir', True) and state['reservoir_pressure'] is not None:
                # Для общего резервуара устанавливаем одинаковое давление для всех скважин
                reservoir_pressure = state['reservoir_pressure']
                for i in range(n_wells):
                    well_state = state['wells'][i]
                    # Для каждой скважины: [давление, деб, добыча, время]
                    well_vector = [
                        reservoir_pressure,
                        well_state['flow_rate'],
                        well_state['cumulative_production'],
                        state['current_time']
                    ]
                    state_vector.extend(well_vector)
            else:
                # Для независимых резервуаров берем давление из состояния каждой скважины
                for i in range(n_wells):
                    well_state = state['wells'][i]
                    well_vector = [
                        well_state['reservoir_pressure'],
                        well_state['flow_rate'],
                        well_state['cumulative_production'],
                        state['current_time']
                    ]
                    state_vector.extend(well_vector)
            
            # Устанавливаем вектор состояния
            simulator.state = np.array(state_vector)
            
            # Запускаем симуляцию на заданное количество шагов
            start_production = state['cumulative_production']
            for _ in range(simulation_steps):
                # Одинаковое действие для всех скважин
                actions = [action_value] * n_wells
                _, reward, done, _ = simulator.step(actions)
                # Если симуляция завершена, прерываем
                if done:
                    break
            
            # Получаем обновленное состояние симулятора
            final_state_vector = simulator.state
            state_size_per_well = len(final_state_vector) // n_wells
            
            # Вычисляем суммарную добычу по всем скважинам
            final_production = 0
            for i in range(n_wells):
                start_idx = i * state_size_per_well
                final_production += final_state_vector[start_idx + 2]  # Индекс 2 - cumulative_production
            
            # Вычисляем добычу за период
            production = final_production - start_production
            
            # Проверяем, является ли это действие лучшим
            if production > best_production:
                best_production = production
                best_action = action_value
                
                # Собираем информацию о состоянии скважин
                well_states = []
                for i in range(n_wells):
                    start_idx = i * state_size_per_well
                    well_vector = final_state_vector[start_idx:start_idx + state_size_per_well]
                    
                    well_states.append({
                        'final_reservoir_pressure': well_vector[0],
                        'final_cumulative_production': well_vector[2],
                        'is_depleted': well_vector[0] <= params.get('initial_bhp', 50.0)
                    })
                
                # Определяем общее пластовое давление для общего резервуара
                final_shared_pressure = None
                if params.get('shared_reservoir', True):
                    # В случае общего резервуара берем давление первой скважины
                    final_shared_pressure = final_state_vector[0]
                
                best_results = {
                    'final_reservoir_pressure': final_shared_pressure,
                    'final_cumulative_production': final_production,
                    'production_delta': production,
                    'steps_simulated': simulation_steps,
                    'wells': well_states
                }
            
            if verbose:
                print(f"Действие {action_value:.1f}: Добыча {production:.2f} м³")
                
        else:
            raise ValueError(f"Неизвестный тип симулятора: {simulator_type}")
    
    if verbose:
        print(f"\n{COLOR_GREEN}Оптимальное действие: {best_action:.1f} → Добыча: {best_production:.2f} м³{COLOR_RESET}")
    
    # Индекс оптимального действия (для дискретизации в обучении)
    action_index = action_space.index(best_action) + 1  # +1 для соответствия нумерации от 1 до 10
    
    # Формируем результаты
    result = {
        'optimal_action': best_action,
        'optimal_action_index': action_index,
        'expected_production': best_production,
        'simulation_details': best_results
    }
    
    return result


def format_prompt(
    simulator_type: str,
    state: Dict,
    forecast_days: int = 1,
    weekly_note: str = "",
    monthly_note: str = ""
) -> str:
    """
    Форматирует промпт для модели на основе состояния скважины.
    
    Args:
        simulator_type: Тип симулятора ('single_well' или 'multi_well')
        state: Словарь с текущим состоянием симулятора
        forecast_days: На сколько дней вперед делается прогноз
        weekly_note: Дополнительное примечание для недельных прогнозов
        monthly_note: Дополнительное примечание для месячных прогнозов
        
    Returns:
        str: Отформатированный промпт
    """
    if simulator_type == 'single_well':
        # Форматируем промпт для одиночной скважины
        prompt = f"""Состояние скважины на день {int(state['current_time'])}:
- Давление в пласте: {state['reservoir_pressure']:.2f} атм
- Текущий дебит: {state['flow_rate']:.2f} м³/сут
- Накопленная добыча: {state['cumulative_production']:.2f} м³
- Прошедшее время: {state['current_time']:.1f} дней
- Истощение резервуара: {state['depletion_ratio']*100:.1f}%

Выберите степень открытия штуцера на следующие {forecast_days} дней, указав ОДНО ЧИСЛО от 1 до 10."""

    elif simulator_type == 'multi_well':
        # Формируем промпт для мультискважинного симулятора
        wells_info = []
        for i, well in enumerate(state['wells']):
            wells_info.append(f"""Скважина #{i+1}:
- Давление в пласте: {well['reservoir_pressure']:.2f} атм
- Текущий дебит: {well['flow_rate']:.2f} м³/сут
- Накопленная добыча: {well['cumulative_production']:.2f} м³""")
        
        wells_text = "\n\n".join(wells_info)
        
        # Общая информация о месторождении
        field_info = f"""Общая информация о месторождении:
- Общее пластовое давление: {state['reservoir_pressure']:.2f} атм
- Общая накопленная добыча: {state['cumulative_production']:.2f} м³
- Прошедшее время: {state['current_time']:.1f} дней
- Истощение резервуара: {state['depletion_ratio']*100:.1f}%

"""
        
        # Если нет общего резервуара, убираем строку с общим давлением
        if state['reservoir_pressure'] is None:
            field_info = field_info.replace(f"- Общее пластовое давление: {state['reservoir_pressure']:.2f} атм\n", "")
        
        prompt = f"""{field_info}{wells_text}

Выберите ЕДИНУЮ степень открытия штуцера для ВСЕХ скважин на следующие {forecast_days} дней, указав ОДНО ЧИСЛО от 1 до 10."""
    
    else:
        raise ValueError(f"Неизвестный тип симулятора: {simulator_type}")
    
    return prompt


def generate_dataset(
    output_file: str,
    num_samples: int = 1000,
    simulator_type: str = 'single_well',
    min_depletion: float = 0.0,
    max_depletion: float = 0.8,
    use_realistic_ranges: bool = True,
    simulation_steps: int = 30,
    forecast_days: int = 1,
    verbose: bool = False,
    **simulator_params
) -> None:
    """
    Генерирует датасет с состояниями скважин и оптимальными действиями.
    
    Args:
        output_file: Путь к файлу для сохранения датасета
        num_samples: Количество генерируемых примеров
        simulator_type: Тип симулятора ('single_well' или 'multi_well')
        min_depletion: Минимальное истощение резервуара (от 0 до 1)
        max_depletion: Максимальное истощение резервуара (от 0 до 1)
        use_realistic_ranges: Использовать реалистичные диапазоны параметров
        simulation_steps: Количество шагов симуляции для оценки действий
        forecast_days: На сколько дней вперед делается прогноз
        verbose: Выводить подробную информацию
        simulator_params: Дополнительные параметры для создания симулятора
    """
    # Настраиваем параметры симулятора на основе forecast_days
    sim_params = simulator_params.copy()
    sim_params['dt'] = forecast_days
    
    # Добавляем max_time, если он не указан
    if 'max_time' not in sim_params:
        sim_params['max_time'] = 365.0  # Значение по умолчанию
    
    # Создаем список для хранения данных
    dataset = []
    
    # Генерируем примеры
    print(f"{COLOR_BLUE}Генерация {num_samples} примеров для {simulator_type}...{COLOR_RESET}")
    
    # Создаем путь к директории datasets и добавляем временную метку к имени файла
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Проверяем, включает ли путь уже директорию datasets
    if not output_file.startswith('datasets/'):
        # Если путь не содержит директорию datasets, добавляем её
        # Также проверяем, является ли путь просто именем файла или полным путем
        base_filename = os.path.basename(output_file)
        name_without_ext, ext = os.path.splitext(base_filename)
        if not ext:  # Если расширение не указано, добавляем .json
            ext = '.json'
        # Формируем новое имя файла с временной меткой
        new_filename = f"{name_without_ext}_{timestamp}{ext}"
        output_file = os.path.join('datasets', new_filename)
    else:
        # Если путь уже содержит datasets, просто добавляем временную метку перед расширением
        dirname = os.path.dirname(output_file)
        basename = os.path.basename(output_file)
        name_without_ext, ext = os.path.splitext(basename)
        if not ext:  # Если расширение не указано, добавляем .json
            ext = '.json'
        # Формируем новое имя файла с временной меткой
        new_filename = f"{name_without_ext}_{timestamp}{ext}"
        output_file = os.path.join(dirname, new_filename)
    
    # Создаем директорию, если её нет
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Устанавливаем seed для воспроизводимости
    random.seed(42)
    np.random.seed(42)
    
    # Используем tqdm для прогресс-бара
    for i in tqdm(range(num_samples)):
        # Генерируем случайное состояние
        state_info = generate_random_state(
            simulator_type=simulator_type,
            min_depletion=min_depletion,
            max_depletion=max_depletion,
            use_realistic_ranges=use_realistic_ranges,
            **sim_params
        )
        
        # Находим оптимальное действие
        optimal_result = find_optimal_action(
            simulator_type=simulator_type,
            state=state_info['state'],
            simulator_params=state_info['simulator_params'],
            simulation_steps=simulation_steps,
            verbose=verbose and (i % 100 == 0)  # Подробный вывод каждые 100 примеров
        )
        
        # Создаем промпт
        prompt = format_prompt(
            simulator_type=simulator_type,
            state=state_info['state'],
            forecast_days=forecast_days
        )
        
        # Формируем пример для датасета
        example = {
            'prompt': prompt,
            'state': state_info['state'],
            'simulator_params': state_info['simulator_params'],
            'optimal_action': optimal_result['optimal_action'],
            'optimal_action_index': optimal_result['optimal_action_index'],
            'expected_production': optimal_result['expected_production'],
            'metadata': {
                'simulator_type': simulator_type,
                'forecast_days': forecast_days,
                'simulation_steps': simulation_steps,
                'created_at': timestamp
            }
        }
        
        # Добавляем пример в датасет
        dataset.append(example)
        
        # Периодически сохраняем датасет (каждые 100 примеров)
        if (i + 1) % 100 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # Сохраняем финальный датасет
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"{COLOR_GREEN}Датасет успешно сгенерирован: {output_file}{COLOR_RESET}")
    print(f"Количество примеров: {len(dataset)}")
    
    # Создаем CSV-версию с основными данными для удобства просмотра
    csv_file = output_path.with_suffix('.csv')
    
    # Преобразуем данные для CSV
    csv_data = []
    
    for example in dataset:
        csv_row = {
            'depletion_ratio': example['state']['depletion_ratio'],
            'reservoir_pressure': example['state']['reservoir_pressure'] if simulator_type == 'single_well' or example['state']['reservoir_pressure'] is not None else 'multiple',
            'current_time': example['state']['current_time'],
            'cumulative_production': example['state']['cumulative_production'],
            'optimal_action': example['optimal_action'],
            'optimal_action_index': example['optimal_action_index'],
            'expected_production': example['expected_production']
        }
        
        # Добавляем информацию о потоке для одиночной скважины
        if simulator_type == 'single_well':
            csv_row['flow_rate'] = example['state']['flow_rate']
        
        csv_data.append(csv_row)
    
    # Сохраняем CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    
    print(f"CSV-версия сохранена: {csv_file}")
    
    # Возвращаем путь к созданному файлу для удобства
    return output_file


def parse_args():
    """Обрабатывает аргументы командной строки."""
    parser = argparse.ArgumentParser(description='Генерация датасетов для обучения LLM управлению нефтяными скважинами')
    
    # Основные параметры
    parser.add_argument('--output', type=str, required=True, help='Имя файла для сохранения датасета (будет сохранен в директории datasets/)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Количество генерируемых примеров')
    parser.add_argument('--simulator_type', type=str, choices=['single_well', 'multi_well'], default='single_well',
                      help='Тип симулятора (одиночная или несколько скважин)')
    
    # Параметры случайных состояний
    parser.add_argument('--min_depletion', type=float, default=0.0, help='Минимальное истощение резервуара (0-1)')
    parser.add_argument('--max_depletion', type=float, default=0.8, help='Максимальное истощение резервуара (0-1)')
    parser.add_argument('--use_realistic_ranges', action='store_true', default=True,
                      help='Использовать реалистичные диапазоны параметров')
    
    # Параметры симуляции
    parser.add_argument('--simulation_steps', type=int, default=30, 
                      help='Количество шагов симуляции для оценки действий')
    parser.add_argument('--forecast_days', type=int, default=1,
                      help='Количество дней для прогноза (1=ежедневно, 7=еженедельно, 30=ежемесячно)')
    
    # Параметры вывода
    parser.add_argument('--verbose', action='store_true', help='Выводить подробную информацию')
    
    # Параметры симулятора - общие
    parser.add_argument('--initial_pressure', type=float, default=200.0, help='Начальное пластовое давление (атм)')
    parser.add_argument('--initial_bhp', type=float, default=50.0, help='Начальное забойное давление (атм)')
    parser.add_argument('--productivity_index', type=float, default=0.1, help='Индекс продуктивности (м3/день/атм)')
    parser.add_argument('--total_volume', type=float, default=1e6, help='Общий объем резервуара (м3)')
    
    # Параметры для мультискважинного симулятора
    parser.add_argument('--n_wells', type=int, default=3, help='Количество скважин (для multi_well)')
    parser.add_argument('--interaction_strength', type=float, default=0.1, 
                      help='Сила взаимодействия между скважинами (0-1)')
    parser.add_argument('--shared_reservoir', action='store_true', default=True,
                      help='Использовать общий резервуар (для multi_well)')
    
    return parser.parse_args()


def main():
    """Основная функция для запуска генерации датасета."""
    args = parse_args()
    
    # Собираем параметры симулятора из аргументов
    simulator_params = {
        'initial_reservoir_pressure': args.initial_pressure,
        'initial_bhp': args.initial_bhp,
        'productivity_index': args.productivity_index,
        'total_volume': args.total_volume,
    }
    
    # Добавляем параметры для мультискважинного симулятора
    if args.simulator_type == 'multi_well':
        simulator_params.update({
            'n_wells': args.n_wells,
            'interaction_strength': args.interaction_strength,
            'shared_reservoir': args.shared_reservoir
        })
    
    # Генерируем датасет
    output_path = generate_dataset(
        output_file=args.output,
        num_samples=args.num_samples,
        simulator_type=args.simulator_type,
        min_depletion=args.min_depletion,
        max_depletion=args.max_depletion,
        use_realistic_ranges=args.use_realistic_ranges,
        simulation_steps=args.simulation_steps,
        forecast_days=args.forecast_days,
        verbose=args.verbose,
        **simulator_params
    )
    
    print(f"Датасет доступен по пути: {output_path}")


if __name__ == "__main__":
    main() 