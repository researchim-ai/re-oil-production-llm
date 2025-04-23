# -*- coding: utf-8 -*-
import os
import torch
import re
import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, GenerationConfig
from simulators.single_well.simulator import SingleWellSimulator
from simulators.multi_well.simulator import MultiWellSimulator
from grpo.prompts import get_first_step_prompt, get_subsequent_step_prompt, BASE_PROMPT_TEMPLATE, get_reasoning_first_step_prompt, get_reasoning_subsequent_step_prompt

# Константы для цветов в консоли
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_CYAN = "\033[96m"
COLOR_MAGENTA = "\033[35m"

class ParallelSimulator:
    """
    Класс для параллельной симуляции нескольких независимых скважин
    для сбора данных в GRPO.
    """
    
    def __init__(
        self,
        n_simulators: int,
        simulator_type: str = "single_well",
        simulator_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
    ):
        """
        Инициализирует набор параллельных симуляторов.
        
        Args:
            n_simulators: Количество параллельных симуляторов
            simulator_type: Тип симулятора ('single_well' или 'multi_well')
            simulator_config: Конфигурация для инициализации симуляторов
            device: Устройство для вычислений
        """
        self.n_simulators = n_simulators
        self.device = device
        self.simulator_type = simulator_type
        
        if simulator_config is None:
            simulator_config = {}
        
        # Создаем набор симуляторов
        self.simulators = []
        for i in range(n_simulators):
            if simulator_type == "single_well":
                simulator = SingleWellSimulator(**simulator_config)
            elif simulator_type == "multi_well":
                simulator = MultiWellSimulator(**simulator_config)
            else:
                raise ValueError(f"Неизвестный тип симулятора: {simulator_type}")
            
            self.simulators.append(simulator)
        
        # Сохраняем текущие состояния для всех симуляторов
        self.current_states = [None] * n_simulators
        self.reset_all()
    
    def reset_all(self):
        """Сбрасывает все симуляторы к начальному состоянию."""
        for i, simulator in enumerate(self.simulators):
            self.current_states[i] = simulator.reset()
        
        return self.current_states
    
    def reset_simulator(self, index: int):
        """Сбрасывает конкретный симулятор по индексу."""
        if 0 <= index < self.n_simulators:
            self.current_states[index] = self.simulators[index].reset()
            return self.current_states[index]
        else:
            raise IndexError(f"Индекс {index} выходит за пределы диапазона симуляторов (0-{self.n_simulators-1})")
    
    def step(self, actions: List[float]) -> List[Tuple[np.ndarray, float, bool, Dict]]:
        """
        Выполняет один шаг параллельно во всех симуляторах.
        
        Args:
            actions: Список действий для всех симуляторов
            
        Returns:
            List[Tuple]: Список кортежей (состояние, награда, флаг_завершения, инфо)
        """
        if len(actions) != self.n_simulators:
            raise ValueError(f"Ожидается {self.n_simulators} действий, получено {len(actions)}")
        
        results = []
        for i, (simulator, action) in enumerate(zip(self.simulators, actions)):
            next_state, reward, done, info = simulator.step(action)
            self.current_states[i] = next_state
            results.append((next_state, reward, done, info))
        
        return results
    
    def get_prompts(self, histories: Optional[List[List[str]]] = None, prompt_type: str = "standard") -> List[str]:
        """
        Формирует промпты для всех симуляторов с учетом их текущих состояний.
        
        Args:
            histories: Список историй для каждого симулятора
            prompt_type: Тип промпта ("standard", "reasoning")
            
        Returns:
            List[str]: Список промптов для всех симуляторов
        """
        if histories is None:
            histories = [[] for _ in range(self.n_simulators)]
        
        prompts = []
        for i, (state, history) in enumerate(zip(self.current_states, histories)):
            simulator = self.simulators[i]
            state_text = self.format_state(state, simulator)
            
            # Проверяем тип симулятора (одна или несколько скважин)
            is_multi_well = hasattr(simulator, 'well_names') and len(getattr(simulator, 'well_names', [])) > 1
            
            # Ограничиваем историю до 2 последних взаимодействий для экономии токенов
            if len(history) > 2:
                history = history[-2:]
            
            # Используем разные шаблоны в зависимости от наличия истории и типа промпта
            if prompt_type == "reasoning":
                if not history:
                    # Первый шаг эпизода с рассуждениями
                    prompt = get_reasoning_first_step_prompt(state_text, is_multi_well)
                else:
                    # Последующие шаги с историей и рассуждениями
                    history_text = ' | '.join(history)
                    prompt = get_reasoning_subsequent_step_prompt(state_text, history_text, is_multi_well)
            else:  # standard
                if not history:
                    # Первый шаг эпизода
                    prompt = get_first_step_prompt(state_text, is_multi_well)
                else:
                    # Последующие шаги с историей
                    history_text = ' | '.join(history)
                    prompt = get_subsequent_step_prompt(state_text, history_text, is_multi_well)
            
            prompts.append(prompt)
        
        return prompts
    
    def format_state(self, state, simulator):
        """
        Форматирует состояние симулятора для вывода в компактном виде.
        """
        # Проверяем тип симулятора (одна или несколько скважин)
        if hasattr(simulator, 'well_names') and len(getattr(simulator, 'well_names', [])) > 1:
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
    
    def format_short_state(self, state):
        """
        Форматирует состояние скважины в компактном виде для истории взаимодействий.
        """
        if len(state) >= 4:
            # Базовое состояние [pressure, flow_rate, production, time]
            return f"P={state[0]:.1f}атм, Q={state[1]:.1f}м³/сут, V={state[2]:.1f}м³, t={state[3]:.1f}д"
        else:
            # Если формат состояния неизвестен, возвращаем просто числа
            return ", ".join([f"{x:.1f}" for x in state])
    
    def parse_llm_action(self, response: str) -> Tuple[Optional[float], Dict[str, float]]:
        """
        Извлекает значение действия из ответа языковой модели.
        
        Args:
            response (str): Текст ответа модели.
            
        Returns:
            Tuple[Optional[float], Dict[str, float]]: Кортеж, содержащий извлеченное значение действия 
            (от 0 до 1 или None если формат некорректен) и словарь наград за форматирование.
        """
        try:
            # Очищаем ответ
            clean_response = response.strip()
            
            # Если ответ пустой, возвращаем None для обозначения некорректного формата
            if not clean_response:
                print(f"{COLOR_RED}Пустой ответ: действие не будет выполнено{COLOR_RESET}")
                return None, {"empty_response": -1.0}
            
            # Проверяем количество тегов <reasoning> и <parameter> в ответе
            opening_reasoning_tags = clean_response.count('<reasoning>')
            closing_reasoning_tags = clean_response.count('</reasoning>')
            opening_parameter_tags = clean_response.count('<parameter>')
            closing_parameter_tags = clean_response.count('</parameter>')
            
            # Если есть множественные теги, считаем формат некорректным
            if opening_reasoning_tags > 1 or closing_reasoning_tags > 1:
                print(f"{COLOR_RED}Неправильный формат ответа. Действие не будет выполнено.{COLOR_RESET}")
                return None, {"multiple_reasoning_tags": -1.0}
            
            if opening_parameter_tags > 1 or closing_parameter_tags > 1:
                print(f"{COLOR_RED}Неправильный формат ответа. Действие не будет выполнено.{COLOR_RESET}")
                return None, {"multiple_parameter_tags": -1.0}
            
            # Проверяем наличие тега reasoning (необязательный)
            has_reasoning = False
            reasoning_content = ""
            reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
            reasoning_match = re.search(reasoning_pattern, clean_response, re.DOTALL)
            
            # Расчет штрафа за длинные размышления
            reasoning_length_penalty = 0.0
            max_allowed_words = 50  # Максимально допустимое количество слов в рассуждении
            
            if reasoning_match:
                reasoning_content = reasoning_match.group(1).strip()
                if len(reasoning_content) > 5:  # Уменьшаем минимальную длину рассуждения
                    has_reasoning = True
                    
                    # Подсчитываем количество слов в рассуждении
                    word_count = len(reasoning_content.split())
                    
                    # Если слов больше максимально допустимого количества, добавляем штраф
                    if word_count > max_allowed_words:
                        # Штраф за каждые 10 слов сверх лимита
                        excess_words = word_count - max_allowed_words
                        reasoning_length_penalty = min(1.0, excess_words / 100)
                        print(f"Слишком длинное рассуждение: {word_count} слов (лимит {max_allowed_words}). Штраф: {reasoning_length_penalty:.2f}")
            
            # Также проверяем наличие открывающего тега reasoning без закрывающего
            # Но только если нет закрытого тега
            if not has_reasoning and opening_reasoning_tags == 1 and closing_reasoning_tags == 0:
                # Извлекаем содержимое после <reasoning>
                reasoning_start = clean_response.find('<reasoning>') + len('<reasoning>')
                reasoning_content = clean_response[reasoning_start:].strip()
                if len(reasoning_content) > 5:
                    has_reasoning = True
                    
                    # Подсчитываем количество слов в рассуждении
                    word_count = len(reasoning_content.split())
                    
                    # Если слов больше максимально допустимого количества, добавляем штраф
                    if word_count > max_allowed_words:
                        # Штраф за каждые 10 слов сверх лимита
                        excess_words = word_count - max_allowed_words
                        reasoning_length_penalty = min(1.0, excess_words / 100)
                        print(f"Слишком длинное незавершенное рассуждение: {word_count} слов (лимит {max_allowed_words}). Штраф: {reasoning_length_penalty:.2f}")
                    
                    # Удаляем неполный тег из ответа для дальнейшей обработки
                    clean_response = clean_response.replace('<reasoning>', '').strip()
            
            # Проверка наличия parameter тега для извлечения действия
            perfect_pattern = r'<parameter>(.*?)</parameter>'
            perfect_match = re.search(perfect_pattern, clean_response, re.DOTALL)
            
            if perfect_match:
                # Извлекаем значение внутри тегов
                value_str = perfect_match.group(1).strip()
                
                # Проверяем, что в теге parameter содержится только число
                if not re.match(r'^\s*\d+(\.\d+)?\s*$', value_str):
                    print(f"{COLOR_YELLOW}Неполный формат тега parameter. Возможно действие не будет выполнено корректно.{COLOR_RESET}")
                    # Попробуем извлечь число из строки
                    numeric_pattern = r'(\d+(\.\d+)?)'
                    numeric_match = re.search(numeric_pattern, value_str)
                    if numeric_match:
                        value_str = numeric_match.group(1)
                        print(f"{COLOR_GREEN}Извлечено число: {value_str}{COLOR_RESET}")
                    else:
                        print(f"{COLOR_RED}Не найдено число в теге parameter. Действие не будет выполнено.{COLOR_RESET}")
                        return None, {"parameter_not_number": -1.0}
                
                try:
                    value = float(value_str)
                    
                    # Ограничиваем диапазон значений
                    value = max(0.0, min(1.0, value))
                    if value < 0.0 or value > 1.0:
                        print(f"Значение в теге <parameter> ({value}) выходит за допустимый диапазон [0, 1]. Применено ограничение: {value}")
                    
                    # Определяем тип награды за формат с учетом штрафа за длину рассуждения
                    if has_reasoning:
                        # Идеальный формат с рассуждением, максимальная награда за минусом штрафа за длину
                        format_reward = 1.2 - reasoning_length_penalty
                        return value, {"reasoning_parameter_format": format_reward}
                    else:
                        # Идеальный формат без рассуждения, хорошая награда
                        return value, {"parameter_format": 1.0}
                except ValueError:
                    # Если содержимое тегов не число, возвращаем None
                    print(f"Ошибка формата: тег <parameter> содержит не число: '{value_str}'. Действие не будет выполнено.")
                    return None, {"parameter_not_number": -1.0}
            
            # Более гибкий паттерн для случаев с незакрытыми тегами parameter
            if opening_parameter_tags == 1 and closing_parameter_tags == 0:
                flexible_parameter_pattern = r'<parameter>(.*?)(?:</parameter|</parameter>|$)'
                flexible_parameter_match = re.search(flexible_parameter_pattern, clean_response, re.DOTALL)
                
                if flexible_parameter_match:
                    # Extract the value inside the tags
                    value_str = flexible_parameter_match.group(1).strip()
                    
                    # Проверяем, что в теге parameter содержится только число
                    if not re.match(r'^\s*\d+(\.\d+)?\s*$', value_str):
                        print(f"{COLOR_YELLOW}Неполный формат тега parameter. Возможно действие не будет выполнено корректно.{COLOR_RESET}")
                        # Попробуем извлечь число из строки
                        numeric_pattern = r'(\d+(\.\d+)?)'
                        numeric_match = re.search(numeric_pattern, value_str)
                        if numeric_match:
                            value_str = numeric_match.group(1)
                            print(f"{COLOR_GREEN}Извлечено число: {value_str}{COLOR_RESET}")
                        else:
                            print(f"{COLOR_RED}Не найдено число в теге parameter. Действие не будет выполнено.{COLOR_RESET}")
                            return None, {"parameter_not_number": -1.0}
                    
                    try:
                        value = float(value_str)
                        # Ограничиваем диапазон
                        value = max(0.0, min(1.0, value))
                        
                        # Определяем тип награды с учетом наличия рассуждения и штрафа за длину
                        if has_reasoning:
                            # Неполный формат с рассуждением, хорошая награда за минусом штрафа за длину
                            format_reward = 0.8 - reasoning_length_penalty
                            print(f"{COLOR_GREEN}Неполный формат с рассуждением. Действие будет выполнено.{COLOR_RESET}")
                            return value, {"reasoning_almost_parameter_format": format_reward}
                        else:
                            # Неполный формат без рассуждения, умеренная награда
                            print(f"{COLOR_YELLOW}Неполный формат без рассуждения. Действие будет выполнено.{COLOR_RESET}")
                            return value, {"almost_parameter_format": 0.6}
                    except ValueError:
                        # If the content inside tags is not a number, use default and give negative reward
                        print(f"{COLOR_RED}Ошибка формата: неполный тег parameter не содержит число. Действие не будет выполнено.{COLOR_RESET}")
                        return None, {"parameter_not_number": -1.0}
            
            # Вариант для случая с одним тегом <reasoning> без параметра - ищем число в тексте рассуждения
            if has_reasoning:
                # Ищем число в тексте рассуждения
                number_pattern = r'(?:^|[^\w])(\d+(?:\.\d+)?)(?:[^\w]|$)'
                number_match = re.search(number_pattern, reasoning_content)
                
                if number_match:
                    value_str = number_match.group(1).strip()
                    try:
                        value = float(value_str)
                        value = max(0.0, min(1.0, value))
                        # Штраф за отсутствие тега параметра плюс штраф за длину
                        format_reward = -0.3 - reasoning_length_penalty
                        print(f"{COLOR_YELLOW}Найдено число {value} в рассуждении. Действие будет выполнено.{COLOR_RESET}")
                        return value, {"reasoning_without_parameter": format_reward}
                    except ValueError:
                        pass
                
                # Если в рассуждении нет числа, даем значение по умолчанию со штрафом и штрафом за длину
                format_reward = -0.5 - reasoning_length_penalty
                print(f"{COLOR_YELLOW}Рассуждение без числа. Используем значение по умолчанию.{COLOR_RESET}")
                return 0.5, {"reasoning_no_parameter": format_reward}
            
            # Если у нас нет рассуждения, ищем число в тексте ответа
            number_pattern = r'(?:^|[^\w])(\d+(?:\.\d+)?)(?:[^\w]|$)'
            number_match = re.search(number_pattern, clean_response)
            
            if number_match:
                value_str = number_match.group(1).strip()
                try:
                    value = float(value_str)
                    value = max(0.0, min(1.0, value))
                    print(f"{COLOR_GREEN}Найдено число: {value}. Действие будет выполнено.{COLOR_RESET}")
                    return value, {"wrong_format_with_number": -0.8}
                except ValueError:
                    pass
            
            # Check for explicit cases of full opening/closing for value extraction
            if any(phrase in clean_response.lower() for phrase in ["fully open", "maximum open", "completely open", "полностью открыть"]):
                print(f"{COLOR_YELLOW}Найдена фраза о полном открытии. Используем значение 1.0{COLOR_RESET}")
                return 1.0, {"wrong_format_open": -0.8}
            elif any(phrase in clean_response.lower() for phrase in ["fully close", "completely close", "close the choke", "полностью закрыть"]):
                print(f"{COLOR_YELLOW}Найдена фраза о полном закрытии. Используем значение 0.0{COLOR_RESET}")
                return 0.0, {"wrong_format_close": -0.8}
            elif any(phrase in clean_response.lower() for phrase in ["no change", "maintain", "keep", "без изменений", "сохранить"]):
                print(f"{COLOR_YELLOW}Найдена фраза о сохранении состояния. Используем значение 0.5{COLOR_RESET}")
                return 0.5, {"wrong_format_maintain": -0.8}
            
            # Если не удалось извлечь значение, возвращаем случайное значение со штрафом
            random_value = 0.5  # Среднее значение в качестве действия по умолчанию
            print(f"{COLOR_RED}Не удалось извлечь значение из ответа. Используем значение по умолчанию.{COLOR_RESET}")
            return random_value, {"wrong_format_default": -1.0}
        except Exception as e:
            # В случае любой ошибки возвращаем None для обозначения некорректного формата
            print(f"{COLOR_RED}Ошибка при обработке ответа: {str(e)}. Используем значение по умолчанию.{COLOR_RESET}")
            return 0.5, {"error": -1.0}


def parallel_rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    parallel_sim: ParallelSimulator,
    n_steps: int = 20,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = True,
    prompt_type: str = "standard",
) -> Tuple[
    List[torch.Tensor],   # all_episode_tokens
    List[torch.Tensor],   # all_action_masks
    List[torch.Tensor],   # all_rewards
    List[Dict]            # all_episode_stats
]:
    """
    Выполняет параллельный прогон эпизодов на множестве симуляторов.
    
    Args:
        model: Модель для генерации ответов
        tokenizer: Токенизатор для модели
        parallel_sim: Параллельный симулятор
        n_steps: Максимальное количество шагов в эпизоде
        temperature: Температура для семплирования
        top_p: Параметр top_p для семплирования
        verbose: Флаг для включения подробного вывода
        prompt_type: Тип промпта для генерации ("standard", "reasoning")
        
    Returns:
        Tuple: Кортеж из списков с токенами, масками действий, наградами и статистикой
    """
    # Засекаем время начала выполнения
    start_time = time.time()
    
    # Инициализируем списки для хранения результатов
    all_episode_tokens = []
    all_action_masks = []
    all_rewards = []
    all_episode_stats = []
    
    # Сбрасываем состояние всех симуляторов
    parallel_sim.reset_all()
    
    # Количество симуляторов
    n_simulators = parallel_sim.n_simulators
    
    # Готовим данные для отслеживания прогресса эпизодов
    episodes_done = [False] * n_simulators
    episode_histories = [[] for _ in range(n_simulators)]
    episode_tokens_list = [[] for _ in range(n_simulators)]
    episode_action_masks_list = [[] for _ in range(n_simulators)]
    episode_rewards_list = [[] for _ in range(n_simulators)]
    episode_format_rewards_list = [[] for _ in range(n_simulators)]
    episode_actions_list = [[] for _ in range(n_simulators)]
    steps_completed = [0] * n_simulators
    episode_total_rewards = [0.0] * n_simulators
    episode_production = [0.0] * n_simulators
    skipped_steps = [0] * n_simulators
    
    # Получаем все возможные индексы симуляторов
    all_indices = list(range(n_simulators))
    
    # Основной цикл по шагам
    for step in range(n_steps):
        if verbose:
            print(f"\n{COLOR_CYAN}==== Шаг {step + 1} ===={COLOR_RESET}")
        
        # Фильтруем активные симуляторы (не завершенные)
        active_indices = [i for i, done in enumerate(episodes_done) if not done]
        
        if not active_indices:
            if verbose:
                print(f"{COLOR_YELLOW}Все эпизоды завершены!{COLOR_RESET}")
            break
        
        if verbose:
            print(f"Активные симуляторы: {[i+1 for i in active_indices]}")
        
        # Получаем промпты для активных симуляторов
        active_histories = [episode_histories[i] for i in active_indices]
        prompts = parallel_sim.get_prompts(active_histories, prompt_type=prompt_type)
        
        # Токенизируем все промпты
        tokenized_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(parallel_sim.device)
        
        # Генерируем действия параллельно
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **tokenized_inputs,
                    generation_config=GenerationConfig(
                        max_new_tokens=128,  # Увеличиваем длину генерации для более качественных ответов
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    ),
                )
            except Exception as e:
                if verbose:
                    print(f"{COLOR_RED}Ошибка при генерации: {e}{COLOR_RESET}")
                # Создаем запасной вариант выходов
                outputs = torch.cat([
                    tokenized_inputs.input_ids,
                    torch.ones((len(active_indices), 1), dtype=torch.long, device=parallel_sim.device) * tokenizer.encode("0.5", add_special_tokens=False)[0]
                ], dim=1)
        
        # Извлекаем новые токены и действия
        actions = []
        valid_actions = []  # Список для хранения валидных действий
        valid_indices = []  # Список для хранения индексов валидных действий
        
        for i, idx in enumerate(active_indices):
            # Добавляем проверку на существование индексов
            if i >= len(outputs) or idx >= n_simulators:
                if verbose:
                    print(f"{COLOR_RED}Ошибка индексации: i={i}, idx={idx}{COLOR_RESET}")
                continue
                
            # Получаем только новые токены (без промпта)
            # Batch size может быть > 1, поэтому берем соответствующую строку
            input_length = tokenized_inputs.input_ids.shape[1]
            if input_length < outputs.shape[1]:
                new_tokens = outputs[i, input_length:]
            else:
                # В случае если по какой-то причине модель не сгенерировала новые токены
                new_tokens = torch.tensor([tokenizer.encode("0.5", add_special_tokens=False)[0]], 
                                       device=parallel_sim.device)
            
            # Декодируем ответ
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            if verbose:
                print(f"Симулятор {idx+1}: Шаг {step+1}/{n_steps} {COLOR_GREEN}Ответ модели:{COLOR_RESET} '{response}'")
            
            # Очищаем ответ от символов, которые модель часто повторяет
            response = re.sub(r'(\*+)', '', response)  # Удаляем звездочки
            response = re.sub(r'`+', '', response)      # Удаляем обратные кавычки
            response = re.sub(r'\s+', ' ', response).strip()  # Нормализуем пробелы
            
            # Извлекаем действие из ответа
            action, format_rewards = parallel_sim.parse_llm_action(response)
            
            # Убедимся, что format_rewards - словарь, даже если parse_llm_action вернул None
            if format_rewards is None:
                format_rewards = {"error": -1.0}
                
            # Инициализируем список действий, если он еще не существует
            if len(episode_actions_list) <= idx:
                # Если каким-то образом индекс оказался за пределами списка
                episode_actions_list.extend([[] for _ in range(idx - len(episode_actions_list) + 1)])
                
            if len(episode_format_rewards_list) <= idx:
                episode_format_rewards_list.extend([[] for _ in range(idx - len(episode_format_rewards_list) + 1)])
            
            # Проверяем, не является ли это первым шагом с неправильным форматом
            if step == 0 and action is None:
                # На первом шаге формат неправильный, прерываем эпизод
                episodes_done[idx] = True
                
                # Добавляем отрицательную награду за неправильный формат
                format_reward = sum(format_rewards.values())
                
                # Инициализируем список наград, если он еще не существует
                if len(episode_rewards_list) <= idx:
                    episode_rewards_list.extend([[] for _ in range(idx - len(episode_rewards_list) + 1)])
                
                episode_rewards_list[idx].append(format_reward)
                episode_total_rewards[idx] += format_reward
                
                # Увеличиваем счетчик шагов, так как это тоже шаг обучения
                steps_completed[idx] += 1
                
                if verbose:
                    print(f"{COLOR_RED}Симулятор {idx+1}: Шаг {step+1}/{n_steps} Эпизод прерван из-за неправильного формата на первом шаге.{COLOR_RESET}")
                    print(f"{COLOR_RED}Симулятор {idx+1}: Шаг {step+1}/{n_steps} Штраф за формат: {format_reward:.4f}{COLOR_RESET}")
                
                # Сохраняем токены для обучения
                episode_tokens_list[idx].append(new_tokens)
                
                # Создаем маску действий
                action_mask = torch.ones_like(new_tokens, dtype=torch.bool)
                episode_action_masks_list[idx].append(action_mask)
                
                # Заполняем дефолтное значение для действия
                actions.append(0.0)
                episode_actions_list[idx].append(0.0)
                episode_format_rewards_list[idx].append(format_rewards)
                
                # Пропускаем дальнейшую обработку для этого симулятора
                continue
            
            if action is not None:
                if verbose:
                    print(f"Симулятор {idx+1}: Шаг {step+1}/{n_steps} {COLOR_YELLOW}Действие:{COLOR_RESET} {action:.4f}")
                    print(f"Симулятор {idx+1}: Шаг {step+1}/{n_steps} {COLOR_BLUE}Награды за форматирование:{COLOR_RESET} {format_rewards}")
                
                # Сохраняем действие и индекс
                valid_actions.append(action)
                valid_indices.append(i)
                
                # Сохраняем токены действия в любом случае (для обучения)
                action_tokens = new_tokens
                
                # Убеждаемся, что тензор не пустой
                if action_tokens.numel() == 0:
                    action_tokens = torch.tensor([tokenizer.encode("0", add_special_tokens=False)[0]], 
                                              device=parallel_sim.device)
                
                # Инициализируем списки токенов и масок, если они еще не существуют
                if len(episode_tokens_list) <= idx:
                    episode_tokens_list.extend([[] for _ in range(idx - len(episode_tokens_list) + 1)])
                    
                if len(episode_action_masks_list) <= idx:
                    episode_action_masks_list.extend([[] for _ in range(idx - len(episode_action_masks_list) + 1)])
                    
                episode_tokens_list[idx].append(action_tokens)
                
                # Создаем маску действий - улучшенная версия
                # Маскируем все токены ответа, независимо от того, валидный формат или нет
                action_mask = torch.ones_like(action_tokens, dtype=torch.bool)
                episode_action_masks_list[idx].append(action_mask)
                
                # Добавляем информацию в историю только если действие было валидным
                if idx < len(parallel_sim.current_states):
                    current_state = parallel_sim.current_states[idx]
                    episode_histories[idx].append(f"Сост:{parallel_sim.format_short_state(current_state)}, Д:{action:.2f}")
                else:
                    if verbose:
                        print(f"{COLOR_RED}Ошибка: Индекс {idx} вне диапазона current_states{COLOR_RESET}")
            else:
                if verbose:
                    print(f"Симулятор {idx+1}: Шаг {step+1}/{n_steps} {COLOR_RED}Действие пропущено из-за неправильного формата{COLOR_RESET}")
                    print(f"Симулятор {idx+1}: Шаг {step+1}/{n_steps} {COLOR_BLUE}Штраф за форматирование:{COLOR_RESET} {format_rewards}")
                # Увеличиваем счетчик пропущенных шагов
                skipped_steps[idx] += 1
                
                # Прерываем эпизод при неправильном формате на любом шаге
                episodes_done[idx] = True
                
                if verbose:
                    print(f"{COLOR_RED}Симулятор {idx+1}: Шаг {step+1}/{n_steps} Эпизод прерван из-за неправильного формата ответа.{COLOR_RESET}")
            
            # Заполняем пустым значением (будет игнорироваться, если шаг пропущен)
            default_action = 0.0
            actions.append(action if action is not None else default_action)
            episode_actions_list[idx].append(action if action is not None else default_action)
            episode_format_rewards_list[idx].append(format_rewards)  # Сохраняем формат наград
        
        # Если есть валидные действия, выполняем шаг для соответствующих симуляторов
        if valid_actions:
            # Заполняем действия для всех симуляторов (нули для симуляторов с невалидными ответами)
            all_actions = [0.0] * n_simulators
            
            # Заполняем только для активных симуляторов с валидными действиями
            for i, idx in enumerate(valid_indices):
                act_idx = active_indices[idx]
                all_actions[act_idx] = valid_actions[i]
            
            # Выполняем шаг параллельно для всех симуляторов
            results = parallel_sim.step(all_actions)
            
            # Обрабатываем результаты
            for idx, (next_state, reward, done, info) in enumerate(results):
                # Пропускаем неактивные симуляторы или те, где формат был неправильный
                if episodes_done[idx] or idx not in [active_indices[i] for i in valid_indices]:
                    continue
                
                # Инициализируем списки, если они еще не существуют для этого индекса
                if len(episode_rewards_list) <= idx:
                    episode_rewards_list.extend([[] for _ in range(idx - len(episode_rewards_list) + 1)])
                
                # Проверяем, есть ли формат наград для этого шага
                format_reward = 0.0
                if idx < len(episode_format_rewards_list) and len(episode_format_rewards_list[idx]) > 0:
                    format_reward = sum(episode_format_rewards_list[idx][-1].values())
                
                # Обновляем данные для активного симулятора
                episode_rewards_list[idx].append(reward + format_reward)
                full_reward = reward + format_reward
                episode_total_rewards[idx] += full_reward
                steps_completed[idx] += 1
                
                # Обновляем информацию о добыче
                if idx < len(parallel_sim.simulators) and hasattr(parallel_sim.simulators[idx], 'cumulative_production'):
                    episode_production[idx] = parallel_sim.simulators[idx].cumulative_production
                
                if verbose:
                    print(f"Симулятор {idx+1}: Шаг {steps_completed[idx]}/{n_steps}, Награда: {reward:.4f}, "
                        f"Формат: {format_reward:.4f}, "
                        f"Общая награда: {episode_total_rewards[idx]:.4f}, "
                        f"Добыча: {episode_production[idx]:.2f} м³")
                
                # Если эпизод завершен, отмечаем его
                episodes_done[idx] = done
                
                if done:
                    if verbose:
                        print(f"{COLOR_MAGENTA}Симулятор {idx+1}: Эпизод завершен после {steps_completed[idx]} шагов.{COLOR_RESET}")
        else:
            # Если нет валидных действий, добавляем штрафы за формат для всех активных симуляторов
            for i, idx in enumerate(active_indices):
                # Проверяем, есть ли формат наград для этого шага
                format_reward = 0.0
                if idx < len(episode_format_rewards_list) and len(episode_format_rewards_list[idx]) > 0:
                    format_reward = sum(episode_format_rewards_list[idx][-1].values())
                
                # Инициализируем список, если он еще не существует для этого индекса
                if len(episode_rewards_list) <= idx:
                    episode_rewards_list.extend([[] for _ in range(idx - len(episode_rewards_list) + 1)])
                
                episode_rewards_list[idx].append(format_reward)
                episode_total_rewards[idx] += format_reward
                
                if verbose:
                    print(f"Симулятор {idx+1}: Шаг {step+1}/{n_steps}, Штраф за формат: {format_reward:.4f}, "
                        f"Общая награда: {episode_total_rewards[idx]:.4f}")
                
                # Помечаем эпизод как завершенный из-за ошибки формата
                episodes_done[idx] = True
                
                if verbose:
                    print(f"{COLOR_RED}Симулятор {idx+1}: Шаг {step+1}/{n_steps} Эпизод прерван из-за неправильного формата ответа.{COLOR_RESET}")
    
    # После завершения всех шагов собираем финальные данные
    for idx in range(n_simulators):
        try:
            # Даже прерванные эпизоды на первом шаге должны быть включены в обучение
            # для передачи негативного сигнала
            
            # Собираем токены для эпизода
            all_tokens = []
            all_masks = []
            
            if idx < len(episode_tokens_list) and idx < len(episode_action_masks_list):
                for step_tokens, step_masks in zip(episode_tokens_list[idx], episode_action_masks_list[idx]):
                    all_tokens.append(step_tokens)
                    all_masks.append(step_masks)
            
            # Проверяем, есть ли хоть какие-то токены
            if not all_tokens:
                if verbose:
                    print(f"{COLOR_RED}Симулятор {idx+1}: Нет сохраненных токенов, пропускаем.{COLOR_RESET}")
                continue
            
            # Объединяем токены и маски в один тензор для эпизода
            try:
                episode_tokens = torch.cat(all_tokens)
                episode_masks = torch.cat(all_masks)
            except Exception as e:
                if verbose:
                    print(f"{COLOR_RED}Симулятор {idx+1}: Ошибка при объединении токенов: {e}{COLOR_RESET}")
                continue
            
            all_episode_tokens.append(episode_tokens)
            all_action_masks.append(episode_masks)
            
            # Преобразуем награды в тензор, проверяя, что список существует
            if idx < len(episode_rewards_list) and episode_rewards_list[idx]:
                episode_rewards = torch.tensor(episode_rewards_list[idx], device=parallel_sim.device)
                all_rewards.append(episode_rewards)
                
                # Собираем статистику эпизода с проверками на существование списков
                episode_stats = {
                    "steps": steps_completed[idx],
                    "reward": episode_total_rewards[idx],
                    "production": episode_production[idx] if idx < len(episode_production) else 0.0,
                    "time": time.time() - start_time,
                    "actions": episode_actions_list[idx] if idx < len(episode_actions_list) else [],
                    "format_rewards": episode_format_rewards_list[idx] if idx < len(episode_format_rewards_list) else [],
                    "skipped_steps": skipped_steps[idx] if idx < len(skipped_steps) else 0,
                    "early_termination": True if skipped_steps[idx] > 0 else False,
                    "first_step_error": step == 0 and action is None if idx < len(episode_actions_list) else False
                }
                all_episode_stats.append(episode_stats)
            else:
                # Если список наград пуст, создаем искусственный список с одной отрицательной наградой
                # Это гарантирует, что даже эпизоды с ошибкой формата на первом шаге будут включены
                if verbose:
                    print(f"{COLOR_YELLOW}Симулятор {idx+1}: Создаем искусственную награду для обучения для эпизода с ошибкой.{COLOR_RESET}")
                
                # Создаем тензор с одной отрицательной наградой
                episode_rewards = torch.tensor([-1.0], device=parallel_sim.device)
                all_rewards.append(episode_rewards)
                
                # Создаем минимальную статистику для этого эпизода
                episode_stats = {
                    "steps": 1,  # Хотя бы один шаг
                    "reward": -1.0,  # Отрицательная награда
                    "production": 0.0,
                    "time": time.time() - start_time,
                    "actions": [0.0],  # Дефолтное действие
                    "format_rewards": [{"wrong_format": -1.0}],  # Штраф за формат
                    "skipped_steps": 1,
                    "early_termination": True,
                    "first_step_error": True
                }
                all_episode_stats.append(episode_stats)
            
        except Exception as e:
            if verbose:
                print(f"{COLOR_RED}Симулятор {idx+1}: Ошибка при сборе финальных данных: {e}{COLOR_RESET}")
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\n{COLOR_GREEN}Роллауты завершены. Время выполнения: {elapsed_time:.2f} с.{COLOR_RESET}")
        print(f"Собрано {len(all_episode_tokens)} эпизодов.")
    
    return all_episode_tokens, all_action_masks, all_rewards, all_episode_stats 