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
from grpo.prompts import get_first_step_prompt, get_subsequent_step_prompt, BASE_PROMPT_TEMPLATE

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
    
    def get_prompts(self, histories: Optional[List[List[Tuple[str, str]]]] = None) -> List[str]:
        """
        Формирует промпты для всех симуляторов с учетом их текущих состояний и полной истории.

        Args:
            histories: Список полных историй для каждого симулятора.
                       Каждая история - список кортежей (форматированное_состояние, форматированное_действие).

        Returns:
            List[str]: Список промптов для всех симуляторов
        """
        if histories is None:
            histories = [[] for _ in range(self.n_simulators)]

        prompts = []
        for i, (state, history) in enumerate(zip(self.current_states, histories)):
            simulator = self.simulators[i]
            state_text = self.format_state(state, simulator)

            # Проверяем тип симулятора
            is_multi_well = hasattr(simulator, 'well_names') and len(getattr(simulator, 'well_names', [])) > 1

            # Используем разные шаблоны в зависимости от наличия истории
            if not history:
                # Первый шаг эпизода
                prompt = get_first_step_prompt(state_text, is_multi_well)
            else:
                # Последующие шаги с ПОЛНОЙ историей
                # Формируем текстовое представление полной истории
                full_history_lines = []
                for step_idx, (hist_state, hist_action) in enumerate(history):
                    full_history_lines.append(f"Шаг {step_idx + 1}: Состояние: {hist_state}, Действие: {hist_action}")
                full_history_text = "\n".join(full_history_lines)

                # Извлекаем последнее действие
                last_action_str = history[-1][1] # Берем строку действия из последнего кортежа

                # Вызываем обновленный get_subsequent_step_prompt
                prompt = get_subsequent_step_prompt(state_text, full_history_text, last_action_str, is_multi_well)

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
        Extracts the action value from the LLM response.
        
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
            
            # Ищем число в тексте для использования как значение
            number_pattern = r'^([0-9]*\.?[0-9]+)$'
            number_match = re.search(number_pattern, clean_response)
            
            if number_match:
                try:
                    value = float(number_match.group(1))
                    # Limit the range
                    value = max(0.0, min(1.0, value))
                    # Идеальный формат, максимальная награда
                    return value, {"correct_format": 1.0}
                except ValueError:
                    # If the content is not a valid number, use default and give negative reward
                    print(f"Ошибка формата: не удалось преобразовать в число: '{clean_response}'. Действие не будет выполнено.")
                    return None, {"not_number": -1.0}
            
            # Более гибкий поиск числа в тексте
            number_pattern = r'(?:^|[^\w])(\d+(?:\.\d+)?)(?:[^\w]|$)'
            number_match = re.search(number_pattern, clean_response)
            if number_match:
                try:
                    value = float(number_match.group(1))
                    # Limit the range
                    value = max(0.0, min(1.0, value))
                    # Не идеальный формат, но нашли число
                    print(f"Найдено число в тексте: {clean_response}. Действие будет выполнено, но в следующий раз используйте только число.")
                    return value, {"almost_correct_format": 0.4}
                except ValueError:
                    print(f"Ошибка формата: не удалось преобразовать в число: '{clean_response}'. Действие не будет выполнено.")
                    return None, {"not_number": -1.0}
            
            # Check for explicit cases of full opening/closing for value extraction
            if any(phrase in clean_response.lower() for phrase in ["fully open", "maximum open", "completely open", "полностью открыть"]):
                print(f"Найдена фраза о полном открытии, но формат неверный. Действие не будет выполнено. Используйте только число 1.0")
                return None, {"wrong_format_open": -0.8}
            elif any(phrase in clean_response.lower() for phrase in ["fully close", "completely close", "close the choke", "полностью закрыть"]):
                print(f"Найдена фраза о полном закрытии, но формат неверный. Действие не будет выполнено. Используйте только число 0.0")
                return None, {"wrong_format_close": -0.8}
            elif any(phrase in clean_response.lower() for phrase in ["no change", "maintain", "keep", "без изменений", "сохранить"]):
                print(f"Найдена фраза о сохранении текущего состояния, но формат неверный. Действие не будет выполнено. Используйте только число (например, 0.5)")
                return None, {"wrong_format_maintain": -0.8}
            
            # If unable to extract a value, use the default
            print(f"Не удалось извлечь значение из ответа: '{clean_response}'. Действие не будет выполнено.")
            return None, {"wrong_format": -1.0}
        
        except Exception as e:
            # Handle any unexpected errors
            print(f"Ошибка при обработке ответа: {e}")
            return None, {"parsing_error": -1.0}


def parallel_rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    parallel_sim: ParallelSimulator,
    n_steps: int = 20,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = True,
) -> Tuple[
    List[torch.Tensor],   # all_episode_tokens
    List[torch.Tensor],   # all_action_masks
    List[torch.Tensor],   # all_rewards
    List[Dict]            # all_episode_stats
]:
    """
    Выполняет параллельные роллауты для всех симуляторов.
    Сохраняет полную историю (состояние, действие) для промптов.
    
    Args:
        model: Языковая модель
        tokenizer: Токенизатор
        parallel_sim: Объект класса ParallelSimulator
        n_steps: Максимальное количество шагов для каждого роллаута
        temperature: Температура генерации
        top_p: Параметр top_p для генерации
        verbose: Выводить ли информацию в консоль
    
    Returns:
        Кортеж из:
            all_episode_tokens: Список тензоров с токенами для каждого эпизода
            all_action_masks: Список тензоров с масками действий
            all_rewards: Список тензоров с наградами
            all_episode_stats: Список словарей со статистикой
    """
    model.eval()
    device = next(model.parameters()).device
    
    n_simulators = parallel_sim.n_simulators
    
    # Инициализируем хранилища для всех эпизодов
    all_episode_tokens = []
    all_action_masks = []
    all_rewards = []
    all_episode_stats = []
    
    # Инициализируем счетчики для отслеживания прогресса
    steps_completed = [0] * n_simulators
    episodes_done = [False] * n_simulators
    skipped_steps = [0] * n_simulators

    # Истории для каждого симулятора (список кортежей: state_str, action_str)
    histories: List[List[Tuple[str, str]]] = [[] for _ in range(n_simulators)]

    # Данные эпизодов
    episode_tokens_list = [[] for _ in range(n_simulators)]
    episode_action_masks_list = [[] for _ in range(n_simulators)]
    episode_rewards_list = [[] for _ in range(n_simulators)]
    episode_actions_list = [[] for _ in range(n_simulators)] # Храним фактические float действия
    episode_total_rewards = [0.0] * n_simulators
    episode_production = [0.0] * n_simulators
    episode_format_rewards_list = [[] for _ in range(n_simulators)]

    start_time = time.time()
    
    # Создаем конфигурацию генерации
    generation_config = GenerationConfig(
        max_new_tokens=10,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Выполняем шаги для всех симуляторов
    for step in range(n_steps):
        if verbose:
            print(f"\n{COLOR_CYAN}Шаг {step+1}/{n_steps}{COLOR_RESET}")
        
        # Получаем список активных симуляторов (не завершенных)
        active_indices = [i for i, done in enumerate(episodes_done) if not done]
        
        if not active_indices:
            if verbose:
                print(f"{COLOR_YELLOW}Все эпизоды завершены, останавливаем симуляцию{COLOR_RESET}")
            break
        
        # Получаем промпты только для активных симуляторов, передавая полные истории
        active_histories = [histories[i] for i in active_indices]
        prompts = parallel_sim.get_prompts(active_histories) # get_prompts теперь использует полные истории
        
        # Токенизируем все промпты
        tokenized_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        
        # Генерируем действия параллельно
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **tokenized_inputs,
                    generation_config=generation_config,
                )
            except Exception as e:
                if verbose:
                    print(f"{COLOR_RED}Ошибка при генерации: {e}{COLOR_RESET}")
                # Создаем запасной вариант выходов
                outputs = torch.cat([
                    tokenized_inputs.input_ids,
                    torch.ones((len(active_indices), 1), dtype=torch.long, device=device) * tokenizer.encode("0.5", add_special_tokens=False)[0]
                ], dim=1)
        
        # Извлекаем новые токены и действия
        actions = [] # Список действий (float или None) на этом шаге для активных симуляторов
        action_strs = [] # Список действий (строка) на этом шаге для активных симуляторов
        valid_action_values = []  # Список для хранения валидных float действий
        valid_sim_indices = [] # Индексы симуляторов (относительно active_indices), где действие было валидным

        current_step_states = [parallel_sim.format_state(parallel_sim.current_states[active_indices[i]], parallel_sim.simulators[active_indices[i]]) for i in range(len(active_indices))]

        for i, idx in enumerate(active_indices): # idx - абсолютный индекс симулятора
            # Получаем только новые токены (без промпта)
            new_tokens = outputs[i, tokenized_inputs.input_ids.shape[1]:]
            
            # Декодируем ответ
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            if verbose:
                print(f"Симулятор {idx+1}: {COLOR_GREEN}Ответ модели:{COLOR_RESET} \'{response}\'")
            
            # Очищаем ответ
            response = re.sub(r'(\\*+)', '', response)
            response = re.sub(r'`+', '', response)
            response = re.sub(r'\s+', ' ', response).strip()
            
            # Извлекаем действие из ответа
            action_value, format_rewards = parallel_sim.parse_llm_action(response)
            action_str = f"{action_value:.4f}" if action_value is not None else "Пропущено (ошибка формата)"
            actions.append(action_value)
            action_strs.append(action_str)
            episode_format_rewards_list[idx].append(format_rewards) # Сохраняем награду за формат

            if action_value is not None:
                if verbose:
                    print(f"Симулятор {idx+1}: {COLOR_YELLOW}Действие:{COLOR_RESET} {action_value:.4f}")
                    print(f"Симулятор {idx+1}: {COLOR_BLUE}Награды за форматирование:{COLOR_RESET} {format_rewards}")

                # Сохраняем валидное действие и индекс
                valid_action_values.append(action_value)
                valid_sim_indices.append(i) # Сохраняем индекс внутри active_indices
                skipped_steps[idx] = 0
            else:
                if verbose:
                    print(f"Симулятор {idx+1}: {COLOR_RED}Действие пропущено из-за неправильного формата{COLOR_RESET}")
                    print(f"Симулятор {idx+1}: {COLOR_BLUE}Штраф за форматирование:{COLOR_RESET} {format_rewards}")

                skipped_steps[idx] += 1
                if skipped_steps[idx] > 5:
                    if verbose:
                        print(f"{COLOR_RED}Симулятор {idx+1}: Слишком много пропущенных шагов подряд, завершаем эпизод.{COLOR_RESET}")
                    episodes_done[idx] = True

            # Сохраняем float действие (или 0.0 если невалидно) для статистики
            episode_actions_list[idx].append(action_value if action_value is not None else 0.0)

            # Сохраняем токены действия в любом случае (для обучения)
            action_tokens = new_tokens
            if action_tokens.numel() == 0:
                action_tokens = torch.tensor([tokenizer.encode("0", add_special_tokens=False)[0]], device=device)
            episode_tokens_list[idx].append(action_tokens)

            # Создаем маску действий
            action_mask = torch.ones_like(action_tokens, dtype=torch.bool)
            episode_action_masks_list[idx].append(action_mask)

            # Добавляем пару (форматированное_состояние, форматированное_действие) в историю
            current_state_str = current_step_states[i] # Берем предрасчитанное состояние
            histories[idx].append((current_state_str, action_str)) # Добавляем кортеж в историю

        # Если есть валидные действия, выполняем шаг для соответствующих симуляторов
        if valid_action_values:
            # Заполняем действия для всех симуляторов (0.0 для невалидных/неактивных)
            step_actions_all_sims = [0.0] * n_simulators
            for i, sim_sub_idx in enumerate(valid_sim_indices):
                abs_idx = active_indices[sim_sub_idx] # Получаем абсолютный индекс
                step_actions_all_sims[abs_idx] = valid_action_values[i] # Применяем действие

            # Выполняем шаг параллельно для всех симуляторов
            results = parallel_sim.step(step_actions_all_sims)

            # Обрабатываем результаты
            for i, (next_state, reward, done, info) in enumerate(results):
                # Обновляем данные только для тех симуляторов, которые были активны и имели валидное действие на этом шаге
                is_valid_step = (i in active_indices) and (active_indices.index(i) in valid_sim_indices)

                if not episodes_done[i] and is_valid_step:
                    # Добавляем награду за шаг (добыча + формат)
                    format_reward_sum = sum(episode_format_rewards_list[i][-1].values())
                    full_reward = reward + format_reward_sum
                    episode_rewards_list[i].append(full_reward)
                    episode_total_rewards[i] += full_reward
                    steps_completed[i] += 1

                    # Обновляем информацию о добыче
                    if hasattr(parallel_sim.simulators[i], 'cumulative_production'):
                        episode_production[i] = parallel_sim.simulators[i].cumulative_production

                    if verbose:
                        print(f"Симулятор {i+1}: Шаг {steps_completed[i]}, Награда: {reward:.4f}, "
                              f"Общая награда: {episode_total_rewards[i]:.4f}, "
                              f"Добыча: {episode_production[i]:.2f} м³")

                    # Если эпизод завершен, отмечаем его
                    episodes_done[i] = done
                    if done and verbose:
                        print(f"{COLOR_MAGENTA}Симулятор {i+1}: Эпизод завершен после {steps_completed[i]} шагов.{COLOR_RESET}")

        # Если не было валидных действий у АКТИВНЫХ симуляторов на этом шаге, добавляем только штрафы за формат
        # Проверяем только те, что были активны, но не попали в valid_sim_indices
        active_but_failed_indices = [active_indices[i] for i, sim_sub_idx in enumerate(active_indices) if i not in valid_sim_indices]
        for idx in active_but_failed_indices:
             if not episodes_done[idx]: # Только если эпизод еще не закончен
                 format_reward_sum = sum(episode_format_rewards_list[idx][-1].values())
                 episode_rewards_list[idx].append(format_reward_sum) # Добавляем только награду за формат
                 episode_total_rewards[idx] += format_reward_sum
                 steps_completed[idx] += 1 # Шаг все равно засчитываем

                 if verbose:
                     print(f"Симулятор {idx+1}: Шаг {steps_completed[idx]} пропущен (невалидное действие), Штраф за формат: {format_reward_sum:.4f}, "
                           f"Общая награда: {episode_total_rewards[idx]:.4f}")

    # После завершения всех шагов собираем финальные данные
    for idx in range(n_simulators):
        try:
            # Пропускаем симуляторы, которые не начали работу
            if steps_completed[idx] == 0:
                continue
                
            # Собираем токены для эпизода
            all_tokens = []
            all_masks = []
            
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
            
            # Преобразуем награды в тензор
            episode_rewards = torch.tensor(episode_rewards_list[idx], device=device)
            all_rewards.append(episode_rewards)
            
            # Собираем статистику эпизода
            episode_stats = {
                "steps": steps_completed[idx],
                "reward": episode_total_rewards[idx],
                "production": episode_production[idx],
                "time": time.time() - start_time,
                "actions": episode_actions_list[idx],
                "format_rewards": episode_format_rewards_list[idx],
                "skipped_steps": skipped_steps[idx]
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