# -*- coding: utf-8 -*-
import os
import torch
import re
import time
import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Any, Union
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, GenerationConfig
from simulators.single_well.simulator import SingleWellSimulator
from simulators.multi_well.simulator import MultiWellSimulator
from grpo.prompts import get_first_step_prompt, get_subsequent_step_prompt, BASE_PROMPT_TEMPLATE

# Импортируем константу для дискретных действий и функции из utils
from grpo.utils import DISCRETE_ACTIONS, parse_llm_action, COLOR_RESET, COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_BLUE, COLOR_CYAN, COLOR_MAGENTA

# Константы для цветов в консоли (больше не нужны, импортированы из utils)
# COLOR_RESET = "\033[0m"
# COLOR_GREEN = "\033[92m"
# COLOR_RED = "\033[91m"
# COLOR_YELLOW = "\033[93m"
# COLOR_BLUE = "\033[94m"
# COLOR_CYAN = "\033[96m"
# COLOR_MAGENTA = "\033[35m"

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
    
    def reset_all_to_random_state(self, min_depletion: float = 0.0, max_depletion: float = 0.9,
                                use_realistic_ranges: bool = True):
        """
        Сбрасывает все симуляторы к случайным промежуточным состояниям.
        
        ВАЖНО: Этот метод следует вызывать только в начале эпизода, а не на каждом шаге,
        чтобы модель могла изучать последовательность взаимосвязанных действий в рамках
        одного эпизода разработки скважины.
        
        Args:
            min_depletion (float): Минимальное значение степени истощения (0.0 = начало разработки)
            max_depletion (float): Максимальное значение степени истощения (1.0 = полное истощение)
            use_realistic_ranges (bool): Использовать ли реалистичные ограничения для параметров
            
        Returns:
            List[np.ndarray]: Список случайных начальных состояний для всех симуляторов
        """
        for i, simulator in enumerate(self.simulators):
            # Проверяем, поддерживает ли симулятор метод reset_to_random_state
            if hasattr(simulator, 'reset_to_random_state'):
                # Для каждого симулятора устанавливаем свой диапазон истощения
                # Это создаст разнообразие в тренировочных данных
                individual_min = max(0.0, min_depletion + random.uniform(-0.1, 0.1))
                individual_max = min(0.95, max_depletion + random.uniform(-0.1, 0.1))
                
                # Гарантируем, что min < max
                if individual_min >= individual_max:
                    individual_min = max(0.0, individual_max - 0.1)
                
                self.current_states[i] = simulator.reset_to_random_state(
                    min_depletion=individual_min,
                    max_depletion=individual_max,
                    use_realistic_ranges=use_realistic_ranges
                )
            else:
                # Если симулятор не поддерживает случайное состояние, используем обычный сброс
                self.current_states[i] = simulator.reset()
        
        return self.current_states
    
    def reset_simulator(self, index: int):
        """Сбрасывает конкретный симулятор по индексу."""
        if 0 <= index < self.n_simulators:
            self.current_states[index] = self.simulators[index].reset()
            return self.current_states[index]
        else:
            raise IndexError(f"Индекс {index} выходит за пределы диапазона симуляторов (0-{self.n_simulators-1})")
    
    def reset_simulator_to_random_state(self, index: int, min_depletion: float = 0.0, max_depletion: float = 0.9,
                                      use_realistic_ranges: bool = True):
        """
        Сбрасывает конкретный симулятор к случайному промежуточному состоянию.
        
        Args:
            index (int): Индекс симулятора для сброса
            min_depletion (float): Минимальное значение степени истощения
            max_depletion (float): Максимальное значение степени истощения
            use_realistic_ranges (bool): Использовать ли реалистичные ограничения для параметров
            
        Returns:
            np.ndarray: Случайное начальное состояние для указанного симулятора
        """
        if 0 <= index < self.n_simulators:
            simulator = self.simulators[index]
            
            # Проверяем, поддерживает ли симулятор метод reset_to_random_state
            if hasattr(simulator, 'reset_to_random_state'):
                self.current_states[index] = simulator.reset_to_random_state(
                    min_depletion=min_depletion,
                    max_depletion=max_depletion,
                    use_realistic_ranges=use_realistic_ranges
                )
            else:
                # Если симулятор не поддерживает случайное состояние, используем обычный сброс
                self.current_states[index] = simulator.reset()
                
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
    
    def get_prompts(self, histories: Optional[List[List[str]]] = None) -> List[str]:
        """
        Формирует промпты для всех симуляторов с учетом их текущих состояний.
        
        Args:
            histories: Список историй для каждого симулятора
            
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
            
            # Используем разные шаблоны в зависимости от наличия истории
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
        Извлекает значения действий из ответа модели.
        
        Args:
            response (str): Текст ответа модели.
            
        Returns:
            Tuple[Optional[float], Dict[str, float]]: Кортеж, содержащий извлеченное значение действия 
            (от 0 до 1 или None, если формат некорректен) и словарь наград за формат.
        """
        # Вызываем функцию из utils вместо реализации метода здесь
        return parse_llm_action(response)


def parallel_rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    parallel_sim: ParallelSimulator,
    n_steps: int = 20,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = True,
    use_random_states: bool = False,
    random_state_min_depletion: float = 0.0,
    random_state_max_depletion: float = 0.9,
    use_realistic_ranges: bool = True,
) -> Tuple[
    List[torch.Tensor],   # all_episode_tokens
    List[torch.Tensor],   # all_action_masks
    List[torch.Tensor],   # all_rewards
    List[Dict]            # all_episode_stats
]:
    """
    Выполняет параллельные роллауты для всех симуляторов.
    
    Args:
        model: Языковая модель
        tokenizer: Токенизатор
        parallel_sim: Объект класса ParallelSimulator
        n_steps: Максимальное количество шагов для каждого роллаута
        temperature: Температура генерации
        top_p: Параметр top_p для генерации
        verbose: Выводить ли информацию в консоль
        use_random_states: Использовать ли случайные начальные состояния для симуляторов
        random_state_min_depletion: Минимальная степень истощения для случайных состояний
        random_state_max_depletion: Максимальная степень истощения для случайных состояний
        use_realistic_ranges: Использовать ли реалистичные ограничения для параметров
    
    Returns:
        Кортеж из:
            all_episode_tokens: Список тензоров с токенами для каждого эпизода
            all_action_masks: Список тензоров с масками действий
            all_rewards: Список тензоров с наградами
            all_episode_stats: Список словарей со статистикой
    """
    model.eval()  # Переводим модель в режим оценки
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
    skipped_steps = [0] * n_simulators  # Счетчик пропущенных шагов для каждого симулятора
    
    # Истории для каждого симулятора
    histories = [[] for _ in range(n_simulators)]
    
    # Данные эпизодов
    episode_tokens_list = [[] for _ in range(n_simulators)]
    episode_action_masks_list = [[] for _ in range(n_simulators)]
    episode_rewards_list = [[] for _ in range(n_simulators)]
    episode_actions_list = [[] for _ in range(n_simulators)]
    episode_total_rewards = [0.0] * n_simulators
    episode_production = [0.0] * n_simulators
    episode_format_rewards_list = [[] for _ in range(n_simulators)]  # Добавляем список для хранения формата наград
    
    start_time = time.time()
    
    # Сбрасываем все симуляторы в начальное состояние (обычное или случайное)
    if use_random_states:
        if verbose:
            print(f"{COLOR_CYAN}Используем случайные начальные состояния "
                  f"(диапазон истощения: {random_state_min_depletion:.2f}-{random_state_max_depletion:.2f}, "
                  f"реалистичные диапазоны: {'Да' if use_realistic_ranges else 'Нет'}){COLOR_RESET}")
        parallel_sim.reset_all_to_random_state(
            min_depletion=random_state_min_depletion,
            max_depletion=random_state_max_depletion,
            use_realistic_ranges=use_realistic_ranges
        )
    else:
        if verbose:
            print(f"{COLOR_CYAN}Используем начальное состояние скважин{COLOR_RESET}")
        parallel_sim.reset_all()
    
    # Создаем конфигурацию генерации
    generation_config = GenerationConfig(
        max_new_tokens=10,  # Ограничиваем длину генерации для эффективности
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
        
        # Получаем промпты только для активных симуляторов
        active_histories = [histories[i] for i in active_indices]
        prompts = parallel_sim.get_prompts(active_histories)
        
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
                # Создаем запасной вариант выходов с числом "5" вместо "0.5"
                # Раньше было: torch.ones(...) * tokenizer.encode("0.5", add_special_tokens=False)[0]
                # Теперь используем "5" как запасной вариант (средний из 10)
                outputs = torch.cat([
                    tokenized_inputs.input_ids,
                    torch.ones((len(active_indices), 1), dtype=torch.long, device=device) * tokenizer.encode("5", add_special_tokens=False)[0]
                ], dim=1)
        
        # Извлекаем новые токены и действия
        actions = []
        valid_actions = []  # Список для хранения валидных действий
        valid_indices = []  # Список для хранения индексов валидных действий
        for i, idx in enumerate(active_indices):
            # Получаем только новые токены (без промпта)
            # Batch size может быть > 1, поэтому берем соответствующую строку
            new_tokens = outputs[i, tokenized_inputs.input_ids.shape[1]:]
            
            # Декодируем ответ
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            if verbose:
                print(f"Симулятор {idx+1}: {COLOR_GREEN}Ответ модели:{COLOR_RESET} '{response}'")
            
            # Очищаем ответ от символов, которые модель часто повторяет
            response = re.sub(r'(\*+)', '', response)  # Удаляем звездочки
            response = re.sub(r'`+', '', response)      # Удаляем обратные кавычки
            response = re.sub(r'\s+', ' ', response).strip()  # Нормализуем пробелы
            
            # Извлекаем действие из ответа
            action, format_rewards = parallel_sim.parse_llm_action(response)
            
            if action is not None:
                if verbose:
                    print(f"Симулятор {idx+1}: {COLOR_YELLOW}Действие:{COLOR_RESET} {action:.4f}")
                    print(f"Симулятор {idx+1}: {COLOR_BLUE}Награды за форматирование:{COLOR_RESET} {format_rewards}")
                
                # Сохраняем действие и индекс
                valid_actions.append(action)
                valid_indices.append(i)
                skipped_steps[idx] = 0  # Сбрасываем счетчик пропущенных шагов
            else:
                if verbose:
                    print(f"Симулятор {idx+1}: {COLOR_RED}Действие пропущено из-за неправильного формата{COLOR_RESET}")
                    print(f"Симулятор {idx+1}: {COLOR_BLUE}Штраф за форматирование:{COLOR_RESET} {format_rewards}")
                
                # Отмечаем шаг как пропущенный и увеличиваем счетчик
                skipped_steps[idx] += 1
                
                # Если слишком много пропущенных шагов подряд, завершаем эпизод
                if skipped_steps[idx] > 5:
                    if verbose:
                        print(f"{COLOR_RED}Симулятор {idx+1}: Слишком много пропущенных шагов подряд, завершаем эпизод.{COLOR_RESET}")
                    episodes_done[idx] = True
            
            # Заполняем пустым значением (будет игнорироваться, если шаг пропущен)
            actions.append(action if action is not None else 0.0)
            episode_actions_list[idx].append(action if action is not None else 0.0)
            episode_format_rewards_list[idx].append(format_rewards)  # Сохраняем формат наград
            
            # Сохраняем токены действия в любом случае (для обучения)
            action_tokens = new_tokens
            
            # Убеждаемся, что тензор не пустой
            if action_tokens.numel() == 0:
                action_tokens = torch.tensor([tokenizer.encode("0", add_special_tokens=False)[0]], 
                                          device=device)
            
            episode_tokens_list[idx].append(action_tokens)
            
            # --- NEW точная маска только на число ---
            num_match = re.search(r'\s*([\d\.]+)\s*', response)
            mask = torch.zeros_like(action_tokens, dtype=torch.bool)
            if num_match:
                num_tokens = tokenizer(num_match.group(1), add_special_tokens=False).input_ids
                # Число идёт последними токенами ответа → помечаем столько же
                mask[-len(num_tokens):] = True
            else:
                # если формат неправильный – оставляем всю маску (штраф уже начислен)
                mask[:] = True
            episode_action_masks_list[idx].append(mask)
            # --- END NEW ---
            
            # Добавляем информацию в историю только если действие было валидным
            if action is not None:
                current_state = parallel_sim.current_states[idx]
                histories[idx].append(f"Сост:{parallel_sim.format_short_state(current_state)}, Д:{action:.2f}")
        
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
                
                # Обновляем данные для активного симулятора
                episode_rewards_list[idx].append(reward + sum(episode_format_rewards_list[idx][-1].values()))
                full_reward = reward + sum(episode_format_rewards_list[idx][-1].values())
                episode_total_rewards[idx] += full_reward
                steps_completed[idx] += 1
                
                # Обновляем информацию о добыче
                if hasattr(parallel_sim.simulators[idx], 'cumulative_production'):
                    episode_production[idx] = parallel_sim.simulators[idx].cumulative_production
                
                if verbose:
                    print(f"Симулятор {idx+1}: Шаг {steps_completed[idx]}, Награда: {reward:.4f}, "
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
                format_reward = sum(episode_format_rewards_list[idx][-1].values())
                episode_rewards_list[idx].append(format_reward)
                episode_total_rewards[idx] += format_reward
                
                if verbose:
                    print(f"Симулятор {idx+1}: Шаг пропущен, Штраф за формат: {format_reward:.4f}, "
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