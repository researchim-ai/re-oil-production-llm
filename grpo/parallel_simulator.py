# -*- coding: utf-8 -*-
import os
import torch
import re
import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, GenerationConfig
from simulators.single_well.simulator import SingleWellSimulator
from simulators.multi_well.simulator import MultiWellSimulator

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
            
            # Формируем промпт в зависимости от типа симулятора
            if hasattr(simulator, 'well_names') and len(getattr(simulator, 'well_names', [])) > 1:
                system_prompt = """ЗАДАЧА: Управление добычей нефти в нескольких скважинах.
ТРЕБУЕТСЯ: Указать степень открытия штуцера от 0 до 1.
ФОРМАТ ОТВЕТА: Только число от 0 до 1, без текста."""
            else:
                system_prompt = """ЗАДАЧА: Управление добычей нефти в одной скважине.
ТРЕБУЕТСЯ: Указать степень открытия штуцера от 0 до 1.
ФОРМАТ ОТВЕТА: Только число от 0 до 1, без текста."""
            
            # Ограничиваем историю до 2 последних взаимодействий для экономии токенов
            if len(history) > 2:
                history = history[-2:]
            
            if not history:
                # Первый шаг эпизода - более директивный промпт
                prompt = f"""{system_prompt}

Состояние: {state_text}

ОТВЕТЬТЕ ТОЛЬКО ЧИСЛОМ от 0 до 1 без пояснений:"""
            else:
                # Последующие шаги с более директивным промптом
                prompt = f"""{system_prompt}

История: {' | '.join(history)}
Состояние: {state_text}

ОТВЕТЬТЕ ТОЛЬКО ЧИСЛОМ от 0 до 1 без пояснений:"""
            
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
    
    def parse_llm_action(self, response: str) -> float:
        """
        Извлекает значение степени открытия штуцера из ответа языковой модели.
        """
        try:
            # Очистка ответа
            clean_response = response.strip()
            
            # Если ответ пустой, возвращаем значение по умолчанию
            if not clean_response:
                return 0.5
            
            # Сначала проверим на явные случаи полного открытия/закрытия
            if any(phrase in clean_response.lower() for phrase in ["полностью открыть", "максимально открыть", "открыть полностью"]):
                return 1.0
            elif any(phrase in clean_response.lower() for phrase in ["полностью закрыть", "закрыть полностью", "закрыть штуцер"]):
                return 0.0
            
            # Пытаемся найти число в ответе - первое и самое строгое соответствие
            strict_match = re.match(r'^\s*(\d+(?:\.\d+)?)\s*$', clean_response)
            if strict_match:
                value = float(strict_match.group(1))
                # Нормализуем значение
                if value > 1 and value <= 100:
                    value /= 100.0
                # Ограничиваем диапазон
                value = max(0.0, min(1.0, value))
                return value
                
            # Если не нашли строгое соответствие, ищем любое число в строке
            number_match = re.search(r'(\d+(?:\.\d+)?)', clean_response)
            if number_match:
                value = float(number_match.group(1))
                # Нормализуем значение
                if value > 1 and value <= 100:
                    value /= 100.0
                # Ограничиваем диапазон
                value = max(0.0, min(1.0, value))
                return value
            
            # Если не удалось извлечь значение, используем значение по умолчанию
            return 0.5
        except Exception as e:
            # При любой ошибке возвращаем безопасное значение
            return 0.5


def parallel_rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    parallel_sim: ParallelSimulator,
    n_steps: int = 10,
    temperature: float = 0.3,
    verbose: bool = True,
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
        verbose: Выводить ли информацию в консоль
    
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
    
    # Истории для каждого симулятора
    histories = [[] for _ in range(n_simulators)]
    
    # Данные эпизодов
    episode_tokens_list = [[] for _ in range(n_simulators)]
    episode_action_masks_list = [[] for _ in range(n_simulators)]
    episode_rewards_list = [[] for _ in range(n_simulators)]
    episode_actions_list = [[] for _ in range(n_simulators)]
    episode_total_rewards = [0.0] * n_simulators
    episode_production = [0.0] * n_simulators
    
    start_time = time.time()
    
    # Создаем конфигурацию генерации
    generation_config = GenerationConfig(
        max_new_tokens=5,  # Нам нужно только короткое число
        do_sample=True,    # Сэмплирование для разнообразия
        temperature=temperature,
        top_p=0.95,
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
                # Создаем запасной вариант выходов
                outputs = torch.cat([
                    tokenized_inputs.input_ids,
                    torch.ones((len(active_indices), 1), dtype=torch.long, device=device) * tokenizer.encode("0.5", add_special_tokens=False)[0]
                ], dim=1)
        
        # Извлекаем новые токены и действия
        actions = []
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
            action = parallel_sim.parse_llm_action(response)
            if verbose:
                print(f"Симулятор {idx+1}: {COLOR_YELLOW}Действие:{COLOR_RESET} {action:.4f}")
            
            actions.append(action)
            episode_actions_list[idx].append(action)
            
            # Обновляем историю
            current_state = parallel_sim.current_states[idx]
            histories[idx].append(f"Сост:{parallel_sim.format_short_state(current_state)}, Д:{action:.2f}")
            
            # Сохраняем токены действия
            action_tokens = new_tokens
            
            # Убеждаемся, что тензор не пустой
            if action_tokens.numel() == 0:
                action_tokens = torch.tensor([tokenizer.encode("0", add_special_tokens=False)[0]], 
                                          device=device)
            
            episode_tokens_list[idx].append(action_tokens)
            
            # Создаем маску действий
            action_mask = torch.ones_like(action_tokens, dtype=torch.bool)
            episode_action_masks_list[idx].append(action_mask)
        
        # Выполняем шаг для активных симуляторов
        # Подготавливаем действия для всех симуляторов (заполняем пустышками для неактивных)
        all_actions = [0.0] * n_simulators
        for i, idx in enumerate(active_indices):
            all_actions[idx] = actions[i]
        
        # Выполняем шаг параллельно для всех симуляторов
        results = parallel_sim.step(all_actions)
        
        # Обрабатываем результаты
        for idx, (next_state, reward, done, info) in enumerate(results):
            # Пропускаем неактивные симуляторы
            if episodes_done[idx]:
                continue
            
            # Обновляем данные для активного симулятора
            episode_rewards_list[idx].append(reward)
            episode_total_rewards[idx] += reward
            steps_completed[idx] += 1
            
            # Обновляем информацию о добыче
            if hasattr(parallel_sim.simulators[idx], 'cumulative_production'):
                episode_production[idx] = parallel_sim.simulators[idx].cumulative_production
            
            if verbose:
                print(f"Симулятор {idx+1}: Шаг {steps_completed[idx]}, Награда: {reward:.4f}, "
                    f"Общая награда: {episode_total_rewards[idx]:.4f}, "
                    f"Добыча: {episode_production[idx]:.2f} м³")
            
            # Проверяем завершение эпизода
            if done:
                episodes_done[idx] = True
                if verbose:
                    print(f"{COLOR_MAGENTA}Симулятор {idx+1}: Эпизод завершен после {steps_completed[idx]} шагов.{COLOR_RESET}")
    
    # После завершения всех шагов собираем финальные данные
    for idx in range(n_simulators):
        # Пропускаем пустые эпизоды (хотя таких быть не должно)
        if not episode_tokens_list[idx]:
            continue
        
        # Собираем токены эпизода
        try:
            episode_tokens = torch.cat(episode_tokens_list[idx])
            episode_action_masks = torch.cat(episode_action_masks_list[idx])
            episode_rewards = torch.tensor(episode_rewards_list[idx], dtype=torch.float32)
            
            all_episode_tokens.append(episode_tokens)
            all_action_masks.append(episode_action_masks)
            all_rewards.append(episode_rewards)
            
            # Собираем статистику эпизода
            episode_stats = {
                "steps": steps_completed[idx],
                "reward": episode_total_rewards[idx],
                "production": episode_production[idx],
                "time": time.time() - start_time,
                "actions": episode_actions_list[idx],
            }
            all_episode_stats.append(episode_stats)
            
        except Exception as e:
            if verbose:
                print(f"{COLOR_RED}Ошибка при обработке эпизода {idx+1}: {e}{COLOR_RESET}")
    
    # Выводим итоговую статистику
    if verbose:
        avg_reward = sum(stats["reward"] for stats in all_episode_stats) / len(all_episode_stats) if all_episode_stats else 0
        avg_steps = sum(stats["steps"] for stats in all_episode_stats) / len(all_episode_stats) if all_episode_stats else 0
        avg_production = sum(stats["production"] for stats in all_episode_stats) / len(all_episode_stats) if all_episode_stats else 0
        
        print(f"\n{COLOR_MAGENTA}Итоги по {len(all_episode_stats)} эпизодам:{COLOR_RESET}")
        print(f"Средняя награда: {avg_reward:.4f}")
        print(f"Среднее количество шагов: {avg_steps:.1f}")
        print(f"Средняя добыча: {avg_production:.2f} м³")
    
    return all_episode_tokens, all_action_masks, all_rewards, all_episode_stats 