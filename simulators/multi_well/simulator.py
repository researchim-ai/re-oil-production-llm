import numpy as np
import random
from simulators.single_well.simulator import SingleWellSimulator

class MultiWellSimulator:
    """
    Симулятор для нескольких нефтяных скважин.
    
    Моделирует группу скважин, которые могут быть независимыми или 
    взаимодействовать через общий резервуар.
    
    Состояние: список состояний всех скважин + состояние резервуара
    Действие: список управляющих воздействий для всех скважин (choke_opening для каждой)
    Награда: сумма дебитов всех скважин на текущем шаге
    """
    
    def __init__(self,
                 n_wells: int = 3,  # количество скважин
                 interaction_strength: float = 0.1,  # параметр взаимодействия между скважинами (0-1)
                 shared_reservoir: bool = True,  # используют ли скважины общий резервуар
                 total_volume: float = 3e6,  # общий объем резервуара (м3)
                 **well_params  # параметры для отдельных скважин
                 ):
        """
        Инициализация мультискважинного симулятора.
        
        Args:
            n_wells: количество скважин
            interaction_strength: степень взаимовлияния между скважинами (0-1)
            shared_reservoir: используют ли скважины общий резервуар
            total_volume: общий объем резервуара (м3)
            well_params: параметры для инициализации скважин
        """
        self.n_wells = n_wells
        self.interaction_strength = np.clip(interaction_strength, 0.0, 1.0)
        self.shared_reservoir = shared_reservoir
        self.total_volume = total_volume
        
        # Добавляем сохранение последних действий
        self.last_actions = [None] * n_wells
        
        # Распределение объема резервуара между скважинами
        if shared_reservoir:
            # При общем резервуаре каждая скважина использует весь объем
            well_volume = total_volume
        else:
            # При независимых резервуарах объем делится между скважинами
            well_volume = total_volume / n_wells
        
        # Сохраняем параметры для будущего использования
        self.well_params = well_params.copy()
        self.well_volume = well_volume
        
        # Создаем симуляторы для каждой скважины
        self.simulators = []
        for i in range(n_wells):
            # Создаем копию параметров, чтобы не изменять оригинальные
            params = well_params.copy()
            # Устанавливаем объем для каждой скважины
            params['total_volume'] = well_volume
            # Создаем симулятор скважины
            simulator = SingleWellSimulator(**params)
            self.simulators.append(simulator)
        
        # Инициализация состояния
        self.state = self.reset()
        
    def reset(self) -> np.ndarray:
        """
        Сбрасывает симулятор к начальному состоянию.
        
        Returns:
            np.ndarray: Начальное состояние системы, включающее состояния всех скважин
        """
        # Сбрасываем состояние каждой скважины
        states = [sim.reset() for sim in self.simulators]
        
        # Сбрасываем последние действия
        self.last_actions = [None] * self.n_wells
        
        # Объединяем все состояния в один вектор
        self.state = np.concatenate(states)
        
        return self.state
        
    def reset_to_random_state(self, min_depletion: float = 0.0, max_depletion: float = 0.9,
                               use_realistic_ranges: bool = True) -> np.ndarray:
        """
        Сбрасывает симулятор к случайному промежуточному состоянию разработки группы скважин.
        
        Args:
            min_depletion (float): Минимальное значение степени истощения (0.0 = начало разработки)
            max_depletion (float): Максимальное значение степени истощения (1.0 = полное истощение)
            use_realistic_ranges (bool): Использовать ли реалистичные ограничения для параметров
            
        Returns:
            np.ndarray: Случайное промежуточное состояние
        """
        # Сначала сбрасываем в начальное состояние, чтобы корректно проинициализировать все переменные
        self.reset()
        
        # Ограничиваем максимальную степень истощения для реалистичности
        if use_realistic_ranges:
            # Обычно месторождение не добывается до полного истощения из-за экономических ограничений
            realistic_max_depletion = min(max_depletion, 0.85)  # Не более 85% от общего объема
            # В начале разработки обычно уже есть минимальный отбор для тестирования
            realistic_min_depletion = max(min_depletion, 0.01)  # Минимум 1% от общего объема
            
            # Используем ограниченные диапазоны
            min_depletion = realistic_min_depletion
            max_depletion = realistic_max_depletion
        
        # В многоскважинной системе скважины обычно имеют разную степень истощения
        # Выбираем общую степень истощения для всего месторождения
        field_depletion_ratio = random.uniform(min_depletion, max_depletion)
        
        # Накопленная добыча и время для всего месторождения
        total_field_production = field_depletion_ratio * self.total_volume
        
        # Рассчитываем время разработки для всего месторождения
        if use_realistic_ranges:
            # Нелинейная зависимость времени от степени истощения
            base_time = self.max_time * (field_depletion_ratio**0.9)
            time_variation = 0.12 * base_time  # 12% вариации
            field_time = min(self.max_time - self.dt, max(0, base_time + random.uniform(-time_variation, time_variation)))
        else:
            # Линейная зависимость
            field_time = field_depletion_ratio * self.max_time
        
        # Расчет пластового давления для общего месторождения
        if self.shared_reservoir:
            if use_realistic_ranges:
                # Нелинейное падение давления для общего резервуара
                pressure_factor = 1.0 - field_depletion_ratio**0.85
                pressure_variation = random.uniform(-0.03, 0.03)  # ±3% вариации
                field_pressure = self.initial_reservoir_pressure * max(0, min(1, pressure_factor + pressure_variation))
            else:
                # Линейное падение давления
                field_pressure = self.initial_reservoir_pressure * (1.0 - field_depletion_ratio)
            
            # Применяем одинаковое давление ко всем скважинам при общем резервуаре
            for i in range(self.n_wells):
                self.reservoir_pressures[i] = field_pressure
        
        # Распределяем добычу между скважинами
        remaining_production = total_field_production
        
        # Первый проход: распределяем добычу между скважинами
        well_production_ratios = []
        
        if use_realistic_ranges:
            # В реальности скважины имеют разные объемы добычи
            # Используем распределение, где некоторые скважины добывают больше других
            for i in range(self.n_wells):
                if i == self.n_wells - 1:
                    # Последняя скважина получает оставшуюся долю
                    well_production_ratios.append(1.0 - sum(well_production_ratios))
                else:
                    # Генерируем случайную долю, но с учетом оставшихся скважин
                    remaining_wells = self.n_wells - i
                    max_share = (1.0 - sum(well_production_ratios)) / remaining_wells * 2
                    min_share = max(0, (1.0 - sum(well_production_ratios)) / remaining_wells * 0.5)
                    well_production_ratios.append(random.uniform(min_share, max_share))
        else:
            # Равномерное распределение с небольшой случайностью
            for i in range(self.n_wells):
                if i == self.n_wells - 1:
                    # Последняя скважина получает оставшуюся долю
                    well_production_ratios.append(1.0 - sum(well_production_ratios))
                else:
                    well_production_ratios.append(random.uniform(0.0, 1.0 / self.n_wells * 1.5))
        
        # Нормализуем доли, чтобы сумма была равна 1
        total_ratio = sum(well_production_ratios)
        well_production_ratios = [ratio / total_ratio for ratio in well_production_ratios]
        
        # Применяем распределение к скважинам
        for i in range(self.n_wells):
            # Рассчитываем добычу для конкретной скважины
            self.cumulative_productions[i] = total_field_production * well_production_ratios[i]
            
            # Устанавливаем время для всех скважин одинаковое
            self.times[i] = field_time
            
            # Для независимых резервуаров рассчитываем давление индивидуально
            if not self.shared_reservoir:
                # Индивидуальное истощение для этой скважины
                well_depletion = self.cumulative_productions[i] / (self.total_volume / self.n_wells)
                
                if use_realistic_ranges:
                    # Нелинейное падение давления
                    pressure_factor = 1.0 - well_depletion**0.85
                    pressure_variation = random.uniform(-0.05, 0.05)  # ±5% вариации
                    self.reservoir_pressures[i] = self.initial_reservoir_pressure * max(0, min(1, pressure_factor + pressure_variation))
                else:
                    # Линейное падение давления
                    self.reservoir_pressures[i] = self.initial_reservoir_pressure * (1.0 - well_depletion)
            
            # Генерируем случайное последнее действие для каждой скважины
            if use_realistic_ranges:
                # В зависимости от индивидуального истощения скважины
                well_depletion = self.cumulative_productions[i] / (self.total_volume / self.n_wells)
                
                if well_depletion < 0.3:
                    # Начальная стадия разработки
                    last_action = random.uniform(0.7, 1.0)
                elif well_depletion < 0.7:
                    # Средняя стадия разработки
                    last_action = random.uniform(0.4, 0.8)
                else:
                    # Поздняя стадия разработки
                    last_action = random.uniform(0.1, 0.5)
            else:
                # Полностью случайное значение
                last_action = random.uniform(0.0, 1.0)
            
            self.last_actions[i] = last_action
            self.valve_openings[i] = last_action
            
            # Рассчитываем текущий дебит для скважины
            delta_p = max(0.0, self.reservoir_pressures[i] - self.bhp)
            self.current_rates[i] = self.productivity_index * delta_p * last_action
            
            # Обратное влияние других скважин при наличии взаимодействия
            if self.interaction_strength > 0:
                for j in range(self.n_wells):
                    if i != j:
                        # Влияние j-й скважины на i-ю
                        self.current_rates[i] -= self.interaction_strength * self.productivity_index * \
                                               delta_p * self.valve_openings[j] / self.n_wells
                
                # Гарантируем, что дебит не отрицательный
                self.current_rates[i] = max(0.0, self.current_rates[i])
        
        # Формируем состояние как вектор
        state = []
        for i in range(self.n_wells):
            well_state = [
                self.reservoir_pressures[i],
                self.current_rates[i],
                self.cumulative_productions[i],
                self.times[i]
            ]
            state.extend(well_state)
        
        self.state = np.array(state)
        return self.state
    
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Выполняет один шаг симуляции для всех скважин.

        Args:
            actions (float or np.ndarray): Управляющие воздействия - choke_openings (от 0 до 1) для каждой скважины.
                                         Если передано одно число, оно применяется ко всем скважинам.

        Returns:
            tuple[np.ndarray, float, bool, dict]: Кортеж (новое_состояние, награда, флаг_завершения, инфо).
                                               новое_состояние: массив состояний всех скважин
                                               награда: суммарный дебит на этом шаге
                                               флаг_завершения: True, если симуляция закончена
                                               инфо: дополнительная информация о шаге
        """
        # Предварительная обработка действий
        if np.isscalar(actions):
            # Если передано одно число, создаем массив с этим числом для всех скважин
            actions = np.ones(self.n_wells) * actions
        else:
            # Иначе преобразуем в массив нужной длины
            actions = np.asarray(actions).flatten()
            if len(actions) < self.n_wells:
                # Если действий меньше, чем скважин, дополняем последним значением
                actions = np.pad(actions, (0, self.n_wells - len(actions)), 'edge')
            elif len(actions) > self.n_wells:
                # Если действий больше, чем скважин, обрезаем
                actions = actions[:self.n_wells]

        # Клиппируем действия в диапазон [0, 1]
        actions = np.clip(actions, 0.0, 1.0)
        self.current_valve_openings = actions  # Сохраняем текущие открытия штуцеров
        
        # Сохраняем последние действия
        for i in range(self.n_wells):
            self.last_actions[i] = actions[i]

        # 1. Обновляем время
        self.time += self.dt

        # 2. Рассчитываем дебиты и добычу для каждой скважины с учетом текущего давления
        current_rates = np.zeros(self.n_wells)
        volumes_produced = np.zeros(self.n_wells)
        total_produced = 0.0

        # 3. Для каждой скважины рассчитываем добычу
        for i in range(self.n_wells):
            # Получаем текущие данные для скважины
            start_idx = i * self.state_dim_per_well
            well_res_pressure = self.state[start_idx]
            well_bhp = self.state[start_idx + 1]
            
            # Рассчитываем дебит для скважины
            delta_p = max(0.0, well_res_pressure - well_bhp)
            rate = self.pi * delta_p * actions[i]
            current_rates[i] = rate
            
            # Рассчитываем добычу за этот шаг
            volume = rate * self.dt
            volumes_produced[i] = volume
            total_produced += volume
            
            # Обновляем накопленную добычу для скважины
            well_cumulative_prod = self.state[start_idx + 2] + volume
            self.state[start_idx + 2] = well_cumulative_prod
            
        # Обновляем накопленную общую добычу
        self.cumulative_production += total_produced
        self.current_rates = current_rates  # Сохраняем текущие дебиты

        # 4. Обновляем пластовое давление для всех скважин
        if self.shared_reservoir:
            # Для общего резервуара падение давления зависит от общей добычи
            depletion_ratio = min(1.0, self.cumulative_production / self.total_volume) if self.total_volume > 0 else 1.0
            new_pressure = self.initial_reservoir_pressure * (1.0 - depletion_ratio)
            new_pressure = max(0.0, new_pressure)
            
            # Применяем новое давление ко всем скважинам
            for i in range(self.n_wells):
                start_idx = i * self.state_dim_per_well
                self.state[start_idx] = new_pressure
        else:
            # Для отдельных резервуаров
            for i in range(self.n_wells):
                start_idx = i * self.state_dim_per_well
                
                # Рассчитываем падение давления от добычи этой скважины
                well_cumulative_prod = self.state[start_idx + 2]
                depletion_ratio = min(1.0, well_cumulative_prod / (self.total_volume / self.n_wells)) if self.total_volume > 0 else 1.0
                new_pressure = self.initial_reservoir_pressure * (1.0 - depletion_ratio)
                
                # Учитываем влияние соседних скважин
                if self.interaction_strength > 0:
                    for j in range(self.n_wells):
                        if i != j:
                            # Влияние j-й скважины на i-ю
                            j_start_idx = j * self.state_dim_per_well
                            j_cumulative_prod = self.state[j_start_idx + 2]
                            j_depletion = j_cumulative_prod / (self.total_volume / self.n_wells) if self.total_volume > 0 else 1.0
                            # Снижаем давление пропорционально интерференции
                            new_pressure -= self.initial_reservoir_pressure * j_depletion * self.interaction_strength / (self.n_wells - 1)
                
                # Гарантируем неотрицательное давление
                new_pressure = max(0.0, new_pressure)
                self.state[start_idx] = new_pressure

        # 5. Обновляем время для всех скважин
        for i in range(self.n_wells):
            self.state[i * self.state_dim_per_well + 3] = self.time

        # 6. Считаем общую награду как сумму дебитов всех скважин
        reward = sum(current_rates)

        # 7. Проверяем условие завершения
        # Симуляция завершается, если:
        # - Вышло максимальное время
        # - Все скважины перестали давать нефть (пластовое давление <= BHP для всех скважин)
        all_wells_depleted = True
        for i in range(self.n_wells):
            start_idx = i * self.state_dim_per_well
            if self.state[start_idx] > self.state[start_idx + 1]:  # res_pressure > bhp
                all_wells_depleted = False
                break

        done = self.time >= self.max_time or all_wells_depleted

        # 8. Создаем словарь с дополнительной информацией
        info = {
            'well_rates': current_rates.copy(),
            'well_volumes': volumes_produced.copy(),
            'total_produced': total_produced,
            'depletion_ratio': self.cumulative_production / self.total_volume if self.total_volume > 0 else 1.0,
            'remaining_time': max(0.0, self.max_time - self.time),
            'valve_openings': actions.copy()
        }

        return self.state, reward, done, info
    
    def get_state_dim(self) -> int:
        """Возвращает размерность вектора состояния."""
        return len(self.state)
    
    def get_action_dim(self) -> int:
        """Возвращает размерность вектора действия."""
        return self.n_wells

# Пример использования
if __name__ == '__main__':
    # Параметры для всех скважин
    well_params = {
        'initial_reservoir_pressure': 200.0,
        'initial_bhp': 50.0,
        'productivity_index': 0.1,
        'dt': 1.0,
        'max_time': 365.0
    }
    
    # Создаем симулятор с 3 скважинами
    simulator = MultiWellSimulator(
        n_wells=3,
        interaction_strength=0.2,
        shared_reservoir=True,
        total_volume=3e6,
        **well_params
    )
    
    state = simulator.reset()
    print(f"Начальное состояние: {state}")
    
    total_reward = 0
    done = False
    step = 0
    
    while not done:
        # Пример стратегии: разное открытие штуцеров для разных скважин
        actions = np.array([0.8, 0.5, 0.3])
        
        next_state, reward, done, info = simulator.step(actions)
        total_reward += reward
        state = next_state
        step += 1
        
        if step % 30 == 0 or done:
            # Разделяем общий вектор состояния на состояния отдельных скважин
            well_states = []
            state_size = len(state) // simulator.n_wells
            for i in range(simulator.n_wells):
                well_state = state[i*state_size:(i+1)*state_size]
                well_states.append(well_state)
            
            print(f"Шаг: {step}, Время: {well_states[0][3]:.1f} дней")
            for i, well_state in enumerate(well_states):
                print(f"  Скважина {i+1}: Давление: {well_state[0]:.2f} атм, Дебит: {well_state[1]:.2f} м3/сут, Добыча: {well_state[2]:.1f} м3")
            print(f"  Суммарный дебит: {reward:.2f} м3/сут, Завершено: {done}")
    
    print(f"Симуляция завершена после {step} шагов ({well_states[0][3]:.1f} дней).")
    
    # Рассчитываем суммарную добычу по всем скважинам
    total_production = sum(well_state[2] for well_state in well_states)
    print(f"Суммарная добыча: {total_production:.1f} м3")
    print(f"Суммарная награда * dt: {total_reward * simulator.simulators[0].dt:.1f}")
