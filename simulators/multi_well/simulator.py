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
        
        # Сохраняем параметры скважин
        self.initial_reservoir_pressure = well_params.get('initial_reservoir_pressure', 200.0)
        self.bhp = well_params.get('initial_bhp', 50.0)
        self.pi = well_params.get('productivity_index', 0.1)
        self.dt = well_params.get('dt', 1.0)
        self.max_time = well_params.get('max_time', 365.0)
        
        # Инициализация атрибутов для отслеживания состояния симуляции
        self.time = 0.0
        self.cumulative_production = 0.0
        self.current_rates = np.zeros(n_wells)
        self.current_valve_openings = np.zeros(n_wells)
        self.reservoir_pressures = np.ones(n_wells) * self.initial_reservoir_pressure
        self.cumulative_productions = np.zeros(n_wells)
        self.times = np.zeros(n_wells)
        self.valve_openings = np.zeros(n_wells)
        
        # Количество элементов в состоянии одной скважины
        self.state_dim_per_well = 4  # [reservoir_pressure, flow_rate, cumulative_production, time]
        
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
        
        # Сбрасываем общие атрибуты симуляции
        self.time = 0.0
        self.cumulative_production = 0.0
        self.current_rates = np.zeros(self.n_wells)
        self.current_valve_openings = np.zeros(self.n_wells)
        self.reservoir_pressures = np.ones(self.n_wells) * self.initial_reservoir_pressure
        self.cumulative_productions = np.zeros(self.n_wells)
        self.times = np.zeros(self.n_wells)
        self.valve_openings = np.zeros(self.n_wells)
        
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
        self.cumulative_production = total_field_production
        
        # Рассчитываем время разработки для всего месторождения
        if use_realistic_ranges:
            # Нелинейная зависимость времени от степени истощения
            base_time = self.max_time * (field_depletion_ratio**0.9)
            time_variation = 0.12 * base_time  # 12% вариации
            field_time = min(self.max_time - self.dt, max(0, base_time + random.uniform(-time_variation, time_variation)))
        else:
            # Линейная зависимость
            field_time = field_depletion_ratio * self.max_time
        
        # Устанавливаем текущее время
        self.time = field_time
        
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
        
        # Формируем вектор состояния
        states = []
        
        # Применяем распределение к скважинам
        for i in range(self.n_wells):
            # Рассчитываем добычу для конкретной скважины
            well_production = total_field_production * well_production_ratios[i]
            self.cumulative_productions[i] = well_production
            
            # Устанавливаем время для всех скважин одинаковое
            self.times[i] = field_time
            
            # Для независимых резервуаров рассчитываем давление индивидуально
            if not self.shared_reservoir:
                # Индивидуальное истощение для этой скважины
                well_volume = self.total_volume / self.n_wells
                well_depletion = well_production / well_volume
                
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
                well_volume = self.total_volume / self.n_wells 
                well_depletion = self.cumulative_productions[i] / well_volume
                
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
            current_rate = self.pi * delta_p * last_action
            
            # Обратное влияние других скважин при наличии взаимодействия
            if self.interaction_strength > 0:
                for j in range(self.n_wells):
                    if i != j:
                        # Влияние j-й скважины на i-ю через дебит
                        j_delta_p = max(0.0, self.reservoir_pressures[j] - self.bhp)
                        j_rate = self.pi * j_delta_p * self.valve_openings[j]
                        # Уменьшаем дебит i-й скважины пропорционально активности j-й
                        current_rate -= self.interaction_strength * j_rate / (self.n_wells - 1)
                
                # Гарантируем, что дебит не отрицательный
                current_rate = max(0.0, current_rate)
            
            self.current_rates[i] = current_rate
            
            # Создаем состояние для скважины
            well_state = [
                self.reservoir_pressures[i],
                current_rate,
                self.cumulative_productions[i],
                field_time
            ]
            states.extend(well_state)
        
        # Сохраняем состояние
        self.state = np.array(states)
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
            
            # Рассчитываем дебит для скважины
            delta_p = max(0.0, well_res_pressure - self.bhp)
            rate = self.pi * delta_p * actions[i]
            
            # Учитываем интерференцию между скважинами
            if self.interaction_strength > 0:
                for j in range(self.n_wells):
                    if i != j:
                        # Влияние j-й скважины на i-ю через дебит
                        j_delta_p = max(0.0, self.state[j * self.state_dim_per_well] - self.bhp)
                        j_rate = self.pi * j_delta_p * actions[j]
                        # Уменьшаем дебит i-й скважины пропорционально активности j-й
                        rate -= self.interaction_strength * j_rate / (self.n_wells - 1)
            
            # Убеждаемся, что дебит не отрицательный
            rate = max(0.0, rate)
            current_rates[i] = rate
            
            # Рассчитываем добычу за этот шаг
            volume = rate * self.dt
            volumes_produced[i] = volume
            total_produced += volume
            
            # Обновляем накопленную добычу для скважины
            well_cumulative_prod = self.state[start_idx + 2] + volume
            self.state[start_idx + 2] = well_cumulative_prod
            self.cumulative_productions[i] = well_cumulative_prod
            
            # Обновляем текущий дебит в состоянии
            self.state[start_idx + 1] = rate
            
            # Обновляем время
            self.state[start_idx + 3] = self.time
            self.times[i] = self.time
            
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
                self.reservoir_pressures[i] = new_pressure
        else:
            # Для отдельных резервуаров с интерференцией
            for i in range(self.n_wells):
                start_idx = i * self.state_dim_per_well
                
                # Рассчитываем базовое падение давления от добычи этой скважины
                well_cumulative_prod = self.state[start_idx + 2]
                well_volume = self.total_volume / self.n_wells
                depletion_ratio = min(1.0, well_cumulative_prod / well_volume) if well_volume > 0 else 1.0
                new_pressure = self.initial_reservoir_pressure * (1.0 - depletion_ratio)
                
                # Учитываем влияние соседних скважин на пластовое давление
                if self.interaction_strength > 0:
                    for j in range(self.n_wells):
                        if i != j:
                            # Влияние j-й скважины на i-ю через давление
                            j_start_idx = j * self.state_dim_per_well
                            j_cumulative_prod = self.state[j_start_idx + 2]
                            j_depletion = j_cumulative_prod / well_volume if well_volume > 0 else 1.0
                            
                            # Снижаем давление пропорционально добыче соседней скважины
                            interference_factor = self.interaction_strength / (self.n_wells - 1)
                            pressure_drop = self.initial_reservoir_pressure * j_depletion * interference_factor
                            new_pressure -= pressure_drop
                
                # Гарантируем неотрицательное давление
                new_pressure = max(0.0, new_pressure)
                self.state[start_idx] = new_pressure
                self.reservoir_pressures[i] = new_pressure

        # 6. Считаем общую награду как сумму дебитов всех скважин
        reward = sum(current_rates)

        # 7. Проверяем условие завершения
        # Симуляция завершается, если:
        # - Вышло максимальное время
        # - Все скважины перестали давать нефть (пластовое давление <= BHP для всех скважин)
        all_wells_depleted = True
        for i in range(self.n_wells):
            start_idx = i * self.state_dim_per_well
            if self.state[start_idx] > self.bhp:  # res_pressure > bhp
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
            'valve_openings': actions.copy(),
            'well_pressures': self.reservoir_pressures.copy(),
            'well_cumulative_productions': self.cumulative_productions.copy()
        }

        return self.state, reward, done, info
    
    def get_state_dim(self) -> int:
        """Возвращает размерность вектора состояния."""
        return len(self.state)
    
    def get_action_dim(self) -> int:
        """Возвращает размерность вектора действия."""
        return self.n_wells
    
    def get_well_states(self) -> list:
        """
        Разделяет общий вектор состояния на индивидуальные состояния скважин.
        
        Returns:
            list: Список массивов numpy, каждый из которых содержит состояние отдельной скважины.
        """
        well_states = []
        for i in range(self.n_wells):
            start_idx = i * self.state_dim_per_well
            end_idx = start_idx + self.state_dim_per_well
            well_state = self.state[start_idx:end_idx]
            well_states.append(well_state)
        return well_states
    
    def calculate_interference_matrix(self) -> np.ndarray:
        """
        Рассчитывает матрицу интерференции между скважинами.
        
        Returns:
            np.ndarray: Матрица размера n_wells x n_wells, где каждый элемент [i, j] 
                       показывает влияние скважины j на скважину i.
        """
        # Инициализируем матрицу интерференции
        interference_matrix = np.zeros((self.n_wells, self.n_wells))
        
        # Заполняем матрицу
        for i in range(self.n_wells):
            for j in range(self.n_wells):
                if i == j:
                    # Влияние скважины на саму себя всегда 1.0
                    interference_matrix[i, j] = 1.0
                else:
                    # Влияние j-й скважины на i-ю определяется интенсивностью взаимодействия
                    # Можно модифицировать для учета пространственного расположения скважин
                    interference_matrix[i, j] = self.interaction_strength / (self.n_wells - 1)
        
        return interference_matrix
    
    def get_field_info(self) -> dict:
        """
        Возвращает полную информацию о текущем состоянии месторождения.
        
        Returns:
            dict: Словарь с информацией о месторождении и скважинах.
        """
        well_states = self.get_well_states()
        
        # Сбор информации о скважинах
        wells_info = []
        for i in range(self.n_wells):
            well_info = {
                'id': i,
                'reservoir_pressure': well_states[i][0],
                'current_rate': well_states[i][1],
                'cumulative_production': well_states[i][2],
                'time': well_states[i][3],
                'valve_opening': self.last_actions[i] if self.last_actions[i] is not None else 0.0
            }
            wells_info.append(well_info)
        
        # Сбор общей информации о месторождении
        field_info = {
            'wells': wells_info,
            'total_wells': self.n_wells,
            'shared_reservoir': self.shared_reservoir,
            'interaction_strength': self.interaction_strength,
            'total_volume': self.total_volume,
            'remaining_volume': self.total_volume - self.cumulative_production,
            'depletion_ratio': self.cumulative_production / self.total_volume if self.total_volume > 0 else 1.0,
            'total_cumulative_production': self.cumulative_production,
            'current_total_rate': sum(well_state[1] for well_state in well_states),
            'current_time': self.time,
            'remaining_time': max(0.0, self.max_time - self.time)
        }
        
        return field_info

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
    
    # Можно также использовать случайное начальное состояние
    # state = simulator.reset_to_random_state(min_depletion=0.1, max_depletion=0.5)
    # print(f"Случайное начальное состояние: {state}")
    
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < 365:  # Ограничиваем максимальным числом шагов
        # Пример стратегии: разное открытие штуцеров для разных скважин
        # В реальности, это место для алгоритма оптимизации
        actions = np.array([0.8, 0.5, 0.3])
        
        next_state, reward, done, info = simulator.step(actions)
        total_reward += reward
        state = next_state
        step += 1
        
        if step % 30 == 0 or done:
            # Получаем состояния скважин
            well_states = simulator.get_well_states()
            
            print(f"Шаг: {step}, Время: {simulator.time:.1f} дней")
            for i, well_state in enumerate(well_states):
                print(f"  Скважина {i+1}: Давление: {well_state[0]:.2f} атм, "
                      f"Дебит: {well_state[1]:.2f} м3/сут, "
                      f"Добыча: {well_state[2]:.1f} м3, "
                      f"Штуцер: {actions[i]:.2f}")
            print(f"  Суммарный дебит: {reward:.2f} м3/сут, Общая добыча: {simulator.cumulative_production:.1f} м3")
            print(f"  Истощение резервуара: {simulator.cumulative_production / simulator.total_volume * 100:.1f}%, Завершено: {done}")
    
    print(f"\nСимуляция завершена после {step} шагов ({simulator.time:.1f} дней).")
    print(f"Суммарная добыча: {simulator.cumulative_production:.1f} м3")
    print(f"Средний дебит: {simulator.cumulative_production / step:.1f} м3/сут")
    
    # Дополнительная информация о интерференции
    if simulator.interaction_strength > 0:
        print("\nМатрица интерференции между скважинами:")
        interference_matrix = simulator.calculate_interference_matrix()
        for i in range(simulator.n_wells):
            row = " ".join([f"{val:.2f}" for val in interference_matrix[i]])
            print(f"  Скважина {i+1} -> {row}")
