import numpy as np
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
        
        # Распределение объема резервуара между скважинами
        if shared_reservoir:
            # При общем резервуаре каждая скважина использует весь объем
            well_volume = total_volume
        else:
            # При независимых резервуарах объем делится между скважинами
            well_volume = total_volume / n_wells
        
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
        Сбрасывает симулятор в начальное состояние.
        
        Returns:
            np.ndarray: Начальное состояние [давление1, bhp1, добыча1, время1, давление2, ...]
        """
        # Обнуляем время и накопленную добычу
        self.time = 0.0
        self.cumulative_production = 0.0
        
        # Инициализируем состояние для каждой скважины
        state_blocks = []
        for i in range(self.n_wells):
            # Для каждой скважины состояние содержит:
            # - пластовое давление
            # - забойное давление (BHP)
            # - накопленную добычу
            # - время (для всех скважин одинаково)
            well_state = np.array([
                self.initial_reservoir_pressure,  # начальное пластовое давление
                self.initial_bhp,                 # начальное забойное давление
                0.0,                              # начальная накопленная добыча
                0.0                               # начальное время
            ])
            state_blocks.append(well_state)
        
        # Объединяем состояния всех скважин в один вектор
        self.state = np.concatenate(state_blocks)
        
        # Инициализируем переменные для отслеживания последовательности действий
        self.previous_actions = np.zeros(self.n_wells)
        self.previous_reward = 0.0
        
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
        base_reward = sum(current_rates)

        # Улучшенная система вознаграждений
        # Фазы разработки месторождения
        early_phase = self.time < 0.25 * self.max_time  # Ранняя фаза
        mid_phase = 0.25 * self.max_time <= self.time < 0.7 * self.max_time  # Средняя фаза
        late_phase = self.time >= 0.7 * self.max_time  # Поздняя фаза
        
        # Бонусы и штрафы зависят от фазы разработки
        phase_multiplier = 1.0
        
        if early_phase:
            # В начале разработки важно не допустить быстрого истощения
            # Штраф за слишком высокую суммарную добычу в начале
            if base_reward > 0.6 * self.n_wells * self.initial_reservoir_pressure * 0.1:  # примерный максимальный дебит
                phase_multiplier *= 0.7
            
            # Бонус за сбалансированную разработку (избегаем перекоса на одну скважину)
            rate_variance = np.var(current_rates) / (np.mean(current_rates) + 1e-6)**2  # нормализованная дисперсия
            balance_factor = np.exp(-5 * rate_variance)  # максимум при равномерной добыче
            phase_multiplier *= 1.0 + 0.5 * balance_factor
            
        elif mid_phase:
            # В средней фазе важно поддерживать стабильную добычу
            # Бонус за стабильность дебита относительно предыдущего шага
            if hasattr(self, 'previous_reward'):
                rate_change = abs(base_reward - self.previous_reward) / (self.previous_reward + 1e-6)
                stability_factor = np.exp(-3 * rate_change)  # максимум при стабильном дебите
                phase_multiplier *= 1.0 + 0.3 * stability_factor
                
        else:  # late_phase
            # В поздней фазе максимизируем добычу
            # Бонус за высокий дебит
            max_theoretical_rate = self.n_wells * (self.initial_reservoir_pressure * 0.3) * 0.1  # приблизительный
            rate_ratio = base_reward / max_theoretical_rate
            phase_multiplier *= 1.0 + 0.6 * rate_ratio
        
        # Бонус за оптимальную стратегию последовательности действий
        strategy_multiplier = 1.0
        
        if hasattr(self, 'previous_actions'):
            # Идеальные изменения зависят от фазы
            if early_phase:
                # В начале мягкое увеличение дебита
                ideal_changes = np.ones(self.n_wells) * 0.05
            elif mid_phase:
                # В середине стабильность
                ideal_changes = np.zeros(self.n_wells)
            else:  # late_phase
                # В конце агрессивное увеличение дебита
                ideal_changes = np.ones(self.n_wells) * 0.1
                
            # Рассчитываем отклонение от идеальной стратегии
            actual_changes = actions - self.previous_actions
            change_deviations = np.abs(actual_changes - ideal_changes)
            mean_deviation = np.mean(change_deviations)
            
            # Штраф за отклонение от идеальной стратегии
            strategy_multiplier = np.exp(-3 * mean_deviation)
            
            # Штраф за слишком резкие изменения
            if np.max(np.abs(actual_changes)) > 0.3:
                strategy_multiplier *= 0.7
                
        # Сохраняем текущие действия и награду для следующего шага
        self.previous_actions = actions.copy()
        self.previous_reward = base_reward
        
        # Применяем множители к базовой награде
        reward = base_reward * phase_multiplier * strategy_multiplier

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
