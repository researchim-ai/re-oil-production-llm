import numpy as np

class SingleWellSimulator:
    """
    Простой симулятор одной нефтяной скважины.

    Состояние: [reservoir_pressure, flow_rate, cumulative_production, time]
    Действие: choke_opening (от 0 до 1)
    Награда: flow_rate на текущем шаге
    """
    def __init__(self,
                 initial_reservoir_pressure: float = 200.0, # атм
                 initial_bhp: float = 50.0, # атм (для примера, упростим зависимость)
                 productivity_index: float = 0.1, # м3/сут/атм
                 total_volume: float = 1e6, # м3 - начальный объем нефти в зоне дренирования
                 dt: float = 1.0, # дней - шаг симуляции
                 max_time: float = 365.0 # дней - максимальное время симуляции
                 ):
        """
        Инициализация симулятора.
        """
        self.initial_reservoir_pressure = initial_reservoir_pressure
        self.initial_bhp = initial_bhp # Упрощение: пока считаем BHP постоянным
        self.pi = productivity_index
        self.total_volume = total_volume # Используется для расчета падения давления
        self.dt = dt
        self.max_time = max_time

        # Инициализация переменных состояния при создании экземпляра
        self.state = self.reset()

    def reset(self) -> np.ndarray:
        """
        Сбрасывает симулятор в начальное состояние.
        
        Returns:
            np.ndarray: Начальное состояние [давление, дебит, добыча, время]
        """
        # Начальные условия
        self.reservoir_pressure = self.initial_reservoir_pressure
        self.flow_rate = 0.0 # Нет начального дебита, пока не открыт штуцер
        self.cumulative_production = 0.0
        self.time = 0.0
        self.previous_choke = 0.0  # Инициализируем значение предыдущего положения штуцера
        
        self.state = np.array([
            self.reservoir_pressure,
            self.flow_rate,
            self.cumulative_production,
            self.time
        ])
        
        return self.state

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        """
        Выполняет один шаг симуляции.

        Args:
            action (float): Управляющее воздействие - choke_opening (от 0 до 1).

        Returns:
            tuple[np.ndarray, float, bool, dict]: Кортеж (новое_состояние, награда, флаг_завершения, инфо).
                                             новое_состояние: [reservoir_pressure, flow_rate, cumulative_production, time]
                                             награда: flow_rate на этом шаге
                                             флаг_завершения: True, если симуляция закончена
                                             инфо: дополнительная информация о шаге
        """
        # 1. Ограничиваем действие
        choke_opening = np.clip(action, 0.0, 1.0)
        self.current_valve_opening = choke_opening  # Сохраняем текущее открытие штуцера

        # 2. Рассчитываем дебит (упрощенная модель притока)
        # Учитываем, что давление в пласте не может быть ниже BHP
        delta_p = max(0.0, self.reservoir_pressure - self.initial_bhp)
        # Дебит зависит от перепада давления и открытия штуцера
        current_flow_rate = self.pi * delta_p * choke_opening
        self.current_rate = current_flow_rate  # Сохраняем текущий дебит

        # 3. Обновляем время и накопленную добычу
        self.time += self.dt
        volume_produced_this_step = current_flow_rate * self.dt
        self.cumulative_production += volume_produced_this_step

        # 4. Обновляем давление в пласте (упрощенная материальная балансовая модель)
        # Давление падает пропорционально доле добытой нефти от общего объема
        # Это очень грубое приближение, но для начала подойдет
        # Добавляем небольшой знаменатель, чтобы избежать деления на ноль, если total_volume = 0
        if self.total_volume > 1e-9:
             depletion_ratio = self.cumulative_production / self.total_volume
             self.reservoir_pressure = self.initial_reservoir_pressure * (1 - depletion_ratio)
        else:
             self.reservoir_pressure = 0.0 # Если объем 0, давление падает до 0

        # Убедимся, что давление не отрицательное
        self.reservoir_pressure = max(0.0, self.reservoir_pressure)

        # 5. Обновляем текущий дебит для состояния
        self.flow_rate = current_flow_rate

        # 6. Формируем новое состояние
        self.state = np.array([
            self.reservoir_pressure,
            self.flow_rate,
            self.cumulative_production,
            self.time
        ])

        # 7. Рассчитываем награду
        reward = current_flow_rate / 10
        
        # Улучшенная функция награды с reward shaping
        
        # Фазы разработки скважины
        early_phase = self.time < 0.25 * self.max_time  # Ранняя фаза
        mid_phase = 0.25 * self.max_time <= self.time < 0.7 * self.max_time  # Средняя фаза
        late_phase = self.time >= 0.7 * self.max_time  # Поздняя фаза
        
        # Стратегический коэффициент для текущей фазы (влияет на размер награды)
        strategic_factor = 1.0
        
        # Бонусы и штрафы зависят от фазы разработки
        if early_phase:
            # Ранняя фаза: поощряем умеренную добычу для сохранения давления
            optimal_choke = 0.3  # Оптимальная степень открытия штуцера в начале
            strategic_factor = 1.0 - 2.0 * abs(choke_opening - optimal_choke)  # Максимум при optimal_choke
            
            # Дополнительный штраф за слишком интенсивную добычу в начале
            if choke_opening > 0.6:
                reward *= 0.5  # Существенный штраф
            
            # Бонус за поддержание высокого давления
            if self.reservoir_pressure > 0.8 * self.initial_reservoir_pressure:
                reward *= 1.5
        
        elif mid_phase:
            # Средняя фаза: поощряем стабильную добычу
            optimal_choke = 0.5  # Оптимальная степень открытия штуцера в середине
            strategic_factor = 1.0 - 1.5 * abs(choke_opening - optimal_choke)
            
            # Бонус за поддержание стабильного дебита
            if 0.4 <= choke_opening <= 0.6:
                reward *= 1.3
        
        else:  # late_phase
            # Поздняя фаза: поощряем максимальную добычу
            optimal_choke = 0.8  # Оптимальная степень открытия штуцера в конце
            strategic_factor = 1.0 - 1.0 * abs(choke_opening - optimal_choke)
            
            # Бонус за максимальную добычу в конце
            if choke_opening > 0.7:
                reward *= 1.4
        
        # Применяем стратегический коэффициент (минимум 0.2)
        strategic_factor = max(0.2, strategic_factor)
        reward *= strategic_factor
        
        # Бонус за оптимальную последовательность действий между шагами
        # Оцениваем изменение открытия штуцера относительно предыдущего шага
        if hasattr(self, 'previous_choke') and self.time > self.dt:
            # Идеальное изменение зависит от фазы
            if early_phase:
                # В начале - плавное увеличение
                ideal_change = 0.05  # Небольшое увеличение
            elif mid_phase:
                # В середине - стабильность
                ideal_change = 0.0  # Минимальное изменение
            else:  # late_phase
                # В конце - более существенное увеличение
                ideal_change = 0.1  # Значительное увеличение
            
            # Рассчитываем фактическое изменение
            actual_change = choke_opening - self.previous_choke
            
            # Оцениваем, насколько хорошо изменение соответствует идеальному
            sequence_factor = 1.0 - 2.0 * abs(actual_change - ideal_change)
            sequence_factor = max(0.3, sequence_factor)  # Ограничиваем минимальный множитель
            
            # Применяем множитель для последовательности
            reward *= sequence_factor
            
            # Штраф за резкие изменения (высокая волатильность вредна для оборудования)
            if abs(actual_change) > 0.3:
                reward *= 0.7  # Существенный штраф за резкие изменения
        
        # Сохраняем текущее открытие штуцера для следующего шага
        self.previous_choke = choke_opening
        
        # 8. Проверяем условие завершения эпизода
        # Завершаем, если время вышло или давление пласта упало ниже BHP (приток прекратился)
        done = self.time >= self.max_time or self.reservoir_pressure <= self.initial_bhp

        # 9. Формируем информационный словарь
        info = {
            'choke_opening': choke_opening,
            'delta_p': delta_p,
            'volume_produced_this_step': volume_produced_this_step,
            'depletion_ratio': self.cumulative_production / self.total_volume if self.total_volume > 1e-9 else 1.0,
            'remaining_time': max(0.0, self.max_time - self.time)
        }

        return self.state, reward, done, info

    def get_state_dim(self) -> int:
        """Возвращает размерность вектора состояния."""
        return len(self.state)

    def get_action_dim(self) -> int:
        """Возвращает размерность вектора действия."""
        # Пока у нас одно действие - choke_opening
        return 1

# Пример использования (можно будет убрать или перенести в тесты/примеры)
if __name__ == '__main__':
    simulator = SingleWellSimulator()
    state = simulator.reset()
    print(f"Initial state: {state}")

    total_reward = 0
    done = False
    step = 0
    while not done:
        # Пример простой стратегии: держать штуцер открытым на 50%
        action = 0.5
        next_state, reward, done, info = simulator.step(action)
        total_reward += reward # Суммируем награды (дебиты)
        state = next_state
        step += 1
        if step % 30 == 0 or done: # Печатаем каждые 30 дней
             print(f"Step: {step}, Time: {state[3]:.1f} days, Reservoir Pressure: {state[0]:.2f} atm, Flow Rate: {state[1]:.2f} m3/day, Cumulative Prod: {state[2]:.1f} m3, Reward: {reward:.2f}, Done: {done}")

    print(f"Simulation finished after {step} steps ({state[3]:.1f} days).")
    print(f"Total production: {state[2]:.1f} m3")
    # Награда у нас равна дебиту за шаг. Суммарная награда за эпизод, умноженная на шаг по времени dt,
    # должна быть равна накопленной добыче.
    print(f"Total reward * dt (should approx equal Total production): {total_reward * simulator.dt:.1f}")
