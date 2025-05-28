import numpy as np
import random

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
        
        # Добавляем поле для хранения последнего действия
        self.last_action = None

        # Инициализация переменных состояния при создании экземпляра
        self.state = self.reset()

    def reset(self) -> np.ndarray:
        """
        Сбрасывает симулятор к начальному состоянию.
        Возвращает:
            np.ndarray: Начальное состояние [reservoir_pressure, flow_rate, cumulative_production, time]
        """
        self.reservoir_pressure = self.initial_reservoir_pressure
        self.bhp = self.initial_bhp
        self.flow_rate = 0.0
        self.cumulative_production = 0.0
        self.time = 0.0
        
        # Сбрасываем последнее действие
        self.last_action = None
        self.current_valve_opening = 0.0
        
        # Возвращаем начальное состояние как numpy массив
        self.state = np.array([
            self.reservoir_pressure,
            self.flow_rate,
            self.cumulative_production,
            self.time
        ])
        return self.state
        
    def reset_to_random_state(self, min_depletion: float = 0.0, max_depletion: float = 0.9, 
                              use_realistic_ranges: bool = True) -> np.ndarray:
        """
        Сбрасывает симулятор к случайному промежуточному состоянию разработки скважины.
        
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
            # Обычно скважина не добывается до полного истощения из-за экономических ограничений
            realistic_max_depletion = min(max_depletion, 0.85)  # Не более 85% от общего объема
            # В начале разработки обычно уже есть минимальный отбор для тестирования
            realistic_min_depletion = max(min_depletion, 0.01)  # Минимум 1% от общего объема
            
            # Используем ограниченные диапазоны
            min_depletion = realistic_min_depletion
            max_depletion = realistic_max_depletion
        
        # Выбираем случайную степень истощения из заданного интервала
        depletion_ratio = random.uniform(min_depletion, max_depletion)
        
        # Рассчитываем накопленную добычу на основе выбранной степени истощения
        self.cumulative_production = depletion_ratio * self.total_volume
        
        # Рассчитываем пластовое давление на основе степени истощения
        # Добавляем небольшую нелинейность для реалистичности
        if use_realistic_ranges:
            # В реальности давление падает нелинейно, обычно быстрее в начале разработки
            pressure_factor = 1.0 - depletion_ratio**0.8  # Нелинейная зависимость
            # Добавляем небольшую случайность для учета непредсказуемых факторов
            pressure_variation = random.uniform(-0.05, 0.05)  # ±5% вариации
            self.reservoir_pressure = self.initial_reservoir_pressure * max(0, min(1, pressure_factor + pressure_variation))
        else:
            # Линейное падение давления (упрощенная модель)
            self.reservoir_pressure = self.initial_reservoir_pressure * (1.0 - depletion_ratio)
        
        # Рассчитываем случайное значение открытия штуцера для последнего действия
        if use_realistic_ranges:
            # В зависимости от степени истощения мы можем ограничить диапазон открытия штуцера
            if depletion_ratio < 0.3:
                # В начале разработки обычно применяют большее открытие
                last_action_value = random.uniform(0.6, 1.0)
            elif depletion_ratio < 0.7:
                # В середине разработки используют средние значения
                last_action_value = random.uniform(0.3, 0.8)
            else:
                # При высоком истощении обычно снижают открытие для поддержания давления
                last_action_value = random.uniform(0.1, 0.5)
        else:
            # Полностью случайное открытие штуцера
            last_action_value = random.uniform(0.0, 1.0)
        
        self.last_action = last_action_value
        self.current_valve_opening = last_action_value
        
        # Рассчитываем текущий дебит с учетом текущего давления и последнего действия
        delta_p = max(0.0, self.reservoir_pressure - self.bhp)
        self.flow_rate = self.pi * delta_p * last_action_value
        self.current_rate = self.flow_rate
        
        # Рассчитываем пройденное время, пропорциональное истощению
        if use_realistic_ranges:
            # В реальности время добычи будет иметь нелинейную зависимость от степени истощения
            # При этом учитываем характер пласта и режим разработки
            base_time = self.max_time * (depletion_ratio**0.9)  # Нелинейная зависимость
            
            # Добавляем реалистичную вариацию времени
            time_variation = 0.15 * base_time  # 15% вариации
            self.time = min(self.max_time - self.dt, max(0, base_time + random.uniform(-time_variation, time_variation)))
        else:
            # Простая линейная зависимость с небольшой случайностью
            proportional_time = depletion_ratio * self.max_time
            time_variation = 0.2 * proportional_time  # 20% вариации
            self.time = min(self.max_time - self.dt, max(0, proportional_time + random.uniform(-time_variation, time_variation)))
        
        # Обновляем вектор состояния
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
        
        # Сохраняем последнее выполненное действие
        self.last_action = choke_opening

        # 2. Рассчитываем дебит (упрощенная модель притока)
        # Учитываем, что давление в пласте не может быть ниже BHP
        delta_p = max(0.0, self.reservoir_pressure - self.bhp)
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

        # # 7. Рассчитываем награду
        # reward = current_flow_rate / 10
        
        # # Улучшенная функция награды с reward shaping
        # # Бонус за поддержание высокого давления пласта (устойчивая добыча)
        # if self.reservoir_pressure > 0.5 * self.initial_reservoir_pressure:
        #     reward *= 1.2  # Бонус за сохранение высокого давления
        
        # # Штраф за слишком интенсивную добычу на ранних стадиях
        # if self.time < 0.3 * self.max_time and current_flow_rate > 0.7 * self.pi * self.initial_reservoir_pressure:
        #     reward *= 0.8  # Штраф за слишком агрессивную эксплуатацию в начале

        reward = current_flow_rate

        # Штраф за слишком интенсивную добычу на ранних стадиях
        # 8. Проверяем условие завершения эпизода
        # Завершаем, если время вышло или давление пласта упало ниже BHP (приток прекратился)
        done = self.time >= self.max_time or self.reservoir_pressure <= self.bhp

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
