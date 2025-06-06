import numpy as np
import random
# Поскольку SingleWellSimulator не будет использоваться напрямую в двухфазной логике MultiWellSimulator,
# и его логика сильно отличается, его импорт можно будет убрать или пересмотреть позже, если он понадобится.
# from simulators.single_well.simulator import SingleWellSimulator 

class MultiWellSimulator:
    """
    Симулятор для нескольких нефтяных скважин с двухфазной моделью (нефть-вода).
    
    Моделирует группу скважин, которые могут быть независимыми или 
    взаимодействовать через общий резервуар.
    
    Состояние: информация о давлении, насыщенности, дебитах и накопленной добыче для каждой скважины.
    Действие: список управляющих воздействий для всех скважин (choke_opening для каждой).
    Награда: суммарный дебит нефти (поверхностный) всех скважин на текущем шаге.
    """
    
    def __init__(self,
                 n_wells: int = 3,
                 interaction_matrix: np.ndarray | None = None,
                 pressure_exponent: float = 0.85,
                 shared_reservoir: bool = True,
                 # total_volume теперь рассматривается как общий ПОРИСТЫЙ объем пласта (м3)
                 total_porous_volume: float = 3e6, 
                 
                 # Параметры двухфазной модели (могут быть частью well_params или заданы глобально)
                 initial_water_saturation: float = 0.2,
                 residual_oil_saturation: float = 0.15,
                 residual_water_saturation: float = 0.1, # Связанная вода
                 oil_viscosity: float = 2.0, # сП (сантипуаз)
                 water_viscosity: float = 1.0, # сП
                 oil_formation_volume_factor: float = 1.1, # м3/м3 (пластовый/поверхностный)
                 water_formation_volume_factor: float = 1.01, # м3/м3
                 koren_exponent_oil: float = 2.0, # Экспонента Кори для нефти (no)
                 koren_exponent_water: float = 2.0, # Экспонента Кори для воды (nw)
                 
                 **well_params # Общие параметры скважин, как раньше
                 ):
        """
        Инициализация мультискважинного симулятора с двухфазной моделью.
        
        Args:
            n_wells: количество скважин
            interaction_matrix: Матрица NxN ... (описание как раньше)
            pressure_exponent: Степень в формуле падения давления ...
            shared_reservoir: используют ли скважины общий резервуар
            total_porous_volume: Общий ПОРИСТЫЙ объем пласта (м3). 
                                 Если shared_reservoir=False, этот объем делится на n_wells.
            initial_water_saturation: Начальная водонасыщенность (доля)
            residual_oil_saturation: Остаточная нефтенасыщенность (доля)
            residual_water_saturation: Связанная/остаточная водонасыщенность (доля)
            oil_viscosity: Вязкость нефти (сП)
            water_viscosity: Вязкость воды (сП)
            oil_formation_volume_factor: Объемный коэффициент нефти Bo
            water_formation_volume_factor: Объемный коэффициент воды Bw
            koren_exponent_oil: Экспонента Кори для нефти (no) для расчета ОФП
            koren_exponent_water: Экспонента Кори для воды (nw) для расчета ОФП
            well_params: Общие параметры для инициализации скважин 
                         (initial_reservoir_pressure, initial_bhp, productivity_index, dt, max_time)
        """
        self.n_wells = n_wells
        self.pressure_exponent = pressure_exponent
        
        if interaction_matrix is None:
            self.interaction_matrix = np.zeros((n_wells, n_wells))
        else:
            # Валидация interaction_matrix (код как раньше)
            if not isinstance(interaction_matrix, np.ndarray):
                raise ValueError("interaction_matrix должна быть объектом np.ndarray.")
            if interaction_matrix.shape != (n_wells, n_wells):
                raise ValueError(f"Размер interaction_matrix должен быть ({n_wells}, {n_wells}), "
                                 f"получен {interaction_matrix.shape}")
            if not np.all((interaction_matrix >= 0) & (interaction_matrix <= 1)):
                raise ValueError("Все значения в interaction_matrix должны быть в диапазоне [0, 1].")
            if not np.all(np.diag(interaction_matrix) == 0): # Диагональные должны быть 0
                raise ValueError("Диагональные элементы interaction_matrix должны быть равны 0.")
            self.interaction_matrix = interaction_matrix
            
        self.shared_reservoir = shared_reservoir
        self.total_porous_volume = total_porous_volume
        self.initial_water_saturation = initial_water_saturation
        self.Sor = residual_oil_saturation
        self.Swr = residual_water_saturation
        
        self.mu_o = oil_viscosity
        self.mu_w = water_viscosity
        self.Bo = oil_formation_volume_factor
        self.Bw = water_formation_volume_factor
        self.no = koren_exponent_oil
        self.nw = koren_exponent_water

        # Общие параметры скважин из well_params
        self.initial_reservoir_pressure = well_params.get('initial_reservoir_pressure', 200.0)
        self.bhp = well_params.get('initial_bhp', 50.0)
        # self.pi теперь будет интерпретироваться как PI по общей жидкости
        self.pi_total_liquid = well_params.get('productivity_index', 0.1) 
        self.dt = well_params.get('dt', 1.0)
        self.max_time = well_params.get('max_time', 365.0)
        
        # Инициализация атрибутов для отслеживания состояния симуляции
        self.time = 0.0
        
        # Давления (остается как есть)
        self.reservoir_pressures = np.ones(n_wells) * self.initial_reservoir_pressure
        
        # Насыщенности
        self.water_saturations = np.ones(n_wells) * self.initial_water_saturation
        
        # Дебиты (теперь раздельные, поверхностные)
        self.current_oil_rates_surface = np.zeros(n_wells)
        self.current_water_rates_surface = np.zeros(n_wells)
        
        # Накопленная добыча (теперь раздельная, поверхностная)
        self.cumulative_oil_production_surface = np.zeros(n_wells)
        self.cumulative_water_production_surface = np.zeros(n_wells)
        # Общая накопленная добыча флюидов из пласта (в пластовых условиях) - для расчета истощения
        self.cumulative_fluid_produced_reservoir_total = 0.0 # Для общего пласта
        self.cumulative_fluid_produced_reservoir_per_well = np.zeros(n_wells) # Для раздельных

        self.current_valve_openings = np.zeros(n_wells) # Остается
        self.last_actions = [None] * n_wells # Остается

        # Размеры состояния для одной скважины:
        # [P_res, Sw, Q_oil_surf, Q_water_surf, Cum_oil_surf, Cum_water_surf, time]
        self.state_dim_per_well = 7
        
        # Распределение порового объема
        if shared_reservoir:
            self.porous_volume_per_well_zone = self.total_porous_volume # Каждая скважина работает на весь объем
        else:
            # При независимых резервуарах объем делится между скважинами
            self.porous_volume_per_well_zone = self.total_porous_volume / n_wells
        
        # Инициализация состояния (будет сделана в reset)
        self.state = np.zeros(self.n_wells * self.state_dim_per_well) # Просто создаем массив нужного размера
        self.reset() # Вызываем reset для корректной инициализации state

        # Сохраняем базовое начальное давление для reset_to_random_state
        self.base_initial_pressure = self.initial_reservoir_pressure

        # Проверка корректности остаточных насыщенностей
        if not (0 <= self.Swr < 1 and 0 <= self.Sor < 1 and self.Swr + self.Sor < 1):
             raise ValueError("Некорректные значения остаточных насыщенностей Swr, Sor.")
        
        if not (self.Swr <= self.initial_water_saturation < 1 - self.Sor):
            raise ValueError(f"Начальная водонасыщенность ({self.initial_water_saturation}) должна быть в диапазоне "
                             f"[{self.Swr}, {1 - self.Sor}).")

    def reset(self) -> np.ndarray:
        """
        Сбрасывает симулятор к начальному состоянию для двухфазной модели.
        """
        self.time = 0.0
        self.last_actions = np.zeros(self.n_wells)
        self.current_valve_openings = np.zeros(self.n_wells)
        
        self.reservoir_pressures.fill(self.initial_reservoir_pressure)
        self.water_saturations.fill(self.initial_water_saturation)
        
        self.current_oil_rates_surface.fill(0.0)
        self.current_water_rates_surface.fill(0.0)
        
        self.cumulative_oil_production_surface.fill(0.0)
        self.cumulative_water_production_surface.fill(0.0)
        self.cumulative_fluid_produced_reservoir_total = 0.0
        self.cumulative_fluid_produced_reservoir_per_well.fill(0.0)
        
        self.current_valve_openings.fill(0.0)
        self.last_actions = [None] * self.n_wells
        
        # Формируем начальный вектор состояния
        current_state_list = []
        for i in range(self.n_wells):
            current_state_list.extend([
                self.reservoir_pressures[i],
                self.water_saturations[i],
                self.current_oil_rates_surface[i],
                self.current_water_rates_surface[i],
                self.cumulative_oil_production_surface[i],
                self.cumulative_water_production_surface[i],
                self.time
            ])
        self.state = np.array(current_state_list)
        return self.state

    def get_state_dim(self) -> int:
        """Возвращает размерность вектора состояния."""
        # self.state инициализируется в reset() до вызова этого метода извне обычно
        if self.state is None or len(self.state) == 0 : # На случай если вызван до первого reset
             return self.n_wells * self.state_dim_per_well
        return len(self.state)

    def get_action_dim(self) -> int:
        """Возвращает размерность вектора действия."""
        return self.n_wells

    def calculate_interference_matrix(self) -> np.ndarray:
        """
        Возвращает матрицу интерференции между скважинами.
        """
        return self.interaction_matrix
    
    def _calculate_krel(self, Sw: float) -> tuple[float, float]:
        """
        Расчет относительных фазовых проницаемостей (ОФП) по модели Кори.
        
        Args:
            Sw (float): Текущая водонасыщенность (доля).
            
        Returns:
            tuple[float, float]: Кортеж (kro, krw) - ОФП нефти и воды.
        """
        if self.Swr >= 1 - self.Sor:
            # Предотвращение деления на ноль, если диапазон подвижной воды нулевой
            return 0.0, 0.0

        # Нормализованная водонасыщенность (от 0 до 1)
        Sn_w = (Sw - self.Swr) / (1 - self.Swr - self.Sor)
        Sn_w = np.clip(Sn_w, 0.0, 1.0) # Ограничиваем на случай ошибок округления

        # ОФП по модели Кори
        kro = (1 - Sn_w) ** self.no
        krw = Sn_w ** self.nw
        
        return kro, krw

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Выполняет один шаг симуляции для всех скважин с двухфазной моделью.

        Args:
            actions (np.ndarray): Управляющие воздействия (choke_openings) для каждой скважины.

        Returns:
            tuple[np.ndarray, float, bool, dict]: (новое_состояние, награда, флаг_завершения, инфо).
        """
        # --- 1. Предварительная обработка действий и обновление времени ---
        actions = np.asarray(actions).flatten()
        actions = np.clip(actions, 0.0, 1.0)
        self.last_actions = actions.copy()
        self.current_valve_openings = actions.copy()
        self.time += self.dt

        # --- 2. Расчет идеальных дебитов для каждой скважины (без интерференции) ---
        ideal_total_liquid_rates_res = np.zeros(self.n_wells)
        fractions_water_res = np.zeros(self.n_wells) # Доля воды в потоке жидкости (пласт. усл.)

        for i in range(self.n_wells):
            p_res = self.reservoir_pressures[i]
            sw = self.water_saturations[i]
            
            delta_p = p_res - self.bhp
            if delta_p <= 0:
                continue # Давления нет, дебита нет

            kro, krw = self._calculate_krel(sw)
            if kro == 0 and krw == 0:
                continue # Нет подвижности флюидов

            lambda_o = kro / self.mu_o
            lambda_w = krw / self.mu_w
            lambda_t = lambda_o + lambda_w
            
            if lambda_t > 1e-9: # Избегаем деления на ноль
                fractions_water_res[i] = lambda_w / lambda_t
            else:
                fractions_water_res[i] = 1.0 if krw > 0 else 0.0

            # PI в модели - это PI по общей жидкости
            ideal_total_liquid_rates_res[i] = self.pi_total_liquid * delta_p * actions[i]
        
        # --- 3. Расчет финальных дебитов с учетом интерференции ---
        final_total_liquid_rates_res = np.copy(ideal_total_liquid_rates_res)
        for i in range(self.n_wells):
            interference_reduction = 0.0
            for j in range(self.n_wells):
                if i == j: continue
                # Уменьшение дебита i-й скважины пропорционально идеальному дебиту j-й
                interference_reduction += self.interaction_matrix[i, j] * ideal_total_liquid_rates_res[j]
            
            final_total_liquid_rates_res[i] -= interference_reduction
        
        final_total_liquid_rates_res = np.maximum(0.0, final_total_liquid_rates_res) # Убеждаемся, что дебиты не отрицательные

        # --- 4. Обновление состояний, давлений и насыщенностей ---
        volumes_produced_res = final_total_liquid_rates_res * self.dt
        
        # Для shared_reservoir, нам нужно обновить общий Sw и P
        if self.shared_reservoir:
            total_volume_produced_res = np.sum(volumes_produced_res)
            
            # Средневзвешенная доля воды в общем потоке
            if np.sum(final_total_liquid_rates_res) > 1e-9:
                avg_fw_res = np.sum(final_total_liquid_rates_res * fractions_water_res) / np.sum(final_total_liquid_rates_res)
            else:
                avg_fw_res = 0.0
            
            # Обновление общей накопленной добычи
            self.cumulative_fluid_produced_reservoir_total += total_volume_produced_res
            
            # Обновление общей насыщенности
            # Sw_new = Sw_old + dV_water / V_pore = Sw_old + (dV_total * fw) / V_pore
            dSw_total = (total_volume_produced_res * avg_fw_res) / self.total_porous_volume
            self.water_saturations.fill(min(self.water_saturations[0] + dSw_total, 1.0 - self.Sor))
            
            # Обновление общего давления
            depletion_ratio = self.cumulative_fluid_produced_reservoir_total / self.total_porous_volume
            new_pressure = self.initial_reservoir_pressure * ((1.0 - depletion_ratio) ** self.pressure_exponent)
            self.reservoir_pressures.fill(max(0.0, new_pressure))

        # Для раздельных резервуаров, обновляем каждую скважину индивидуально
        else:
            for i in range(self.n_wells):
                # Обновление накопленной добычи для i-й ячейки
                self.cumulative_fluid_produced_reservoir_per_well[i] += volumes_produced_res[i]
                
                # Обновление насыщенности для i-й ячейки
                dSw_i = (volumes_produced_res[i] * fractions_water_res[i]) / self.porous_volume_per_well_zone
                self.water_saturations[i] = min(self.water_saturations[i] + dSw_i, 1.0 - self.Sor)

                # Обновление давления для i-й ячейки
                depletion_own = self.cumulative_fluid_produced_reservoir_per_well[i] / self.porous_volume_per_well_zone
                new_pressure_base = self.initial_reservoir_pressure * ((1.0 - depletion_own) ** self.pressure_exponent)
                
                # Учет падения давления от соседей
                pressure_drop_interference = 0.0
                for j in range(self.n_wells):
                    if i == j: continue
                    depletion_neighbor = self.cumulative_fluid_produced_reservoir_per_well[j] / self.porous_volume_per_well_zone
                    pressure_drop_interference += self.initial_reservoir_pressure * depletion_neighbor * self.interaction_matrix[i, j]
                
                self.reservoir_pressures[i] = max(0.0, new_pressure_base - pressure_drop_interference)

        # --- 5. Финализация дебитов, накопленной добычи и состояния ---
        current_state_list = []
        for i in range(self.n_wells):
            # Поверхностные дебиты
            oil_rate_res = final_total_liquid_rates_res[i] * (1.0 - fractions_water_res[i])
            water_rate_res = final_total_liquid_rates_res[i] * fractions_water_res[i]
            
            self.current_oil_rates_surface[i] = oil_rate_res / self.Bo
            self.current_water_rates_surface[i] = water_rate_res / self.Bw
            
            # Поверхностная накопленная добыча
            self.cumulative_oil_production_surface[i] += self.current_oil_rates_surface[i] * self.dt
            self.cumulative_water_production_surface[i] += self.current_water_rates_surface[i] * self.dt
            
            # Формирование нового состояния
            current_state_list.extend([
                self.reservoir_pressures[i],
                self.water_saturations[i],
                self.current_oil_rates_surface[i],
                self.current_water_rates_surface[i],
                self.cumulative_oil_production_surface[i],
                self.cumulative_water_production_surface[i],
                self.time
            ])
        self.state = np.array(current_state_list)
        
        # --- 6. Расчет награды и проверка завершения ---
        reward = np.sum(self.current_oil_rates_surface) # Награда - суммарный дебит НЕФТИ
        
        # Условие завершения: вышло время, или ВСЕ скважины истощены
        done = self.time >= self.max_time
        if not done:
            all_wells_depleted = True
            for i in range(self.n_wells):
                # Скважина истощена, если давление ниже забойного ИЛИ она полностью обводнилась
                is_pressure_depleted = self.reservoir_pressures[i] <= self.bhp
                is_watered_out = self.water_saturations[i] >= (1.0 - self.Sor)
                if not (is_pressure_depleted or is_watered_out):
                    all_wells_depleted = False
                    break
            if all_wells_depleted:
                done = True

        # --- 7. Формирование инфо-словаря ---
        info = {
            'oil_rates_surface': self.current_oil_rates_surface.copy(),
            'water_rates_surface': self.current_water_rates_surface.copy(),
            'total_oil_rate_surface': reward,
            'water_cuts_surface': (self.current_water_rates_surface / (self.current_oil_rates_surface + self.current_water_rates_surface + 1e-9)).copy(),
            'well_pressures': self.reservoir_pressures.copy(),
            'well_water_saturations': self.water_saturations.copy(),
            'cumulative_oil_surface': self.cumulative_oil_production_surface.copy(),
            'cumulative_water_surface': self.cumulative_water_production_surface.copy(),
            'valve_openings': self.current_valve_openings.copy(),
        }

        return self.state, reward, done, info

    def get_well_states(self) -> list[np.ndarray]:
        """
        Возвращает состояния всех скважин в виде списка массивов.
        Каждый массив представляет состояние одной скважины.
        """
        # self.state - это плоский массив [P0, Sw0, ..., time0, P1, Sw1, ..., time1, ...]
        return np.split(self.state, self.n_wells)

    def get_field_info(self) -> dict:
        """
        Возвращает подробную информацию о состоянии всего месторождения и каждой скважины.
        """
        wells_info = []
        well_states = self.get_well_states()
        for i in range(self.n_wells):
            state_vec = well_states[i]
            well_info = {
                'reservoir_pressure': state_vec[0],
                'water_saturation': state_vec[1],
                'oil_rate_surface': state_vec[2],
                'water_rate_surface': state_vec[3],
                'cumulative_oil_surface': state_vec[4],
                'cumulative_water_surface': state_vec[5],
                'valve_opening': self.current_valve_openings[i]
            }
            wells_info.append(well_info)

        field_info = {
            'time': self.time,
            'total_oil_rate_surface': np.sum(self.current_oil_rates_surface),
            'total_water_rate_surface': np.sum(self.current_water_rates_surface),
            'wells': wells_info,
            'interaction_matrix': self.interaction_matrix.tolist()
        }
        return field_info

    def reset_to_random_state(self, seed: int | None = None) -> np.ndarray:
        """
        Сбрасывает симулятор в случайное начальное состояние.
        Изменяет начальное давление и насыщенность в небольшом диапазоне.

        Args:
            seed (int | None): Seed для генератора случайных чисел.

        Returns:
            np.ndarray: Начальное состояние.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Задаем диапазоны для случайных значений
        # Давление: +/- 10% от исходного
        pressure_range = [self.base_initial_pressure * 0.9, self.base_initial_pressure * 1.1]
        # Насыщенность: от Swr до (1 - Sor - небольшой буфер)
        saturation_range = [self.Swr, 1 - self.Sor - 0.05]

        # Генерируем случайные значения
        self.initial_reservoir_pressure = np.random.uniform(*pressure_range)
        self.initial_water_saturation = np.random.uniform(*saturation_range)
        
        # Убедимся, что Swi не выходит за пределы
        self.initial_water_saturation = np.clip(
            self.initial_water_saturation, 
            self.Swr, 
            1 - self.Sor
        )
        
        # Вызываем обычный reset, который использует обновленные initial_... параметры
        return self.reset()

# Пример использования (потребует обновления после полной реализации двухфазной модели)
if __name__ == '__main__':
    print("Запуск примера для MultiWellSimulator (двухфазная модель)")
    
    n_wells_example = 2
    interaction_example = np.array([[0.0, 0.05], [0.05, 0.0]])

    sim_params = {
        'n_wells': n_wells_example,
        'interaction_matrix': interaction_example,
        'shared_reservoir': True,
        'total_porous_volume': 5e6, 'initial_water_saturation': 0.25,
        'residual_oil_saturation': 0.2, 'residual_water_saturation': 0.15,
        'oil_viscosity': 2.5, 'water_viscosity': 0.8,
        'oil_formation_volume_factor': 1.15, 'water_formation_volume_factor': 1.02,
        'koren_exponent_oil': 2.5, 'koren_exponent_water': 2.2,
        'initial_reservoir_pressure': 220.0, 'initial_bhp': 40.0,
        'productivity_index': 0.15, 'dt': 1.0, 'max_time': 365.0
    }

    simulator = MultiWellSimulator(**sim_params)
    state = simulator.reset()
    
    print(f"Симулятор для {simulator.n_wells} скважин создан.")
    print(f"Начальное давление: {simulator.reservoir_pressures[0]:.2f}, Начальная Sw: {simulator.water_saturations[0]:.3f}")

    total_oil_produced = 0
    for step_num in range(int(sim_params['max_time'])):
        actions = np.ones(n_wells_example) * 0.7 # Простое действие
        state, reward, done, info = simulator.step(actions)
        
        total_oil_produced += np.sum(info['oil_rates_surface']) * simulator.dt
        
        if step_num % 30 == 0 or done:
            print(f"\n--- Шаг {step_num+1}, Время {simulator.time:.1f} дней ---")
            print(f"Давление: {info['well_pressures'][0]:.2f} атм | Sw: {info['well_water_saturations'][0]:.3f}")
            for i in range(simulator.n_wells):
                print(f"  Скважина {i+1}: Q_oil={info['oil_rates_surface'][i]:.2f} м3/сут, "
                      f"Q_water={info['water_rates_surface'][i]:.2f} м3/сут, "
                      f"Обводненность={info['water_cuts_surface'][i]*100:.1f}%")
            print(f"Награда (общий Q_oil): {reward:.2f} м3/сут")

        if done:
            print(f"\nСимуляция завершена на шаге {step_num+1}.")
            break
            
    print(f"\nИтоговая накопленная добыча нефти: {np.sum(simulator.cumulative_oil_production_surface):.1f} м3")

# Удаляем старые методы, которые были после `calculate_interference_matrix`
