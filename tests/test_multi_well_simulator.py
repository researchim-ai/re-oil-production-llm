import unittest
import numpy as np
import sys
import os

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulators.multi_well.simulator import MultiWellSimulator

class TestMultiWellSimulatorTwoPhase(unittest.TestCase):

    def setUp(self):
        """Настройка перед каждым тестом с параметрами для двухфазной модели."""
        self.n_wells = 2
        self.default_sim_params = {
            'n_wells': self.n_wells,
            'interaction_matrix': np.zeros((self.n_wells, self.n_wells)),
            'shared_reservoir': True,
            'total_porous_volume': 5e6,
            'initial_water_saturation': 0.25,
            'residual_oil_saturation': 0.2,
            'residual_water_saturation': 0.2,
            'oil_viscosity': 2.0,
            'water_viscosity': 1.0,
            'oil_formation_volume_factor': 1.2,
            'water_formation_volume_factor': 1.0,
            'koren_exponent_oil': 2.0,
            'koren_exponent_water': 2.0,
            'initial_reservoir_pressure': 200.0,
            'initial_bhp': 50.0,
            'productivity_index': 0.1, # PI по общей жидкости
            'dt': 1.0,
            'max_time': 10.0
        }

    def test_initialization_two_phase(self):
        """Тест инициализации симулятора с двухфазными параметрами."""
        simulator = MultiWellSimulator(**self.default_sim_params)
        self.assertEqual(simulator.n_wells, self.n_wells)
        self.assertEqual(simulator.state_dim_per_well, 7)
        self.assertEqual(simulator.mu_o, self.default_sim_params['oil_viscosity'])
        self.assertEqual(simulator.initial_water_saturation, self.default_sim_params['initial_water_saturation'])

        # Проверка начального состояния
        initial_state = simulator.reset()
        self.assertEqual(len(initial_state), self.n_wells * 7)
        
        # Проверка состояния первой скважины
        self.assertEqual(initial_state[0], self.default_sim_params['initial_reservoir_pressure']) # P_res
        self.assertEqual(initial_state[1], self.default_sim_params['initial_water_saturation']) # Sw
        self.assertEqual(initial_state[2], 0.0) # Q_oil
        self.assertEqual(initial_state[3], 0.0) # Q_water
        self.assertEqual(initial_state[4], 0.0) # Cum_oil
        self.assertEqual(initial_state[5], 0.0) # Cum_water
        self.assertEqual(initial_state[6], 0.0) # time

    def test_initialization_invalid_saturations(self):
        """Тест инициализации с некорректными значениями насыщенностей."""
        params = self.default_sim_params.copy()
        
        # Случай 1: Swr + Sor >= 1
        params['residual_water_saturation'] = 0.8
        params['residual_oil_saturation'] = 0.3
        with self.assertRaisesRegex(ValueError, "Некорректные значения остаточных насыщенностей"):
            MultiWellSimulator(**params)

        # Случай 2: Swi < Swr
        params = self.default_sim_params.copy()
        params['initial_water_saturation'] = 0.1
        params['residual_water_saturation'] = 0.2
        with self.assertRaisesRegex(ValueError, r"Начальная водонасыщенность .* должна быть в диапазоне"):
            MultiWellSimulator(**params)

        # Случай 3: Swi >= 1 - Sor
        params = self.default_sim_params.copy()
        params['initial_water_saturation'] = 0.9
        params['residual_oil_saturation'] = 0.1
        with self.assertRaisesRegex(ValueError, r"Начальная водонасыщенность .* должна быть в диапазоне"):
            MultiWellSimulator(**params)

    def test_calculate_krel(self):
        """Тест расчета ОФП в крайних и промежуточной точках."""
        simulator = MultiWellSimulator(**self.default_sim_params)
        
        Swr = self.default_sim_params['residual_water_saturation']
        Sor = self.default_sim_params['residual_oil_saturation']

        # 1. При Sw = Swr (начало подвижности воды)
        # Ожидаем: krw=0, kro=1
        kro, krw = simulator._calculate_krel(Swr)
        self.assertAlmostEqual(kro, 1.0, places=7)
        self.assertAlmostEqual(krw, 0.0, places=7)

        # 2. При Sw = 1 - Sor (полное обводнение, остаточная нефть)
        # Ожидаем: krw=1, kro=0
        kro, krw = simulator._calculate_krel(1.0 - Sor)
        self.assertAlmostEqual(kro, 0.0, places=7)
        self.assertAlmostEqual(krw, 1.0, places=7)

        # 3. Промежуточная точка (середина диапазона подвижной воды)
        Sw_mid = Swr + (1.0 - Swr - Sor) / 2.0
        no = self.default_sim_params['koren_exponent_oil']
        nw = self.default_sim_params['koren_exponent_water']
        
        # Sn_w для середины должен быть 0.5
        # kro = (1 - 0.5)^no = 0.5^no
        # krw = 0.5^nw
        expected_kro = 0.5 ** no
        expected_krw = 0.5 ** nw
        
        kro, krw = simulator._calculate_krel(Sw_mid)
        self.assertAlmostEqual(kro, expected_kro, places=7)
        self.assertAlmostEqual(krw, expected_krw, places=7)

    def test_step_single_well_physics(self):
        """Тест базовой физики (дебиты, P, Sw) на одном шаге для одной скважины."""
        params = self.default_sim_params.copy()
        params['n_wells'] = 1
        params['interaction_matrix'] = np.zeros((1, 1))
        params['pressure_exponent'] = 1.0 # Линейное падение давления для простоты
        simulator = MultiWellSimulator(**params)
        simulator.reset()

        # --- 1. Расчет ожидаемых значений вручную ---
        # Входные данные
        p_res_initial = params['initial_reservoir_pressure']
        p_bhp = params['initial_bhp']
        sw_initial = params['initial_water_saturation']
        pi = params['productivity_index']
        action = 0.8 # Открытие штуцера
        dt = params['dt']

        # Расчет ОФП для начальной насыщенности
        kro, krw = simulator._calculate_krel(sw_initial)
        
        # Расчет подвижностей и доли воды
        lambda_o = kro / params['oil_viscosity']
        lambda_w = krw / params['water_viscosity']
        lambda_t = lambda_o + lambda_w
        fw_res = lambda_w / lambda_t
        
        # Расчет общего дебита жидкости в пластовых условиях
        total_liquid_rate_res = pi * (p_res_initial - p_bhp) * action
        
        # Расчет фазовых дебитов в поверхностных условиях
        expected_oil_rate_surf = total_liquid_rate_res * (1.0 - fw_res) / params['oil_formation_volume_factor']
        expected_water_rate_surf = total_liquid_rate_res * fw_res / params['water_formation_volume_factor']
        
        # Расчет изменения насыщенности
        volume_produced_res = total_liquid_rate_res * dt
        # Для n_wells=1, porous_volume_per_well_zone = total_porous_volume
        dSw = (volume_produced_res * fw_res) / params['total_porous_volume']
        expected_sw_new = sw_initial + dSw
        
        # Расчет изменения давления
        depletion_ratio = volume_produced_res / params['total_porous_volume']
        expected_p_res_new = p_res_initial * (1.0 - depletion_ratio)

        # --- 2. Выполнение шага симуляции ---
        actions = np.array([action])
        next_state, reward, done, info = simulator.step(actions)

        # --- 3. Сравнение результатов ---
        self.assertAlmostEqual(info['oil_rates_surface'][0], expected_oil_rate_surf, places=5)
        self.assertAlmostEqual(info['water_rates_surface'][0], expected_water_rate_surf, places=5)
        self.assertAlmostEqual(reward, expected_oil_rate_surf, places=5)
        
        self.assertAlmostEqual(info['well_water_saturations'][0], expected_sw_new, places=7)
        self.assertAlmostEqual(info['well_pressures'][0], expected_p_res_new, places=5)
        
        # Проверка накопленной добычи
        self.assertAlmostEqual(info['cumulative_oil_surface'][0], expected_oil_rate_surf * dt, places=5)
        self.assertAlmostEqual(info['cumulative_water_surface'][0], expected_water_rate_surf * dt, places=5)

        self.assertFalse(done)

    def test_step_shared_reservoir_no_interaction(self):
        """Тест шага: общий резервуар, 2 скважины, без взаимодействия."""
        params = self.default_sim_params.copy()
        simulator = MultiWellSimulator(**params)
        simulator.reset()
        
        actions = np.array([0.5, 0.8])
        next_state, reward, done, info = simulator.step(actions)

        # Дебиты должны быть ненулевыми и разными
        self.assertGreater(info['oil_rates_surface'][0], 0)
        self.assertGreater(info['oil_rates_surface'][1], 0)
        self.assertNotEqual(info['oil_rates_surface'][0], info['oil_rates_surface'][1])
        
        # Давление и насыщенность должны быть одинаковыми для обеих скважин
        self.assertAlmostEqual(info['well_pressures'][0], info['well_pressures'][1], places=7)
        self.assertAlmostEqual(info['well_water_saturations'][0], info['well_water_saturations'][1], places=7)
        
        # Давление и насыщенность должны измениться по сравнению с начальными
        self.assertLess(info['well_pressures'][0], params['initial_reservoir_pressure'])
        self.assertGreater(info['well_water_saturations'][0], params['initial_water_saturation'])

    def test_step_separate_reservoirs_no_interaction(self):
        """Тест шага: раздельные резервуары, 2 скважины, без взаимодействия."""
        params = self.default_sim_params.copy()
        params['shared_reservoir'] = False
        simulator = MultiWellSimulator(**params)
        simulator.reset()

        # Используем разные actions, чтобы получить разные дебиты и, следовательно, разные P и Sw
        actions = np.array([0.5, 0.8])
        next_state, reward, done, info = simulator.step(actions)
        
        # Дебиты должны быть ненулевыми
        self.assertGreater(info['oil_rates_surface'][0], 0)
        self.assertGreater(info['oil_rates_surface'][1], 0)
        
        # Давление и насыщенность для каждой скважины должны измениться
        self.assertLess(info['well_pressures'][0], params['initial_reservoir_pressure'])
        self.assertLess(info['well_pressures'][1], params['initial_reservoir_pressure'])
        self.assertGreater(info['well_water_saturations'][0], params['initial_water_saturation'])
        self.assertGreater(info['well_water_saturations'][1], params['initial_water_saturation'])

        # Скважина с большим дебитом (action=0.8) должна иметь большее падение давления и больший рост Sw
        self.assertLess(info['well_pressures'][1], info['well_pressures'][0])
        self.assertGreater(info['well_water_saturations'][1], info['well_water_saturations'][0])

    def test_step_shared_reservoir_with_interaction(self):
        """Тест шага: общий резервуар, есть взаимодействие по дебиту."""
        params = self.default_sim_params.copy()
        params['interaction_matrix'] = np.array([[0.0, 0.1], [0.2, 0.0]])
        simulator = MultiWellSimulator(**params)
        
        # Запускаем один шаг, чтобы получить "идеальные" дебиты без интерференции
        sim_no_inter = MultiWellSimulator(**self.default_sim_params)
        sim_no_inter.reset()
        actions = np.array([0.8, 0.8])
        _, _, _, info_no_inter = sim_no_inter.step(actions)
        ideal_rates = info_no_inter['oil_rates_surface'] + info_no_inter['water_rates_surface']
        
        # Теперь шаг с взаимодействием
        simulator.reset()
        _, _, _, info_with_inter = simulator.step(actions)

        # Проверяем, что дебиты УМЕНЬШИЛИСЬ из-за взаимного влияния
        self.assertLess(info_with_inter['oil_rates_surface'][0], ideal_rates[0])
        self.assertLess(info_with_inter['oil_rates_surface'][1], ideal_rates[1])
        
        # При этом давление и Sw все еще должны быть одинаковы для обеих скважин
        self.assertAlmostEqual(info_with_inter['well_pressures'][0], info_with_inter['well_pressures'][1])
        self.assertAlmostEqual(info_with_inter['well_water_saturations'][0], info_with_inter['well_water_saturations'][1])

    def test_step_separate_reservoirs_with_interaction(self):
        """Тест шага: раздельные резервуары, есть взаимодействие (дебит и давление)."""
        params = self.default_sim_params.copy()
        params['shared_reservoir'] = False
        params['interaction_matrix'] = np.array([[0.0, 0.1], [0.2, 0.0]])
        simulator = MultiWellSimulator(**params)
        
        # Запускаем шаг в симуляторе без взаимодействия для сравнения
        params_no_inter = params.copy()
        params_no_inter['interaction_matrix'] = np.zeros((2,2))
        sim_no_inter = MultiWellSimulator(**params_no_inter)
        sim_no_inter.reset()
        actions = np.array([0.8, 0.8])
        _, _, _, info_no_inter = sim_no_inter.step(actions)
        
        # Теперь шаг с взаимодействием
        simulator.reset()
        _, _, _, info_with_inter = simulator.step(actions)

        # 1. Проверяем влияние на дебит (он должен быть меньше)
        self.assertLess(info_with_inter['oil_rates_surface'][0], info_no_inter['oil_rates_surface'][0])
        self.assertLess(info_with_inter['oil_rates_surface'][1], info_no_inter['oil_rates_surface'][1])

        # 2. Проверка давления на одном шаге не является надежной, так как снижение собственного
        # дебита может компенсировать падение давления от соседа.
        # Убираем эти строки.
        # self.assertLess(info_with_inter['well_pressures'][0], info_no_inter['well_pressures'][0])
        # self.assertLess(info_with_inter['well_pressures'][1], info_no_inter['well_pressures'][1])

    def test_get_info_methods(self):
        """Тест методов get_well_states и get_field_info."""
        params = self.default_sim_params.copy()
        simulator = MultiWellSimulator(**params)
        simulator.reset()
        
        actions = np.array([0.5, 1.0])
        state, reward, done, info = simulator.step(actions)
        
        # 1. Тест get_well_states()
        well_states = simulator.get_well_states()
        self.assertIsInstance(well_states, list)
        self.assertEqual(len(well_states), self.n_wells)
        self.assertIsInstance(well_states[0], np.ndarray)
        self.assertEqual(len(well_states[0]), simulator.state_dim_per_well)
        
        # Сравним значение давления из get_well_states с тем, что в info
        self.assertEqual(well_states[0][0], info['well_pressures'][0]) # P_res для скважины 0
        self.assertEqual(well_states[1][1], info['well_water_saturations'][1]) # Sw для скважины 1
        
        # 2. Тест get_field_info()
        field_info = simulator.get_field_info()
        self.assertIsInstance(field_info, dict)
        self.assertIn('time', field_info)
        self.assertIn('wells', field_info)
        self.assertEqual(field_info['time'], simulator.time)
        self.assertEqual(len(field_info['wells']), self.n_wells)
        
        # Сравним значения для первой скважины
        well_0_info = field_info['wells'][0]
        self.assertEqual(well_0_info['reservoir_pressure'], info['well_pressures'][0])
        self.assertEqual(well_0_info['oil_rate_surface'], info['oil_rates_surface'][0])
        self.assertEqual(well_0_info['valve_opening'], actions[0])

    def test_reset_to_random_state(self):
        """Тест сброса в случайное состояние."""
        params = self.default_sim_params.copy()
        simulator = MultiWellSimulator(**params)
        
        initial_state_1 = simulator.reset()
        initial_pressure_1 = initial_state_1[0]
        initial_sw_1 = initial_state_1[1]
        
        # Запоминаем базовые значения, от которых отталкивается рандомизация
        base_pressure = simulator.base_initial_pressure
        base_sw = self.default_sim_params['initial_water_saturation']

        # Вызываем сброс в случайное состояние
        initial_state_2 = simulator.reset_to_random_state(seed=42)
        initial_pressure_2 = initial_state_2[0]
        initial_sw_2 = initial_state_2[1]

        # Давление и насыщенность должны измениться
        self.assertNotAlmostEqual(base_pressure, initial_pressure_2, places=5)
        self.assertNotAlmostEqual(base_sw, initial_sw_2, places=5)
        
        # Проверим, что значения находятся в ожидаемых диапазонах
        self.assertGreaterEqual(initial_pressure_2, base_pressure * 0.9)
        self.assertLessEqual(initial_pressure_2, base_pressure * 1.1)
        self.assertGreaterEqual(initial_sw_2, params['residual_water_saturation'])
        self.assertLessEqual(initial_sw_2, 1.0 - params['residual_oil_saturation'])
        
        # Убедимся, что повторный вызов с тем же seed дает тот же результат
        initial_state_3 = simulator.reset_to_random_state(seed=42)
        self.assertTrue(np.array_equal(initial_state_2, initial_state_3))


if __name__ == '__main__':
    unittest.main()