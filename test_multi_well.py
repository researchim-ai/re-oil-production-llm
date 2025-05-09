from simulators.multi_well.simulator import MultiWellSimulator
import numpy as np

def test_different_configurations():
    """Тестирование различных конфигураций многоскважинного симулятора"""
    print("=== Тест 1: 5 скважин с сильным взаимодействием, раздельные резервуары ===")
    sim = MultiWellSimulator(
        n_wells=5, 
        interaction_strength=0.4, 
        shared_reservoir=False, 
        total_volume=5e6
    )
    
    # Установим случайное начальное состояние
    state = sim.reset_to_random_state(min_depletion=0.2, max_depletion=0.6)
    
    # Получим информацию о месторождении
    field_info = sim.get_field_info()
    print(f'Количество скважин: {field_info["total_wells"]}')
    print(f'Степень истощения: {field_info["depletion_ratio"]:.2f}')
    print(f'Текущее время: {field_info["current_time"]:.1f} дней')
    print('Дебиты скважин:')
    for i, well in enumerate(field_info["wells"]):
        print(f'  Скважина {i+1}: {well["current_rate"]:.2f} м3/сут')
    
    # Рассчитаем интерференцию между скважинами
    print("\nМатрица интерференции:")
    interference = sim.calculate_interference_matrix()
    for i in range(sim.n_wells):
        print(f'  Скважина {i+1} -> {" ".join([f"{val:.2f}" for val in interference[i]])}')
    
    # Выполним шаги симуляции
    print("\nПроводим симуляцию 30 дней:")
    total_reward = 0
    for i in range(30):
        # Используем разную стратегию для каждой скважины
        actions = [0.9, 0.7, 0.5, 0.3, 0.1]  # от агрессивной к консервативной
        state, reward, done, info = sim.step(actions)
        total_reward += reward
        
        if i % 10 == 9 or done:  # каждые 10 дней или при завершении
            print(f'День {sim.time:.0f}:')
            for j, rate in enumerate(info['well_rates']):
                print(f'  Скважина {j+1}: Дебит = {rate:.2f} м3/сут, Штуцер = {actions[j]:.1f}')
            print(f'  Суммарный дебит: {reward:.2f} м3/сут')
            
        if done:
            print("Симуляция завершена раньше!")
            break
    
    print(f"\nОбщая добыча за 30 дней: {total_reward * sim.dt:.2f} м3")
    
    print("\n=== Тест 2: 3 скважины с общим резервуаром ===")
    sim2 = MultiWellSimulator(
        n_wells=3, 
        interaction_strength=0.0,  # нет дополнительного взаимодействия
        shared_reservoir=True,     # общий резервуар
        total_volume=3e6
    )
    
    # Начнем с начального состояния
    sim2.reset()
    
    print("Проводим симуляцию 365 дней с разными стратегиями:")
    
    # Симуляция 1 год с разными стратегиями
    total_days = 365
    step = 0
    
    # Стратегия 1: Все скважины открыты на 80%
    strategy1_days = total_days // 3
    print(f"\nСтратегия 1 ({strategy1_days} дней): Все скважины открыты на 80%")
    actions1 = [0.8, 0.8, 0.8]
    
    for i in range(strategy1_days):
        state, reward, done, info = sim2.step(actions1)
        step += 1
        
        if i % 30 == 29 or done:  # каждые 30 дней или при завершении
            print(f'День {sim2.time:.0f}: Суммарный дебит = {reward:.2f} м3/сут, Добыча = {sim2.cumulative_production:.0f} м3')
            
        if done:
            print("Симуляция завершена раньше!")
            break
    
    # Стратегия 2: Первая скважина на максимум, остальные на минимум
    strategy2_days = total_days // 3
    print(f"\nСтратегия 2 ({strategy2_days} дней): Первая скважина 100%, остальные 20%")
    actions2 = [1.0, 0.2, 0.2]
    
    for i in range(strategy2_days):
        if step >= total_days or done:
            break
            
        state, reward, done, info = sim2.step(actions2)
        step += 1
        
        if i % 30 == 29 or done:  # каждые 30 дней или при завершении
            print(f'День {sim2.time:.0f}: Суммарный дебит = {reward:.2f} м3/сут, Добыча = {sim2.cumulative_production:.0f} м3')
            
        if done:
            print("Симуляция завершена раньше!")
            break
    
    # Стратегия 3: Переменное открытие штуцеров
    print(f"\nСтратегия 3 (оставшиеся дни): Переменные штуцеры в зависимости от времени")
    remaining_days = total_days - step
    
    for i in range(remaining_days):
        if done:
            break
            
        # Динамическое изменение штуцеров в зависимости от времени
        day_factor = (sim2.time / sim2.max_time)  # нормализованное время (0-1)
        actions3 = [
            0.5 + 0.4 * np.sin(day_factor * np.pi),  # от 0.5 до 0.9 и обратно
            0.5 - 0.3 * np.sin(day_factor * np.pi),  # от 0.5 до 0.2 и обратно 
            0.5 + 0.3 * np.cos(day_factor * np.pi)   # от 0.8 до 0.2 и обратно
        ]
        
        state, reward, done, info = sim2.step(actions3)
        step += 1
        
        if i % 30 == 29 or i == remaining_days - 1 or done:  # каждые 30 дней или в конце
            print(f'День {sim2.time:.0f}: Штуцеры = [{actions3[0]:.2f}, {actions3[1]:.2f}, {actions3[2]:.2f}]')
            print(f'  Суммарный дебит = {reward:.2f} м3/сут, Добыча = {sim2.cumulative_production:.0f} м3')
    
    # Итоговая статистика
    print("\nИтоговая статистика:")
    print(f"Общая добыча: {sim2.cumulative_production:.0f} м3")
    print(f"Степень истощения: {sim2.cumulative_production / sim2.total_volume * 100:.1f}%")
    print(f"Средний дебит: {sim2.cumulative_production / step:.1f} м3/сут")

if __name__ == "__main__":
    test_different_configurations() 