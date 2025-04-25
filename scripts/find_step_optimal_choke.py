import argparse
import numpy as np
import json
from pathlib import Path
import copy
from tqdm import tqdm

# Добавляем путь к симуляторам, предполагая, что скрипт запускается из корня проекта
import sys
sys.path.append(str(Path(__file__).parent.parent))

from simulators.single_well.simulator import SingleWellSimulator
# Пока поддерживаем только SingleWell, т.к. логика копирования/состояния проще
# from simulators.multi_well.simulator import MultiWellSimulator

def parse_args():
    parser = argparse.ArgumentParser(description='Find step-optimal choke opening strategy.')

    # --- Аргументы Симулятора (копируем из trainer.py) ---
    parser.add_argument('--initial_pressure', type=float, default=200.0, help='Initial reservoir pressure (atm)')
    parser.add_argument('--initial_bhp', type=float, default=50.0, help='Initial bottom hole pressure (atm)')
    parser.add_argument('--productivity_index', type=float, default=0.1, help='Productivity index (m3/day/atm)')
    parser.add_argument('--total_volume', type=float, default=1e6, help='Total reservoir volume (m3)')
    parser.add_argument('--simulation_dt', type=float, default=1.0, help='Base simulation step size (days)')
    parser.add_argument('--simulation_max_time', type=float, default=4.0, help='Maximum simulation time (days)') # Уменьшаем по умолчанию для скорости
    parser.add_argument('--forecast_days', type=int, default=1, help='Number of days per simulation step (multiplies simulation_dt)')

    # --- Аргументы Скрипта Поиска ---
    parser.add_argument('--num_action_steps', type=int, default=11, help='Number of discrete choke values to test (e.g., 11 for 0.0, 0.1, ..., 1.0)')
    parser.add_argument('--output_file', type=str, default='optimal_strategy.json', help='Output JSON file to save the strategy')

    args = parser.parse_args()
    return args

def run_simulation_with_policy(simulator_config: dict, policy: callable, max_steps: int) -> tuple[float, list]:
    """
    Запускает симуляцию, используя заданную политику для выбора действия на каждом шаге.
    Возвращает итоговую добычу и историю (шаг, состояние, действие).
    """
    sim = SingleWellSimulator(**simulator_config)
    state = sim.reset()
    history = []
    done = False
    step = 0
    
    while not done and step < max_steps:
        action = policy(step, state, sim) # Политика определяет действие
        next_state, reward, done, info = sim.step(action)
        history.append({
            "step": step,
            "state_before": state.tolist(),
            "action": action,
            "reward": reward,
            "state_after": next_state.tolist(),
            "done": done
        })
        state = next_state
        step += 1
        
    final_production = state[2] # Накопленная добыча - 3й элемент состояния
    return final_production, history

def find_optimal_strategy(args):
    """
    Находит пошагово-оптимальную стратегию.
    """
    simulator_config = {
        'initial_reservoir_pressure': args.initial_pressure,
        'initial_bhp': args.initial_bhp,
        'productivity_index': args.productivity_index,
        'total_volume': args.total_volume,
        'dt': args.simulation_dt * args.forecast_days,
        'max_time': args.simulation_max_time
    }
    
    main_sim = SingleWellSimulator(**simulator_config)
    current_state = main_sim.reset()
    optimal_actions_log = []
    done = False
    step = 0
    max_steps = int(args.simulation_max_time / (args.simulation_dt * args.forecast_days))
    
    possible_actions = np.linspace(0.0, 1.0, args.num_action_steps)

    print(f"Finding optimal strategy for {max_steps} steps...")
    
    progress_bar = tqdm(total=max_steps, desc="Simulating steps")

    while not done and step < max_steps:
        best_action_for_step = None
        max_final_production = -np.inf

        # Пробуем каждое возможное действие на текущем шаге
        for action_candidate in possible_actions:
            # Создаем копию симулятора или восстанавливаем состояние
            # Важно: нужно именно копировать объект или его состояние,
            # чтобы предсказания не влияли друг на друга.
            # Создание нового объекта проще всего.
            temp_sim = SingleWellSimulator(**simulator_config)
            # Устанавливаем текущее состояние во временный симулятор
            temp_sim.state = copy.deepcopy(current_state)
            temp_sim.time = current_state[3] # Восстанавливаем время
            temp_sim.reservoir_pressure = current_state[0]
            temp_sim.cumulative_production = current_state[2]
            
            # 1. Делаем первый шаг с action_candidate
            temp_state, _, temp_done, _ = temp_sim.step(action_candidate)
            
            # 2. Симулируем оставшиеся шаги с ПОСТОЯННЫМ действием action_candidate
            current_step_in_temp = step + 1
            while not temp_done and current_step_in_temp < max_steps:
                 temp_state, _, temp_done, _ = temp_sim.step(action_candidate)
                 current_step_in_temp += 1

            final_production_for_candidate = temp_sim.state[2]

            # Обновляем лучшее действие для текущего шага
            if final_production_for_candidate > max_final_production:
                max_final_production = final_production_for_candidate
                best_action_for_step = action_candidate

        # Записываем найденное лучшее действие и состояние *перед* этим шагом
        optimal_actions_log.append({
            "step": step,
            "state_before": current_state.tolist(),
            "optimal_action": best_action_for_step,
            "predicted_max_final_production": max_final_production # Добыча, предсказанная при поиске
        })

        # Делаем реальный шаг в основном симуляторе с лучшим найденным действием
        if best_action_for_step is None:
             print(f"Warning: No best action found for step {step}. Using 0.0")
             best_action_for_step = 0.0
             
        current_state, _, done, _ = main_sim.step(best_action_for_step)
        step += 1
        progress_bar.update(1)

    progress_bar.close()
    print(f"Optimal strategy search completed. Total steps: {step}")
    final_production_actual = main_sim.state[2]
    print(f"Final production achieved with this strategy: {final_production_actual:.2f} m³")

    # Сохраняем результаты в JSON
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "parameters": vars(args),
            "final_production_actual": final_production_actual,
            "strategy": optimal_actions_log
        }, f, indent=4, ensure_ascii=False)
        
    print(f"Optimal strategy saved to: {output_path.resolve()}")

if __name__ == "__main__":
    args = parse_args()
    find_optimal_strategy(args) 