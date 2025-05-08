# -*- coding: utf-8 -*-
import re
from typing import Optional, Dict, Tuple, Any, List

# Константы для цветов в консоли
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_CYAN = "\033[96m"
COLOR_MAGENTA = "\033[35m"

# Константа для дискретных действий
DISCRETE_ACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def parse_llm_action(response: str) -> Tuple[Optional[float], Dict[str, float]]:
    """
    Извлекает значения действий из ответа модели.
    
    Args:
        response (str): Текст ответа модели.
        
    Returns:
        Tuple[Optional[float], Dict[str, float]]: Кортеж, содержащий извлеченное значение действия 
        (от 0 до 1 или None, если формат некорректен) и словарь наград за формат.
    """
    try:
        # Очищаем ответ
        clean_response = response.strip()
        
        # Если ответ пустой, возвращаем None для обозначения некорректного формата
        if not clean_response:
            print(f"Пустой ответ: действие не будет выполнено")
            return None, {"empty_response": -1.0}
        
        # Проверяем на специфические форматы с "=" или другими признаками неправильного ответа
        if re.search(r'(?:штуцер|choke|valve|параметр|степень)\s*[=:]', clean_response.lower()):
            print(f"Обнаружен неправильный формат с присваиванием: '{clean_response}'. Используйте просто число от 1 до 10.")
            return None, {"assignment_format": -1.0}
        
        # Пытаемся извлечь число от 1 до 10 из ответа
        # Измененный паттерн, который требует чтобы число стояло отдельно или в начале/конце строки
        number_pattern = r'(?:^|\s)([1-9]|10)(?:\s|$)'
        number_match = re.search(number_pattern, clean_response)
        
        if number_match:
            # Извлекаем число и преобразуем в индекс (0-9)
            index_str = number_match.group(1)
            try:
                index = int(index_str)
                if 1 <= index <= 10:
                    # Преобразуем индекс (1-10) в индекс массива (0-9)
                    array_index = index - 1
                    # Получаем значение действия из массива DISCRETE_ACTIONS
                    value = DISCRETE_ACTIONS[array_index]
                    print(f"Выбран вариант {index} (значение штуцера {value})")
                    return value, {"correct_format": 1.0}
                else:
                    print(f"Число {index} не входит в диапазон от 1 до 10. Действие не будет выполнено.")
                    return None, {"out_of_range": -0.8}
            except ValueError:
                # Если не удалось преобразовать в число
                print(f"Ошибка при преобразовании '{index_str}' в число. Действие не будет выполнено.")
                return None, {"not_a_number": -1.0}
        
        # Проверяем шаблоны текстового ответа для более гибкого распознавания
        # (Например, "вариант 5", "выбираю 3", и т.д.)
        text_pattern = r'(?:вариант|option|выбираю|choose|pick|выбор|вар\.?|var\.?)\s*[#]?\s*([1-9]|10)(?:[^\d]|$)'
        text_match = re.search(text_pattern, clean_response.lower())
        
        if text_match:
            index_str = text_match.group(1)
            try:
                index = int(index_str)
                if 1 <= index <= 10:
                    array_index = index - 1
                    value = DISCRETE_ACTIONS[array_index]
                    print(f"Выбран вариант {index} (значение штуцера {value}) через текстовое описание")
                    return value, {"text_format": 0.8}  # Чуть меньшая награда за текстовый формат
                else:
                    print(f"Число {index} в текстовом описании не входит в диапазон от 1 до 10. Действие не будет выполнено.")
                    return None, {"text_out_of_range": -0.8}
            except ValueError:
                print(f"Ошибка при преобразовании '{index_str}' из текстового описания в число. Действие не будет выполнено.")
                return None, {"text_not_a_number": -1.0}
        
        # Если не удалось извлечь индекс, проверяем на прямое указание значения штуцера
        # Только если значение идет отдельно, а не в составе других конструкций
        direct_value_pattern = r'^(0\.[1-9]|1\.0)$'
        direct_value_match = re.search(direct_value_pattern, clean_response)
        
        if direct_value_match:
            value_str = direct_value_match.group(1)
            try:
                value = float(value_str)
                if value in DISCRETE_ACTIONS:
                    index = DISCRETE_ACTIONS.index(value) + 1
                    print(f"Указано прямое значение штуцера {value} (вариант {index})")
                    return value, {"direct_value": 0.6}  # Еще меньшая награда за прямое значение
                else:
                    print(f"Указанное значение {value} не соответствует дискретным вариантам. Действие не будет выполнено.")
                    return None, {"invalid_direct_value": -0.8}
            except ValueError:
                print(f"Ошибка при преобразовании '{value_str}' в число. Действие не будет выполнено.")
                return None, {"direct_value_error": -1.0}
        
        # Проверка на старый формат с тегами <parameter>
        # Это для обратной совместимости
        parameter_pattern = r'<parameter>(.*?)</parameter>'
        parameter_match = re.search(parameter_pattern, clean_response, re.DOTALL)
        
        if parameter_match:
            print(f"Обнаружен устаревший формат с тегами <parameter>. Используйте просто число от 1 до 10.")
            # Проверяем, содержится ли там число от 0 до 1
            param_value_str = parameter_match.group(1).strip()
            try:
                param_value = float(param_value_str)
                if 0 <= param_value <= 1:
                    # Находим ближайшее дискретное значение
                    closest_idx = min(range(len(DISCRETE_ACTIONS)), 
                                   key=lambda i: abs(DISCRETE_ACTIONS[i] - param_value))
                    value = DISCRETE_ACTIONS[closest_idx]
                    index = closest_idx + 1
                    print(f"Преобразовано в дискретное значение: вариант {index} (значение штуцера {value})")
                    return value, {"old_format_converted": 0.3}  # Низкая награда за старый формат
            except ValueError:
                pass
            
            return None, {"old_parameter_format": -0.5}
        
        # Если ничего не подошло, выводим общую ошибку
        print(f"Не удалось определить номер варианта из ответа: '{clean_response}'. Ожидается число от 1 до 10.")
        return None, {"wrong_format": -1.0}
    except Exception as e:
        # В случае любой ошибки возвращаем None
        print(f"Ошибка при обработке ответа: {str(e)}. Действие не будет выполнено.")
        return None, {"error": -1.0}

def format_state(state, simulator):
    """
    Форматирует состояние симулятора для вывода в компактном виде.
    
    Args:
        state: состояние симулятора
        simulator: объект симулятора
        
    Returns:
        str: форматированное состояние
    """
    # Проверяем тип симулятора (одна или несколько скважин)
    if hasattr(simulator, 'well_names') and len(simulator.well_names) > 1:
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
            
            # Добавляем информацию о последнем действии, если оно есть
            if hasattr(simulator, 'last_actions') and i < len(simulator.last_actions):
                action_value = simulator.last_actions[i]
                if action_value is not None:
                    # Находим ближайший индекс в DISCRETE_ACTIONS
                    if action_value in DISCRETE_ACTIONS:
                        action_index = DISCRETE_ACTIONS.index(action_value) + 1
                        well_info += f", ПредДейст: {action_index}"
                    else:
                        # Если значение не точно соответствует, находим ближайшее
                        closest_idx = min(range(len(DISCRETE_ACTIONS)), 
                                          key=lambda j: abs(DISCRETE_ACTIONS[j] - action_value))
                        action_index = closest_idx + 1
                        well_info += f", ПредДейст≈{action_index}"
            
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
        
        # Добавляем информацию о последнем действии, если оно есть
        if hasattr(simulator, 'last_action') and simulator.last_action is not None:
            action_value = simulator.last_action
            # Находим ближайший индекс в DISCRETE_ACTIONS
            if action_value in DISCRETE_ACTIONS:
                action_index = DISCRETE_ACTIONS.index(action_value) + 1
                result += f", ПредДейст: {action_index}"
            else:
                # Если значение не точно соответствует, находим ближайшее
                closest_idx = min(range(len(DISCRETE_ACTIONS)), 
                                  key=lambda j: abs(DISCRETE_ACTIONS[j] - action_value))
                action_index = closest_idx + 1
                result += f", ПредДейст≈{action_index}"
        
        # Добавляем информацию о максимальном времени симуляции
        if hasattr(simulator, 'max_time'):
            remaining_time = simulator.max_time - time
            result += f", ост.время={remaining_time:.1f}д"
        
        return result 