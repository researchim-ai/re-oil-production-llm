#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Основной скрипт для запуска обучения LLM для оптимизации добычи нефти.
"""

from .trainer import main
import sys

if __name__ == "__main__":
    # Добавляем параметры для стабильной генерации, если они не указаны
    if '--temperature' not in sys.argv:
        sys.argv.extend(['--temperature', '0.1'])
    
    main() 