# System Capability Engine
# Optimizes Ryzen 9 3950X (16c/32t) + 128GB DDR4 + RTX 3080Ti for maximum parallelism

from .detector import SystemCapabilities
from .optimizer import get_optimal_config, TaskType
from .report import capability_report

__all__ = ['SystemCapabilities', 'get_optimal_config', 'TaskType', 'capability_report']
__version__ = '1.0.0'
