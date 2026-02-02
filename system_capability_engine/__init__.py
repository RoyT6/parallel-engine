# System Capability Engine
# Optimizes Ryzen 9 3950X (16c/32t) + 128GB DDR4 + RTX 3080Ti for maximum parallelism

from .detector import SystemCapabilities
from .optimizer import get_optimal_config, TaskType
from .report import capability_report
from .env_setup import (
    setup_parallel_env,
    setup_cpu_intensive,
    setup_io_intensive,
    setup_gpu_compute,
    setup_rapids,
    get_parallel_env_vars,
    print_current_env,
)

__all__ = [
    'SystemCapabilities',
    'get_optimal_config',
    'TaskType',
    'capability_report',
    'setup_parallel_env',
    'setup_cpu_intensive',
    'setup_io_intensive',
    'setup_gpu_compute',
    'setup_rapids',
    'get_parallel_env_vars',
    'print_current_env',
]
__version__ = '1.1.0'
