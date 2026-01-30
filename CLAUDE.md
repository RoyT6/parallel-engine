# Parallel Engine - System Capability Engine

## Project Overview
A Python engine that detects hardware capabilities and provides optimal parallelism configurations for a Ryzen 9 3950X + 128GB DDR4 + RTX 3080Ti system.

## Hardware Profile (Detected)
- **CPU**: AMD Ryzen 9 3950X (16 cores / 32 threads, 64MB L3 cache, 4 CCX)
- **RAM**: 128 GB DDR4-3200, quad-channel
- **GPU**: NVIDIA RTX 3080Ti (12GB VRAM, 10240 CUDA cores, 80 SMs, Compute 8.6)
- **CUDA**: 13.1

## Optimal Worker Counts
| Task Type | Workers | Rationale |
|-----------|---------|-----------|
| CPU-Bound | 14 processes | Physical cores minus 2 for system overhead |
| I/O-Bound | 128 threads | 4x logical cores, RAM allows aggressive threading |
| Scraping | 100 concurrent | Balanced for respectful rate limiting |
| GPU Compute | 4 CPU feeders + 16 CUDA streams | Keeps GPU saturated |
| Data Pipeline | 14 processes | Full CPU utilization for ETL |

## Files
- `detector.py` - Hardware detection (CPU, RAM, GPU, WSL)
- `optimizer.py` - Task-specific parallelism configs and ParallelExecutor class
- `report.py` - Human-readable capability reports
- `quick_config.py` - One-liner constants for quick access
- `examples.py` - Usage examples for common scenarios
- `__init__.py` / `__main__.py` - Package entry points

## Usage
```python
# Quick constants
from system_capability_engine.quick_config import CPU_WORKERS, IO_WORKERS, ASYNC_LIMIT

# Full detection
from system_capability_engine import SystemCapabilities, get_optimal_config, TaskType
caps = SystemCapabilities.detect()
config = get_optimal_config(TaskType.SCRAPING)

# Run report
python -m system_capability_engine
```

## Session: 2026-01-19

### Accomplished
- Created complete system capability detection engine
- Implemented auto-detection for CPU (with Ryzen CCX topology), RAM, GPU/CUDA, and WSL
- Built task-specific optimization profiles (CPU_BOUND, IO_BOUND, SCRAPING, GPU_COMPUTE, DATA_PIPELINE, MEMORY_INTENSIVE)
- Created ParallelExecutor unified interface
- Added quick_config.py for one-liner access to optimal settings
- Tested and verified all detection working correctly
- Moved all files to `Downloads\Parallel Engine\` directory

### Key Design Decisions
- Worker counts leave 2 cores for system overhead
- Ryzen CCX awareness for L3 cache efficiency
- Scraping limited to 100 concurrent to be respectful
- GPU configs use pin_memory and prefetch for throughput
- Process pools use 'spawn' context for Windows compatibility
