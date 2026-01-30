"""
Capability Report Generator
Generates human-readable and machine-parseable reports of system capabilities.
"""

from typing import Dict, Any, Optional
from .detector import SystemCapabilities
from .optimizer import TaskType, get_optimal_config, get_scraping_config, get_gpu_config


def capability_report(caps: Optional[SystemCapabilities] = None, verbose: bool = True) -> str:
    """
    Generate a comprehensive capability report that informs incoming tasks
    of the system's capabilities and optimal usage patterns.
    """
    if caps is None:
        caps = SystemCapabilities.detect()

    lines = []
    lines.append("=" * 70)
    lines.append("SYSTEM CAPABILITY ENGINE - HARDWARE PROFILE")
    lines.append("=" * 70)
    lines.append("")

    # CPU Section
    lines.append("CPU CAPABILITIES")
    lines.append("-" * 40)
    lines.append(f"  Model: {caps.cpu.model_name or 'AMD Ryzen 9 3950X'}")
    lines.append(f"  Physical Cores: {caps.cpu.physical_cores or 16}")
    lines.append(f"  Logical Cores (Threads): {caps.cpu.logical_cores or 32}")
    lines.append(f"  L3 Cache: {caps.cpu.l3_cache_mb or 64} MB (4-way)")
    lines.append(f"  Architecture: {caps.cpu.architecture or 'x86_64'}")
    if caps.cpu.is_ryzen:
        lines.append(f"  CCX Count: {caps.cpu.ccx_count or 4} (cores per CCX: {caps.cpu.cores_per_ccx or 4})")
    lines.append("")

    # Memory Section
    lines.append("MEMORY CAPABILITIES")
    lines.append("-" * 40)
    lines.append(f"  Total RAM: {caps.memory.total_gb:.1f} GB")
    lines.append(f"  Available RAM: {caps.memory.available_gb:.1f} GB")
    lines.append(f"  Memory Channels: {caps.memory.channels}")
    lines.append(f"  Memory Speed: DDR4-{caps.memory.speed_mhz}")
    lines.append(f"  Memory per Core: {caps.memory.memory_per_core_gb:.1f} GB")
    lines.append("")

    # GPU Section
    lines.append("GPU CAPABILITIES")
    lines.append("-" * 40)
    if caps.gpu.is_available:
        lines.append(f"  Model: {caps.gpu.name}")
        lines.append(f"  VRAM: {caps.gpu.vram_gb:.1f} GB")
        lines.append(f"  CUDA Cores: {caps.gpu.cuda_cores}")
        lines.append(f"  Tensor Cores: {caps.gpu.tensor_cores}")
        lines.append(f"  Streaming Multiprocessors: {caps.gpu.sm_count}")
        lines.append(f"  Compute Capability: {caps.gpu.compute_capability}")
        lines.append(f"  Driver Version: {caps.gpu.driver_version}")
        if caps.gpu.cuda_version:
            lines.append(f"  CUDA Version: {caps.gpu.cuda_version}")
    else:
        lines.append("  GPU: Not detected (assuming RTX 3080Ti)")
        lines.append("  VRAM: 12 GB")
        lines.append("  CUDA Cores: 10240")
        lines.append("  Tensor Cores: 320")
        lines.append("  SM Count: 80")
    lines.append("")

    # WSL Section
    if caps.wsl.is_wsl:
        lines.append("WSL ENVIRONMENT")
        lines.append("-" * 40)
        lines.append(f"  WSL Version: {caps.wsl.wsl_version}")
        lines.append(f"  Distribution: {caps.wsl.distro}")
        lines.append(f"  GPU Access: {'Yes' if caps.wsl.can_access_gpu else 'No'}")
        lines.append("")

    # Optimal Worker Counts
    lines.append("=" * 70)
    lines.append("OPTIMAL PARALLELISM CONFIGURATION")
    lines.append("=" * 70)
    lines.append("")
    lines.append("RECOMMENDED WORKER COUNTS")
    lines.append("-" * 40)
    lines.append(f"  CPU-Bound Workers: {caps.optimal_cpu_workers}")
    lines.append(f"  I/O-Bound Workers: {caps.optimal_io_workers}")
    lines.append(f"  GPU Workers: {caps.optimal_gpu_workers}")
    lines.append(f"  Process Pool Size: {caps.optimal_process_pool_size}")
    lines.append(f"  Thread Pool Size: {caps.optimal_thread_pool_size}")
    lines.append("")

    if verbose:
        # Task-specific recommendations
        lines.append("TASK-SPECIFIC CONFIGURATIONS")
        lines.append("-" * 40)
        lines.append("")

        for task_type in TaskType:
            config = get_optimal_config(task_type)
            lines.append(f"  {task_type.name}:")
            lines.append(f"    Process Workers: {config.process_workers}")
            lines.append(f"    Thread Workers: {config.thread_workers}")
            lines.append(f"    Async Semaphore: {config.async_semaphore}")
            lines.append(f"    Batch Size: {config.optimal_batch_size}")
            lines.append("")

        # Scraping-specific
        lines.append("WEB SCRAPING CONFIGURATION")
        lines.append("-" * 40)
        scraping = get_scraping_config()
        lines.append(f"  Concurrent Requests: {scraping['scrapy']['CONCURRENT_REQUESTS']}")
        lines.append(f"  Connection Pool: {scraping['aiohttp']['connector_limit']}")
        lines.append(f"  HTTP/2 Enabled: {scraping['httpx']['http2']}")
        lines.append(f"  Semaphore Limit: {scraping['semaphore_limit']}")
        lines.append("")

        # GPU-specific
        lines.append("GPU/CUDA CONFIGURATION")
        lines.append("-" * 40)
        gpu = get_gpu_config()
        lines.append(f"  DataLoader Workers: {gpu['pytorch_dataloader']['num_workers']}")
        lines.append(f"  CUDA Streams: {gpu['cuda']['num_streams']}")
        lines.append(f"  Pin Memory: {gpu['pytorch_dataloader']['pin_memory']}")
        lines.append(f"  FP16 Batch Size: {gpu['batch_sizes']['fp16']}")
        lines.append("")

    # Usage Instructions
    lines.append("=" * 70)
    lines.append("HOW TO USE THIS ENGINE")
    lines.append("=" * 70)
    lines.append("""
QUICK START:
    from system_capability_engine import (
        SystemCapabilities, get_optimal_config, TaskType
    )

    # Detect system capabilities
    caps = SystemCapabilities.detect()

    # Get optimal config for your task type
    config = get_optimal_config(TaskType.SCRAPING)

    # Use the recommended settings
    workers = config.thread_workers  # 64 for scraping
    semaphore = asyncio.Semaphore(config.async_semaphore)  # 150

PARALLEL EXECUTION:
    from system_capability_engine.optimizer import ParallelExecutor

    with ParallelExecutor(TaskType.CPU_BOUND) as executor:
        results = list(executor.map_cpu_bound(process_func, items))

SCRAPING:
    from system_capability_engine import get_scraping_config

    config = get_scraping_config()
    connector = aiohttp.TCPConnector(**config['aiohttp'])

GPU WORKLOADS:
    from system_capability_engine import get_gpu_config

    config = get_gpu_config()
    dataloader = DataLoader(dataset, **config['pytorch_dataloader'])
""")

    return "\n".join(lines)


def get_capability_dict(caps: Optional[SystemCapabilities] = None) -> Dict[str, Any]:
    """
    Get system capabilities as a dictionary for programmatic use.
    Pass this to incoming tasks to inform them of optimal settings.
    """
    if caps is None:
        caps = SystemCapabilities.detect()

    return {
        'cpu': {
            'model': caps.cpu.model_name or 'AMD Ryzen 9 3950X',
            'physical_cores': caps.cpu.physical_cores or 16,
            'logical_cores': caps.cpu.logical_cores or 32,
            'l3_cache_mb': caps.cpu.l3_cache_mb or 64,
            'is_ryzen': caps.cpu.is_ryzen,
            'ccx_count': caps.cpu.ccx_count or 4,
        },
        'memory': {
            'total_gb': caps.memory.total_gb or 128,
            'available_gb': caps.memory.available_gb or 100,
            'channels': caps.memory.channels or 4,
            'speed_mhz': caps.memory.speed_mhz or 3200,
        },
        'gpu': {
            'name': caps.gpu.name or 'NVIDIA RTX 3080Ti',
            'vram_gb': caps.gpu.vram_gb or 12,
            'cuda_cores': caps.gpu.cuda_cores or 10240,
            'tensor_cores': caps.gpu.tensor_cores or 320,
            'sm_count': caps.gpu.sm_count or 80,
            'is_available': caps.gpu.is_available,
        },
        'platform': caps.platform or 'Windows',
        'optimal_workers': {
            'cpu_bound': caps.optimal_cpu_workers or 14,
            'io_bound': caps.optimal_io_workers or 128,
            'gpu': caps.optimal_gpu_workers or 80,
            'process_pool': caps.optimal_process_pool_size or 15,
            'thread_pool': caps.optimal_thread_pool_size or 32,
        },
        'configs': {
            task_type.name.lower(): get_optimal_config(task_type).to_dict()
            for task_type in TaskType
        },
        'scraping': get_scraping_config(),
        'gpu_compute': get_gpu_config(),
    }


def print_report():
    """Print the capability report to stdout"""
    print(capability_report())


if __name__ == "__main__":
    print_report()
