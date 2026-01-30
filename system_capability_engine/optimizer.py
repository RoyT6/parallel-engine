"""
Task Optimizer
Provides optimal parallelism configuration based on task type and system capabilities.
Optimized for: Ryzen 9 3950X (16c/32t, 64MB L3), 128GB DDR4, RTX 3080Ti
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import asyncio
import multiprocessing


class TaskType(Enum):
    """Types of tasks with different parallelism requirements"""
    CPU_BOUND = auto()          # Heavy computation
    IO_BOUND = auto()           # Network, file I/O
    SCRAPING = auto()           # Web scraping (mixed I/O + parsing)
    GPU_COMPUTE = auto()        # CUDA/GPU workloads
    MIXED = auto()              # General purpose
    DATA_PIPELINE = auto()      # ETL, data processing
    MEMORY_INTENSIVE = auto()   # Large dataset processing


@dataclass
class OptimalConfig:
    """Optimal configuration for a task"""
    # Worker counts
    process_workers: int = 14           # For CPU-bound (physical cores - 2)
    thread_workers: int = 32            # For I/O-bound (logical cores)
    async_semaphore: int = 128          # For asyncio concurrency
    gpu_streams: int = 16               # CUDA streams

    # Pool configurations
    process_pool_chunk_size: int = 1000
    thread_pool_queue_size: int = 10000
    connection_pool_size: int = 100

    # Memory settings
    max_memory_per_worker_gb: float = 4.0
    use_memory_mapping: bool = True
    pin_gpu_memory: bool = True

    # Affinity settings
    use_core_affinity: bool = True
    spread_across_ccx: bool = True      # For Ryzen CCX topology

    # Batch settings
    optimal_batch_size: int = 1000
    prefetch_factor: int = 2

    # Scraping specific
    concurrent_requests: int = 100
    dns_cache_size: int = 1000
    use_http2: bool = True
    request_timeout: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy unpacking"""
        return {
            'process_workers': self.process_workers,
            'thread_workers': self.thread_workers,
            'async_semaphore': self.async_semaphore,
            'gpu_streams': self.gpu_streams,
            'process_pool_chunk_size': self.process_pool_chunk_size,
            'thread_pool_queue_size': self.thread_pool_queue_size,
            'connection_pool_size': self.connection_pool_size,
            'max_memory_per_worker_gb': self.max_memory_per_worker_gb,
            'use_memory_mapping': self.use_memory_mapping,
            'pin_gpu_memory': self.pin_gpu_memory,
            'use_core_affinity': self.use_core_affinity,
            'spread_across_ccx': self.spread_across_ccx,
            'optimal_batch_size': self.optimal_batch_size,
            'prefetch_factor': self.prefetch_factor,
            'concurrent_requests': self.concurrent_requests,
            'dns_cache_size': self.dns_cache_size,
            'use_http2': self.use_http2,
            'request_timeout': self.request_timeout,
        }


# Optimal configurations per task type for Ryzen 9 3950X + 128GB + RTX 3080Ti
TASK_CONFIGS = {
    TaskType.CPU_BOUND: OptimalConfig(
        process_workers=14,             # Physical cores minus system overhead
        thread_workers=2,               # Minimal threads, use processes
        async_semaphore=14,
        process_pool_chunk_size=10000,  # Larger chunks for CPU work
        max_memory_per_worker_gb=8.0,   # 128GB / 16 cores
        use_core_affinity=True,
        spread_across_ccx=True,         # Spread across 4 CCX for L3 efficiency
        optimal_batch_size=10000,
    ),

    TaskType.IO_BOUND: OptimalConfig(
        process_workers=4,              # Few processes
        thread_workers=128,             # Many threads for I/O
        async_semaphore=256,            # High async concurrency
        connection_pool_size=200,
        concurrent_requests=200,
        use_core_affinity=False,        # Let OS schedule I/O threads
        optimal_batch_size=100,
    ),

    TaskType.SCRAPING: OptimalConfig(
        process_workers=8,              # For parsing
        thread_workers=64,              # For requests
        async_semaphore=150,            # Controlled concurrency
        connection_pool_size=100,
        concurrent_requests=100,        # Respectful scraping rate
        dns_cache_size=2000,
        use_http2=True,
        request_timeout=30,
        optimal_batch_size=50,
    ),

    TaskType.GPU_COMPUTE: OptimalConfig(
        process_workers=4,              # CPU workers to feed GPU
        thread_workers=8,               # For data loading
        gpu_streams=16,                 # CUDA streams for RTX 3080Ti
        pin_gpu_memory=True,
        prefetch_factor=4,              # Prefetch batches for GPU
        optimal_batch_size=2048,        # Large batches for GPU efficiency
        max_memory_per_worker_gb=2.0,   # Keep CPU memory low, use VRAM
    ),

    TaskType.MIXED: OptimalConfig(
        process_workers=8,
        thread_workers=32,
        async_semaphore=64,
        gpu_streams=8,
        connection_pool_size=50,
        optimal_batch_size=500,
    ),

    TaskType.DATA_PIPELINE: OptimalConfig(
        process_workers=14,             # Full CPU utilization
        thread_workers=16,
        async_semaphore=32,
        use_memory_mapping=True,        # Memory-map large files
        max_memory_per_worker_gb=8.0,
        process_pool_chunk_size=50000,  # Large chunks for ETL
        optimal_batch_size=50000,
        spread_across_ccx=True,
    ),

    TaskType.MEMORY_INTENSIVE: OptimalConfig(
        process_workers=8,              # Fewer workers, more memory each
        thread_workers=8,
        max_memory_per_worker_gb=14.0,  # ~112GB for workers, rest for system
        use_memory_mapping=True,
        process_pool_chunk_size=100000,
        optimal_batch_size=100000,
        spread_across_ccx=False,        # Keep memory local to CCX
    ),
}


def get_optimal_config(task_type: TaskType = TaskType.MIXED) -> OptimalConfig:
    """Get optimal configuration for a task type"""
    return TASK_CONFIGS.get(task_type, TASK_CONFIGS[TaskType.MIXED])


def create_process_pool(config: Optional[OptimalConfig] = None) -> ProcessPoolExecutor:
    """Create an optimally configured process pool"""
    if config is None:
        config = get_optimal_config(TaskType.CPU_BOUND)

    return ProcessPoolExecutor(
        max_workers=config.process_workers,
        mp_context=multiprocessing.get_context('spawn')  # Safer on Windows
    )


def create_thread_pool(config: Optional[OptimalConfig] = None) -> ThreadPoolExecutor:
    """Create an optimally configured thread pool"""
    if config is None:
        config = get_optimal_config(TaskType.IO_BOUND)

    return ThreadPoolExecutor(
        max_workers=config.thread_workers,
        thread_name_prefix='capability_engine_'
    )


def create_async_semaphore(config: Optional[OptimalConfig] = None) -> asyncio.Semaphore:
    """Create an optimally configured async semaphore"""
    if config is None:
        config = get_optimal_config(TaskType.IO_BOUND)

    return asyncio.Semaphore(config.async_semaphore)


class ParallelExecutor:
    """
    Unified parallel executor that automatically selects the best
    parallelism strategy based on task type.
    """

    def __init__(self, task_type: TaskType = TaskType.MIXED):
        self.task_type = task_type
        self.config = get_optimal_config(task_type)
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._thread_pool: Optional[ThreadPoolExecutor] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    @property
    def process_pool(self) -> ProcessPoolExecutor:
        if self._process_pool is None:
            self._process_pool = create_process_pool(self.config)
        return self._process_pool

    @property
    def thread_pool(self) -> ThreadPoolExecutor:
        if self._thread_pool is None:
            self._thread_pool = create_thread_pool(self.config)
        return self._thread_pool

    def map_cpu_bound(self, func: Callable, items, chunksize: Optional[int] = None):
        """Execute CPU-bound work across process pool"""
        if chunksize is None:
            chunksize = self.config.process_pool_chunk_size
        return self.process_pool.map(func, items, chunksize=chunksize)

    def map_io_bound(self, func: Callable, items):
        """Execute I/O-bound work across thread pool"""
        return self.thread_pool.map(func, items)

    async def gather_async(self, coros, limit: Optional[int] = None):
        """Execute async coroutines with optimal concurrency"""
        if limit is None:
            limit = self.config.async_semaphore

        semaphore = asyncio.Semaphore(limit)

        async def limited(coro):
            async with semaphore:
                return await coro

        return await asyncio.gather(*[limited(c) for c in coros])

    def shutdown(self, wait: bool = True):
        """Shutdown all pools"""
        if self._process_pool:
            self._process_pool.shutdown(wait=wait)
        if self._thread_pool:
            self._thread_pool.shutdown(wait=wait)


def get_scraping_config() -> Dict[str, Any]:
    """Get optimal configuration for web scraping libraries"""
    config = get_optimal_config(TaskType.SCRAPING)

    return {
        # aiohttp settings
        'aiohttp': {
            'connector_limit': config.connection_pool_size,
            'connector_limit_per_host': 10,
            'timeout_total': config.request_timeout,
            'timeout_connect': 10,
            'ttl_dns_cache': 300,
        },
        # httpx settings
        'httpx': {
            'max_connections': config.connection_pool_size,
            'max_keepalive_connections': config.connection_pool_size // 2,
            'timeout': config.request_timeout,
            'http2': config.use_http2,
        },
        # requests/urllib3 settings
        'requests': {
            'pool_connections': config.connection_pool_size // 10,
            'pool_maxsize': config.connection_pool_size,
            'max_retries': 3,
        },
        # Scrapy settings
        'scrapy': {
            'CONCURRENT_REQUESTS': config.concurrent_requests,
            'CONCURRENT_REQUESTS_PER_DOMAIN': 16,
            'DOWNLOAD_DELAY': 0.1,
            'REACTOR_THREADPOOL_MAXSIZE': config.thread_workers,
            'DNS_TIMEOUT': 10,
        },
        # General async settings
        'semaphore_limit': config.async_semaphore,
        'batch_size': config.optimal_batch_size,
    }


def get_gpu_config() -> Dict[str, Any]:
    """Get optimal configuration for GPU/CUDA workloads"""
    config = get_optimal_config(TaskType.GPU_COMPUTE)

    return {
        # PyTorch DataLoader
        'pytorch_dataloader': {
            'num_workers': config.process_workers,
            'pin_memory': config.pin_gpu_memory,
            'prefetch_factor': config.prefetch_factor,
            'persistent_workers': True,
        },
        # CUDA settings
        'cuda': {
            'num_streams': config.gpu_streams,
            'enable_cudnn_benchmark': True,
            'allow_tf32': True,  # RTX 30 series
        },
        # Batch sizes for different VRAM usage (RTX 3080Ti 12GB)
        'batch_sizes': {
            'fp32': 512,    # ~6GB VRAM
            'fp16': 1024,   # ~6GB VRAM
            'int8': 2048,   # ~6GB VRAM
        },
        'optimal_batch_size': config.optimal_batch_size,
    }
