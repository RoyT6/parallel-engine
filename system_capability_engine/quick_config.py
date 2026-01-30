"""
Quick Configuration Access
One-liner access to optimal settings for common use cases.

Usage:
    from system_capability_engine.quick_config import *

    # CPU-bound multiprocessing
    with ProcessPoolExecutor(max_workers=CPU_WORKERS) as pool:
        results = pool.map(func, data)

    # I/O-bound threading
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        results = pool.map(func, urls)

    # Async scraping
    semaphore = asyncio.Semaphore(ASYNC_LIMIT)

    # PyTorch DataLoader
    loader = DataLoader(dataset, num_workers=DATALOADER_WORKERS, pin_memory=PIN_MEMORY)
"""

# =============================================================================
# RYZEN 9 3950X + 128GB DDR4 + RTX 3080Ti OPTIMAL SETTINGS
# =============================================================================

# CPU Configuration
CPU_PHYSICAL_CORES = 16
CPU_LOGICAL_CORES = 32
L3_CACHE_MB = 64
CCX_COUNT = 4
CORES_PER_CCX = 4

# Memory Configuration
TOTAL_RAM_GB = 128
MEMORY_CHANNELS = 4
MEMORY_SPEED_MHZ = 3200

# GPU Configuration
GPU_NAME = "RTX 3080Ti"
VRAM_GB = 12
CUDA_CORES = 10240
TENSOR_CORES = 320
SM_COUNT = 80
COMPUTE_CAPABILITY = "8.6"

# =============================================================================
# OPTIMAL WORKER COUNTS
# =============================================================================

# CPU-bound tasks (use physical cores - 2 for system)
CPU_WORKERS = 14

# I/O-bound tasks (4x logical cores with this much RAM)
IO_WORKERS = 128

# Mixed workloads
MIXED_CPU_WORKERS = 8
MIXED_IO_WORKERS = 32

# GPU data loading
DATALOADER_WORKERS = 4
PIN_MEMORY = True

# =============================================================================
# POOL SIZES
# =============================================================================

PROCESS_POOL_SIZE = 15      # For ProcessPoolExecutor
THREAD_POOL_SIZE = 32       # For ThreadPoolExecutor

# =============================================================================
# ASYNC / SCRAPING LIMITS
# =============================================================================

ASYNC_LIMIT = 150           # General async semaphore
SCRAPING_LIMIT = 100        # Concurrent HTTP requests
CONNECTION_POOL = 100       # HTTP connection pool size
DNS_CACHE_SIZE = 2000       # DNS cache entries

# High-throughput I/O
HIGH_IO_ASYNC_LIMIT = 256   # For pure I/O operations

# =============================================================================
# BATCH SIZES
# =============================================================================

CPU_BATCH_SIZE = 10000      # CPU-bound processing
IO_BATCH_SIZE = 100         # I/O-bound processing
SCRAPING_BATCH_SIZE = 50    # Web scraping batches
GPU_BATCH_SIZE = 2048       # GPU processing (FP32)
GPU_BATCH_FP16 = 1024       # GPU processing (FP16)
PIPELINE_BATCH_SIZE = 50000 # Data pipeline batches

# =============================================================================
# CHUNK SIZES (for multiprocessing map)
# =============================================================================

CPU_CHUNK_SIZE = 1000       # ProcessPoolExecutor chunksize
PIPELINE_CHUNK_SIZE = 10000 # Data pipeline chunksize

# =============================================================================
# TIMEOUTS
# =============================================================================

HTTP_TIMEOUT = 30           # HTTP request timeout (seconds)
CONNECT_TIMEOUT = 10        # Connection timeout (seconds)

# =============================================================================
# SCRAPY SETTINGS (copy to your settings.py)
# =============================================================================

SCRAPY_SETTINGS = {
    'CONCURRENT_REQUESTS': 100,
    'CONCURRENT_REQUESTS_PER_DOMAIN': 16,
    'DOWNLOAD_DELAY': 0.1,
    'REACTOR_THREADPOOL_MAXSIZE': 64,
    'DNS_TIMEOUT': 10,
    'DOWNLOAD_TIMEOUT': 30,
}

# =============================================================================
# AIOHTTP SETTINGS
# =============================================================================

AIOHTTP_CONNECTOR = {
    'limit': 100,
    'limit_per_host': 10,
    'ttl_dns_cache': 300,
    'enable_cleanup_closed': True,
}

AIOHTTP_TIMEOUT = {
    'total': 30,
    'connect': 10,
}

# =============================================================================
# HTTPX SETTINGS
# =============================================================================

HTTPX_LIMITS = {
    'max_connections': 100,
    'max_keepalive_connections': 50,
}

# =============================================================================
# PYTORCH DATALOADER SETTINGS
# =============================================================================

PYTORCH_DATALOADER = {
    'num_workers': 4,
    'pin_memory': True,
    'prefetch_factor': 4,
    'persistent_workers': True,
}

# =============================================================================
# CUDA SETTINGS
# =============================================================================

CUDA_STREAMS = 16
ENABLE_CUDNN_BENCHMARK = True
ALLOW_TF32 = True  # RTX 30 series optimization

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_process_pool():
    """Get optimally configured ProcessPoolExecutor"""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    return ProcessPoolExecutor(
        max_workers=CPU_WORKERS,
        mp_context=multiprocessing.get_context('spawn')
    )


def get_thread_pool():
    """Get optimally configured ThreadPoolExecutor"""
    from concurrent.futures import ThreadPoolExecutor
    return ThreadPoolExecutor(max_workers=IO_WORKERS)


def get_semaphore(limit: int = ASYNC_LIMIT):
    """Get asyncio semaphore with optimal limit"""
    import asyncio
    return asyncio.Semaphore(limit)


def get_aiohttp_connector():
    """Get optimally configured aiohttp connector"""
    try:
        import aiohttp
        return aiohttp.TCPConnector(**AIOHTTP_CONNECTOR)
    except ImportError:
        raise ImportError("aiohttp not installed. Run: pip install aiohttp")


def get_httpx_client():
    """Get optimally configured httpx AsyncClient"""
    try:
        import httpx
        return httpx.AsyncClient(
            limits=httpx.Limits(**HTTPX_LIMITS),
            timeout=HTTP_TIMEOUT,
            http2=True,
        )
    except ImportError:
        raise ImportError("httpx not installed. Run: pip install httpx")


# =============================================================================
# PRINT QUICK REFERENCE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RYZEN 9 3950X + 128GB + RTX 3080Ti QUICK REFERENCE")
    print("=" * 60)
    print(f"\nCPU Workers (multiprocessing): {CPU_WORKERS}")
    print(f"I/O Workers (threading):       {IO_WORKERS}")
    print(f"Async Semaphore:               {ASYNC_LIMIT}")
    print(f"Scraping Concurrency:          {SCRAPING_LIMIT}")
    print(f"DataLoader Workers:            {DATALOADER_WORKERS}")
    print(f"GPU Batch Size (FP16):         {GPU_BATCH_FP16}")
    print("\nImport with: from system_capability_engine.quick_config import *")
