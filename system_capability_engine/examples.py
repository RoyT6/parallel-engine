"""
Example usage of the System Capability Engine
Demonstrates how to use optimal parallelism for various tasks.
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Any

from .detector import SystemCapabilities
from .optimizer import (
    TaskType, get_optimal_config, ParallelExecutor,
    get_scraping_config, get_gpu_config
)
from .report import capability_report, get_capability_dict


# ============================================================================
# EXAMPLE 1: CPU-Bound Processing
# ============================================================================

def cpu_bound_example():
    """
    Example: Process large dataset with CPU-bound operations.
    Uses 14 process workers (physical cores - 2 for system).
    """
    config = get_optimal_config(TaskType.CPU_BOUND)

    print(f"CPU-Bound Configuration:")
    print(f"  Workers: {config.process_workers}")
    print(f"  Chunk Size: {config.process_pool_chunk_size}")

    def heavy_computation(x):
        """Simulated heavy computation"""
        result = 0
        for i in range(1000000):
            result += x * i
        return result

    # Use optimal worker count
    with ProcessPoolExecutor(max_workers=config.process_workers) as executor:
        items = range(100)
        results = list(executor.map(
            heavy_computation,
            items,
            chunksize=config.process_pool_chunk_size // 100
        ))

    return results


# ============================================================================
# EXAMPLE 2: I/O-Bound Processing
# ============================================================================

def io_bound_example():
    """
    Example: Handle many I/O operations concurrently.
    Uses 128 thread workers for maximum I/O parallelism.
    """
    config = get_optimal_config(TaskType.IO_BOUND)

    print(f"I/O-Bound Configuration:")
    print(f"  Thread Workers: {config.thread_workers}")
    print(f"  Async Semaphore: {config.async_semaphore}")

    def io_operation(url):
        """Simulated I/O operation"""
        import time
        time.sleep(0.01)  # Simulate network delay
        return f"Fetched: {url}"

    with ThreadPoolExecutor(max_workers=config.thread_workers) as executor:
        urls = [f"https://example.com/page/{i}" for i in range(1000)]
        results = list(executor.map(io_operation, urls))

    return results


# ============================================================================
# EXAMPLE 3: Async Web Scraping
# ============================================================================

async def scraping_example():
    """
    Example: Web scraping with optimal concurrency.
    Uses configuration optimized for respectful, high-performance scraping.
    """
    config = get_optimal_config(TaskType.SCRAPING)
    scraping_cfg = get_scraping_config()

    print(f"Scraping Configuration:")
    print(f"  Concurrent Requests: {config.concurrent_requests}")
    print(f"  Semaphore Limit: {scraping_cfg['semaphore_limit']}")

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(config.async_semaphore)

    async def fetch_url(url: str) -> str:
        """Fetch a URL with rate limiting"""
        async with semaphore:
            # Simulated async fetch
            await asyncio.sleep(0.01)
            return f"Scraped: {url}"

    urls = [f"https://example.com/item/{i}" for i in range(500)]
    tasks = [fetch_url(url) for url in urls]
    results = await asyncio.gather(*tasks)

    return results


# ============================================================================
# EXAMPLE 4: aiohttp Configuration
# ============================================================================

def get_aiohttp_session_config():
    """
    Get optimal aiohttp session configuration for your system.

    Usage:
        import aiohttp
        config = get_aiohttp_session_config()
        connector = aiohttp.TCPConnector(**config['connector'])
        timeout = aiohttp.ClientTimeout(**config['timeout'])
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            ...
    """
    scraping_cfg = get_scraping_config()
    aiohttp_cfg = scraping_cfg['aiohttp']

    return {
        'connector': {
            'limit': aiohttp_cfg['connector_limit'],
            'limit_per_host': aiohttp_cfg['connector_limit_per_host'],
            'ttl_dns_cache': aiohttp_cfg['ttl_dns_cache'],
            'enable_cleanup_closed': True,
        },
        'timeout': {
            'total': aiohttp_cfg['timeout_total'],
            'connect': aiohttp_cfg['timeout_connect'],
        }
    }


# ============================================================================
# EXAMPLE 5: httpx Configuration
# ============================================================================

def get_httpx_client_config():
    """
    Get optimal httpx client configuration for your system.

    Usage:
        import httpx
        config = get_httpx_client_config()
        async with httpx.AsyncClient(**config) as client:
            ...
    """
    scraping_cfg = get_scraping_config()
    httpx_cfg = scraping_cfg['httpx']

    return {
        'limits': {
            'max_connections': httpx_cfg['max_connections'],
            'max_keepalive_connections': httpx_cfg['max_keepalive_connections'],
        },
        'timeout': httpx_cfg['timeout'],
        'http2': httpx_cfg['http2'],
    }


# ============================================================================
# EXAMPLE 6: Scrapy Settings
# ============================================================================

def get_scrapy_settings():
    """
    Get optimal Scrapy settings for your system.

    Usage in settings.py:
        from system_capability_engine.examples import get_scrapy_settings
        settings = get_scrapy_settings()
        CONCURRENT_REQUESTS = settings['CONCURRENT_REQUESTS']
        ...
    """
    scraping_cfg = get_scraping_config()
    return scraping_cfg['scrapy']


# ============================================================================
# EXAMPLE 7: PyTorch DataLoader
# ============================================================================

def get_pytorch_dataloader_config():
    """
    Get optimal PyTorch DataLoader configuration for your system.

    Usage:
        from torch.utils.data import DataLoader
        config = get_pytorch_dataloader_config()
        loader = DataLoader(dataset, batch_size=config['batch_size'], **config['loader_args'])
    """
    gpu_cfg = get_gpu_config()

    return {
        'loader_args': gpu_cfg['pytorch_dataloader'],
        'batch_size': gpu_cfg['optimal_batch_size'],
        'batch_sizes': gpu_cfg['batch_sizes'],
    }


# ============================================================================
# EXAMPLE 8: Mixed Workload with ParallelExecutor
# ============================================================================

def mixed_workload_example():
    """
    Example: Handle mixed CPU and I/O workloads with unified executor.
    """
    with ParallelExecutor(TaskType.MIXED) as executor:
        print(f"Mixed Workload Configuration:")
        print(f"  Process Workers: {executor.config.process_workers}")
        print(f"  Thread Workers: {executor.config.thread_workers}")

        # CPU-bound work
        def compute(x):
            return x ** 2

        cpu_results = list(executor.map_cpu_bound(compute, range(1000)))

        # I/O-bound work
        def io_op(x):
            import time
            time.sleep(0.001)
            return x * 2

        io_results = list(executor.map_io_bound(io_op, range(100)))

    return cpu_results, io_results


# ============================================================================
# EXAMPLE 9: Inform Incoming Task of Capabilities
# ============================================================================

def get_task_context() -> dict:
    """
    Generate a context dictionary to pass to incoming tasks.
    This informs the task of available resources and optimal settings.

    Usage:
        context = get_task_context()
        my_task.run(context=context)
        # Task can then use context['optimal_workers']['cpu_bound'] etc.
    """
    caps = SystemCapabilities.detect()
    return get_capability_dict(caps)


# ============================================================================
# MAIN: Run all examples and print report
# ============================================================================

def main():
    """Run examples and print system report"""
    print(capability_report())
    print("\n" + "=" * 70)
    print("RUNNING EXAMPLES")
    print("=" * 70 + "\n")

    # CPU-bound
    print("1. CPU-Bound Example:")
    cpu_bound_example()
    print("   Complete!\n")

    # I/O-bound
    print("2. I/O-Bound Example:")
    io_bound_example()
    print("   Complete!\n")

    # Async scraping
    print("3. Async Scraping Example:")
    asyncio.run(scraping_example())
    print("   Complete!\n")

    # Mixed workload
    print("4. Mixed Workload Example:")
    mixed_workload_example()
    print("   Complete!\n")

    # Show task context
    print("5. Task Context (for informing incoming tasks):")
    context = get_task_context()
    print(f"   CPU cores: {context['cpu']['physical_cores']}/{context['cpu']['logical_cores']}")
    print(f"   RAM: {context['memory']['total_gb']} GB")
    print(f"   GPU: {context['gpu']['name']}")
    print(f"   Optimal CPU workers: {context['optimal_workers']['cpu_bound']}")
    print(f"   Optimal I/O workers: {context['optimal_workers']['io_bound']}")


if __name__ == "__main__":
    main()
