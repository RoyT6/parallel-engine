"""
System Capability Detector
Detects and reports hardware capabilities for optimal parallelism configuration.
Optimized for: Ryzen 9 3950X (16c/32t, 64MB L3), 128GB DDR4, RTX 3080Ti
"""

import os
import sys
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import multiprocessing


@dataclass
class CPUInfo:
    """CPU capability information"""
    physical_cores: int = 0
    logical_cores: int = 0
    l3_cache_mb: int = 0
    architecture: str = ""
    model_name: str = ""
    is_ryzen: bool = False
    numa_nodes: int = 1

    # Ryzen 9 3950X specifics
    ccx_count: int = 0  # Core Complex count (4 for 3950X)
    cores_per_ccx: int = 0

    @property
    def hyperthreading_ratio(self) -> float:
        if self.physical_cores == 0:
            return 1.0
        return self.logical_cores / self.physical_cores


@dataclass
class MemoryInfo:
    """Memory capability information"""
    total_gb: float = 0
    available_gb: float = 0
    channels: int = 0
    speed_mhz: int = 0

    @property
    def memory_per_core_gb(self) -> float:
        """Memory available per logical core"""
        return self.total_gb / max(1, 32)  # Assuming 32 threads


@dataclass
class GPUInfo:
    """GPU capability information"""
    name: str = ""
    vram_gb: float = 0
    cuda_cores: int = 0
    compute_capability: str = ""
    is_available: bool = False
    driver_version: str = ""
    cuda_version: str = ""

    # RTX 3080Ti specifics
    tensor_cores: int = 0
    sm_count: int = 0  # Streaming Multiprocessors


@dataclass
class WSLInfo:
    """WSL environment information"""
    is_wsl: bool = False
    wsl_version: int = 0
    distro: str = ""
    can_access_gpu: bool = False


@dataclass
class SystemCapabilities:
    """Complete system capability report"""
    cpu: CPUInfo = field(default_factory=CPUInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    gpu: GPUInfo = field(default_factory=GPUInfo)
    wsl: WSLInfo = field(default_factory=WSLInfo)
    platform: str = ""
    python_version: str = ""

    # Derived optimal settings
    optimal_cpu_workers: int = 0
    optimal_io_workers: int = 0
    optimal_gpu_workers: int = 0
    optimal_process_pool_size: int = 0
    optimal_thread_pool_size: int = 0

    @classmethod
    def detect(cls) -> 'SystemCapabilities':
        """Detect all system capabilities"""
        caps = cls()
        caps.platform = platform.system()
        caps.python_version = platform.python_version()

        caps.cpu = _detect_cpu()
        caps.memory = _detect_memory()
        caps.gpu = _detect_gpu()
        caps.wsl = _detect_wsl()

        # Calculate optimal worker counts
        caps._calculate_optimal_workers()

        return caps

    def _calculate_optimal_workers(self):
        """Calculate optimal worker counts based on detected hardware"""
        # For Ryzen 9 3950X with 16 cores / 32 threads
        physical = self.cpu.physical_cores or 16
        logical = self.cpu.logical_cores or 32

        # CPU-bound tasks: use physical cores (avoid HT overhead)
        # Leave 1-2 cores for system
        self.optimal_cpu_workers = max(1, physical - 2)  # 14 for 3950X

        # I/O-bound tasks: can use more than logical cores
        # With 128GB RAM, we can go aggressive
        self.optimal_io_workers = logical * 4  # 128 for 3950X

        # GPU workers: based on SM count for RTX 3080Ti (80 SMs)
        if self.gpu.is_available:
            self.optimal_gpu_workers = min(self.gpu.sm_count or 80, 80)

        # Process pool: physical cores minus overhead
        self.optimal_process_pool_size = max(1, physical - 1)  # 15 for 3950X

        # Thread pool: logical cores for mixed workloads
        self.optimal_thread_pool_size = logical  # 32 for 3950X

    def get_config_for_task(self, task_type: str) -> Dict[str, Any]:
        """Get optimal configuration for a specific task type"""
        configs = {
            'cpu_bound': {
                'workers': self.optimal_cpu_workers,
                'use_processes': True,
                'chunk_size': self.cpu.l3_cache_mb * 1024 * 1024 // self.optimal_cpu_workers,
                'affinity_strategy': 'spread_ccx',  # Spread across CCX for L3 efficiency
            },
            'io_bound': {
                'workers': self.optimal_io_workers,
                'use_processes': False,  # Threads for I/O
                'connection_pool_size': self.optimal_io_workers,
                'semaphore_limit': self.optimal_io_workers * 2,
            },
            'scraping': {
                'concurrent_requests': min(self.optimal_io_workers, 100),
                'connection_pool_size': 50,
                'dns_cache_size': 1000,
                'use_http2': True,
                'workers': self.optimal_io_workers // 2,
            },
            'gpu_compute': {
                'cuda_streams': min(self.gpu.sm_count or 80, 16),
                'batch_size': int(self.gpu.vram_gb * 1024) if self.gpu.vram_gb else 0,
                'workers': 4,  # CPU workers to feed GPU
                'pin_memory': True,
            },
            'mixed': {
                'cpu_workers': self.optimal_cpu_workers,
                'io_workers': self.optimal_io_workers // 2,
                'gpu_enabled': self.gpu.is_available,
            },
        }
        return configs.get(task_type, configs['mixed'])


def _detect_cpu() -> CPUInfo:
    """Detect CPU capabilities"""
    info = CPUInfo()

    try:
        info.logical_cores = multiprocessing.cpu_count()
    except:
        info.logical_cores = os.cpu_count() or 1

    if platform.system() == 'Windows':
        info = _detect_cpu_windows(info)
    else:
        info = _detect_cpu_linux(info)

    # Detect if Ryzen and configure CCX info
    if 'ryzen' in info.model_name.lower() or 'amd' in info.model_name.lower():
        info.is_ryzen = True
        # Ryzen 9 3950X has 4 CCX with 4 cores each
        if info.physical_cores >= 16:
            info.ccx_count = 4
            info.cores_per_ccx = info.physical_cores // 4

    return info


def _detect_cpu_windows(info: CPUInfo) -> CPUInfo:
    """Windows-specific CPU detection"""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        info.model_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        winreg.CloseKey(key)
    except:
        pass

    try:
        # Use WMIC for detailed info
        result = subprocess.run(
            ['wmic', 'cpu', 'get', 'NumberOfCores,L3CacheSize', '/format:csv'],
            capture_output=True, text=True, timeout=5
        )
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        if len(lines) > 1:
            parts = lines[-1].split(',')
            if len(parts) >= 3:
                info.l3_cache_mb = int(parts[1]) // 1024 if parts[1].isdigit() else 64
                info.physical_cores = int(parts[2]) if parts[2].isdigit() else info.logical_cores // 2
    except:
        # Default for Ryzen 9 3950X
        info.physical_cores = 16
        info.l3_cache_mb = 64

    info.architecture = platform.machine()
    return info


def _detect_cpu_linux(info: CPUInfo) -> CPUInfo:
    """Linux-specific CPU detection"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()

        for line in cpuinfo.split('\n'):
            if 'model name' in line:
                info.model_name = line.split(':')[1].strip()
            elif 'cpu cores' in line:
                info.physical_cores = int(line.split(':')[1].strip())
            elif 'cache size' in line and 'L3' in line:
                cache_str = line.split(':')[1].strip()
                info.l3_cache_mb = int(''.join(filter(str.isdigit, cache_str))) // 1024
    except:
        info.physical_cores = info.logical_cores // 2
        info.l3_cache_mb = 64

    info.architecture = platform.machine()
    return info


def _detect_memory() -> MemoryInfo:
    """Detect memory capabilities"""
    info = MemoryInfo()

    if platform.system() == 'Windows':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', c_ulonglong),
                    ('ullAvailPhys', c_ulonglong),
                    ('ullTotalPageFile', c_ulonglong),
                    ('ullAvailPageFile', c_ulonglong),
                    ('ullTotalVirtual', c_ulonglong),
                    ('ullAvailVirtual', c_ulonglong),
                    ('ullAvailExtendedVirtual', c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))

            info.total_gb = stat.ullTotalPhys / (1024**3)
            info.available_gb = stat.ullAvailPhys / (1024**3)
        except:
            info.total_gb = 128  # Default for your system
            info.available_gb = 100
    else:
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            for line in meminfo.split('\n'):
                if 'MemTotal' in line:
                    info.total_gb = int(line.split()[1]) / (1024**2)
                elif 'MemAvailable' in line:
                    info.available_gb = int(line.split()[1]) / (1024**2)
        except:
            info.total_gb = 128
            info.available_gb = 100

    # Assume DDR4-3200 quad channel for Ryzen 9 3950X
    info.channels = 4 if info.total_gb >= 64 else 2
    info.speed_mhz = 3200

    return info


def _detect_gpu() -> GPUInfo:
    """Detect GPU capabilities"""
    info = GPUInfo()

    # Try nvidia-smi first
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            info.name = parts[0].strip()
            vram_str = parts[1].strip()
            info.vram_gb = float(''.join(filter(lambda x: x.isdigit() or x == '.', vram_str))) / 1024
            info.driver_version = parts[2].strip() if len(parts) > 2 else ""
            info.is_available = True

            # RTX 3080Ti specifics
            if '3080' in info.name:
                info.cuda_cores = 10240
                info.tensor_cores = 320
                info.sm_count = 80
                info.compute_capability = "8.6"
    except:
        pass

    # Try CUDA toolkit
    if info.is_available:
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
            if 'release' in result.stdout:
                info.cuda_version = result.stdout.split('release')[1].split(',')[0].strip()
        except:
            pass

    return info


def _detect_wsl() -> WSLInfo:
    """Detect WSL environment"""
    info = WSLInfo()

    # Check if running in WSL
    try:
        if os.path.exists('/proc/version'):
            with open('/proc/version', 'r') as f:
                version = f.read().lower()
            if 'microsoft' in version or 'wsl' in version:
                info.is_wsl = True
                info.wsl_version = 2 if 'wsl2' in version else 1

                # Get distro name
                try:
                    result = subprocess.run(['lsb_release', '-d'], capture_output=True, text=True)
                    info.distro = result.stdout.split(':')[1].strip()
                except:
                    info.distro = "Unknown"

                # Check GPU access in WSL2
                if info.wsl_version == 2:
                    try:
                        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
                        info.can_access_gpu = result.returncode == 0
                    except:
                        pass
    except:
        pass

    return info
