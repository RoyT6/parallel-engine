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

## Session: 2026-02-01

### Accomplished

**Parallel Environment Variables Setup**

1. **Created `env_setup.py`** - Python module for setting optimal environment variables
   - `setup_parallel_env()` - General parallel setup (14 CPU threads, GPU enabled)
   - `setup_cpu_intensive()` - CPU-bound workloads
   - `setup_io_intensive()` - I/O-bound workloads
   - `setup_gpu_compute()` - GPU computing
   - `setup_rapids()` - RAPIDS/cuDF optimized for trillion-parameter calculations
   - Auto-configures on import

2. **Created shell scripts**
   - `set_parallel_env.bat` - Windows batch file
   - `set_parallel_env.ps1` - PowerShell script

3. **Environment variables configured**:
   - OpenMP, MKL, OpenBLAS, NumExpr, BLIS, Polars (CPU threading)
   - CUDA, cuDNN, TensorFlow, PyTorch (GPU)
   - RAPIDS: RMM pool allocator, cuDF spilling, cuML, cuGraph
   - Memory management with 128GB host RAM spill support

### Usage

```python
# Option 1: Auto-configure on import
import system_capability_engine.env_setup

# Option 2: Task-specific setup
from system_capability_engine import setup_rapids
setup_rapids(verbose=True)

# Option 3: Check current environment
from system_capability_engine import print_current_env
print_current_env()
```

```powershell
# PowerShell
. .\set_parallel_env.ps1
```

```batch
REM Command Prompt
set_parallel_env.bat
```

---

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

## Session: 2026-01-30

### Accomplished

**IMDB ID Validation & Correction - 98.1% Accuracy Achieved**

1. **Validated and corrected IMDB IDs across 12 Netflix files**
   - Before: 83,941 / 98,263 (85.4%)
   - After: 96,419 / 98,263 (98.1%)
   - Filled: +12,478 new IMDB IDs
   - Target was 97%, exceeded by 1.1%

2. **Applied fc_uid corrections from CHANGE_THESE files**
   - Loaded 2,659 title->fc_uid mappings
   - Updated 9,911 fc_uid values across 12 files
   - Used GPU-accelerated cuDF merge (32.62s)

3. **Validated BFD_VIEWS_V27.75_SCHEMA_COMPLIANT.parquet**
   - 559,872 rows processed
   - 5,454 IMDB IDs corrected
   - 338 null values filled

### Files Created

| File | Location | Purpose |
|------|----------|---------|
| `validate_imdb_ids.py` | Parallel Engine | CPU-based IMDB validator with 14 workers |
| `validate_parquet.py` | Parallel Engine | Parquet file validator |
| `apply_fc_uid.py` | Parallel Engine | fc_uid applicator with 14 workers |
| `apply_fc_uid_gpu.py` | GPU Engine | GPU-accelerated fc_uid applicator |
| `imdb_validator_97.py` | GPU Engine | GPU-accelerated IMDB validator achieving 98.1% |

### Key Technical Decisions

1. **No fuzzy matching** - Exact normalized title joins only for precision
2. **Suffix stripping** - Remove (Year), (Language), (Country) patterns
3. **Vote-based disambiguation** - Prefer higher-voted IMDB entries when duplicates exist
4. **Curated sources priority** - BFD and IMDB Seasons Allocated take precedence over raw IMDB
5. **GPU memory management** - Free GPU memory immediately after each file to prevent WSL2 disconnects

### Data Sources Used

| Source | Records | Purpose |
|--------|---------|---------|
| title.basics.tsv.gz | 1,626,199 | Primary/original titles |
| title.ratings.tsv.gz | 1,619,764 | Vote counts for disambiguation |
| IMDB Seasons Allocated | 223,466 | Curated TV with seasons |
| BFD Parquet | 559,858 | Already-matched reference |

### Per-File Results

| File | Coverage |
|------|----------|
| netflix_films_h1_2023 | 97.5% |
| netflix_films_h2_2023 | 99.6% |
| netflix_films_h1_2024 | 98.7% |
| netflix_films_h2_2024 | 99.1% |
| netflix_films_h1_2025 | 98.7% |
| netflix_films_h2_2025 | 98.9% |
| netflix_tv_h1_2023 | 98.7% |
| netflix_tv_h2_2023 | 97.5% |
| netflix_tv_h1_2024 | 97.5% |
| netflix_tv_h2_2024 | 96.9% |
| netflix_tv_h1_2025 | 96.5% |
| netflix_tv_h2_2025 | 97.2% |

### Remaining Gaps (1,844 records without IMDB IDs)

- Likely regional/international content not in IMDB
- Very new releases not yet indexed
- Non-standard title formats requiring manual lookup

## Session: 2026-02-01

### Research: Netflix Views + IMDb Metadata Integration

**Objective**: Determine optimal approach to integrate IMDb metadata (ratings, genres, runtime) into the Netflix Views Database.

### Files Analyzed

| File | Rows | IMDb ID Coverage |
|------|------|------------------|
| `imdb_metadata.parquet` | 10,717,142 | 100% (source) |
| `netflix_views_database.parquet` | 98,546 | 0% (needs integration) |
| `netflix_stage2_*.parquet` | 83,291 | 76% (63,309 matched) |

### IMDb Metadata Schema (19 columns)

- **Identification**: imdb_id, parent_imdb_id
- **Titles**: title, original_title, title_type, show_title, episode_title
- **Episodes**: season_number, episode_number
- **Dates**: start_year, end_year, show_start_year, show_end_year
- **Content**: runtime_minutes, genres, show_genres, is_adult
- **Ratings**: imdb_score, imdb_vote_count

**File Integrity**: SHA-256 `1bf8612adf8317903247833fd2e08cf297d709fbd32fe0928c8b55756c192624`

### Netflix Views Database Structure

- 98,546 rows (56,465 films + 42,081 TV seasons)
- 119 columns (mostly time-series viewing data)
- `fc_uid` column exists but is empty (0% populated)
- No IMDb IDs currently

### IMDb Data Structure for TV

| title_type | Count | Structure |
|------------|-------|-----------|
| tv_show | 416,395 | One row per show (parent) |
| tv_episode | 9,411,239 | One row per episode |
| film | 889,508 | One row per film |

No season-level rows exist; episodes link to parent via `parent_imdb_id`.

### Integration Path Analysis

| Approach | Title Coverage |
|----------|----------------|
| Stage 2 exact match only | 27.8% |
| Direct IMDb match (base title parsing) | 75.3% |
| **Combined 3-tier approach** | **78.7%** |

### Recommended Integration Strategy

**3-Tier Matching:**
1. **Tier 1**: Stage 2 exact match (highest confidence, already validated)
2. **Tier 2**: Base title + year + type match against IMDb
   - Parse "Show: Season 1" to extract "Show"
   - Match against tv_show records (not episodes)
3. **Tier 3**: Fuzzy match for remaining ~21%

**Target Output Schema:**
```
Identification:  imdb_id (78%+ populated)
Netflix Core:    title, title_type, season_number, premiere_date
IMDb Enrichment: imdb_score, imdb_vote_count, genres, runtime_minutes
Views Data:      hours_* and views_* (119 time columns)
```

### Pending Work

- [ ] Build 3-tier IMDb ID lookup script
- [ ] Join IMDb metadata to Views Database
- [ ] Output enriched parquet file
- [ ] Handle remaining ~21% unmatched titles (likely regional content)
