"""
Views ID Merge Engine
Fills missing tmdb_id and imdb_id in BFD_VIEWS using parallel processing.
Optimized for: Ryzen 9 3950X (16c/32t), 128GB DDR4, RTX 3080Ti
"""

import pandas as pd
import numpy as np
import re
import time
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import multiprocessing

warnings.filterwarnings('ignore')

# Import system capability engine
import sys
sys.path.insert(0, str(Path(__file__).parent))
from system_capability_engine.quick_config import (
    CPU_WORKERS, IO_WORKERS, PIPELINE_BATCH_SIZE, CPU_CHUNK_SIZE
)

# =============================================================================
# CONFIGURATION
# =============================================================================

VIEWS_FILE = r"C:\Users\RoyT6\Downloads\BFD_VIEWS_V27.75.parquet"
SOURCES_DIR = Path(r"C:\Users\RoyT6\Downloads\Freckles\00_ACTIVE_SOURCES")

# Source files
IMDB_SEASONS = SOURCES_DIR / "IMDB" / "IMDB Seasons Allocated.csv"
IMDB_BASICS = SOURCES_DIR / "IMDB" / "title_basics_filtered.parquet"
TMDB_MOVIES = SOURCES_DIR / "TMDB" / "TMDB_MOVIES_GDHD.csv"
TMDB_TV = SOURCES_DIR / "TMDB" / "TMDB_TV_22_MAY24_GDHD.csv"
TMDB_CHECKPOINTS_DIR = SOURCES_DIR / "TMDB" / "checkpoints_v1.5"

# Output
OUTPUT_FILE = r"C:\Users\RoyT6\Downloads\BFD_VIEWS_V27.75_ID_MERGED.parquet"

# Required columns to validate
REQUIRED_COLS = [
    'title', 'original_title', 'title_type', 'fc_uid', 'row_hash',
    'imdb_id', 'tmdb_id', 'imdb_url', 'status', 'overview',
    'tagline', 'max_seasons', 'season_number'
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_imdb_from_fc_uid(fc_uid: str) -> Optional[str]:
    """Extract imdb_id from fc_uid if it follows tt* pattern"""
    if pd.isna(fc_uid):
        return None
    fc_uid = str(fc_uid)
    # Pattern: tt1234567 or tt1234567_s1
    match = re.match(r'^(tt\d+)', fc_uid)
    if match:
        return match.group(1)
    return None

def clean_imdb_id(imdb_id) -> Optional[str]:
    """Clean and validate imdb_id"""
    if pd.isna(imdb_id) or imdb_id == 'None' or imdb_id == '':
        return None
    imdb_id = str(imdb_id).strip()
    if imdb_id.startswith('tt') and len(imdb_id) >= 9:
        return imdb_id
    # Try to fix malformed IDs
    if imdb_id.isdigit():
        return f"tt{imdb_id}"
    return None

def clean_tmdb_id(tmdb_id) -> Optional[int]:
    """Clean and validate tmdb_id"""
    if pd.isna(tmdb_id):
        return None
    try:
        val = int(float(tmdb_id))
        return val if val > 0 else None
    except (ValueError, TypeError):
        return None

def print_progress(phase: str, current: int, total: int, extra: str = ""):
    """Print progress bar"""
    pct = (current / total) * 100 if total > 0 else 0
    bar_len = 40
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = '#' * filled + '-' * (bar_len - filled)
    print(f"\r  {phase}: [{bar}] {pct:5.1f}% ({current:,}/{total:,}) {extra}", end='', flush=True)

# =============================================================================
# DATA LOADING (PARALLEL)
# =============================================================================

def load_views_file() -> pd.DataFrame:
    """Load the main views parquet file"""
    print("Loading views file...")
    start = time.time()
    df = pd.read_parquet(VIEWS_FILE)
    print(f"  Loaded {len(df):,} rows in {time.time()-start:.1f}s")
    return df

def load_imdb_seasons() -> pd.DataFrame:
    """Load IMDB Seasons Allocated"""
    print("Loading IMDB Seasons Allocated...")
    start = time.time()
    df = pd.read_csv(IMDB_SEASONS, usecols=['fc_uid', 'imdb_id', 'imdb_url', 'original_title'])
    df['imdb_id'] = df['imdb_id'].apply(clean_imdb_id)
    print(f"  Loaded {len(df):,} rows in {time.time()-start:.1f}s")
    return df

def load_imdb_basics() -> pd.DataFrame:
    """Load IMDB title basics"""
    print("Loading IMDB title_basics_filtered...")
    start = time.time()
    df = pd.read_parquet(IMDB_BASICS)
    df = df.rename(columns={'tconst': 'imdb_id', 'primaryTitle': 'title_imdb'})
    print(f"  Loaded {len(df):,} rows in {time.time()-start:.1f}s")
    return df

def load_tmdb_movies() -> pd.DataFrame:
    """Load TMDB Movies"""
    print("Loading TMDB Movies GDHD...")
    start = time.time()
    df = pd.read_csv(TMDB_MOVIES, usecols=['tmdb_id', 'imdb_id'])
    df['imdb_id'] = df['imdb_id'].apply(clean_imdb_id)
    df['tmdb_id'] = df['tmdb_id'].apply(clean_tmdb_id)
    df = df.dropna(subset=['imdb_id', 'tmdb_id'])
    print(f"  Loaded {len(df):,} valid mappings in {time.time()-start:.1f}s")
    return df

def load_tmdb_tv() -> pd.DataFrame:
    """Load TMDB TV"""
    print("Loading TMDB TV GDHD...")
    start = time.time()
    df = pd.read_csv(TMDB_TV, usecols=['tmdb_id', 'imdb_id'])
    df['imdb_id'] = df['imdb_id'].apply(clean_imdb_id)
    df['tmdb_id'] = df['tmdb_id'].apply(clean_tmdb_id)
    df = df.dropna(subset=['imdb_id', 'tmdb_id'])
    print(f"  Loaded {len(df):,} valid mappings in {time.time()-start:.1f}s")
    return df

def load_single_checkpoint(filepath: Path) -> pd.DataFrame:
    """Load a single TMDB checkpoint file"""
    try:
        df = pd.read_parquet(filepath, columns=['tmdb_id', 'imdb_id'])
        df['imdb_id'] = df['imdb_id'].apply(clean_imdb_id)
        df['tmdb_id'] = df['tmdb_id'].apply(clean_tmdb_id)
        return df.dropna(subset=['imdb_id', 'tmdb_id'])
    except Exception as e:
        return pd.DataFrame(columns=['tmdb_id', 'imdb_id'])

def load_tmdb_checkpoints() -> pd.DataFrame:
    """Load all TMDB checkpoints in parallel"""
    print("Loading TMDB Checkpoints v1.5...")
    start = time.time()

    checkpoint_files = sorted(TMDB_CHECKPOINTS_DIR.glob("checkpoint_*.parquet"))
    print(f"  Found {len(checkpoint_files)} checkpoint files")

    dfs = []
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as executor:
        futures = {executor.submit(load_single_checkpoint, f): f for f in checkpoint_files}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            df = future.result()
            if len(df) > 0:
                dfs.append(df)
            print_progress("Loading checkpoints", completed, len(checkpoint_files))

    print()  # newline after progress

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        # Deduplicate, keeping first occurrence
        combined = combined.drop_duplicates(subset=['imdb_id'], keep='first')
        print(f"  Combined {len(combined):,} unique imdb->tmdb mappings in {time.time()-start:.1f}s")
        return combined
    return pd.DataFrame(columns=['tmdb_id', 'imdb_id'])

def load_all_sources_parallel() -> Dict[str, pd.DataFrame]:
    """Load all source files in parallel"""
    print("\n" + "="*70)
    print("PHASE 0: LOADING ALL DATA SOURCES (PARALLEL)")
    print("="*70 + "\n")

    start = time.time()
    sources = {}

    # Use thread pool for I/O bound loading
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(load_views_file): 'views',
            executor.submit(load_imdb_seasons): 'imdb_seasons',
            executor.submit(load_imdb_basics): 'imdb_basics',
            executor.submit(load_tmdb_movies): 'tmdb_movies',
            executor.submit(load_tmdb_tv): 'tmdb_tv',
        }

        for future in as_completed(futures):
            name = futures[future]
            sources[name] = future.result()

    # Load checkpoints separately (they use their own parallelism)
    sources['tmdb_checkpoints'] = load_tmdb_checkpoints()

    print(f"\nAll sources loaded in {time.time()-start:.1f}s")
    return sources

# =============================================================================
# MERGE PHASES
# =============================================================================

def phase1_fc_uid_match(df: pd.DataFrame, imdb_seasons: pd.DataFrame) -> pd.DataFrame:
    """Phase 1: Direct fc_uid match from IMDB Seasons Allocated"""
    print("\n" + "="*70)
    print("PHASE 1: DIRECT fc_uid MATCH (IMDB Seasons Allocated)")
    print("="*70 + "\n")

    start = time.time()

    # Count missing before
    imdb_missing_before = df['imdb_id'].isna().sum() + (df['imdb_id'] == 'None').sum()

    # Create lookup dict for speed
    fc_uid_to_imdb = dict(zip(imdb_seasons['fc_uid'], imdb_seasons['imdb_id']))
    fc_uid_to_url = dict(zip(imdb_seasons['fc_uid'], imdb_seasons['imdb_url']))
    fc_uid_to_orig = dict(zip(imdb_seasons['fc_uid'], imdb_seasons['original_title']))

    print(f"  Lookup table size: {len(fc_uid_to_imdb):,} entries")

    # Find rows needing update
    needs_imdb = (df['imdb_id'].isna()) | (df['imdb_id'] == 'None')
    mask = needs_imdb & df['fc_uid'].isin(fc_uid_to_imdb)

    # Apply updates
    matched_fc_uids = df.loc[mask, 'fc_uid']
    df.loc[mask, 'imdb_id'] = matched_fc_uids.map(fc_uid_to_imdb)

    # Also update imdb_url if missing
    needs_url = df['imdb_url'].isna()
    url_mask = needs_url & df['fc_uid'].isin(fc_uid_to_url)
    df.loc[url_mask, 'imdb_url'] = df.loc[url_mask, 'fc_uid'].map(fc_uid_to_url)

    # Update original_title if missing
    needs_orig = df['original_title'].isna()
    orig_mask = needs_orig & df['fc_uid'].isin(fc_uid_to_orig)
    df.loc[orig_mask, 'original_title'] = df.loc[orig_mask, 'fc_uid'].map(fc_uid_to_orig)

    imdb_missing_after = df['imdb_id'].isna().sum() + (df['imdb_id'] == 'None').sum()
    filled = imdb_missing_before - imdb_missing_after

    print(f"  imdb_id filled: {filled:,}")
    print(f"  imdb_url updated: {url_mask.sum():,}")
    print(f"  original_title updated: {orig_mask.sum():,}")
    print(f"  Phase 1 completed in {time.time()-start:.1f}s")

    return df

def phase2_extract_imdb_from_fc_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2: Extract imdb_id from fc_uid for films"""
    print("\n" + "="*70)
    print("PHASE 2: EXTRACT imdb_id FROM fc_uid")
    print("="*70 + "\n")

    start = time.time()

    # Count missing before
    imdb_missing_before = df['imdb_id'].isna().sum() + (df['imdb_id'] == 'None').sum()

    # Find rows needing extraction
    needs_imdb = (df['imdb_id'].isna()) | (df['imdb_id'] == 'None')

    # Apply extraction using vectorized operation
    extracted = df['fc_uid'].apply(extract_imdb_from_fc_uid)

    # Update where needed and extraction succeeded
    mask = needs_imdb & extracted.notna()
    df.loc[mask, 'imdb_id'] = extracted[mask]

    # Generate imdb_url for newly filled imdb_ids
    url_mask = mask & df['imdb_url'].isna()
    df.loc[url_mask, 'imdb_url'] = 'https://www.imdb.com/title/' + df.loc[url_mask, 'imdb_id'] + '/'

    imdb_missing_after = df['imdb_id'].isna().sum() + (df['imdb_id'] == 'None').sum()
    filled = imdb_missing_before - imdb_missing_after

    print(f"  imdb_id extracted: {filled:,}")
    print(f"  imdb_url generated: {url_mask.sum():,}")
    print(f"  Phase 2 completed in {time.time()-start:.1f}s")

    return df

def phase3_tmdb_enrichment(df: pd.DataFrame, tmdb_movies: pd.DataFrame,
                           tmdb_tv: pd.DataFrame, tmdb_checkpoints: pd.DataFrame) -> pd.DataFrame:
    """Phase 3: Enrich tmdb_id from TMDB sources"""
    print("\n" + "="*70)
    print("PHASE 3: tmdb_id ENRICHMENT FROM TMDB SOURCES")
    print("="*70 + "\n")

    start = time.time()

    # Count missing before
    tmdb_missing_before = df['tmdb_id'].isna().sum()

    # Combine all TMDB sources (checkpoints have priority - most complete)
    print("  Building combined imdb->tmdb lookup...")

    # Start with movies and TV
    combined = pd.concat([tmdb_movies, tmdb_tv], ignore_index=True)
    combined = combined.drop_duplicates(subset=['imdb_id'], keep='first')

    # Add checkpoints (these override if there's a conflict since they're more recent)
    all_tmdb = pd.concat([combined, tmdb_checkpoints], ignore_index=True)
    all_tmdb = all_tmdb.drop_duplicates(subset=['imdb_id'], keep='last')  # checkpoints win

    # Create lookup dict
    imdb_to_tmdb = dict(zip(all_tmdb['imdb_id'], all_tmdb['tmdb_id']))
    print(f"  Combined lookup: {len(imdb_to_tmdb):,} imdb->tmdb mappings")

    # Clean the imdb_id column in views for matching
    df['imdb_id_clean'] = df['imdb_id'].apply(clean_imdb_id)

    # Find rows needing tmdb_id
    needs_tmdb = df['tmdb_id'].isna()
    has_imdb = df['imdb_id_clean'].notna()

    # Apply lookup
    mask = needs_tmdb & has_imdb
    lookups = df.loc[mask, 'imdb_id_clean'].map(imdb_to_tmdb)

    # Update where lookup succeeded
    found_mask = mask & lookups.notna()
    df.loc[found_mask, 'tmdb_id'] = lookups[found_mask]

    # Clean up temp column
    df = df.drop(columns=['imdb_id_clean'])

    tmdb_missing_after = df['tmdb_id'].isna().sum()
    filled = tmdb_missing_before - tmdb_missing_after

    print(f"  tmdb_id filled: {filled:,}")
    print(f"  tmdb_id still missing: {tmdb_missing_after:,}")
    print(f"  Phase 3 completed in {time.time()-start:.1f}s")

    return df

def phase4_validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 4: Final validation and cleanup"""
    print("\n" + "="*70)
    print("PHASE 4: VALIDATION AND CLEANUP")
    print("="*70 + "\n")

    start = time.time()

    # Clean up 'None' strings
    print("  Cleaning 'None' strings...")
    for col in ['imdb_id', 'tmdb_id', 'imdb_url', 'original_title', 'row_hash']:
        if col in df.columns:
            none_count = (df[col] == 'None').sum()
            if none_count > 0:
                df.loc[df[col] == 'None', col] = None
                print(f"    {col}: converted {none_count:,} 'None' strings to null")

    # Ensure tmdb_id is numeric
    print("  Converting tmdb_id to numeric...")
    df['tmdb_id'] = pd.to_numeric(df['tmdb_id'], errors='coerce')

    # Validate TV shows have season_number
    print("  Validating TV content...")
    tv_types = ['tv_season', 'tvSeries', 'episode']
    tv_mask = df['title_type'].isin(tv_types)
    tv_no_season = tv_mask & df['season_number'].isna()
    if tv_no_season.sum() > 0:
        print(f"    WARNING: {tv_no_season.sum():,} TV rows missing season_number")
    else:
        print(f"    All {tv_mask.sum():,} TV rows have season_number")

    print(f"  Phase 4 completed in {time.time()-start:.1f}s")

    return df

# =============================================================================
# FINAL REPORT
# =============================================================================

def print_final_report(df: pd.DataFrame):
    """Print final integrity report"""
    print("\n" + "="*70)
    print("FINAL INTEGRITY REPORT")
    print("="*70 + "\n")

    total = len(df)

    print(f"Total rows: {total:,}\n")
    print("Required Columns Status:")
    print("-" * 60)

    for col in REQUIRED_COLS:
        if col in df.columns:
            null_count = df[col].isna().sum()
            none_count = (df[col] == 'None').sum() if df[col].dtype == 'object' else 0
            missing = null_count + none_count
            pct = ((total - missing) / total) * 100
            status = "[OK]" if pct >= 99.0 else "[--]" if pct >= 90.0 else "[!!]"
            print(f"  {status} {col:20} | {total - missing:>10,} / {total:,} ({pct:>6.2f}%)")
        else:
            print(f"  [!!] {col:20} | MISSING COLUMN!")

    # TV season check
    print("\n" + "-" * 60)
    tv_types = ['tv_season', 'tvSeries', 'episode']
    tv_df = df[df['title_type'].isin(tv_types)]
    tv_with_season = tv_df['season_number'].notna().sum()
    print(f"TV Content: {tv_with_season:,} / {len(tv_df):,} have season_number ({(tv_with_season/len(tv_df)*100):.2f}%)")

    # Unique key check
    print(f"\nfc_uid uniqueness: {df['fc_uid'].nunique():,} / {total:,}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  VIEWS ID MERGE ENGINE")
    print("  Parallel Processing: {} CPU workers, {} I/O workers".format(CPU_WORKERS, IO_WORKERS))
    print("="*70)

    total_start = time.time()

    # Load all sources
    sources = load_all_sources_parallel()

    # Get main dataframe
    df = sources['views']

    # Execute merge phases
    df = phase1_fc_uid_match(df, sources['imdb_seasons'])
    df = phase2_extract_imdb_from_fc_uid(df)
    df = phase3_tmdb_enrichment(df, sources['tmdb_movies'], sources['tmdb_tv'], sources['tmdb_checkpoints'])
    df = phase4_validate_and_clean(df)

    # Final report
    print_final_report(df)

    # Save output
    print("\n" + "="*70)
    print("SAVING OUTPUT")
    print("="*70 + "\n")

    print(f"Writing to: {OUTPUT_FILE}")
    start = time.time()
    df.to_parquet(OUTPUT_FILE, index=False, compression='snappy')
    print(f"  Saved in {time.time()-start:.1f}s")

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  MERGE COMPLETE - Total time: {total_time:.1f}s")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
