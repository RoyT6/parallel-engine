"""
Views ID Merge Engine v2
Fixes imdb_id format normalization issue.
Properly handles tt0158429 (views) vs tt158429 (checkpoints) format differences.
"""

import pandas as pd
import numpy as np
import re
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings('ignore')

# Import system capability engine
import sys
sys.path.insert(0, str(Path(__file__).parent))
from system_capability_engine.quick_config import CPU_WORKERS, IO_WORKERS

# =============================================================================
# CONFIGURATION
# =============================================================================

VIEWS_FILE = r"C:\Users\RoyT6\Downloads\BFD_VIEWS_V27.75.parquet"
SOURCES_DIR = Path(r"C:\Users\RoyT6\Downloads\Freckles\00_ACTIVE_SOURCES")

TMDB_MOVIES = SOURCES_DIR / "TMDB" / "TMDB_MOVIES_GDHD.csv"
TMDB_CHECKPOINTS_DIR = SOURCES_DIR / "TMDB" / "checkpoints_v1.5"

OUTPUT_FILE = r"C:\Users\RoyT6\Downloads\BFD_VIEWS_V27.75_ID_MERGED.parquet"

# =============================================================================
# IMDB ID NORMALIZATION - KEY FIX
# =============================================================================

def normalize_imdb_id(imdb_id) -> str:
    """
    Normalize imdb_id to consistent format: tt + 7 digit number with leading zeros.
    Handles: tt0158429, tt158429, 158429, 0158429
    Returns: tt0158429 format (always 9 chars) or None if invalid
    """
    if pd.isna(imdb_id) or imdb_id == 'None' or imdb_id == '':
        return None

    imdb_str = str(imdb_id).strip()

    # Extract numeric part
    match = re.search(r'tt?(\d+)', imdb_str)
    if match:
        num = match.group(1)
        # Pad to 7 digits
        num_padded = num.zfill(7)
        return f'tt{num_padded}'

    # Try pure numeric
    if imdb_str.isdigit():
        return f'tt{imdb_str.zfill(7)}'

    return None

def clean_tmdb_id(tmdb_id) -> int:
    """Clean and validate tmdb_id"""
    if pd.isna(tmdb_id):
        return None
    try:
        val = int(float(tmdb_id))
        return val if val > 0 else None
    except (ValueError, TypeError):
        return None

# =============================================================================
# DATA LOADING
# =============================================================================

def load_views_file() -> pd.DataFrame:
    """Load the main views parquet file"""
    print("Loading views file...")
    start = time.time()
    df = pd.read_parquet(VIEWS_FILE)
    print(f"  Loaded {len(df):,} rows in {time.time()-start:.1f}s")
    return df

def load_tmdb_movies() -> pd.DataFrame:
    """Load TMDB Movies with normalized imdb_id"""
    print("Loading TMDB Movies GDHD...")
    start = time.time()
    df = pd.read_csv(TMDB_MOVIES, usecols=['tmdb_id', 'imdb_id'], low_memory=False)

    # Normalize imdb_id
    df['imdb_id_norm'] = df['imdb_id'].apply(normalize_imdb_id)
    df['tmdb_id'] = df['tmdb_id'].apply(clean_tmdb_id)

    # Keep only valid mappings
    df = df[df['imdb_id_norm'].notna() & df['tmdb_id'].notna()]
    df = df[['imdb_id_norm', 'tmdb_id']].rename(columns={'imdb_id_norm': 'imdb_id'})
    df = df.drop_duplicates(subset=['imdb_id'], keep='first')

    print(f"  Loaded {len(df):,} valid mappings in {time.time()-start:.1f}s")
    return df

def load_single_checkpoint(filepath: Path) -> pd.DataFrame:
    """Load a single TMDB checkpoint file with normalized imdb_id"""
    try:
        df = pd.read_parquet(filepath, columns=['imdb_id', 'tmdb_id'])

        # Normalize imdb_id
        df['imdb_id_norm'] = df['imdb_id'].apply(normalize_imdb_id)
        df['tmdb_id'] = df['tmdb_id'].apply(clean_tmdb_id)

        # Keep only valid mappings
        df = df[df['imdb_id_norm'].notna() & df['tmdb_id'].notna()]
        return df[['imdb_id_norm', 'tmdb_id']].rename(columns={'imdb_id_norm': 'imdb_id'})
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return pd.DataFrame(columns=['imdb_id', 'tmdb_id'])

def load_tmdb_checkpoints() -> pd.DataFrame:
    """Load ALL TMDB checkpoints with normalized imdb_id"""
    print("Loading ALL TMDB Checkpoints v1.5...")
    start = time.time()

    checkpoint_files = sorted(TMDB_CHECKPOINTS_DIR.glob("checkpoint_*.parquet"))
    print(f"  Found {len(checkpoint_files)} checkpoint files")

    all_dfs = []
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as executor:
        futures = {executor.submit(load_single_checkpoint, f): f for f in checkpoint_files}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            df = future.result()
            if len(df) > 0:
                all_dfs.append(df)
            if completed % 10 == 0 or completed == len(checkpoint_files):
                print(f"    Processed {completed}/{len(checkpoint_files)} files...")

    if all_dfs:
        print("  Concatenating all checkpoints...")
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"  Total rows before dedup: {len(combined):,}")

        # Deduplicate - keep first (arbitrary but consistent)
        combined = combined.drop_duplicates(subset=['imdb_id'], keep='first')
        print(f"  Unique imdb->tmdb mappings: {len(combined):,}")
        print(f"  Completed in {time.time()-start:.1f}s")
        return combined

    return pd.DataFrame(columns=['imdb_id', 'tmdb_id'])

# =============================================================================
# MERGE PROCESS
# =============================================================================

def merge_tmdb_ids(views_df: pd.DataFrame, tmdb_movies: pd.DataFrame,
                   tmdb_checkpoints: pd.DataFrame) -> pd.DataFrame:
    """Merge tmdb_id into views using normalized imdb_id matching"""
    print("\n" + "="*70)
    print("MERGING tmdb_id")
    print("="*70 + "\n")

    start = time.time()

    # Count initial state
    tmdb_before = views_df['tmdb_id'].notna().sum()
    print(f"tmdb_id populated before: {tmdb_before:,}")

    # Normalize imdb_id in views
    print("\nNormalizing imdb_id in views file...")
    views_df['imdb_id_norm'] = views_df['imdb_id'].apply(normalize_imdb_id)
    norm_count = views_df['imdb_id_norm'].notna().sum()
    print(f"  Normalized {norm_count:,} imdb_ids")

    # Combine all TMDB sources
    print("\nCombining TMDB sources...")
    all_tmdb = pd.concat([tmdb_movies, tmdb_checkpoints], ignore_index=True)
    all_tmdb = all_tmdb.drop_duplicates(subset=['imdb_id'], keep='first')
    print(f"  Combined lookup table: {len(all_tmdb):,} imdb->tmdb mappings")

    # Create lookup dict
    imdb_to_tmdb = dict(zip(all_tmdb['imdb_id'], all_tmdb['tmdb_id']))

    # Find rows needing tmdb_id
    needs_tmdb = views_df['tmdb_id'].isna()
    has_imdb = views_df['imdb_id_norm'].notna()
    mask = needs_tmdb & has_imdb

    print(f"\nRows needing tmdb_id with valid imdb_id: {mask.sum():,}")

    # Apply lookup
    print("Applying lookup...")
    lookups = views_df.loc[mask, 'imdb_id_norm'].map(imdb_to_tmdb)
    found = lookups.notna().sum()
    print(f"  Matches found: {found:,}")

    # Update
    views_df.loc[mask & lookups.notna(), 'tmdb_id'] = lookups[lookups.notna()]

    # Clean up temp column
    views_df = views_df.drop(columns=['imdb_id_norm'])

    tmdb_after = views_df['tmdb_id'].notna().sum()
    print(f"\ntmdb_id populated after: {tmdb_after:,}")
    print(f"tmdb_id filled: {tmdb_after - tmdb_before:,}")
    print(f"Merge completed in {time.time()-start:.1f}s")

    return views_df

def generate_imdb_urls(df: pd.DataFrame) -> pd.DataFrame:
    """Generate imdb_url from imdb_id where missing"""
    print("\n" + "="*70)
    print("GENERATING imdb_url")
    print("="*70 + "\n")

    # Clean 'None' strings first
    none_count = (df['imdb_url'] == 'None').sum()
    if none_count > 0:
        df.loc[df['imdb_url'] == 'None', 'imdb_url'] = None
        print(f"Cleaned {none_count:,} 'None' strings from imdb_url")

    # Generate URLs
    mask = df['imdb_url'].isna() & df['imdb_id'].notna()
    df.loc[mask, 'imdb_url'] = 'https://www.imdb.com/title/' + df.loc[mask, 'imdb_id'].astype(str) + '/'
    print(f"Generated {mask.sum():,} imdb_urls")

    return df

def clean_none_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Clean 'None' strings from key columns"""
    print("\n" + "="*70)
    print("CLEANING DATA")
    print("="*70 + "\n")

    for col in ['imdb_id', 'original_title', 'row_hash', 'tagline']:
        if col in df.columns:
            none_count = (df[col] == 'None').sum()
            if none_count > 0:
                df.loc[df[col] == 'None', col] = None
                print(f"  {col}: cleaned {none_count:,} 'None' strings")

    # Ensure tmdb_id is numeric
    df['tmdb_id'] = pd.to_numeric(df['tmdb_id'], errors='coerce')

    return df

def print_final_report(df: pd.DataFrame):
    """Print final integrity report"""
    print("\n" + "="*70)
    print("FINAL INTEGRITY REPORT")
    print("="*70 + "\n")

    total = len(df)
    required = [
        'title', 'original_title', 'title_type', 'fc_uid', 'row_hash',
        'imdb_id', 'tmdb_id', 'imdb_url', 'status', 'overview',
        'tagline', 'max_seasons', 'season_number'
    ]

    print(f"Total rows: {total:,}\n")
    print("-" * 70)

    for col in required:
        if col in df.columns:
            null_count = df[col].isna().sum()
            none_count = (df[col] == 'None').sum() if df[col].dtype == 'object' else 0
            missing = null_count + none_count
            pct = ((total - missing) / total) * 100
            status = "[OK]" if pct >= 99.0 else "[--]" if pct >= 90.0 else "[!!]"
            print(f"  {status} {col:20} | {total - missing:>10,} / {total:,} ({pct:>6.2f}%)")

    # TV validation
    print("\n" + "-" * 70)
    tv_types = ['tv_season', 'tvSeries', 'episode']
    tv_df = df[df['title_type'].isin(tv_types)]
    tv_with_season = tv_df['season_number'].notna().sum()
    print(f"TV Content: {tv_with_season:,} / {len(tv_df):,} have season_number ({tv_with_season/len(tv_df)*100:.2f}%)")

    # tmdb_id by type
    print("\n" + "-" * 70)
    print("tmdb_id coverage by title_type:")
    for ttype in df['title_type'].unique():
        subset = df[df['title_type'] == ttype]
        has_tmdb = subset['tmdb_id'].notna().sum()
        pct = (has_tmdb / len(subset)) * 100
        print(f"  {ttype:12} | {has_tmdb:>7,} / {len(subset):>7,} ({pct:>5.1f}%)")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  VIEWS ID MERGE ENGINE v2 - FULL DATA LOAD")
    print("  Parallel Processing: {} CPU workers, {} I/O workers".format(CPU_WORKERS, IO_WORKERS))
    print("="*70)

    total_start = time.time()

    # Load data
    print("\n" + "="*70)
    print("LOADING ALL DATA SOURCES")
    print("="*70 + "\n")

    views_df = load_views_file()
    tmdb_movies = load_tmdb_movies()
    tmdb_checkpoints = load_tmdb_checkpoints()

    # Merge
    views_df = merge_tmdb_ids(views_df, tmdb_movies, tmdb_checkpoints)

    # Generate URLs
    views_df = generate_imdb_urls(views_df)

    # Clean
    views_df = clean_none_strings(views_df)

    # Report
    print_final_report(views_df)

    # Save
    print("\n" + "="*70)
    print("SAVING OUTPUT")
    print("="*70 + "\n")

    print(f"Writing to: {OUTPUT_FILE}")
    start = time.time()
    views_df.to_parquet(OUTPUT_FILE, index=False, compression='snappy')
    print(f"  Saved in {time.time()-start:.1f}s")

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  MERGE COMPLETE - Total time: {total_time:.1f}s")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
