"""Apply fc_uid changes to Netflix files - 14 Workers"""
import pandas as pd
import re
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

NUM_WORKERS = 14

CHANGE_FILES = [
    r'C:\Users\RoyT6\Downloads\netflix_2.20_movies_CHANGE_THESE_fc_uid.xlsx',
    r'C:\Users\RoyT6\Downloads\netflix_2.20_movies_CHANGE_THESE_fc_uid  V2.xlsx',
]

TARGET_FILES = [
    r'C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30\netflix_tv_h1_2023_v2.30.xlsx',
    r'C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30\netflix_films_h2_2025_v2.30.xlsx',
    r'C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30\netflix_films_h2_2024_v2.30.xlsx',
    r'C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30\netflix_films_h2_2023_v2.30.xlsx',
    r'C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30\netflix_films_h1_2025_v2.30.xlsx',
    r'C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30\netflix_films_h1_2024_v2.30.xlsx',
    r'C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30\netflix_films_h1_2023_v2.30.xlsx',
    r'C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30\netflix_tv_h2_2025_v2.30.xlsx',
    r'C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30\netflix_tv_h2_2024_v2.30.xlsx',
    r'C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30\netflix_tv_h2_2023_v2.30.xlsx',
    r'C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30\netflix_tv_h1_2025_v2.30.xlsx',
    r'C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30\netflix_tv_h1_2024_v2.30.xlsx',
]

def norm(t):
    if not t or pd.isna(t): return ''
    return re.sub(r'[^\w]', '', str(t).lower())

def process_file(args):
    filepath, title_to_fcuid = args
    fname = Path(filepath).name
    if not Path(filepath).exists():
        return fname, 0, "not found"

    df = pd.read_excel(filepath)
    if 'title' not in df.columns or 'fc_uid' not in df.columns:
        return fname, 0, "missing columns"

    updated = 0
    for idx, row in df.iterrows():
        title = row.get('title', '')
        nt = norm(title)
        if nt in title_to_fcuid:
            new_fcuid = title_to_fcuid[nt]
            old_fcuid = row.get('fc_uid')
            if pd.isna(old_fcuid) or str(old_fcuid) != str(new_fcuid):
                df.at[idx, 'fc_uid'] = new_fcuid
                updated += 1

    df.to_excel(filepath, index=False)
    return fname, updated, "ok"

def main():
    print('='*60)
    print(f'Apply fc_uid Changes - {NUM_WORKERS} Workers')
    print('='*60, flush=True)

    start = time.time()

    # Load change files
    print('\nLoading CHANGE_THESE files...', flush=True)
    title_to_fcuid = {}
    for cf in CHANGE_FILES:
        df = pd.read_excel(cf)
        for _, row in df.iterrows():
            title = row.get('title', '')
            fcuid = row.get('fc_uid')
            nt = norm(title)
            if nt and fcuid and pd.notna(fcuid):
                title_to_fcuid[nt] = fcuid
    print(f'Loaded {len(title_to_fcuid)} title->fc_uid mappings', flush=True)

    # Process files in parallel
    print(f'\nProcessing {len(TARGET_FILES)} files with {NUM_WORKERS} workers...', flush=True)
    args = [(f, title_to_fcuid) for f in TARGET_FILES]

    total_updated = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=multiprocessing.get_context('spawn')) as ex:
        results = list(ex.map(process_file, args))

    print(f'\nResults ({time.time()-start:.1f}s):')
    for fname, updated, status in results:
        print(f'  {fname}: {updated} updated ({status})')
        total_updated += updated

    print(f'\nTOTAL: {total_updated} fc_uid values updated')
    print('Done!')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
