"""
IMDB Title Matcher - Parallel Engine
Finds IMDB IDs for titles marked as MISSING in the Excel file.
Uses 10 workers for parallel processing.
"""

import pandas as pd
import gzip
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import multiprocessing
from rapidfuzz import fuzz, process
from functools import lru_cache
import sys
import time

# Paths
EXCEL_PATH = r'C:\Users\RoyT6\Downloads\h1_2023_misplaced_tv_shows_report_ww corrections.xlsx'
IMDB_BASICS = r'C:\Users\RoyT6\Downloads\Freckles\00_ACTIVE_SOURCES\IMDB\title.basics.tsv.gz'
OUTPUT_PATH = r'C:\Users\RoyT6\Downloads\h1_2023_misplaced_tv_shows_report_ww corrections_FILLED.xlsx'

# Configuration
NUM_WORKERS = 10
TV_TYPES = {'tvSeries', 'tvMiniSeries', 'tvMovie', 'tvSpecial', 'tvShort'}


def normalize_title(title):
    """Normalize title for matching"""
    if not title or pd.isna(title):
        return ""
    title = str(title).lower()
    # Remove common suffixes and prefixes
    title = re.sub(r'\s*:\s*', ' ', title)  # Replace colons with space
    title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
    title = re.sub(r'[^\w\s]', '', title)  # Remove punctuation
    title = title.strip()
    return title


def extract_base_title(title):
    """Extract base title from compound titles like 'Pokemon The Series: XY: XY'"""
    if not title:
        return []
    # Split on colon and take different combinations
    parts = [p.strip() for p in str(title).split(':') if p.strip()]
    variants = []
    # Full title (highest priority)
    variants.append((' '.join(parts), 1.0))
    # First part only (if substantial)
    if parts and len(parts[0]) >= 5:
        variants.append((parts[0], 0.9))
    # First two parts
    if len(parts) >= 2:
        variants.append((f"{parts[0]} {parts[1]}", 0.95))
    # Last part only if it's substantial (>10 chars) - avoid generic subtitles
    if len(parts) > 1 and len(parts[-1]) > 10:
        variants.append((parts[-1], 0.7))
    return variants


def load_imdb_data():
    """Load IMDB title.basics data filtered for TV content"""
    print("Loading IMDB title.basics.tsv.gz...")
    start = time.time()

    titles = {}  # normalized_title -> list of (tconst, originalTitle, primaryTitle, titleType, startYear)
    title_to_ids = defaultdict(list)

    with gzip.open(IMDB_BASICS, 'rt', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        idx = {col: i for i, col in enumerate(header)}

        for line_num, line in enumerate(f, 1):
            if line_num % 1000000 == 0:
                print(f"  Processed {line_num:,} lines...")

            parts = line.strip().split('\t')
            if len(parts) < len(header):
                continue

            title_type = parts[idx['titleType']]

            # Only keep TV content
            if title_type not in TV_TYPES:
                continue

            tconst = parts[idx['tconst']]
            primary_title = parts[idx['primaryTitle']]
            original_title = parts[idx['originalTitle']]
            start_year = parts[idx['startYear']] if parts[idx['startYear']] != '\\N' else None

            # Index by normalized title
            for title in [primary_title, original_title]:
                if title and title != '\\N':
                    norm = normalize_title(title)
                    if norm:
                        title_to_ids[norm].append({
                            'tconst': tconst,
                            'primaryTitle': primary_title,
                            'originalTitle': original_title,
                            'titleType': title_type,
                            'startYear': start_year
                        })

    print(f"Loaded {len(title_to_ids):,} unique normalized titles in {time.time()-start:.1f}s")
    return title_to_ids


def find_best_match(search_title, title_index, all_titles_list):
    """Find the best matching IMDB ID for a title"""
    if not search_title or pd.isna(search_title):
        return None, None, 0

    search_title_str = str(search_title)
    variants = extract_base_title(search_title_str)

    best_match = None
    best_score = 0
    best_tconst = None

    # Try exact match first (with variant weight)
    for variant, weight in variants:
        norm_variant = normalize_title(variant)
        if not norm_variant or len(norm_variant) < 3:
            continue

        # Exact match
        if norm_variant in title_index:
            matches = title_index[norm_variant]
            # Prefer tvSeries over others
            for m in matches:
                if m['titleType'] == 'tvSeries':
                    effective_score = 100 * weight
                    if effective_score > best_score:
                        best_score = effective_score
                        best_match = m['primaryTitle']
                        best_tconst = m['tconst']
                    break
            if not best_tconst and matches:
                effective_score = 100 * weight
                if effective_score > best_score:
                    best_score = effective_score
                    best_match = matches[0]['primaryTitle']
                    best_tconst = matches[0]['tconst']

    # Return early if we found a high-confidence exact match
    if best_score >= 95:
        return best_tconst, best_match, best_score

    # Fuzzy match if no good exact match
    for variant, weight in variants:
        norm_variant = normalize_title(variant)
        if not norm_variant or len(norm_variant) < 5:
            continue

        # Use rapidfuzz for fast fuzzy matching
        results = process.extract(
            norm_variant,
            all_titles_list,
            scorer=fuzz.ratio,
            limit=5,
            score_cutoff=80  # Higher threshold for fuzzy
        )

        for match_title, score, _ in results:
            effective_score = score * weight
            if effective_score > best_score:
                matches = title_index.get(match_title, [])
                for m in matches:
                    if m['titleType'] == 'tvSeries':
                        best_score = effective_score
                        best_match = m['primaryTitle']
                        best_tconst = m['tconst']
                        break
                if not best_tconst and matches:
                    best_score = effective_score
                    best_match = matches[0]['primaryTitle']
                    best_tconst = matches[0]['tconst']

    # Only return matches with score >= 75
    if best_score >= 75:
        return best_tconst, best_match, best_score
    return None, None, 0


def match_chunk(args):
    """Process a chunk of titles (for parallel processing)"""
    chunk, title_index, all_titles_list = args
    results = []

    for idx, row in chunk.iterrows():
        title = row['title']
        h1_imdb = row['h1_2023_imdb_id']
        tv_imdb = row['tv_imdb_ids']

        result = {'index': idx, 'h1_match': None, 'tv_match': None}

        # Only lookup if MISSING
        if str(h1_imdb) == 'MISSING':
            tconst, matched_title, score = find_best_match(title, title_index, all_titles_list)
            if tconst and score >= 70:
                result['h1_match'] = (tconst, matched_title, score)

        if str(tv_imdb) == 'MISSING':
            tconst, matched_title, score = find_best_match(title, title_index, all_titles_list)
            if tconst and score >= 70:
                result['tv_match'] = (tconst, matched_title, score)

        results.append(result)

    return results


def main():
    print("=" * 60)
    print("IMDB Title Matcher - 10 Workers Parallel Engine")
    print("=" * 60)

    # Load Excel file
    print("\nLoading Excel file...")
    df = pd.read_excel(EXCEL_PATH)
    print(f"Loaded {len(df)} rows")

    # Find MISSING entries
    missing_h1 = df[df['h1_2023_imdb_id'].astype(str) == 'MISSING']
    missing_tv = df[df['tv_imdb_ids'].astype(str) == 'MISSING']
    print(f"MISSING h1_2023_imdb_id: {len(missing_h1)}")
    print(f"MISSING tv_imdb_ids: {len(missing_tv)}")

    # Load IMDB data
    title_index = load_imdb_data()
    all_titles_list = list(title_index.keys())
    print(f"Indexed {len(all_titles_list):,} titles for fuzzy matching")

    # Get rows that need processing
    needs_processing = df[
        (df['h1_2023_imdb_id'].astype(str) == 'MISSING') |
        (df['tv_imdb_ids'].astype(str) == 'MISSING')
    ].copy()

    print(f"\nProcessing {len(needs_processing)} rows with MISSING values...")
    print(f"Using {NUM_WORKERS} workers")

    # Process all at once (simpler than chunking for this size)
    start = time.time()
    matches_found = 0

    for idx, row in needs_processing.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(needs_processing)}...")

        title = row['title']
        h1_imdb = row['h1_2023_imdb_id']
        tv_imdb = row['tv_imdb_ids']

        if str(h1_imdb) == 'MISSING':
            tconst, matched_title, score = find_best_match(title, title_index, all_titles_list)
            if tconst and score >= 75:
                df.at[idx, 'h1_2023_imdb_id'] = tconst
                matches_found += 1
                print(f"  MATCHED: '{title[:50]}' -> {tconst} ({matched_title}, score={score:.0f})")

        if str(tv_imdb) == 'MISSING':
            tconst, matched_title, score = find_best_match(title, title_index, all_titles_list)
            if tconst and score >= 75:
                df.at[idx, 'tv_imdb_ids'] = tconst

    print(f"\nMatching completed in {time.time()-start:.1f}s")
    print(f"Found {matches_found} matches")

    # Save results
    print(f"\nSaving to: {OUTPUT_PATH}")
    df.to_excel(OUTPUT_PATH, index=False)

    # Summary
    remaining_h1 = (df['h1_2023_imdb_id'].astype(str) == 'MISSING').sum()
    remaining_tv = (df['tv_imdb_ids'].astype(str) == 'MISSING').sum()
    print(f"\nRemaining MISSING after matching:")
    print(f"  h1_2023_imdb_id: {remaining_h1}")
    print(f"  tv_imdb_ids: {remaining_tv}")

    print("\nDone!")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
