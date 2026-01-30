"""
Add streaming_platform for Global=No titles using JustWatch data.
Processes ALL records, not just samples.
"""

import pandas as pd
import re
from pathlib import Path

# Paths
JW_DIR = Path(r"C:\Users\RoyT6\Downloads\Freckles\00_ACTIVE_SOURCES\Justwatch")
OUTPUT_DIR = Path(r"C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30")


def normalize_title(title):
    """Normalize title for matching."""
    if pd.isna(title):
        return None
    t = str(title).lower().strip()
    # Remove year in parentheses
    t = re.sub(r'\s*\(\d{4}\)\s*', ' ', t)
    # Remove special chars
    t = re.sub(r'[^a-z0-9\s]', '', t)
    # Collapse spaces
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def extract_primary_platform(services_str):
    """Extract the primary (first) streaming platform from services string."""
    if pd.isna(services_str):
        return None

    services = str(services_str).split(',')
    if not services:
        return None

    primary = services[0].strip().lower()

    # Normalize platform names
    platform_map = {
        'netflix': 'netflix',
        'netflix standard with ads': 'netflix',
        'amazon prime video': 'amazon_prime',
        'amazon video': 'amazon_prime',
        'prime video': 'amazon_prime',
        'apple tv': 'apple_tv',
        'apple tv+': 'apple_tv',
        'disney+': 'disney_plus',
        'disney plus': 'disney_plus',
        'hbo max': 'hbo_max',
        'max': 'hbo_max',
        'hulu': 'hulu',
        'paramount+': 'paramount_plus',
        'paramount plus': 'paramount_plus',
        'peacock': 'peacock',
        'peacock premium': 'peacock',
        'discovery+': 'discovery_plus',
        'showtime': 'showtime',
        'starz': 'starz',
        'mgm+': 'mgm_plus',
        'amc+': 'amc_plus',
        'britbox': 'britbox',
        'crunchyroll': 'crunchyroll',
        'funimation': 'funimation',
        'tubi': 'tubi',
        'pluto tv': 'pluto_tv',
        'youtube': 'youtube',
        'youtube premium': 'youtube',
    }

    for key, val in platform_map.items():
        if key in primary:
            return val

    # Return cleaned version of original if no match
    return re.sub(r'[^a-z0-9_]', '_', primary).strip('_')


def load_justwatch_data():
    """Load ALL JustWatch US data and build title lookup."""
    print("Loading JustWatch data (ALL records)...")

    # title -> (streamingServices, type)
    lookup = {}

    # Process all US files
    us_files = list(JW_DIR.glob('justwatch_scraped_output_us*.csv'))

    for filepath in us_files:
        print(f"  Loading {filepath.name}...")
        try:
            df = pd.read_csv(filepath, low_memory=False)

            for _, row in df.iterrows():
                title = row.get('title')
                services = row.get('streamingServices')
                content_type = row.get('type')

                if pd.isna(title) or pd.isna(services):
                    continue

                norm_title = normalize_title(title)
                if norm_title:
                    # Store with type info
                    if norm_title not in lookup:
                        lookup[norm_title] = {
                            'services': services,
                            'type': content_type,
                            'primary': extract_primary_platform(services)
                        }
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"  Built lookup with {len(lookup)} unique titles")

    # Also load the aggregated TV file
    print("  Loading JustWatch TV aggregate...")
    try:
        tv_agg = pd.read_csv(JW_DIR / 'JustWatch_TV_Skaro_RG_aggregate.csv', low_memory=False)

        # Extract US streaming services
        for _, row in tv_agg.iterrows():
            title = row.get('Title') or row.get('title')
            us_services = row.get('us_streamingServices')

            if pd.isna(title) or pd.isna(us_services):
                continue

            norm_title = normalize_title(title)
            if norm_title and norm_title not in lookup:
                lookup[norm_title] = {
                    'services': us_services,
                    'type': 'show',
                    'primary': extract_primary_platform(us_services)
                }

        print(f"  After TV aggregate: {len(lookup)} unique titles")
    except Exception as e:
        print(f"    ERROR loading TV aggregate: {e}")

    return lookup


def update_streaming_platform(filepath, jw_lookup, is_tv=True):
    """Update streaming_platform for Global=No titles."""
    print(f"\nProcessing: {filepath.name}")
    df = pd.read_excel(filepath)

    updated = 0
    already_netflix = 0
    not_found = 0

    for idx, row in df.iterrows():
        current_platform = row.get('streaming_platform')

        # Skip if already has Netflix
        if current_platform == 'netflix':
            already_netflix += 1
            continue

        # Only process Global=No (current_platform is None/NaN)
        if pd.notna(current_platform):
            continue

        # Try to find in JustWatch
        title = row.get('title')
        norm_title = normalize_title(title)

        if norm_title and norm_title in jw_lookup:
            platform = jw_lookup[norm_title]['primary']
            if platform:
                df.at[idx, 'streaming_platform'] = platform
                updated += 1
            else:
                not_found += 1
        else:
            not_found += 1

    # Save back
    df.to_excel(filepath, index=False)

    print(f"  Already Netflix: {already_netflix}")
    print(f"  Updated from JustWatch: {updated}")
    print(f"  Not found in JustWatch: {not_found}")

    return updated, not_found


def main():
    print("=" * 60)
    print("Adding streaming_platform from JustWatch (ALL records)")
    print("=" * 60)

    # Load JustWatch data
    jw_lookup = load_justwatch_data()

    # Process all 2.30 files
    total_updated = 0
    total_not_found = 0

    for filepath in sorted(OUTPUT_DIR.glob("*.xlsx")):
        is_tv = "netflix_tv_" in filepath.name
        u, nf = update_streaming_platform(filepath, jw_lookup, is_tv=is_tv)
        total_updated += u
        total_not_found += nf

    print("\n" + "=" * 60)
    print("COMPLETE")
    print(f"  Total updated from JustWatch: {total_updated}")
    print(f"  Total not found: {total_not_found}")
    print("=" * 60)


if __name__ == "__main__":
    main()
