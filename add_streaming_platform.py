"""
Add streaming_platform column to Netflix 2.30 files.
- Uses 'Available Globally?' from original Netflix engagement reports
- Global = Yes -> streaming_platform = 'netflix'
- Global = No -> streaming_platform = None (to be determined later)
"""

import pandas as pd
import re
from pathlib import Path

# Paths
ORIGINAL_REPORTS = [
    Path(r"C:\Users\RoyT6\Downloads\Training Data\What_We_Watched_A_Netflix_Engagement_Report_2023Jan-Jun.xlsx"),
    Path(r"C:\Users\RoyT6\Downloads\Training Data\What_We_Watched_A_Netflix_Engagement_Report_2023Jul-Dec.xlsx"),
    Path(r"C:\Users\RoyT6\Downloads\Training Data\What_We_Watched_A_Netflix_Engagement_Report_2024Jan-Jun.xlsx"),
    Path(r"C:\Users\RoyT6\Downloads\Training Data\What_We_Watched_A_Netflix_Engagement_Report_2024Jul-Dec.xlsx"),
    Path(r"C:\Users\RoyT6\Downloads\Training Data\What_We_Watched_A_Netflix_Engagement_Report_2025Jan-Jun.xlsx"),
    Path(r"C:\Users\RoyT6\Downloads\Training Data\What_We_Watched_A_Netflix_Engagement_Report_2025Jul-Dec.xlsx"),
]
OUTPUT_DIR = Path(r"C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30")


def normalize_title(title):
    """Normalize title for matching."""
    if pd.isna(title):
        return None
    t = str(title).lower().strip()
    # Handle bilingual titles (e.g., "Berlin: Season 1 // Berlín: Temporada 1")
    if '//' in t:
        t = t.split('//')[0].strip()
    # Remove season/series/part suffixes
    t = re.sub(r':\s*(season|limited series|part|volume|chapter|miniseries|series)\s*\d*\s*$', '', t, flags=re.IGNORECASE)
    # Remove year in parentheses at end
    t = re.sub(r'\s*\(\d{4}\)\s*$', '', t)
    # Remove special chars except spaces
    t = re.sub(r'[^a-z0-9\s]', '', t)
    # Collapse multiple spaces
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def extract_season_number(title):
    """Extract season number from title like 'Show: Season 3'."""
    if pd.isna(title):
        return 1
    match = re.search(r'season\s*(\d+)', str(title), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Limited series / miniseries are season 1
    if re.search(r'limited series|miniseries', str(title), flags=re.IGNORECASE):
        return 1
    return 1


def build_global_lookup():
    """Build lookup from original reports: (normalized_title, season) -> is_global."""
    print("Building global availability lookup from original reports...")

    lookup = {}  # (norm_title, season) -> 'Yes' or 'No'
    title_lookup = {}  # norm_title -> 'Yes' or 'No' (for films without season)

    for report_path in ORIGINAL_REPORTS:
        if not report_path.exists():
            print(f"  WARNING: {report_path.name} not found, skipping")
            continue

        print(f"  Loading {report_path.name}...")
        df = pd.read_excel(report_path, header=5)

        for _, row in df.iterrows():
            title = row.get('Title')
            global_status = row.get('Available Globally?')

            if pd.isna(title) or pd.isna(global_status):
                continue

            norm_title = normalize_title(title)
            season = extract_season_number(title)

            if norm_title:
                # Store with season for TV
                lookup[(norm_title, season)] = global_status
                # Also store just by title for films
                if norm_title not in title_lookup:
                    title_lookup[norm_title] = global_status

    print(f"  Built lookup with {len(lookup)} title+season entries")
    print(f"  Built lookup with {len(title_lookup)} title-only entries")

    # Count Yes vs No
    yes_count = sum(1 for v in lookup.values() if v == 'Yes')
    no_count = sum(1 for v in lookup.values() if v == 'No')
    print(f"  Global=Yes: {yes_count}, Global=No: {no_count}")

    return lookup, title_lookup


def add_streaming_platform(filepath, lookup, title_lookup, is_tv=True):
    """Add streaming_platform column to xlsx file."""
    print(f"\nProcessing: {filepath.name}")
    df = pd.read_excel(filepath)

    streaming_platforms = []
    netflix_count = 0
    non_netflix_count = 0
    unknown_count = 0

    for _, row in df.iterrows():
        title = row.get('title')
        season_num = row.get('season_number', 1)

        if pd.isna(season_num):
            season_num = 1
        else:
            season_num = int(season_num)

        norm_title = normalize_title(title)
        platform = None

        if norm_title:
            # Try with season first (for TV)
            if (norm_title, season_num) in lookup:
                global_status = lookup[(norm_title, season_num)]
                if global_status == 'Yes':
                    platform = 'netflix'
                    netflix_count += 1
                else:
                    non_netflix_count += 1
            # Fall back to title-only lookup
            elif norm_title in title_lookup:
                global_status = title_lookup[norm_title]
                if global_status == 'Yes':
                    platform = 'netflix'
                    netflix_count += 1
                else:
                    non_netflix_count += 1
            else:
                unknown_count += 1
        else:
            unknown_count += 1

        streaming_platforms.append(platform)

    # Find position to insert (after first_air_date)
    if 'first_air_date' in df.columns:
        insert_pos = df.columns.get_loc('first_air_date') + 1
    else:
        insert_pos = 4

    # Insert streaming_platform column
    df.insert(insert_pos, 'streaming_platform', streaming_platforms)

    # Save back
    df.to_excel(filepath, index=False)

    total = len(df)
    print(f"  Netflix (Global=Yes): {netflix_count} ({100*netflix_count/total:.1f}%)")
    print(f"  Non-Netflix (Global=No): {non_netflix_count} ({100*non_netflix_count/total:.1f}%)")
    print(f"  Unknown (not in reports): {unknown_count} ({100*unknown_count/total:.1f}%)")

    return netflix_count, non_netflix_count, unknown_count


def main():
    print("=" * 60)
    print("Adding streaming_platform Column to Netflix 2.30")
    print("=" * 60)

    # Build lookup
    lookup, title_lookup = build_global_lookup()

    # Process all 2.30 files
    total_netflix = 0
    total_non_netflix = 0
    total_unknown = 0

    for filepath in sorted(OUTPUT_DIR.glob("*.xlsx")):
        is_tv = "netflix_tv_" in filepath.name
        n, nn, u = add_streaming_platform(filepath, lookup, title_lookup, is_tv=is_tv)
        total_netflix += n
        total_non_netflix += nn
        total_unknown += u

    print("\n" + "=" * 60)
    print("COMPLETE")
    total = total_netflix + total_non_netflix + total_unknown
    print(f"  Netflix (Global=Yes): {total_netflix} ({100*total_netflix/total:.1f}%)")
    print(f"  Non-Netflix (Global=No): {total_non_netflix} ({100*total_non_netflix/total:.1f}%)")
    print(f"  Unknown: {total_unknown} ({100*total_unknown/total:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
