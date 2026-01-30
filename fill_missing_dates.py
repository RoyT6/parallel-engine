"""
Fill missing first_air_date using JustWatch dateCreated and Rating Graph sources.
Processes ALL records row by row.
"""

import pandas as pd
import json
import re
from pathlib import Path

# Paths
JW_DIR = Path(r"C:\Users\RoyT6\Downloads\Freckles\00_ACTIVE_SOURCES\Justwatch")
RG_DIR = Path(r"C:\Users\RoyT6\Downloads\Freckles\00_ACTIVE_SOURCES\Rating_Graph")
OUTPUT_DIR = Path(r"C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30")


def normalize_title(title):
    """Normalize title for matching."""
    if pd.isna(title):
        return None
    t = str(title).lower().strip()
    t = re.sub(r'\s*\(\d{4}\)\s*', ' ', t)
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def normalize_date(date_val):
    """Normalize date to YYYY-MM-DD format."""
    if pd.isna(date_val):
        return None

    date_str = str(date_val).strip()

    # Already in YYYY-MM-DD format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str

    # YYYY only -> YYYY-01-01
    if re.match(r'^\d{4}$', date_str):
        return f"{date_str}-01-01"

    # MM/DD/YYYY or M/D/YYYY
    match = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', date_str)
    if match:
        m, d, y = match.groups()
        return f"{y}-{m.zfill(2)}-{d.zfill(2)}"

    # DD-MM-YY
    match = re.match(r'^(\d{2})-(\d{2})-(\d{2})$', date_str)
    if match:
        d, m, y = match.groups()
        year = int(y)
        year = 1900 + year if year > 50 else 2000 + year
        return f"{year}-{m}-{d}"

    # Try pandas parsing
    try:
        parsed = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(parsed):
            return parsed.strftime('%Y-%m-%d')
    except:
        pass

    return None


def load_justwatch_dates():
    """Load dateCreated from ALL JustWatch US files."""
    print("Loading JustWatch dateCreated (ALL records)...")

    lookup = {}
    us_files = list(JW_DIR.glob('justwatch_scraped_output_us*.csv'))

    for filepath in us_files:
        print(f"  {filepath.name}...")
        try:
            df = pd.read_csv(filepath, low_memory=False)

            for _, row in df.iterrows():
                title = row.get('title')
                date_created = row.get('dateCreated')

                if pd.isna(title) or pd.isna(date_created):
                    continue

                norm_title = normalize_title(title)
                norm_date = normalize_date(date_created)

                if norm_title and norm_date and norm_title not in lookup:
                    lookup[norm_title] = norm_date

        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"  JustWatch dates: {len(lookup)} titles")
    return lookup


def load_rating_graph_dates():
    """Load dates from ALL Rating Graph files."""
    print("Loading Rating Graph dates (ALL records)...")

    lookup = {}

    # RG_AGG_p1 through p4
    for i in range(1, 5):
        filepath = RG_DIR / f"RG_AGG_p{i}.csv"
        if filepath.exists():
            print(f"  {filepath.name}...")
            try:
                df = pd.read_csv(filepath, low_memory=False)

                for _, row in df.iterrows():
                    title = row.get('Title')

                    # Try multiple date columns
                    date = None
                    for col in ['Release Date', 'Release_Date', 'Start Year', 'tmdb_last_air_date', 'ppb_YEAR']:
                        if col in df.columns and pd.notna(row.get(col)):
                            date = normalize_date(row.get(col))
                            if date:
                                break

                    if pd.notna(title) and date:
                        norm_title = normalize_title(title)
                        if norm_title and norm_title not in lookup:
                            lookup[norm_title] = date

            except Exception as e:
                print(f"    ERROR: {e}")

    # Rating Graph Consolidated v4
    filepath = RG_DIR / "Rating Graph Consolidated v4.csv"
    if filepath.exists():
        print(f"  {filepath.name}...")
        try:
            df = pd.read_csv(filepath, low_memory=False)

            for _, row in df.iterrows():
                title = row.get('Title')

                date = None
                for col in ['Release_Date', 'Year', 'release_date_streaming']:
                    if col in df.columns and pd.notna(row.get(col)):
                        date = normalize_date(row.get(col))
                        if date:
                            break

                if pd.notna(title) and date:
                    norm_title = normalize_title(title)
                    if norm_title and norm_title not in lookup:
                        lookup[norm_title] = date

        except Exception as e:
            print(f"    ERROR: {e}")

    # INCOMING_metadata_RG_29.25_tv.csv - season level dates
    filepath = RG_DIR / "INCOMING_metadata_RG_29.25_tv.csv"
    if filepath.exists():
        print(f"  {filepath.name}...")
        try:
            df = pd.read_csv(filepath, low_memory=False)

            for _, row in df.iterrows():
                name = row.get('name')
                start = row.get('start')
                seasons_json = row.get('tmdb_seasons_json')

                if pd.isna(name):
                    continue

                norm_title = normalize_title(name)
                if not norm_title:
                    continue

                # Try to get season-level dates
                if pd.notna(seasons_json):
                    try:
                        seasons = json.loads(seasons_json)
                        for season in seasons:
                            season_num = season.get('season_number', 0)
                            air_date = season.get('air_date')
                            if air_date and season_num > 0:
                                # Store as title_s{season}
                                key = f"{norm_title}_s{season_num}"
                                if key not in lookup:
                                    lookup[key] = air_date
                    except:
                        pass

                # Also store show-level date
                if pd.notna(start) and norm_title not in lookup:
                    date = normalize_date(start)
                    if date:
                        lookup[norm_title] = date

        except Exception as e:
            print(f"    ERROR: {e}")

    # JSON files
    for json_file in ['rating_graph_12_28_2025_tv_shows.json', 'ratinggraph_tv_shows_20251231.json']:
        filepath = RG_DIR / json_file
        if filepath.exists():
            print(f"  {filepath.name}...")
            try:
                with open(filepath, encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        title = item.get('name') or item.get('title') or item.get('Title')
                        date = item.get('first_air_date') or item.get('start') or item.get('Year')

                        if title and date:
                            norm_title = normalize_title(title)
                            norm_date = normalize_date(date)
                            if norm_title and norm_date and norm_title not in lookup:
                                lookup[norm_title] = norm_date

            except Exception as e:
                print(f"    ERROR: {e}")

    print(f"  Rating Graph dates: {len(lookup)} titles")
    return lookup


def update_missing_dates(filepath, jw_dates, rg_dates):
    """Update missing first_air_date in xlsx file."""
    print(f"\nProcessing: {filepath.name}")
    df = pd.read_excel(filepath)

    updated_jw = 0
    updated_rg = 0
    still_missing = 0
    already_has = 0

    for idx, row in df.iterrows():
        current_date = row.get('first_air_date')

        # Skip if already has date
        if pd.notna(current_date):
            already_has += 1
            continue

        title = row.get('title')
        season_num = row.get('season_number')

        norm_title = normalize_title(title)
        if not norm_title:
            still_missing += 1
            continue

        new_date = None
        source = None

        # Try season-specific lookup first (for TV)
        if pd.notna(season_num):
            season_key = f"{norm_title}_s{int(season_num)}"
            if season_key in rg_dates:
                new_date = rg_dates[season_key]
                source = 'rg_season'

        # Try title lookup in Rating Graph
        if not new_date and norm_title in rg_dates:
            new_date = rg_dates[norm_title]
            source = 'rg'

        # Try JustWatch
        if not new_date and norm_title in jw_dates:
            new_date = jw_dates[norm_title]
            source = 'jw'

        if new_date:
            df.at[idx, 'first_air_date'] = new_date
            if source == 'jw':
                updated_jw += 1
            else:
                updated_rg += 1
        else:
            still_missing += 1

    # Save back
    df.to_excel(filepath, index=False)

    print(f"  Already had date: {already_has}")
    print(f"  Updated from RG: {updated_rg}")
    print(f"  Updated from JW: {updated_jw}")
    print(f"  Still missing: {still_missing}")

    return updated_jw + updated_rg, still_missing


def main():
    print("=" * 60)
    print("Filling Missing first_air_date (ALL records)")
    print("=" * 60)

    # Load date sources
    jw_dates = load_justwatch_dates()
    rg_dates = load_rating_graph_dates()

    # Merge lookups (RG takes priority)
    combined = {**jw_dates, **rg_dates}
    print(f"\nCombined lookup: {len(combined)} unique titles")

    # Process all 2.30 files
    total_updated = 0
    total_missing = 0

    for filepath in sorted(OUTPUT_DIR.glob("*.xlsx")):
        u, m = update_missing_dates(filepath, jw_dates, rg_dates)
        total_updated += u
        total_missing += m

    print("\n" + "=" * 60)
    print("COMPLETE")
    print(f"  Total updated: {total_updated}")
    print(f"  Total still missing: {total_missing}")
    print("=" * 60)


if __name__ == "__main__":
    main()
