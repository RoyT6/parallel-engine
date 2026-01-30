"""
Final date filling script - handles all edge cases:
- Title variations (Stranger Things 4 vs Stranger Things: Season 4)
- Arc/Book/Part naming conventions
- Multi-language titles
- Uses IMDB IDs when available
- Falls back to comprehensive title matching
"""

import pandas as pd
import json
import re
from pathlib import Path
from collections import defaultdict

# Paths
JW_DIR = Path(r"C:\Users\RoyT6\Downloads\Freckles\00_ACTIVE_SOURCES\Justwatch")
RG_DIR = Path(r"C:\Users\RoyT6\Downloads\Freckles\00_ACTIVE_SOURCES\Rating_Graph")
IMDB_DIR = Path(r"C:\Users\RoyT6\Downloads\Freckles\00_ACTIVE_SOURCES\IMDB")
TMDB_DIR = Path(r"C:\Users\RoyT6\Downloads\Freckles\00_ACTIVE_SOURCES\TMDB")
OUTPUT_DIR = Path(r"C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30")


def normalize_title(title):
    """Basic title normalization."""
    if pd.isna(title):
        return None
    t = str(title).lower().strip()
    # Remove year in parentheses
    t = re.sub(r'\s*\(\d{4}\)\s*', ' ', t)
    # Remove special chars except spaces
    t = re.sub(r'[^a-z0-9\s]', '', t)
    # Collapse spaces
    t = re.sub(r'\s+', ' ', t).strip()
    return t if t else None


def extract_base_title_and_season(title):
    """
    Extract base title and season/part number from various title formats.
    Returns: (base_title, season_number)
    """
    if pd.isna(title):
        return None, None

    t = str(title)
    season = None

    # Pattern 1: "Show Name 4" (number at end without season/part)
    match = re.search(r'^(.+?)\s+(\d+)$', t)
    if match and int(match.group(2)) <= 30:  # Reasonable season number
        base = match.group(1)
        season = int(match.group(2))
        return normalize_title(base), season

    # Pattern 2: "Show: Season 3" or "Show: Part 2"
    match = re.search(r'^(.+?):\s*(?:season|part|series|volume|chapter|book)\s*(\d+)', t, re.IGNORECASE)
    if match:
        base = match.group(1)
        season = int(match.group(2))
        return normalize_title(base), season

    # Pattern 3: "Show: Book One/Two/Three"
    match = re.search(r'^(.+?):\s*(?:book|part|volume)\s*(one|two|three|four|five|six|i|ii|iii|iv|v|vi)', t, re.IGNORECASE)
    if match:
        base = match.group(1)
        num_map = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
                   'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6}
        season = num_map.get(match.group(2).lower(), 1)
        return normalize_title(base), season

    # Pattern 4: "Show: Subtitle" - extract base
    match = re.search(r'^(.+?):\s*.+$', t)
    if match:
        base = match.group(1)
        return normalize_title(base), None

    return normalize_title(t), None


def normalize_date(date_val, require_month_day=True):
    """Normalize date to YYYY-MM-DD format."""
    if pd.isna(date_val):
        return None

    date_str = str(date_val).strip()

    # Skip year-only if required
    if require_month_day and re.match(r'^\d{4}$', date_str):
        return None

    # Already YYYY-MM-DD
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str

    # YYYY only
    if re.match(r'^\d{4}$', date_str):
        return f"{date_str}-01-01"

    # DD-MM-YY or DD-MM-YYYY
    match = re.match(r'^(\d{2})-(\d{2})-(\d{2,4})$', date_str)
    if match:
        d, m, y = match.groups()
        if len(y) == 2:
            y = int(y)
            y = 1900 + y if y > 50 else 2000 + y
        return f"{y}-{m}-{d}"

    # MM/DD/YYYY
    match = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{2,4})$', date_str)
    if match:
        m, d, y = match.groups()
        if len(y) == 2:
            y = int(y)
            y = 1900 + y if y > 50 else 2000 + y
        return f"{y}-{m.zfill(2)}-{d.zfill(2)}"

    # Try pandas
    try:
        parsed = pd.to_datetime(date_str, errors='coerce', dayfirst=True)
        if pd.notna(parsed):
            return parsed.strftime('%Y-%m-%d')
    except:
        pass

    return None


def build_comprehensive_lookup():
    """Build comprehensive lookup with multiple keys."""
    print("Building comprehensive date lookup...")

    # Lookups
    by_imdb = {}  # imdb_id -> date (most reliable)
    by_title = {}  # norm_title -> date
    by_title_season = {}  # (norm_title, season) -> date

    # Load TMDB TV (has first_air_date)
    print("  Loading TMDB TV...")
    tmdb_tv_path = TMDB_DIR / "TMDB_TV_22_MAY24.csv"
    if tmdb_tv_path.exists():
        df = pd.read_csv(tmdb_tv_path, low_memory=False)
        for _, row in df.iterrows():
            name = row.get('name') or row.get('original_name')
            date = normalize_date(row.get('first_air_date'))
            imdb_id = row.get('imdb_id')

            if date:
                if pd.notna(imdb_id):
                    imdb_str = str(imdb_id).strip()
                    if imdb_str not in by_imdb:
                        by_imdb[imdb_str] = date

                norm_title = normalize_title(name)
                if norm_title and norm_title not in by_title:
                    by_title[norm_title] = date

    # Load TMDB Movies
    print("  Loading TMDB Movies...")
    tmdb_movie_path = TMDB_DIR / "TMDB_MOVIES_GDHD.csv"
    if tmdb_movie_path.exists():
        df = pd.read_csv(tmdb_movie_path, low_memory=False)
        for _, row in df.iterrows():
            title = row.get('title') or row.get('original_title')
            date = normalize_date(row.get('release_date'))
            imdb_id = row.get('imdb_id')

            if date:
                if pd.notna(imdb_id):
                    imdb_str = str(imdb_id).strip()
                    if imdb_str not in by_imdb:
                        by_imdb[imdb_str] = date

                norm_title = normalize_title(title)
                if norm_title and norm_title not in by_title:
                    by_title[norm_title] = date

    # Load Rating Graph metadata with season dates
    print("  Loading Rating Graph TV metadata (seasons)...")
    rg_tv_path = RG_DIR / "INCOMING_metadata_RG_29.25_tv.csv"
    if rg_tv_path.exists():
        df = pd.read_csv(rg_tv_path, low_memory=False)
        for _, row in df.iterrows():
            name = row.get('name')
            start = row.get('start')
            seasons_json = row.get('tmdb_seasons_json')
            imdb_id = row.get('imdb_id')

            norm_title = normalize_title(name)
            if not norm_title:
                continue

            # Show-level date
            date = normalize_date(start)
            if date:
                if pd.notna(imdb_id):
                    imdb_str = str(imdb_id).strip()
                    if imdb_str not in by_imdb:
                        by_imdb[imdb_str] = date
                if norm_title not in by_title:
                    by_title[norm_title] = date

            # Season-level dates from JSON
            if pd.notna(seasons_json):
                try:
                    seasons = json.loads(seasons_json)
                    for season in seasons:
                        sn = season.get('season_number', 0)
                        air_date = season.get('air_date')
                        if sn > 0 and air_date:
                            key = (norm_title, sn)
                            if key not in by_title_season:
                                by_title_season[key] = air_date
                except:
                    pass

    # Load IMDB Seasons Allocated
    print("  Loading IMDB Seasons Allocated...")
    imdb_path = IMDB_DIR / "IMDB Seasons Allocated.csv"
    if imdb_path.exists():
        df = pd.read_csv(imdb_path, low_memory=False)
        for _, row in df.iterrows():
            title = row.get('title')
            orig_title = row.get('original_title')
            start_year = row.get('start_year')
            tconst = row.get('tconst')

            # Use year as date
            date = normalize_date(start_year, require_month_day=False)
            if not date:
                continue

            if pd.notna(tconst):
                tconst_str = str(tconst).strip()
                if tconst_str not in by_imdb:
                    by_imdb[tconst_str] = date

            for t in [title, orig_title]:
                norm_title = normalize_title(t)
                if norm_title and norm_title not in by_title:
                    by_title[norm_title] = date

    # Load JustWatch files
    print("  Loading JustWatch files...")
    for filepath in JW_DIR.glob('justwatch_scraped_output_us*.csv'):
        print(f"    {filepath.name}...")
        try:
            df = pd.read_csv(filepath, low_memory=False)
            for _, row in df.iterrows():
                title = row.get('title')
                date_created = row.get('dateCreated')

                date = normalize_date(date_created)
                if not date:
                    continue

                norm_title = normalize_title(title)
                if norm_title and norm_title not in by_title:
                    by_title[norm_title] = date
        except Exception as e:
            print(f"      ERROR: {e}")

    # Load Rating Graph aggregate files
    print("  Loading Rating Graph aggregates...")
    rg_files = [
        ("flixpatrol_TV_RG_aggregate.csv", ['PREMIERE', 'Release Date']),
        ("Rating Graph Consolidated v4.csv", ['Release_Date', 'release_date_streaming']),
    ]
    for i in range(1, 5):
        rg_files.append((f"RG_AGG_p{i}.csv", ['Release Date', 'Release_Date', 'RT_Release Date (Streaming)']))

    for filename, date_cols in rg_files:
        filepath = RG_DIR / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, low_memory=False)
                for _, row in df.iterrows():
                    title = row.get('Title')

                    date = None
                    for col in date_cols:
                        if col in df.columns and pd.notna(row.get(col)):
                            date = normalize_date(row.get(col))
                            if date:
                                break

                    if not date:
                        continue

                    norm_title = normalize_title(title)
                    if norm_title and norm_title not in by_title:
                        by_title[norm_title] = date
            except Exception as e:
                pass

    print(f"\n  Lookup sizes:")
    print(f"    by_imdb: {len(by_imdb)}")
    print(f"    by_title: {len(by_title)}")
    print(f"    by_title_season: {len(by_title_season)}")

    return by_imdb, by_title, by_title_season


def find_date(title, imdb_id, fc_uid, season_num, by_imdb, by_title, by_title_season):
    """
    Multi-strategy date lookup.
    """
    # Strategy 1: IMDB ID lookup (most reliable)
    if pd.notna(imdb_id):
        imdb_str = str(imdb_id).strip()
        if imdb_str in by_imdb:
            return by_imdb[imdb_str], 'imdb_id'

    # Strategy 2: fc_uid (extract imdb portion)
    if pd.notna(fc_uid):
        fc_str = str(fc_uid).strip()
        # fc_uid like tt123456_s01 or tt123456
        imdb_part = fc_str.split('_')[0]
        if imdb_part in by_imdb:
            return by_imdb[imdb_part], 'fc_uid'

    # Extract base title and potential season
    base_title, extracted_season = extract_base_title_and_season(title)
    if not base_title:
        return None, 'no_title'

    # Use extracted season if we don't have one
    search_season = season_num if pd.notna(season_num) else extracted_season

    # Strategy 3: Base title + season
    if search_season:
        key = (base_title, int(search_season))
        if key in by_title_season:
            return by_title_season[key], 'title_season'

    # Strategy 4: Base title (show-level date)
    if base_title in by_title:
        return by_title[base_title], 'title_base'

    # Strategy 5: Full normalized title
    full_norm = normalize_title(title)
    if full_norm and full_norm in by_title:
        return by_title[full_norm], 'title_full'

    # Strategy 6: Try variations
    variations = generate_title_variations(title)
    for var in variations:
        if var in by_title:
            return by_title[var], 'title_var'

    return None, 'not_found'


def generate_title_variations(title):
    """Generate possible title variations for matching."""
    if pd.isna(title):
        return []

    t = str(title)
    variations = []

    # Remove common suffixes
    suffixes_to_remove = [
        r':\s*season\s*\d+.*$',
        r':\s*part\s*\d+.*$',
        r':\s*book\s*\d+.*$',
        r':\s*volume\s*\d+.*$',
        r':\s*chapter\s*\d+.*$',
        r':\s*the\s*movie.*$',
        r'\s*\d+$',  # trailing number
        r'\s*\(.*\)$',  # parenthetical
    ]

    current = t
    for pattern in suffixes_to_remove:
        stripped = re.sub(pattern, '', current, flags=re.IGNORECASE).strip()
        norm = normalize_title(stripped)
        if norm and norm not in variations:
            variations.append(norm)

    # Handle language variants like "(Hindi)"
    lang_stripped = re.sub(r'\s*\((?:hindi|english|spanish|french|german|japanese|korean|chinese|arabic|portuguese|italian|turkish|thai|indonesian|polish|dutch|swedish|norwegian|danish|finnish|russian|ukrainian|greek|czech|romanian|hungarian|vietnamese|tagalog|malay)\).*$', '', t, flags=re.IGNORECASE)
    norm = normalize_title(lang_stripped)
    if norm and norm not in variations:
        variations.append(norm)

    # Handle "Through My Window 2: ..." -> "Through My Window"
    match = re.match(r'^(.+?)\s*\d+:\s*.+$', t)
    if match:
        norm = normalize_title(match.group(1))
        if norm and norm not in variations:
            variations.append(norm)

    return variations


def update_missing_dates(filepath, by_imdb, by_title, by_title_season):
    """Update missing first_air_date in xlsx file."""
    print(f"\nProcessing: {filepath.name}")
    df = pd.read_excel(filepath)

    stats = defaultdict(int)

    for idx, row in df.iterrows():
        current_date = row.get('first_air_date')

        if pd.notna(current_date):
            stats['already_has'] += 1
            continue

        title = row.get('title')
        imdb_id = row.get('imdb_id')
        fc_uid = row.get('fc_uid')
        season_num = row.get('season_number')

        date, source = find_date(title, imdb_id, fc_uid, season_num, by_imdb, by_title, by_title_season)

        if date:
            df.at[idx, 'first_air_date'] = date
            stats[source] += 1
        else:
            stats['not_found'] += 1

    # Save
    df.to_excel(filepath, index=False)

    print(f"  Already had: {stats['already_has']}")
    print(f"  imdb_id: {stats['imdb_id']}, fc_uid: {stats['fc_uid']}")
    print(f"  title_season: {stats['title_season']}, title_base: {stats['title_base']}")
    print(f"  title_full: {stats['title_full']}, title_var: {stats['title_var']}")
    print(f"  NOT FOUND: {stats['not_found']}")

    total_updated = sum(v for k, v in stats.items() if k not in ['already_has', 'not_found', 'no_title'])
    return total_updated, stats['not_found']


def main():
    print("=" * 70)
    print("Final Date Fill - All Strategies Combined")
    print("=" * 70)

    # Build lookups
    by_imdb, by_title, by_title_season = build_comprehensive_lookup()

    # Process all files
    total_updated = 0
    total_missing = 0

    for filepath in sorted(OUTPUT_DIR.glob("*.xlsx")):
        u, m = update_missing_dates(filepath, by_imdb, by_title, by_title_season)
        total_updated += u
        total_missing += m

    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"  Total updated: {total_updated}")
    print(f"  Total still missing: {total_missing}")
    print("=" * 70)


if __name__ == "__main__":
    main()
