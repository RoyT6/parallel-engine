"""
Add first_air_date column to Netflix crown jewels xlsx files.
- For TV: Uses season-level air_date from Rating Graph metadata
         Fallback 1: TMDB TV show-level first_air_date
         Fallback 2: fc_uid matching to IMDB Seasons Allocated -> Rating Graph
         Fallback 3: title + start_year + season_number matching
- For Films: Uses release_date from TMDB movies data
"""

import pandas as pd
import json
import re
from pathlib import Path

# Paths
INPUT_DIR = Path(r"C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.20")
OUTPUT_DIR = Path(r"C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30")
RG_METADATA = Path(r"C:\Users\RoyT6\Downloads\Freckles\00_ACTIVE_SOURCES\Rating_Graph\INCOMING_metadata_RG_29.25_tv.csv")
TMDB_MOVIES = Path(r"C:\Users\RoyT6\Downloads\Freckles\00_ACTIVE_SOURCES\TMDB\TMDB_MOVIES_GDHD.csv")
TMDB_TV = Path(r"C:\Users\RoyT6\Downloads\Freckles\00_ACTIVE_SOURCES\TMDB\TMDB_TV_22_MAY24.csv")
IMDB_SEASONS = Path(r"C:\Users\RoyT6\Downloads\Freckles\00_ACTIVE_SOURCES\IMDB\IMDB Seasons Allocated.csv")


def normalize_fc_uid(fc_uid):
    """Normalize fc_uid: tt123_s01 -> tt123_s1"""
    if pd.isna(fc_uid):
        return None
    return re.sub(r'_s0*(\d+)$', r'_s\1', str(fc_uid))


def normalize_title(title):
    """Normalize title for matching: lowercase, remove special chars"""
    if pd.isna(title):
        return None
    return re.sub(r'[^a-z0-9\s]', '', str(title).lower()).strip()


def build_tv_air_date_lookup():
    """Build lookup: imdb_id -> season_number -> air_date, and imdb_id -> tmdb_id mapping"""
    print("Loading Rating Graph TV metadata...")
    df = pd.read_csv(RG_METADATA)

    season_lookup = {}
    imdb_to_tmdb = {}
    name_year_lookup = {}  # For title-based matching: (normalized_name, start_year) -> imdb_id

    for _, row in df.iterrows():
        imdb_id = row.get('imdb_id')
        tmdb_id = row.get('tmdb_id')
        name = row.get('name')
        start_year = row.get('start')
        seasons_json = row.get('tmdb_seasons_json')

        if pd.isna(imdb_id):
            continue

        # Store imdb -> tmdb mapping
        if pd.notna(tmdb_id):
            imdb_to_tmdb[imdb_id] = int(tmdb_id)

        # Store name + year -> imdb_id mapping
        if pd.notna(name) and pd.notna(start_year):
            norm_name = normalize_title(name)
            if norm_name:
                name_year_lookup[(norm_name, int(start_year))] = imdb_id

        if pd.isna(seasons_json):
            continue

        try:
            seasons = json.loads(seasons_json)
            season_lookup[imdb_id] = {}
            for season in seasons:
                season_num = season.get('season_number')
                air_date = season.get('air_date')
                if season_num is not None and air_date:
                    season_lookup[imdb_id][season_num] = air_date
        except (json.JSONDecodeError, TypeError):
            continue

    print(f"  Loaded {len(season_lookup)} TV shows with season air dates")
    print(f"  Loaded {len(imdb_to_tmdb)} imdb->tmdb mappings")
    print(f"  Loaded {len(name_year_lookup)} name+year->imdb mappings")
    return season_lookup, imdb_to_tmdb, name_year_lookup


def build_tmdb_tv_fallback_lookup():
    """Build lookup: tmdb_id -> first_air_date (show-level fallback)"""
    print("Loading TMDB TV metadata (fallback)...")
    df = pd.read_csv(TMDB_TV)

    lookup = {}
    for _, row in df.iterrows():
        tmdb_id = row.get('id')
        first_air_date = row.get('first_air_date')

        if pd.notna(tmdb_id) and pd.notna(first_air_date):
            lookup[int(tmdb_id)] = first_air_date

    print(f"  Loaded {len(lookup)} TV shows with first_air_date")
    return lookup


def build_imdb_seasons_lookup():
    """Build lookups from IMDB Seasons Allocated:
    - fc_uid -> imdb_id
    - (title, start_year, season_number) -> imdb_id
    """
    print("Loading IMDB Seasons Allocated...")
    df = pd.read_csv(IMDB_SEASONS)

    fc_uid_to_imdb = {}
    title_year_season_to_imdb = {}

    for _, row in df.iterrows():
        fc_uid = row.get('fc_uid')
        imdb_id = row.get('imdb_id')
        title = row.get('title')
        start_year = row.get('start_year')
        season_num = row.get('season_number')

        if pd.isna(imdb_id):
            continue

        # fc_uid -> imdb_id lookup
        if pd.notna(fc_uid):
            norm_fc_uid = normalize_fc_uid(fc_uid)
            if norm_fc_uid:
                fc_uid_to_imdb[norm_fc_uid] = imdb_id

        # title + start_year + season_number -> imdb_id lookup
        if pd.notna(title) and pd.notna(start_year) and pd.notna(season_num):
            norm_title = normalize_title(title)
            if norm_title:
                key = (norm_title, int(start_year), int(season_num))
                title_year_season_to_imdb[key] = imdb_id

    print(f"  Loaded {len(fc_uid_to_imdb)} fc_uid->imdb mappings")
    print(f"  Loaded {len(title_year_season_to_imdb)} title+year+season->imdb mappings")
    return fc_uid_to_imdb, title_year_season_to_imdb


def build_film_release_date_lookup():
    """Build lookup: imdb_id -> release_date"""
    print("Loading TMDB Movies metadata...")
    df = pd.read_csv(TMDB_MOVIES, low_memory=False)

    lookup = {}
    for _, row in df.iterrows():
        imdb_id = row.get('imdb_id')
        release_date = row.get('release_date')

        if pd.notna(imdb_id) and pd.notna(release_date):
            lookup[imdb_id] = release_date

    print(f"  Loaded {len(lookup)} films with release dates")
    return lookup


def get_air_date_for_tv(imdb_id, season_num, fc_uid, title,
                        season_lookup, imdb_to_tmdb, tmdb_fallback,
                        fc_uid_to_imdb, title_year_season_to_imdb, name_year_lookup):
    """Try multiple strategies to get air_date for a TV show season."""

    # Strategy 1: Direct imdb_id + season lookup in Rating Graph
    if pd.notna(imdb_id) and imdb_id in season_lookup:
        if season_num in season_lookup[imdb_id]:
            return season_lookup[imdb_id][season_num], 'rg_season'

    # Strategy 2: TMDB show-level fallback via imdb->tmdb mapping
    if pd.notna(imdb_id) and imdb_id in imdb_to_tmdb:
        tmdb_id = imdb_to_tmdb[imdb_id]
        if tmdb_id in tmdb_fallback:
            return tmdb_fallback[tmdb_id], 'tmdb_show'

    # Strategy 3: fc_uid matching to IMDB Seasons -> get imdb_id -> lookup
    if pd.notna(fc_uid):
        norm_fc_uid = normalize_fc_uid(fc_uid)
        if norm_fc_uid and norm_fc_uid in fc_uid_to_imdb:
            resolved_imdb = fc_uid_to_imdb[norm_fc_uid]
            # Try season lookup with resolved imdb_id
            if resolved_imdb in season_lookup and season_num in season_lookup[resolved_imdb]:
                return season_lookup[resolved_imdb][season_num], 'fc_uid_season'
            # Try TMDB fallback with resolved imdb_id
            if resolved_imdb in imdb_to_tmdb:
                tmdb_id = imdb_to_tmdb[resolved_imdb]
                if tmdb_id in tmdb_fallback:
                    return tmdb_fallback[tmdb_id], 'fc_uid_tmdb'

    # Strategy 4: Title + year + season matching via IMDB Seasons Allocated
    # We need to try multiple years since Netflix doesn't have start_year
    if pd.notna(title):
        norm_title = normalize_title(title)
        if norm_title:
            # Try to find any year match for this title+season
            for year in range(2000, 2027):
                key = (norm_title, year, season_num)
                if key in title_year_season_to_imdb:
                    resolved_imdb = title_year_season_to_imdb[key]
                    # Try season lookup
                    if resolved_imdb in season_lookup and season_num in season_lookup[resolved_imdb]:
                        return season_lookup[resolved_imdb][season_num], 'title_match_season'
                    # Try TMDB fallback
                    if resolved_imdb in imdb_to_tmdb:
                        tmdb_id = imdb_to_tmdb[resolved_imdb]
                        if tmdb_id in tmdb_fallback:
                            return tmdb_fallback[tmdb_id], 'title_match_tmdb'

    return None, 'unmatched'


def process_tv_file(filepath, season_lookup, imdb_to_tmdb, tmdb_fallback,
                    fc_uid_to_imdb, title_year_season_to_imdb, name_year_lookup, output_dir):
    """Process a TV xlsx file, adding first_air_date column E"""
    print(f"\nProcessing TV: {filepath.name}")
    df = pd.read_excel(filepath)

    # Build first_air_date column
    first_air_dates = []
    stats = {'rg_season': 0, 'tmdb_show': 0, 'fc_uid_season': 0, 'fc_uid_tmdb': 0,
             'title_match_season': 0, 'title_match_tmdb': 0, 'unmatched': 0}

    for _, row in df.iterrows():
        imdb_id = row.get('imdb_id')
        season_num = row.get('season_number')
        fc_uid = row.get('fc_uid')
        title = row.get('title')

        if pd.notna(season_num):
            season_num = int(season_num)
        else:
            season_num = 1  # Default to season 1

        air_date, source = get_air_date_for_tv(
            imdb_id, season_num, fc_uid, title,
            season_lookup, imdb_to_tmdb, tmdb_fallback,
            fc_uid_to_imdb, title_year_season_to_imdb, name_year_lookup
        )

        first_air_dates.append(air_date)
        stats[source] += 1

    # Insert as column E (index 4)
    df.insert(4, 'first_air_date', first_air_dates)

    # Update filename version
    new_name = filepath.name.replace('v2.20', 'v2.30')
    output_path = output_dir / new_name

    df.to_excel(output_path, index=False)

    matched = sum(v for k, v in stats.items() if k != 'unmatched')
    print(f"  RG season: {stats['rg_season']}, TMDB show: {stats['tmdb_show']}")
    print(f"  fc_uid->season: {stats['fc_uid_season']}, fc_uid->TMDB: {stats['fc_uid_tmdb']}")
    print(f"  title match->season: {stats['title_match_season']}, title match->TMDB: {stats['title_match_tmdb']}")
    print(f"  TOTAL MATCHED: {matched}, Unmatched: {stats['unmatched']}")
    print(f"  Saved to: {output_path}")
    return matched, stats['unmatched']


def process_film_file(filepath, film_lookup, output_dir):
    """Process a Films xlsx file, adding first_air_date (release_date) column E"""
    print(f"\nProcessing Film: {filepath.name}")
    df = pd.read_excel(filepath)

    # Build first_air_date column (using release_date for films)
    first_air_dates = []
    matched = 0
    unmatched = 0

    for _, row in df.iterrows():
        imdb_id = row.get('imdb_id')

        release_date = None
        if pd.notna(imdb_id) and imdb_id in film_lookup:
            release_date = film_lookup[imdb_id]
            matched += 1
        else:
            unmatched += 1

        first_air_dates.append(release_date)

    # Insert after fc_uid (index 3)
    df.insert(3, 'first_air_date', first_air_dates)

    # Update filename version
    new_name = filepath.name.replace('v2.20', 'v2.30')
    output_path = output_dir / new_name

    df.to_excel(output_path, index=False)
    print(f"  Matched: {matched}, Unmatched: {unmatched}")
    print(f"  Saved to: {output_path}")
    return matched, unmatched


def main():
    print("=" * 60)
    print("Adding first_air_date to Netflix Crown Jewels v2.30")
    print("=" * 60)

    # Build lookups
    season_lookup, imdb_to_tmdb, name_year_lookup = build_tv_air_date_lookup()
    tmdb_fallback = build_tmdb_tv_fallback_lookup()
    fc_uid_to_imdb, title_year_season_to_imdb = build_imdb_seasons_lookup()
    film_lookup = build_film_release_date_lookup()

    # Process all files
    total_matched = 0
    total_unmatched = 0

    for filepath in INPUT_DIR.glob("*.xlsx"):
        if "netflix_tv_" in filepath.name:
            m, u = process_tv_file(filepath, season_lookup, imdb_to_tmdb, tmdb_fallback,
                                   fc_uid_to_imdb, title_year_season_to_imdb, name_year_lookup, OUTPUT_DIR)
        elif "netflix_films_" in filepath.name:
            m, u = process_film_file(filepath, film_lookup, OUTPUT_DIR)
        else:
            print(f"\nSkipping unknown file: {filepath.name}")
            continue

        total_matched += m
        total_unmatched += u

    print("\n" + "=" * 60)
    print(f"COMPLETE - Total matched: {total_matched}, Total unmatched: {total_unmatched}")
    match_rate = 100 * total_matched / (total_matched + total_unmatched)
    print(f"Overall match rate: {match_rate:.1f}%")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
