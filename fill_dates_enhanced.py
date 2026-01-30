"""
Enhanced date filling using multiple matching strategies:
- title + title_type + start_year
- title + genre
- title + cast/producer
Processes ALL records row by row.
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
OUTPUT_DIR = Path(r"C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30")


def normalize_title(title):
    """Normalize title for matching."""
    if pd.isna(title):
        return None
    t = str(title).lower().strip()
    # Remove year in parentheses
    t = re.sub(r'\s*\(\d{4}\)\s*', ' ', t)
    # Remove season/part suffixes
    t = re.sub(r':\s*(season|part|volume|chapter)\s*\d+.*$', '', t, flags=re.IGNORECASE)
    # Remove special chars
    t = re.sub(r'[^a-z0-9\s]', '', t)
    # Collapse spaces
    t = re.sub(r'\s+', ' ', t).strip()
    return t if t else None


def normalize_genre(genre):
    """Normalize genre string to set of genres."""
    if pd.isna(genre):
        return set()
    genres = re.split(r'[,;|/]', str(genre).lower())
    return {g.strip() for g in genres if g.strip()}


def normalize_cast(cast):
    """Normalize cast string to set of names."""
    if pd.isna(cast):
        return set()
    # Split by common delimiters
    names = re.split(r'[,;|]', str(cast).lower())
    # Take first 3 names (main cast)
    result = set()
    for name in names[:5]:
        name = re.sub(r'[^a-z\s]', '', name.strip())
        if name and len(name) > 2:
            result.add(name)
    return result


def normalize_date(date_val, require_month_day=True):
    """Normalize date to YYYY-MM-DD format."""
    if pd.isna(date_val):
        return None

    date_str = str(date_val).strip()

    # Skip year-only if we require month/day
    if require_month_day and re.match(r'^\d{4}$', date_str):
        return None

    # Already in YYYY-MM-DD format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str

    # YYYY only -> YYYY-01-01 (only if not requiring month/day)
    if re.match(r'^\d{4}$', date_str):
        return f"{date_str}-01-01"

    # DD-MM-YY format (e.g., 07-07-24, 23-09-1994)
    match = re.match(r'^(\d{2})-(\d{2})-(\d{2,4})$', date_str)
    if match:
        d, m, y = match.groups()
        if len(y) == 2:
            y = int(y)
            y = 1900 + y if y > 50 else 2000 + y
        return f"{y}-{m}-{d}"

    # MM/DD/YYYY or M/D/YYYY or MM/DD/YY
    match = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{2,4})$', date_str)
    if match:
        m, d, y = match.groups()
        if len(y) == 2:
            y = int(y)
            y = 1900 + y if y > 50 else 2000 + y
        return f"{y}-{m.zfill(2)}-{d.zfill(2)}"

    # Try pandas parsing
    try:
        parsed = pd.to_datetime(date_str, errors='coerce', dayfirst=True)
        if pd.notna(parsed):
            return parsed.strftime('%Y-%m-%d')
    except:
        pass

    return None


def get_title_type(is_tv_file):
    """Get title type based on file."""
    return 'tv_show' if is_tv_file else 'movie'


def build_comprehensive_lookup():
    """Build comprehensive lookup with multiple matching keys."""
    print("Building comprehensive lookup (ALL records)...")

    # Multiple lookup dictionaries
    by_title = {}  # norm_title -> date
    by_title_year = {}  # (norm_title, year) -> date
    by_title_type_year = {}  # (norm_title, type, year) -> date
    by_title_genre = defaultdict(list)  # norm_title -> [(genres, date), ...]
    by_title_cast = defaultdict(list)  # norm_title -> [(cast_set, date), ...]

    # Load IMDB Seasons Allocated (best source with all fields)
    print("  Loading IMDB Seasons Allocated...")
    imdb_path = IMDB_DIR / "IMDB Seasons Allocated.csv"
    if imdb_path.exists():
        df = pd.read_csv(imdb_path, low_memory=False)
        print(f"    {len(df)} rows")

        for _, row in df.iterrows():
            title = row.get('title')
            orig_title = row.get('original_title')
            title_type = row.get('title_type')
            start_year = row.get('start_year')
            genres = row.get('genres')
            cast = row.get('cast')
            directors = row.get('directors')

            # Use start_year as date proxy
            date = normalize_date(start_year)
            if not date:
                continue

            for t in [title, orig_title]:
                norm_title = normalize_title(t)
                if not norm_title:
                    continue

                # Simple title lookup
                if norm_title not in by_title:
                    by_title[norm_title] = date

                # Title + year
                if pd.notna(start_year):
                    year = int(start_year)
                    key = (norm_title, year)
                    if key not in by_title_year:
                        by_title_year[key] = date

                # Title + type + year
                if pd.notna(title_type) and pd.notna(start_year):
                    t_type = 'tv_show' if 'tv' in str(title_type).lower() else 'movie'
                    year = int(start_year)
                    key = (norm_title, t_type, year)
                    if key not in by_title_type_year:
                        by_title_type_year[key] = date

                # Title + genre
                norm_genres = normalize_genre(genres)
                if norm_genres:
                    by_title_genre[norm_title].append((norm_genres, date))

                # Title + cast/directors
                norm_cast = normalize_cast(cast)
                norm_dirs = normalize_cast(directors)
                combined_cast = norm_cast | norm_dirs
                if combined_cast:
                    by_title_cast[norm_title].append((combined_cast, date))

    # Load Rating Graph files
    print("  Loading Rating Graph files...")

    # Date columns to check (priority order) - skip tmdb_last_air_date (year only)
    rg_date_cols = ['Release Date', 'Release_Date', 'RT_Release Date (Streaming)',
                    'release_date_streaming', 'PREMIERE', 'Start Year']

    for i in range(1, 5):
        filepath = RG_DIR / f"RG_AGG_p{i}.csv"
        if filepath.exists():
            print(f"    {filepath.name}...")
            try:
                df = pd.read_csv(filepath, low_memory=False)

                for _, row in df.iterrows():
                    title = row.get('Title')
                    start_year = row.get('Start Year')
                    genres = row.get('Genres')
                    directors = row.get('Directors')
                    producers = row.get('Producers')
                    actors = row.get('ppb_Top 5 Actors')

                    # Get date from various columns (require month/day)
                    date = None
                    for col in rg_date_cols:
                        if col in df.columns and pd.notna(row.get(col)):
                            date = normalize_date(row.get(col), require_month_day=True)
                            if date:
                                break

                    if not date:
                        continue

                    norm_title = normalize_title(title)
                    if not norm_title:
                        continue

                    if norm_title not in by_title:
                        by_title[norm_title] = date

                    if pd.notna(start_year):
                        try:
                            year = int(float(start_year))
                            key = (norm_title, year)
                            if key not in by_title_year:
                                by_title_year[key] = date
                        except:
                            pass

                    norm_genres = normalize_genre(genres)
                    if norm_genres:
                        by_title_genre[norm_title].append((norm_genres, date))

                    combined_cast = normalize_cast(actors) | normalize_cast(directors) | normalize_cast(producers)
                    if combined_cast:
                        by_title_cast[norm_title].append((combined_cast, date))

            except Exception as e:
                print(f"      ERROR: {e}")

    # Load FlixPatrol TV aggregate (has PREMIERE column)
    filepath = RG_DIR / "flixpatrol_TV_RG_aggregate.csv"
    if filepath.exists():
        print(f"    {filepath.name}...")
        try:
            df = pd.read_csv(filepath, low_memory=False)

            for _, row in df.iterrows():
                title = row.get('Title')
                premiere = row.get('PREMIERE')
                start_year = row.get('Start Year')

                date = normalize_date(premiere, require_month_day=True)
                if not date:
                    continue

                norm_title = normalize_title(title)
                if not norm_title:
                    continue

                if norm_title not in by_title:
                    by_title[norm_title] = date

                if pd.notna(start_year):
                    try:
                        year = int(float(start_year))
                        key = (norm_title, year)
                        if key not in by_title_year:
                            by_title_year[key] = date
                    except:
                        pass

        except Exception as e:
            print(f"      ERROR: {e}")

    # Load TVShows_RG_Aggregated_Consolidated_v2.csv
    filepath = RG_DIR / "TVShows_RG_Aggregated_Consolidated_v2.csv"
    if filepath.exists():
        print(f"    {filepath.name}...")
        try:
            df = pd.read_csv(filepath, low_memory=False)

            for _, row in df.iterrows():
                title = row.get('Title')
                start_year = row.get('Start Year')

                date = None
                for col in rg_date_cols:
                    if col in df.columns and pd.notna(row.get(col)):
                        date = normalize_date(row.get(col), require_month_day=True)
                        if date:
                            break

                if not date:
                    continue

                norm_title = normalize_title(title)
                if not norm_title:
                    continue

                if norm_title not in by_title:
                    by_title[norm_title] = date

                if pd.notna(start_year):
                    try:
                        year = int(float(start_year))
                        key = (norm_title, year)
                        if key not in by_title_year:
                            by_title_year[key] = date
                    except:
                        pass

        except Exception as e:
            print(f"      ERROR: {e}")

    # Rating Graph Consolidated v4
    filepath = RG_DIR / "Rating Graph Consolidated v4.csv"
    if filepath.exists():
        print(f"    {filepath.name}...")
        try:
            df = pd.read_csv(filepath, low_memory=False)

            for _, row in df.iterrows():
                title = row.get('Title')
                year_val = row.get('Year')

                date = None
                for col in ['Release_Date', 'release_date_streaming']:
                    if col in df.columns and pd.notna(row.get(col)):
                        date = normalize_date(row.get(col), require_month_day=True)
                        if date:
                            break

                if not date:
                    continue

                norm_title = normalize_title(title)
                if not norm_title:
                    continue

                if norm_title not in by_title:
                    by_title[norm_title] = date

                if pd.notna(year_val):
                    try:
                        year = int(float(year_val))
                        key = (norm_title, year)
                        if key not in by_title_year:
                            by_title_year[key] = date
                    except:
                        pass

        except Exception as e:
            print(f"      ERROR: {e}")

    # IMDB 2024 TOP 500 CLEAN.csv (has Release_Date)
    filepath = IMDB_DIR / "IMDB 2024 TOP 500 CLEAN.csv"
    if filepath.exists():
        print(f"    {filepath.name}...")
        try:
            df = pd.read_csv(filepath, low_memory=False)

            for _, row in df.iterrows():
                title = row.get('Title') or row.get('title')
                release_date = row.get('Release_Date')

                date = normalize_date(release_date, require_month_day=True)
                if not date:
                    continue

                norm_title = normalize_title(title)
                if not norm_title:
                    continue

                if norm_title not in by_title:
                    by_title[norm_title] = date

        except Exception as e:
            print(f"      ERROR: {e}")

    # Load JustWatch
    print("  Loading JustWatch files...")
    for filepath in JW_DIR.glob('justwatch_scraped_output_us*.csv'):
        print(f"    {filepath.name}...")
        try:
            df = pd.read_csv(filepath, low_memory=False)

            for _, row in df.iterrows():
                title = row.get('title')
                date_created = row.get('dateCreated')
                content_type = row.get('type')
                director = row.get('director')

                date = normalize_date(date_created)
                if not date:
                    continue

                norm_title = normalize_title(title)
                if not norm_title:
                    continue

                if norm_title not in by_title:
                    by_title[norm_title] = date

                # Extract year from date
                try:
                    year = int(date[:4])
                    key = (norm_title, year)
                    if key not in by_title_year:
                        by_title_year[key] = date

                    if pd.notna(content_type):
                        t_type = 'tv_show' if 'show' in str(content_type).lower() else 'movie'
                        key = (norm_title, t_type, year)
                        if key not in by_title_type_year:
                            by_title_type_year[key] = date
                except:
                    pass

                norm_dirs = normalize_cast(director)
                if norm_dirs:
                    by_title_cast[norm_title].append((norm_dirs, date))

        except Exception as e:
            print(f"      ERROR: {e}")

    print(f"\n  Lookup sizes:")
    print(f"    by_title: {len(by_title)}")
    print(f"    by_title_year: {len(by_title_year)}")
    print(f"    by_title_type_year: {len(by_title_type_year)}")
    print(f"    by_title_genre: {len(by_title_genre)}")
    print(f"    by_title_cast: {len(by_title_cast)}")

    return by_title, by_title_year, by_title_type_year, by_title_genre, by_title_cast


def find_date(title, season_num, content_category, is_tv,
              by_title, by_title_year, by_title_type_year, by_title_genre, by_title_cast):
    """Try multiple matching strategies to find date."""

    norm_title = normalize_title(title)
    if not norm_title:
        return None, 'no_title'

    title_type = 'tv_show' if is_tv else 'movie'

    # Strategy 1: title + type + year (try multiple years)
    for year in range(2025, 1990, -1):
        key = (norm_title, title_type, year)
        if key in by_title_type_year:
            return by_title_type_year[key], 'title_type_year'

    # Strategy 2: title + year (any type)
    for year in range(2025, 1990, -1):
        key = (norm_title, year)
        if key in by_title_year:
            return by_title_year[key], 'title_year'

    # Strategy 3: title + genre match
    if norm_title in by_title_genre:
        # Use content_category as genre hint if available
        if pd.notna(content_category):
            target_genres = normalize_genre(content_category)
            if target_genres:
                best_match = None
                best_overlap = 0
                for genres, date in by_title_genre[norm_title]:
                    overlap = len(target_genres & genres)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = date
                if best_match:
                    return best_match, 'title_genre'

        # Just use first genre match
        if by_title_genre[norm_title]:
            return by_title_genre[norm_title][0][1], 'title_genre_first'

    # Strategy 4: title + cast overlap
    if norm_title in by_title_cast and by_title_cast[norm_title]:
        return by_title_cast[norm_title][0][1], 'title_cast'

    # Strategy 5: Simple title match
    if norm_title in by_title:
        return by_title[norm_title], 'title_only'

    return None, 'not_found'


def update_missing_dates(filepath, by_title, by_title_year, by_title_type_year, by_title_genre, by_title_cast):
    """Update missing first_air_date in xlsx file."""
    print(f"\nProcessing: {filepath.name}")
    df = pd.read_excel(filepath)

    is_tv = "netflix_tv_" in filepath.name

    stats = defaultdict(int)

    for idx, row in df.iterrows():
        current_date = row.get('first_air_date')

        if pd.notna(current_date):
            stats['already_has'] += 1
            continue

        title = row.get('title')
        season_num = row.get('season_number')
        content_category = row.get('content_category')

        date, source = find_date(
            title, season_num, content_category, is_tv,
            by_title, by_title_year, by_title_type_year, by_title_genre, by_title_cast
        )

        if date:
            df.at[idx, 'first_air_date'] = date
            stats[source] += 1
        else:
            stats['not_found'] += 1

    # Save back
    df.to_excel(filepath, index=False)

    print(f"  Already had: {stats['already_has']}")
    print(f"  title_type_year: {stats['title_type_year']}")
    print(f"  title_year: {stats['title_year']}")
    print(f"  title_genre: {stats['title_genre']} + first: {stats['title_genre_first']}")
    print(f"  title_cast: {stats['title_cast']}")
    print(f"  title_only: {stats['title_only']}")
    print(f"  NOT FOUND: {stats['not_found']}")

    total_updated = sum(v for k, v in stats.items() if k not in ['already_has', 'not_found', 'no_title'])
    return total_updated, stats['not_found']


def main():
    print("=" * 70)
    print("Enhanced Date Filling - Multiple Matching Strategies")
    print("=" * 70)

    # Build lookups
    by_title, by_title_year, by_title_type_year, by_title_genre, by_title_cast = build_comprehensive_lookup()

    # Process all 2.30 files
    total_updated = 0
    total_missing = 0

    for filepath in sorted(OUTPUT_DIR.glob("*.xlsx")):
        u, m = update_missing_dates(filepath, by_title, by_title_year, by_title_type_year, by_title_genre, by_title_cast)
        total_updated += u
        total_missing += m

    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"  Total updated: {total_updated}")
    print(f"  Total still missing: {total_missing}")
    print("=" * 70)


if __name__ == "__main__":
    main()
