"""
Merge authoritative first_air_date from senior file into Netflix 2.30 sheets.
- Cleans unicode issues by transliterating to ASCII
- Uses fc_uid as primary key for matching
- Updates first_air_date column in all 2.30 xlsx files
"""

import pandas as pd
import unicodedata
import re
from pathlib import Path

# Paths
SENIOR_FILE = Path(r"C:\Users\RoyT6\Downloads\netflix_2.20_with_dates_cleaned.csv")
OUTPUT_DIR = Path(r"C:\Users\RoyT6\Downloads\Training Data\netflix totals\Best Set\2.30")


def transliterate_to_ascii(text):
    """Convert unicode characters to closest ASCII equivalent."""
    if pd.isna(text):
        return text
    # Normalize unicode and convert to ASCII
    text = str(text)
    # NFD decomposes characters (é -> e + combining accent)
    normalized = unicodedata.normalize('NFD', text)
    # Remove combining characters (accents, diacritics)
    ascii_text = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    # Encode to ASCII, replacing any remaining non-ASCII
    ascii_text = ascii_text.encode('ascii', 'ignore').decode('ascii')
    return ascii_text


def normalize_fc_uid(fc_uid):
    """Normalize fc_uid for matching: remove _s## suffix variations."""
    if pd.isna(fc_uid):
        return None
    fc_uid = str(fc_uid).strip()
    # Remove season suffix if present (for films)
    # tt123456_s01 -> tt123456 (base) and also keep full for TV
    return fc_uid


def convert_date_format(date_str):
    """Convert DD-MM-YY to YYYY-MM-DD format."""
    if pd.isna(date_str):
        return None
    try:
        date_str = str(date_str).strip()
        # Parse DD-MM-YY
        parts = date_str.split('-')
        if len(parts) == 3:
            day, month, year = parts
            # Handle 2-digit year
            year = int(year)
            if year > 50:
                year = 1900 + year
            else:
                year = 2000 + year
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    except:
        pass
    return date_str


def load_and_clean_senior_file():
    """Load senior file with correct encoding and clean it."""
    print("Loading and cleaning senior file...")

    # Read with latin-1 encoding
    df = pd.read_csv(SENIOR_FILE, encoding='latin-1')
    print(f"  Total rows: {len(df)}")
    print(f"  With premiere_date: {df['premiere_date'].notna().sum()}")

    # Clean titles - transliterate to ASCII
    df['title_clean'] = df['title'].apply(transliterate_to_ascii)

    # Convert date format from DD-MM-YY to YYYY-MM-DD
    df['premiere_date_iso'] = df['premiere_date'].apply(convert_date_format)

    # Build lookup: fc_uid -> premiere_date
    # For TV shows with season suffix, extract base fc_uid too
    lookup_full = {}  # Full fc_uid (tt123_s01)
    lookup_base = {}  # Base fc_uid (tt123456)

    for _, row in df.iterrows():
        fc_uid = row['fc_uid']
        date = row['premiere_date_iso']

        if pd.isna(fc_uid) or pd.isna(date):
            continue

        fc_uid = str(fc_uid).strip()
        lookup_full[fc_uid] = date

        # Also store base (without season suffix) for films
        if '_s' not in fc_uid:
            lookup_base[fc_uid] = date

    print(f"  Built lookup with {len(lookup_full)} full fc_uids")
    print(f"  Built lookup with {len(lookup_base)} base fc_uids")

    # Save cleaned version
    cleaned_path = SENIOR_FILE.parent / "netflix_2.20_with_dates_ascii.csv"
    df[['fc_uid', 'title_clean', 'premiere_date_iso', 'finale_date']].to_csv(cleaned_path, index=False)
    print(f"  Saved cleaned file to: {cleaned_path}")

    return lookup_full, lookup_base


def merge_into_xlsx(filepath, lookup_full, lookup_base, is_tv=True):
    """Merge first_air_date from senior file into xlsx."""
    print(f"\nProcessing: {filepath.name}")
    df = pd.read_excel(filepath)

    updated = 0
    already_had = 0
    still_missing = 0

    # Check current first_air_date column
    if 'first_air_date' not in df.columns:
        print("  ERROR: first_air_date column not found!")
        return

    new_dates = []
    for idx, row in df.iterrows():
        current_date = row.get('first_air_date')
        fc_uid = row.get('fc_uid')

        # If already has date, keep it
        if pd.notna(current_date):
            new_dates.append(current_date)
            already_had += 1
            continue

        # Try to find date from senior file
        new_date = None
        if pd.notna(fc_uid):
            fc_uid_str = str(fc_uid).strip()

            # Try full fc_uid first (for TV with season)
            if fc_uid_str in lookup_full:
                new_date = lookup_full[fc_uid_str]
            # Try normalized version (remove leading zeros in season)
            elif re.sub(r'_s0*(\d+)$', r'_s\1', fc_uid_str) in lookup_full:
                new_date = lookup_full[re.sub(r'_s0*(\d+)$', r'_s\1', fc_uid_str)]
            # For films, try base fc_uid
            elif not is_tv and fc_uid_str in lookup_base:
                new_date = lookup_base[fc_uid_str]

        if new_date:
            new_dates.append(new_date)
            updated += 1
        else:
            new_dates.append(None)
            still_missing += 1

    # Update the column
    df['first_air_date'] = new_dates

    # Save back
    df.to_excel(filepath, index=False)

    print(f"  Already had date: {already_had}")
    print(f"  Updated from senior: {updated}")
    print(f"  Still missing: {still_missing}")

    return updated, still_missing


def main():
    print("=" * 60)
    print("Merging Senior Dates into Netflix 2.30")
    print("=" * 60)

    # Load and clean senior file
    lookup_full, lookup_base = load_and_clean_senior_file()

    # Process all 2.30 files
    total_updated = 0
    total_missing = 0

    for filepath in sorted(OUTPUT_DIR.glob("*.xlsx")):
        is_tv = "netflix_tv_" in filepath.name
        u, m = merge_into_xlsx(filepath, lookup_full, lookup_base, is_tv=is_tv)
        total_updated += u
        total_missing += m

    print("\n" + "=" * 60)
    print(f"COMPLETE")
    print(f"  Total updated from senior file: {total_updated}")
    print(f"  Total still missing: {total_missing}")
    print("=" * 60)


if __name__ == "__main__":
    main()
