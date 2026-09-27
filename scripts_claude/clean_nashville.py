"""Build the cleaned Nashville base table - the single input for every feature step.

Applies exactly the changelog in nashville_cleaning_protocol.md Part 2.0, nothing more:
  A1  drop rows where search_conducted is NA
  A2  drop duplicates on (date, time, officer_id_hash, subject_age, subject_sex,
      subject_race, location)
  A3  drop the column `violation` (duplicate of reason_for_stop)
  A4  drop 2019-03 (partial month; extract ends 2019-03-24)

Values are never altered: ages are kept as recorded, coordinates are kept even
when they fall outside Davidson County (a flag is added instead, so the decision
to discard them stays with the feature step). Only types are converted.

Outputs (under opp_data/, which is git-ignored):
  opp_data/processed/nashville_clean.parquet
  opp_data/processed/nashville_clean_manifest.json   (row counts per step, schema)
"""
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
from pyarrow import csv as pacsv

ROOT = Path(__file__).resolve().parents[1]
# the raw file may be kept zipped, or extracted (folder of the same name, or a bare CSV)
SRC_CANDIDATES = [ROOT / 'opp_data' / 'tn_nashville_2020_04_01.csv.zip',
                  ROOT / 'opp_data' / 'tn_nashville_2020_04_01.csv' / 'tn_nashville_2020_04_01.csv',
                  ROOT / 'opp_data' / 'tn_nashville_2020_04_01.csv']
OUT_DIR = ROOT / 'opp_data' / 'processed'
OUT = OUT_DIR / 'nashville_clean.parquet'
MANIFEST = OUT_DIR / 'nashville_clean_manifest.json'

# Expected counts, from the diagnostics. The script stops if reality differs.
EXPECTED = {'raw': 3_092_351, 'A1': 39, 'A2': 3, 'A4': 4_023, 'final': 3_088_286}

DUP_KEY = ['date', 'time', 'officer_id_hash', 'subject_age', 'subject_sex',
           'subject_race', 'location']
BOOL_COLS = ['arrest_made', 'citation_issued', 'warning_issued', 'contraband_found',
             'contraband_drugs', 'contraband_weapons', 'frisk_performed',
             'search_conducted', 'search_person', 'search_vehicle',
             'raw_verbal_warning_issued', 'raw_written_warning_issued',
             'raw_traffic_citation_issued', 'raw_misd_state_citation_issued',
             'raw_driver_searched', 'raw_passenger_searched', 'raw_search_consent',
             'raw_search_arrest', 'raw_search_warrant', 'raw_search_inventory',
             'raw_search_plain_view']
NUM_COLS = ['subject_age', 'lat', 'lng']
DAVIDSON_BOX = {'lat': (35.9, 36.5), 'lng': (-87.2, -86.4)}


def is_missing(s):
    return s.isna() | s.isin(['', 'NA'])


def to_bool(s):
    """'TRUE'/'FALSE' strings -> nullable boolean; anything else -> <NA>."""
    low = s.str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype='boolean')
    out[low.eq('true')] = True
    out[low.eq('false')] = False
    return out


def find_source():
    for p in SRC_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError('raw Nashville file not found; looked for:\n  '
                            + '\n  '.join(str(p) for p in SRC_CANDIDATES))


def read_raw(src):
    """Read every column as text, from the zip or the extracted CSV."""
    if src.suffix == '.zip':
        z = zipfile.ZipFile(src)
        info = z.infolist()[0]
        with z.open(info) as f:
            cols = next(io.TextIOWrapper(f, 'utf8', errors='replace')).strip().split(',')
        opener = lambda: z.open(info)
    else:
        with open(src, encoding='utf-8-sig', errors='replace') as f:
            cols = next(f).strip().split(',')
        opener = lambda: open(src, 'rb')
    with opener() as f:
        return pacsv.read_csv(
            f, read_options=pacsv.ReadOptions(block_size=1 << 26),
            convert_options=pacsv.ConvertOptions(
                column_types={c: pa.string() for c in cols})).to_pandas()


def main():
    src = find_source()
    print(f'source: {src.relative_to(ROOT)}', flush=True)
    df = read_raw(src)

    log = {'source': str(src.relative_to(ROOT)), 'steps': []}

    def record(step, removed, rule):
        log['steps'].append({'step': step, 'rule': rule, 'rows_removed': int(removed),
                             'rows_after': int(len(df))})
        print(f'{step}: removed {removed:,} -> {len(df):,} rows', flush=True)

    assert len(df) == EXPECTED['raw'], f"raw rows {len(df):,} != {EXPECTED['raw']:,}"
    log['rows_raw'] = int(len(df))

    # A1 - search_conducted NA
    m = is_missing(df['search_conducted'])
    df = df[~m]
    record('A1', m.sum(), 'drop rows where search_conducted is NA')
    assert m.sum() == EXPECTED['A1']

    # A2 - duplicates on the key (first occurrence kept)
    m = df.duplicated(subset=DUP_KEY, keep='first')
    df = df[~m]
    record('A2', m.sum(), 'drop duplicates on ' + ', '.join(DUP_KEY) + ' (keep first)')
    assert m.sum() == EXPECTED['A2']

    # A3 - duplicate column
    df = df.drop(columns=['violation'])
    log['steps'].append({'step': 'A3', 'rule': 'drop column violation (= reason_for_stop)',
                         'rows_removed': 0, 'rows_after': int(len(df))})
    print('A3: dropped column violation', flush=True)

    # A4 - partial final month
    date = pd.to_datetime(df['date'], errors='coerce')
    m = date >= '2019-03-01'
    df = df[~m]
    record('A4', m.sum(), 'drop stops dated 2019-03-01 or later (partial month)')
    assert m.sum() == EXPECTED['A4']
    assert len(df) == EXPECTED['final'], f"final rows {len(df):,} != {EXPECTED['final']:,}"

    # --- type conversion only; no value is changed ---
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for c in BOOL_COLS:
        if c in df:
            df[c] = to_bool(df[c])
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c].where(~is_missing(df[c])), errors='coerce')
    # Text columns: literal 'NA' / '' -> real missing. pandas 3 stores text as `str`, not `object`,
    # so test with is_string_dtype (an `== object` test silently skipped every text column).
    na_converted = {}
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            m = is_missing(df[c]) & df[c].notna()
            if m.any():
                na_converted[c] = int(m.sum())
            df[c] = df[c].where(~is_missing(df[c])).astype('string')
    leftover = {c: int(df[c].isin(['NA', '']).sum()) for c in df.columns
                if pd.api.types.is_string_dtype(df[c]) and df[c].isin(['NA', '']).any()}
    assert not leftover, f"literal 'NA' / '' left in text columns: {leftover}"
    log['text_NA_converted_to_missing'] = na_converted

    # --- flags and a stable key (added columns, originals untouched) ---
    df = df.reset_index(drop=True)
    df.insert(0, 'stop_id', range(len(df)))
    inside = (df['lat'].between(*DAVIDSON_BOX['lat']) & df['lng'].between(*DAVIDSON_BOX['lng']))
    df['geo_outside_davidson_box'] = (df['lat'].notna() & ~inside).astype('boolean')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    log['rows_final'] = int(len(df))
    log['added_columns'] = {'stop_id': 'row key, 0..n-1, stable across feature steps',
                            'geo_outside_davidson_box': 'lat/lng present but outside '
                            f"lat {DAVIDSON_BOX['lat']}, lng {DAVIDSON_BOX['lng']}"}
    log['n_outside_davidson_box'] = int(df['geo_outside_davidson_box'].sum())
    log['schema'] = {c: str(t) for c, t in df.dtypes.items()}
    MANIFEST.write_text(json.dumps(log, indent=1))
    print(f'wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) and {MANIFEST.name}', flush=True)


if __name__ == '__main__':
    main()
