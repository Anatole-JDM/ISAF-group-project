"""Fetch 2010 census-block populations for Davidson County (weights for the 800 m circles).

The TIGER block file (tl_2010_47037_tabblock10.zip) has geometry and interior points
but no population. This pulls P001001 (total population, 2010 Census SF1) for every
Davidson block and checks it against the shapefile and the official county total.

Why Davidson only: an 800 m circle reaches outside the county for 1.05% of stops
(0.46% of consent searches); those get a truncation flag in the build step rather
than a 199 MB statewide download.

Usage:   python scripts_claude/fetch_block_pop.py      (needs CENSUS_API_KEY)
Output:  opp_data/external/block_pop_2010_davidson.parquet  (GEOID10, pop10)
"""
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_acs import get  # same key handling and redaction

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = ROOT / 'opp_data' / 'external' / 'tl_2010_47037_tabblock10.zip'
OUT = ROOT / 'opp_data' / 'external' / 'block_pop_2010_davidson.parquet'
API = 'https://api.census.gov/data/2010/dec/sf1'
DAVIDSON_2010_CENSUS_POP = 626_681     # official 2010 Census count, Davidson County, TN


def rows_to_df(rows):
    d = pd.DataFrame(rows[1:], columns=rows[0])
    d['GEOID10'] = d['state'] + d['county'] + d['tract'] + d['block']
    return d[['GEOID10', 'P001001']].rename(columns={'P001001': 'pop10'})


def main():
    key = os.environ.get('CENSUS_API_KEY')
    if not key:
        sys.exit('CENSUS_API_KEY is not set.')
    shp = gpd.read_file(f'zip://{BLOCKS}', ignore_geometry=True)
    tracts = sorted(shp['TRACTCE10'].unique())

    try:   # one request with a tract wildcard, if the API accepts it
        rows = get(API, {'get': 'P001001', 'for': 'block:*',
                         'in': 'state:47 county:037 tract:*', 'key': key}).json()
        df = rows_to_df(rows)
        print('fetched with a single wildcard request', flush=True)
    except RuntimeError:   # otherwise one request per tract
        parts = []
        for i, t in enumerate(tracts, 1):
            rows = get(API, {'get': 'P001001', 'for': 'block:*',
                             'in': f'state:47 county:037 tract:{t}', 'key': key}).json()
            parts.append(rows_to_df(rows))
            if i % 40 == 0:
                print(f'  {i}/{len(tracts)} tracts', flush=True)
        df = pd.concat(parts, ignore_index=True)

    df['pop10'] = pd.to_numeric(df['pop10'], errors='raise').astype('int64')

    # checks: every shapefile block has a population and vice versa, total matches the census
    missing_pop = set(shp['GEOID10']) - set(df['GEOID10'])
    missing_geom = set(df['GEOID10']) - set(shp['GEOID10'])
    total = int(df['pop10'].sum())
    print(f'blocks: shapefile {len(shp):,}, api {len(df):,}; '
          f'without population {len(missing_pop)}, without geometry {len(missing_geom)}')
    print(f'Davidson population: {total:,} (official 2010 count {DAVIDSON_2010_CENSUS_POP:,})')
    print(f'blocks with zero population: {(df["pop10"] == 0).sum():,} of {len(df):,}')
    assert not missing_pop and not missing_geom, 'block sets differ'
    assert total == DAVIDSON_2010_CENSUS_POP, 'population total does not match the 2010 census'

    df.to_parquet(OUT, index=False)
    print(f'wrote {OUT.name}')


if __name__ == '__main__':
    main()
