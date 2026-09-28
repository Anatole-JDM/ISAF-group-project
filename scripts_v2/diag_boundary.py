"""How many stops sit on (or next to) a census-tract boundary?  Decides D2.

Read-only. Applies cleaning A1/A2/A4 in memory, keeps coordinates inside the
Davidson bounding box, assigns each stop to its 2010 tract (point-in-polygon)
and measures its distance to the nearest tract boundary, in metres.
Writes diag_boundary.json only.
"""
import io, json, zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import csv as pacsv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'opp_data' / 'tn_nashville_2020_04_01.csv.zip'
TIGER = ROOT / 'opp_data' / 'external' / 'tl_2010_47_tract10.zip'
OUT = ROOT / 'scripts_claude' / 'diag_boundary.json'
METRIC = 'EPSG:32616'   # UTM 16N, metres - covers Nashville

use = ['date', 'time', 'officer_id_hash', 'subject_age', 'subject_sex', 'subject_race',
       'location', 'lat', 'lng', 'search_conducted', 'contraband_found',
       'raw_search_consent', 'raw_search_arrest', 'raw_search_warrant',
       'raw_search_inventory', 'raw_search_plain_view']
z = zipfile.ZipFile(SRC)
with z.open(z.infolist()[0]) as f:
    df = pacsv.read_csv(f, read_options=pacsv.ReadOptions(block_size=1 << 26),
                        convert_options=pacsv.ConvertOptions(
                            include_columns=use, column_types={c: pa.string() for c in use})).to_pandas()


def B(s):
    return s.str.lower().eq('true')


# cleaning A1, A2, A4 (in memory only)
df = df[~(df['search_conducted'].isna() | df['search_conducted'].isin(['', 'NA']))]
df = df[~df.duplicated(subset=['date', 'time', 'officer_id_hash', 'subject_age',
                               'subject_sex', 'subject_race', 'location'])]
df = df[pd.to_datetime(df['date'], errors='coerce') < '2019-03-01']
n_clean = len(df)

lat = pd.to_numeric(df['lat'], errors='coerce')
lng = pd.to_numeric(df['lng'], errors='coerce')
ok = lat.between(35.9, 36.5) & lng.between(-87.2, -86.4)
df = df[ok].copy()
df['lat'], df['lng'] = lat[ok], lng[ok]
srch = B(df['search_conducted'])
others = ['raw_search_arrest', 'raw_search_warrant', 'raw_search_inventory', 'raw_search_plain_view']
df['consent_excl'] = srch & B(df['raw_search_consent']) & ~pd.concat([B(df[c]) for c in others], axis=1).any(axis=1)
loc = df['location'].fillna('').str.upper()
df['kind'] = np.where(loc.str.contains('&', regex=False), 'intersection',
              np.where(loc.str.match(r'^\d+ '), 'address', 'other'))
df['interstate'] = loc.str.contains(r'\bI[- ]?(?:24|40|65|440)\b|INTERSTATE', regex=True)
print(f'clean rows {n_clean:,}; with usable coordinates {len(df):,}', flush=True)

# unique coordinates (6 dp ~ 0.1 m) to keep the geometry work small
df['key'] = df['lat'].round(6).astype(str) + ',' + df['lng'].round(6).astype(str)
pts = df.groupby('key').agg(lat=('lat', 'first'), lng=('lng', 'first')).reset_index()
g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts['lng'], pts['lat']), crs='EPSG:4326').to_crs(METRIC)
print(f'unique coordinates {len(g):,}', flush=True)

tracts = gpd.read_file(f'zip://{TIGER}').to_crs(METRIC)
dav = tracts[tracts['COUNTYFP10'] == '037']
R = {'rows_clean': int(n_clean), 'rows_with_coords_in_box': int(len(df)),
     'unique_coords': int(len(g)), 'davidson_tracts_2010': int(len(dav)),
     'tiger_crs_original': 'EPSG:4269 (NAD83)'}

# point-in-polygon against all TN tracts (so near-county-line points still resolve)
pip = gpd.sjoin(g, tracts[['GEOID10', 'COUNTYFP10', 'geometry']], how='left', predicate='within')
pip = pip[~pip.index.duplicated(keep='first')]
g['GEOID10'] = pip['GEOID10']
g['COUNTYFP10'] = pip['COUNTYFP10']

# distance to the nearest tract boundary (tracts near the box only)
near = tracts[tracts.intersects(g.union_all().envelope.buffer(5000))]
lines = near.boundary.explode(index_parts=False).reset_index(drop=True)
lines = gpd.GeoDataFrame(geometry=lines, crs=METRIC)
nn = gpd.sjoin_nearest(g[['geometry']], lines, how='left', distance_col='dist_m')
nn = nn[~nn.index.duplicated(keep='first')]
g['dist_m'] = nn['dist_m']
print('distances done', flush=True)

df = df.merge(g[['key', 'GEOID10', 'COUNTYFP10', 'dist_m']], on='key', how='left')
R['stops_assigned_to_tract_pct'] = round(100 * float(df['GEOID10'].notna().mean()), 2)
R['stops_in_davidson_tracts_pct'] = round(100 * float(df['COUNTYFP10'].eq('037').mean()), 2)
R['stops_in_other_tn_counties'] = int((df['GEOID10'].notna() & df['COUNTYFP10'].ne('037')).sum())
R['distinct_tracts_with_stops'] = int(df['GEOID10'].nunique())

TH = [10, 25, 50, 100, 200, 400]


def within(sub):
    return {f'<= {t} m': round(100 * float((sub['dist_m'] <= t).mean()), 1) for t in TH}


R['share_within_distance_of_boundary'] = {
    'all_stops': within(df),
    'consent_exclusive_searches': within(df[df['consent_excl']]),
    'intersections': within(df[df['kind'] == 'intersection']),
    'addresses': within(df[df['kind'] == 'address']),
    'interstate': within(df[df['interstate']]),
    'non_interstate': within(df[~df['interstate']]),
}
R['dist_m_quantiles_all'] = {str(q): round(float(df['dist_m'].quantile(q)), 1) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
tc = df.groupby('GEOID10').size()
R['stops_per_tract_quantiles'] = {str(q): int(tc.quantile(q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]}
cs = df[df['consent_excl']].groupby('GEOID10').size()
R['consent_searches_per_tract_quantiles'] = {str(q): int(cs.quantile(q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]}
R['tracts_with_lt30_consent_searches'] = int((cs < 30).sum())

OUT.write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
