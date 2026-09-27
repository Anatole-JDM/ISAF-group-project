"""Read-only geography and plate profiling for Nashville, on the cleaned stops.

Cleaning applied first (protocol Part 2.0): drop NA search_conducted, drop the
3 key duplicates, drop 2019-03. Nothing is written except diag_geo.json.
"""
import zipfile, io, json, re
import pandas as pd
import pyarrow as pa
from pyarrow import csv as pacsv

ROOT = 'C:/Users/smaye/OneDrive/Bureau/X-HEC/HEC_courses/algo_fairness/ISAF_gp/ISAF-group-project/'
P = ROOT + 'opp_data/tn_nashville_2020_04_01.csv.zip'
OUT = ROOT + 'scripts_claude/diag_geo.json'
use = ['date', 'time', 'location', 'lat', 'lng', 'precinct', 'zone', 'reporting_area',
       'subject_age', 'subject_sex', 'subject_race', 'officer_id_hash', 'search_conducted',
       'contraband_found', 'vehicle_registration_state', 'raw_search_consent',
       'raw_search_arrest', 'raw_search_warrant', 'raw_search_inventory', 'raw_search_plain_view']
z = zipfile.ZipFile(P)
with z.open(z.infolist()[0]) as f:
    df = pacsv.read_csv(f, read_options=pacsv.ReadOptions(block_size=1 << 26),
                        convert_options=pacsv.ConvertOptions(
                            include_columns=use, column_types={c: pa.string() for c in use})).to_pandas()


def B(s):
    return s.str.lower().eq('true')


def NA(s):
    return s.isna() | s.eq('') | s.isin(['NA', 'unknown', 'n/a'])


# --- cleaning A1, A2, A4 ---
n0 = len(df)
df = df[~NA(df['search_conducted'])]
df = df[~df.duplicated(subset=['date', 'time', 'officer_id_hash', 'subject_age',
                               'subject_sex', 'subject_race', 'location'])]
df['d'] = pd.to_datetime(df['date'], errors='coerce')
df = df[df['d'] < '2019-03-01']
df['year'] = df['d'].dt.year
print(f'rows {n0:,} -> {len(df):,} after cleaning', flush=True)
R = {'rows_after_cleaning': int(len(df))}

srch = B(df['search_conducted'])
flags = ['raw_search_arrest', 'raw_search_warrant', 'raw_search_inventory', 'raw_search_plain_view']
consent = srch & B(df['raw_search_consent']) & ~pd.concat([B(df[c]) for c in flags], axis=1).any(axis=1)
hit = B(df['contraband_found'])
R['consent_exclusive_n'] = int(consent.sum())

# --- 1. what are the most frequent coordinates? ---
lat = pd.to_numeric(df['lat'], errors='coerce')
lng = pd.to_numeric(df['lng'], errors='coerce')
df['c4'] = list(zip(lat.round(4), lng.round(4)))
top = df.loc[lat.notna(), 'c4'].value_counts().head(15)
rows = []
for c, n in top.items():
    sub = df[df['c4'] == c]
    rows.append({
        'coord': [float(c[0]), float(c[1])],
        'stops': int(n), 'share_pct': round(100 * n / len(df), 2),
        'share_of_searches_pct': round(100 * float(srch[sub.index].sum()) / max(int(srch.sum()), 1), 2),
        'n_distinct_location_strings': int(sub['location'].nunique()),
        'top_location_strings': {str(k): int(v) for k, v in sub['location'].value_counts().head(3).items()},
        'precinct_mode': str(sub['precinct'].mode().iat[0]) if sub['precinct'].notna().any() else None,
        'years': {int(k): int(v) for k, v in sub['year'].value_counts().sort_index().items()},
    })
R['top_coords'] = rows
print('top coords done', flush=True)

# --- 2. coordinate precision regime by year ---
dec = df['lat'].where(lat.notna()).str.split('.').str[1].str.len()
R['decimals_by_year'] = {int(y): {int(k): int(v) for k, v in g.value_counts().items()}
                         for y, g in dec.groupby(df['year'])}
R['latlng_missing_by_year'] = {int(y): round(100 * float(g.isna().mean()), 1)
                               for y, g in lat.groupby(df['year'])}
box = lat.between(35.9, 36.5) & lng.between(-87.2, -86.4)
out = df[(~box) & lat.notna()]
R['outside_box'] = {'n': int(len(out)),
                    'lat_range': [float(lat[out.index].min()), float(lat[out.index].max())],
                    'lng_range': [float(lng[out.index].min()), float(lng[out.index].max())],
                    'sample': [[float(a), float(b)] for a, b in zip(lat[out.index].head(8), lng[out.index].head(8))]}
print('precision done', flush=True)

# --- 3. what does `location` contain? (road-network features without external data) ---
loc = df['location'].fillna('').str.upper()
R['location_n_distinct'] = int(loc.nunique())
R['location_samples'] = [str(x) for x in df['location'].dropna().sample(20, random_state=0)]
patterns = {
    'I-24': r'\bI[- ]?24\b', 'I-40': r'\bI[- ]?40\b', 'I-65': r'\bI[- ]?65\b', 'I-440': r'\bI[- ]?440\b',
    'any_interstate': r'\bI[- ]?(24|40|65|440)\b|INTERSTATE',
    'Briley_Pkwy': r'BRILEY', 'Ellington_Pkwy': r'ELLINGTON',
    'intersection_marker': r'&|/| AT | AND ',
    'house_number_start': r'^\d+ ',
}
R['location_patterns'] = {}
for k, p in patterns.items():
    m = loc.str.contains(p, regex=True)
    R['location_patterns'][k] = {
        'stops_pct': round(100 * float(m.mean()), 2),
        'searches_pct': round(100 * float(m[srch].mean()), 2),
        'consent_hit_pct': round(100 * float(hit[consent & m].mean()), 1) if (consent & m).sum() > 50 else None,
        'consent_n': int((consent & m).sum()),
    }
print('location done', flush=True)

# --- 4. registration state ---
rs = df['vehicle_registration_state'].fillna('NA').str.upper().str.strip()
vc = rs.value_counts()
R['regstate_n_distinct'] = int(len(vc))
R['regstate_top30'] = {str(k): int(v) for k, v in vc.head(30).items()}
R['regstate_rare_codes'] = {str(k): int(v) for k, v in vc[vc < 20].items()}
R['regstate_TN_share_by_year'] = {int(y): round(100 * float(g.eq('TN').mean()), 1)
                                  for y, g in rs.groupby(df['year'])}
bord = ['KY', 'AL', 'GA', 'MS', 'AR', 'MO', 'NC', 'VA']
grp = pd.Series('other_state', index=df.index)
grp[rs.eq('TN')] = 'TN'
grp[rs.isin(bord)] = 'border_state'
grp[NA(df['vehicle_registration_state'])] = 'missing'
R['regstate_group'] = {}
for g in ['TN', 'border_state', 'other_state', 'missing']:
    m = grp.eq(g)
    R['regstate_group'][g] = {
        'stops': int(m.sum()),
        'search_rate_pct': round(100 * float(srch[m].mean()), 2),
        'consent_n': int((consent & m).sum()),
        'consent_hit_pct': round(100 * float(hit[consent & m].mean()), 1) if (consent & m).sum() > 50 else None,
    }
print('regstate done', flush=True)

json.dump(R, open(OUT, 'w'), indent=1, default=str)
print('ALL DONE', flush=True)
