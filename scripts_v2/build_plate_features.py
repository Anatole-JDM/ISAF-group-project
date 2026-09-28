"""Step 3 - registration-plate features for every Nashville stop.

Features (feature_engineering_plan.md, Step 3):
  plate_state          USPS code of the plate state; 'missing' when not recorded
  plate_missing        plate state not recorded (kept as its own signal: highest search rate)
  plate_out_of_state   plate state recorded and not TN (<NA> when missing)
  plate_group          TN / border_state / other_state / missing   (TN borders 8 states)
  plate_region         TN / South (other) / Midwest / Northeast / West / missing  (Census regions)
  plate_distance_km    great-circle distance from the plate state's 2010 centre of population
                       to downtown Nashville; <NA> when missing

Not built here: `plate_state_cannabis_legal_at_stop`. The plan lists it as a hypothesis only
(ethically loaded, thin cells, dates need sourcing); it needs a team decision first.

The report is descriptive only (counts, search rates, hit rates) - no model is fitted.

Inputs : opp_data/processed/nashville_clean.parquet
         opp_data/external/CenPop2010_Mean_ST.txt   (Census 2010 state centres of population)
Outputs: opp_data/features/plate_features.parquet   one row per stop_id
         opp_data/features/plate_manifest.json      coverage, drift, descriptive rates, thin cells
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STOPS = ROOT / 'opp_data' / 'processed' / 'nashville_clean.parquet'
CENPOP = ROOT / 'opp_data' / 'external' / 'CenPop2010_Mean_ST.txt'
OUT = ROOT / 'opp_data' / 'features'
NASHVILLE = (36.1627, -86.7816)         # downtown Nashville
THIN = 30                               # consent searches below this -> thin cell

FIPS_TO_USPS = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE',
    '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA',
    '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN',
    '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM',
    '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
    '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA',
    '54': 'WV', '55': 'WI', '56': 'WY', '72': 'PR'}
BORDER = {'KY', 'VA', 'NC', 'GA', 'AL', 'MS', 'AR', 'MO'}
REGION = {
    'Northeast': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
    'Midwest': ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'],
    'South': ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'DC', 'WV', 'AL', 'KY', 'MS', 'TN',
              'AR', 'LA', 'OK', 'TX'],
    'West': ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA'],
}
STATE_REGION = {s: r for r, ss in REGION.items() for s in ss}
assert len(STATE_REGION) == 51


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def load_centres():
    c = pd.read_csv(CENPOP, dtype={'STATEFP': str}, encoding='utf-8-sig')
    c.columns = [x.strip().upper() for x in c.columns]
    c['STATEFP'] = c['STATEFP'].str.zfill(2)
    c['state'] = c['STATEFP'].map(FIPS_TO_USPS)
    assert c.loc[c['STATEFP'] == '47', 'STNAME'].iat[0].strip() == 'Tennessee', 'FIPS mapping check failed'
    assert c['state'].notna().all(), f"unmapped FIPS: {c.loc[c['state'].isna(), 'STATEFP'].tolist()}"
    c['dist_km'] = haversine_km(c['LATITUDE'], c['LONGITUDE'], *NASHVILLE)
    return c.set_index('state')['dist_km']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(STOPS, columns=[
        'stop_id', 'date', 'vehicle_registration_state', 'search_conducted', 'contraband_found',
        'raw_search_consent', 'raw_search_arrest', 'raw_search_warrant', 'raw_search_inventory',
        'raw_search_plain_view'])
    dist = load_centres()

    code = s['vehicle_registration_state'].str.upper().str.strip()
    unknown = sorted(set(code.dropna()) - set(STATE_REGION))
    assert not unknown, f'plate codes outside the 50 states + DC: {unknown}'

    f = pd.DataFrame({'stop_id': s['stop_id']})
    f['plate_missing'] = code.isna()
    f['plate_state'] = code.fillna('missing').astype('category')
    f['plate_out_of_state'] = pd.array(np.where(code.isna(), pd.NA, code.ne('TN')), dtype='boolean')
    is_na = code.isna().to_numpy()
    is_tn = code.eq('TN').fillna(False).to_numpy(dtype=bool)
    is_border = code.isin(BORDER).fillna(False).to_numpy(dtype=bool)
    grp = pd.Series('other_state', index=f.index, dtype=object)
    grp[is_border] = 'border_state'
    grp[is_tn] = 'TN'
    grp[is_na] = 'missing'
    f['plate_group'] = grp
    reg = code.map(STATE_REGION).astype(object)
    reg[is_tn] = 'TN'
    reg[is_na] = 'missing'
    f['plate_region'] = reg.to_numpy()
    f['plate_distance_km'] = code.map(dist).astype('float32')
    for c in ['plate_group', 'plate_region']:
        f[c] = f[c].astype('category')
    assert len(f) == len(s) and f['stop_id'].is_unique
    f.to_parquet(OUT / 'plate_features.parquet', index=False)

    # ---------------- descriptive report (no model)
    srch = s['search_conducted'].fillna(False)
    others = ['raw_search_arrest', 'raw_search_warrant', 'raw_search_inventory', 'raw_search_plain_view']
    consent = srch & s['raw_search_consent'].fillna(False) & ~s[others].fillna(False).any(axis=1)
    hit = s['contraband_found'].fillna(False)
    year = s['date'].dt.year

    def table(col):
        rows = {}
        for g, idx in f.groupby(col, observed=True).groups.items():
            n_c = int(consent[idx].sum())
            rows[str(g)] = {'stops': int(len(idx)), 'share_of_stops_pct': round(100 * len(idx) / len(f), 2),
                            'search_rate_pct': round(100 * float(srch[idx].mean()), 2),
                            'consent_searches': n_c,
                            'consent_hit_rate_pct': round(100 * float(hit[idx][consent[idx]].mean()), 1) if n_c else None,
                            'thin_cell': n_c < THIN}
        return rows

    man = {'built_utc': datetime.now(timezone.utc).isoformat(), 'rows': int(len(f)),
           'nashville_reference_point': NASHVILLE,
           'by_plate_group': table('plate_group'), 'by_plate_region': table('plate_region'),
           'out_of_state_share_by_year_pct': {int(y): round(100 * float(v), 1) for y, v in
                                              f['plate_out_of_state'].astype('float').groupby(year).mean().items()},
           'missing_share_by_year_pct': {int(y): round(100 * float(v), 2) for y, v in
                                         f['plate_missing'].groupby(year).mean().items()},
           'distance_km_TN': round(float(dist['TN']), 1)}
    st = table('plate_state')
    man['plate_states_with_thin_consent_cells'] = sorted(k for k, v in st.items() if v['thin_cell'])
    man['by_plate_state'] = st
    (OUT / 'plate_manifest.json').write_text(json.dumps(man, indent=1))

    print(f'wrote plate_features.parquet ({len(f):,} rows, {f.shape[1] - 1} features)')
    for k in ['by_plate_group', 'by_plate_region']:
        print(f'\n{k}:')
        for g, v in man[k].items():
            print(f"  {g:13} stops {v['stops']:>9,} ({v['share_of_stops_pct']:>5}%)  search {v['search_rate_pct']:>5}%  "
                  f"consent n {v['consent_searches']:>6,}  hit {v['consent_hit_rate_pct']}%" + ('  THIN' if v['thin_cell'] else ''))
    print('\nout-of-state share by year (%):', man['out_of_state_share_by_year_pct'])
    print('missing share by year (%):', man['missing_share_by_year_pct'])
    print(f"plate states with < {THIN} consent searches: {len(man['plate_states_with_thin_consent_cells'])} of 51")


if __name__ == '__main__':
    main()
