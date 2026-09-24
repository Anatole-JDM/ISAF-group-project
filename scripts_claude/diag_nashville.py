"""Nashville (OPP) data diagnostics - read-only profiling, no modelling.

Runs the eight diagnostic groups agreed with the team and dumps everything to
diag_nashville.json so the cleaning protocol can be written from observation.
"""
import zipfile, io, json
import pandas as pd
import pyarrow as pa
from pyarrow import csv as pacsv

P = ('C:/Users/smaye/OneDrive/Bureau/X-HEC/HEC_courses/algo_fairness/'
     'ISAF_gp/ISAF-group-project/opp_data/tn_nashville_2020_04_01.csv.zip')
OUT = ('C:/Users/smaye/OneDrive/Bureau/X-HEC/HEC_courses/algo_fairness/'
       'ISAF_gp/ISAF-group-project/scripts_claude/diag_nashville.json')

WANT = ['date', 'time', 'location', 'lat', 'lng', 'precinct', 'zone', 'reporting_area',
        'subject_age', 'subject_race', 'subject_sex', 'officer_id_hash', 'type', 'violation',
        'arrest_made', 'citation_issued', 'warning_issued', 'outcome', 'contraband_found',
        'contraband_drugs', 'contraband_weapons', 'frisk_performed', 'search_conducted',
        'search_person', 'search_vehicle', 'search_basis', 'reason_for_stop',
        'vehicle_registration_state', 'notes', 'raw_search_consent', 'raw_search_arrest',
        'raw_search_warrant', 'raw_search_inventory', 'raw_search_plain_view']

z = zipfile.ZipFile(P)
info = z.infolist()[0]
with z.open(info) as f:
    have = next(io.TextIOWrapper(f, 'utf8', errors='replace')).strip().split(',')
use = [c for c in WANT if c in have]
print('columns loaded:', len(use), '| absent:', [c for c in WANT if c not in have], flush=True)

with z.open(info) as f:
    tbl = pacsv.read_csv(
        f,
        read_options=pacsv.ReadOptions(block_size=1 << 26),
        convert_options=pacsv.ConvertOptions(
            include_columns=use, column_types={c: pa.string() for c in use}))
df = tbl.to_pandas()
del tbl
print('rows:', len(df), flush=True)

R = {}


def B(s):
    return s.str.lower().eq('true')


def NA(s):
    return s.isna() | s.eq('') | s.isin(['NA', 'unknown', 'n/a'])


d = pd.to_datetime(df['date'], errors='coerce')
df['ym'] = d.dt.to_period('M').astype(str)
df['year'] = d.dt.year
srch = B(df['search_conducted']).fillna(False)
hit = B(df['contraband_found']).fillna(False)

# 1. row integrity
key = [k for k in ['date', 'time', 'officer_id_hash', 'subject_age', 'subject_sex',
                   'subject_race', 'location'] if k in df]
R['1_row_integrity'] = {
    'n_rows': int(len(df)),
    'dup_key': key,
    'dup_on_key': int(df.duplicated(subset=key).sum()),
    'search_conducted_NA': int(NA(df['search_conducted']).sum()),
    'type_counts': {str(k): int(v) for k, v in df['type'].value_counts(dropna=False).head(6).items()},
}
print('1 done', flush=True)

# 2. target
g = df[srch].groupby('ym').agg(n=('contraband_found', 'size'),
                               hits=('contraband_found', lambda s: int(B(s).sum())))
g['rate'] = 100 * g['hits'] / g['n']
off = df[srch].groupby('officer_id_hash').agg(n=('contraband_found', 'size'),
                                              h=('contraband_found', lambda s: int(B(s).sum())))
off = off[off['n'] >= 30]
off['rate'] = off['h'] / off['n']
R['2_target'] = {
    'n_search': int(srch.sum()),
    'contraband_TRUE_when_no_search': int((hit & ~srch).sum()),
    'contraband_NA_among_searches': int(NA(df['contraband_found'][srch]).sum()),
    'monthly_series': {str(k): [int(v['n']), round(float(v['rate']), 1)] for k, v in g.iterrows()},
    'officers_ge30': int(len(off)),
    'officers_0pct': int((off['rate'] == 0).sum()),
    'searches_of_0pct_officers': int(off[off['rate'] == 0]['n'].sum()),
    'officers_100pct': int((off['rate'] == 1).sum()),
    'officer_rate_quantiles': {str(q): round(float(off['rate'].quantile(q)), 3)
                               for q in [0.05, 0.25, 0.5, 0.75, 0.95]},
}
print('2 done', flush=True)

# 3. protected attributes
age = pd.to_numeric(df['subject_age'], errors='coerce')
byyr = df.groupby('year')['subject_race'].value_counts(normalize=True).unstack().fillna(0)
R['3_protected'] = {
    'race_counts': {str(k): int(v) for k, v in df['subject_race'].value_counts(dropna=False).items()},
    'race_share_by_year': {int(y): {str(c): round(float(v) * 100, 1) for c, v in row.items()}
                           for y, row in byyr.iterrows()},
    'age_missing_pct': round(100 * float(age.isna().mean()), 2),
    'age_lt15': int((age < 15).sum()),
    'age_gt90': int((age > 90).sum()),
    'age_quantiles': {str(q): float(age.quantile(q)) for q in [0.01, 0.05, 0.5, 0.95, 0.99]},
    'age_top_values': {int(k): int(v) for k, v in age.value_counts().head(8).items()},
    'sex_counts': {str(k): int(v) for k, v in df['subject_sex'].value_counts(dropna=False).items()},
}
print('3 done', flush=True)

# 4. date / time
m = df.groupby('ym').size()
full = pd.period_range(d.min(), d.max(), freq='M').astype(str)
tm = df['time'].where(~NA(df['time']))
R['4_datetime'] = {
    'date_min': str(d.min().date()), 'date_max': str(d.max().date()),
    'months_expected': int(len(full)), 'months_present': int(len(m)),
    'missing_months': [x for x in full if x not in set(m.index)],
    'low_months': [[str(k), int(v)] for k, v in m[m < m.median() * 0.4].items()][:24],
    'stops_per_year': {int(y): int(v) for y, v in df.groupby('year').size().items()},
    'time_missing_pct': round(100 * float(tm.isna().mean()), 2),
    'time_missing_by_year': {int(y): round(100 * float(NA(gg['time']).mean()), 1)
                             for y, gg in df.groupby('year')},
    'minute_top': {str(k): round(100 * float(v) / max(len(tm.dropna()), 1), 1)
                   for k, v in tm.str.slice(3, 5).value_counts().head(6).items()},
    'hour_top': {str(k): round(100 * float(v) / max(len(tm.dropna()), 1), 1)
                 for k, v in tm.str.slice(0, 2).value_counts().head(6).items()},
}
print('4 done', flush=True)

# 5. geography
lat = pd.to_numeric(df['lat'], errors='coerce')
lng = pd.to_numeric(df['lng'], errors='coerce')
box = lat.between(35.9, 36.5) & lng.between(-87.2, -86.4)
coords = pd.Series(list(zip(lat.round(4), lng.round(4))))
vc = coords.value_counts()
geo = {
    'latlng_missing_pct': round(100 * float(lat.isna().mean()), 2),
    'outside_davidson_box': int(((~box) & lat.notna()).sum()),
    'distinct_coords_4dp': int(vc.shape[0]),
    'lat_decimal_places': {int(k): int(v) for k, v in
                           lat.dropna().astype(str).str.split('.').str[1].str.len()
                           .value_counts().head(5).items()},
    'top10_coord_share_pct': round(100 * float(vc.head(10).sum()) / len(df), 2),
    'coords_used_once_pct': round(100 * float((vc == 1).sum()) / max(vc.shape[0], 1), 1),
}
for c in ['precinct', 'zone', 'reporting_area']:
    if c in df:
        geo[c + '_missing_pct'] = round(100 * float(NA(df[c]).mean()), 2)
        geo[c + '_n_levels'] = int(df[c][~NA(df[c])].nunique())
        geo[c + '_missing_by_year'] = {int(y): round(100 * float(NA(gg[c]).mean()), 1)
                                       for y, gg in df.groupby('year')}
R['5_geography'] = geo
print('5 done', flush=True)

# 6. categoricals
rs = df.groupby('year')['reason_for_stop'].value_counts(normalize=True).unstack().fillna(0)
vcv = df['violation'].value_counts(normalize=True)
osz = df.groupby('officer_id_hash').size()
pre = set(df['officer_id_hash'][df['year'] < 2017])
post_rows = df[df['year'] >= 2017]
R['6_categoricals'] = {
    'reason_counts': {str(k): int(v) for k, v in df['reason_for_stop'].value_counts(dropna=False).items()},
    'reason_share_by_year': {int(y): {str(c): round(float(v) * 100, 1) for c, v in row.items() if v > 0.005}
                             for y, row in rs.iterrows()},
    'violation_n_levels': int(df['violation'].nunique()),
    'violation_top20_coverage_pct': round(100 * float(vcv.head(20).sum()), 1),
    'violation_top8': {str(k): round(float(v) * 100, 1) for k, v in vcv.head(8).items()},
    'n_officers': int(df['officer_id_hash'].nunique()),
    'officer_stops_quantiles': {str(q): int(osz.quantile(q)) for q in [0.25, 0.5, 0.75, 0.95]},
    'officers_pre2017': len(pre),
    'officers_post2017': int(post_rows['officer_id_hash'].nunique()),
    'post2017_stops_by_unseen_officers_pct': round(
        100 * float((~post_rows['officer_id_hash'].isin(pre)).mean()), 1) if len(post_rows) else None,
    'regstate_TN_pct': round(100 * float(df['vehicle_registration_state'].eq('TN').mean()), 1),
    'regstate_missing_pct': round(100 * float(NA(df['vehicle_registration_state']).mean()), 2),
}
print('6 done', flush=True)

# 7. sample definition
sb = df[srch]
bb = sb.groupby('year')['search_basis'].value_counts(normalize=True).unstack().fillna(0)
samp = {
    'basis_counts': {str(k): int(v) for k, v in sb['search_basis'].value_counts(dropna=False).items()},
    'basis_share_by_year': {int(y): {str(c): round(float(v) * 100, 1) for c, v in row.items()}
                            for y, row in bb.iterrows()},
}
raws = [c for c in ['raw_search_consent', 'raw_search_arrest', 'raw_search_warrant',
                    'raw_search_inventory', 'raw_search_plain_view'] if c in df]
if raws:
    flags = pd.concat([B(sb[c]).fillna(False) for c in raws], axis=1)
    flags.columns = raws
    others = [c for c in raws if c != 'raw_search_consent']
    samp['raw_flag_counts'] = {c: int(flags[c].sum()) for c in raws}
    samp['n_flags_per_search'] = {int(k): int(v) for k, v in
                                  flags.sum(axis=1).value_counts().sort_index().items()}
    samp['consent_flag_total'] = int(flags['raw_search_consent'].sum())
    samp['consent_AND_other_flag'] = int((flags['raw_search_consent'] & flags[others].any(axis=1)).sum())
cc = df[srch & df['search_basis'].eq('consent')]
samp['consent_n'] = int(len(cc))
samp['consent_by_year_race'] = {int(y): {str(k): int(v) for k, v in gg['subject_race'].value_counts().items()}
                                for y, gg in cc.groupby('year')}
samp['consent_hit_by_year'] = {int(y): [int(len(gg)), round(100 * float(B(gg['contraband_found']).mean()), 1)]
                               for y, gg in cc.groupby('year')}
samp['consent_hit_by_race'] = {str(k): [int(len(gg)), round(100 * float(B(gg['contraband_found']).mean()), 1)]
                               for k, gg in cc.groupby('subject_race')}
R['7_sample'] = samp
print('7 done', flush=True)

# 8. drift within the consent sample
drift = {}
for y, gg in cc.groupby('year'):
    a = pd.to_numeric(gg['subject_age'], errors='coerce')
    drift[int(y)] = {
        'n': int(len(gg)),
        'hit_pct': round(100 * float(B(gg['contraband_found']).mean()), 1),
        'male_pct': round(100 * float(gg['subject_sex'].eq('male').mean()), 1),
        'age_mean': round(float(a.mean()), 1),
        'black_pct': round(100 * float(gg['subject_race'].eq('black').mean()), 1),
        'top_reason': str(gg['reason_for_stop'].mode().iat[0]) if len(gg) else None,
    }
R['8_drift'] = drift

json.dump(R, open(OUT, 'w'), indent=1, default=str)
print('WROTE', OUT, flush=True)
print('ALL DONE', flush=True)
