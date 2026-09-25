"""Join the cleaned stops and every feature table into the master tables.

Outputs
  data/nashville_consent_searches.parquet   modelling sample (consent-only searches), all columns - TRACKED in git
  data/nashville_columns.csv                role of every column in that file                       - TRACKED in git
  opp_data/processed/nashville_master.parquet  all 3,088,286 stops (audit model) - local only (~400 MB)

Column roles (see nashville_columns.csv)
  id / target / protected / protected_and_feature / feature / feature_D3_open / robustness /
  quality_flag / split_helper / analysis_helper / post_decision_DO_NOT_USE
Every column recorded at or after the search decision is renamed with the prefix `post_` so it
cannot be mistaken for a predictor. The build stops if any column has no role.

Consent-only sample: search_conducted and raw_search_consent TRUE, and none of the raw
arrest / warrant / inventory / plain-view flags set (cleaning protocol, Step 5).
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'opp_data' / 'processed'
FEAT = ROOT / 'opp_data' / 'features'
DATA = ROOT / 'data'

BASE_KEEP = ['stop_id', 'raw_row_number', 'date', 'location', 'lat', 'lng', 'precinct', 'zone',
             'reporting_area', 'subject_age', 'subject_race', 'subject_sex', 'officer_id_hash',
             'reason_for_stop', 'contraband_found', 'search_conducted', 'geo_outside_davidson_box']
POST = ['search_basis', 'search_person', 'search_vehicle', 'frisk_performed', 'contraband_drugs',
        'contraband_weapons', 'arrest_made', 'citation_issued', 'warning_issued', 'outcome',
        'raw_driver_searched', 'raw_passenger_searched', 'raw_search_consent', 'raw_search_arrest',
        'raw_search_warrant', 'raw_search_inventory', 'raw_search_plain_view']
# dropped on purpose: time (-> minute_of_day), type (always 'vehicular'),
# vehicle_registration_state (-> plate_state), notes (79% empty free text, post-decision),
# other raw_* passthroughs (warning/citation sub-types, raw ethnicity)

EXPECTED = {'stops': 3_088_286, 'consent': 58_939}

TIME_FEATURES = ['hour', 'hour_missing', 'minute_of_day', 'hour_sin', 'hour_cos', 'day_of_week',
                 'is_weekend', 'month', 'month_sin', 'month_cos', 'is_federal_holiday',
                 'is_holiday_window', 'is_dst', 'sun_elevation_deg', 'light_period', 'is_dark']
RACE_SHARES = ['share_black', 'share_white', 'share_hispanic', 'share_asian']
NBH_FLAGS = ['circle_truncated', 'low_residents', 'income_weight_missing', 'residents_2010']


def role(c):
    if c == 'stop_id':
        return 'id', 'row key shared by every table'
    if c == 'raw_row_number':
        return 'id', 'row number in the original OPP file (traceability)'
    if c == 'contraband_found':
        return 'target', 'contraband found in the search (main target)'
    if c == 'search_conducted':
        return 'target', 'search conducted (target of the audit model, all-stops table only)'
    if c == 'subject_race':
        return 'protected', 'officer-perceived race - for fairness evaluation, not a feature'
    if c in ('subject_age', 'subject_sex'):
        return 'protected_and_feature', 'protected attribute that is also a legitimate predictor - state it'
    if c == 'consent_sample':
        return 'split_helper', 'row belongs to the consent-only modelling sample'
    if c.startswith('post_'):
        return 'post_decision_DO_NOT_USE', 'recorded at or after the search decision - leakage if used as a feature'
    if c in ('reason_for_stop', 'precinct', 'zone', 'reporting_area'):
        return 'feature', 'from the cleaned table'
    if c in ('date', 'year', 'officer_id_hash'):
        return 'split_helper', 'for time splits, grouped validation, officer-level analysis - not a feature'
    if c in ('lat', 'lng', 'location', 'plate_state', 'minutes_after_civil_dusk', 'in_intertwilight'):
        return 'analysis_helper', 'maps, veil-of-darkness test, descriptive tables - not a feature by default'
    if c in ('geo_outside_davidson_box', 'nbh_geo_source', 'geo_bucket', 'acs_vintage',
             'acs_window_overlaps_stop', 'time_heaped', 'dst_ambiguous', 'dst_nonexistent', 'plate_missing'):
        note = ('recording change in 2017 - a date proxy, never a predictor' if c == 'plate_missing'
                else 'data-quality flag')
        return 'quality_flag', note
    for r in ('nbh800_', 'nbh1200_'):
        if c.startswith(r):
            base = c[len(r):]
            if base in NBH_FLAGS:
                return 'quality_flag', 'neighbourhood-circle quality flag'
            if r == 'nbh1200_':
                return 'robustness', 'same feature on a 1,200 m circle - robustness check only'
            if base == 'unemployment_rate':
                return 'analysis_helper', 'tracks the business cycle (a date proxy) - use unemployment_rel'
            if base.endswith('_approx'):
                return 'feature', 'approximation (weighted mean of tract medians) - prefer income_pctile for income'
            if base in RACE_SHARES:
                return 'feature_D3_open', 'neighbourhood racial make-up: model input or fairness-only (decision D3)'
            return 'feature', 'neighbourhood (ACS), 800 m population-weighted circle'
    if c in ('plate_group', 'plate_region', 'plate_distance_km', 'plate_out_of_state'):
        return 'feature', 'registration plate' + (' (share rises over time - monitor)' if c == 'plate_out_of_state' else '')
    if c in TIME_FEATURES:
        return 'feature', 'time of the stop'
    return None, None


def main():
    DATA.mkdir(exist_ok=True)
    base = pd.read_parquet(PROC / 'nashville_clean.parquet', columns=BASE_KEEP + POST)
    base = base.rename(columns={c: f'post_{c}' for c in POST})
    nbh = pd.read_parquet(FEAT / 'nbh_features.parquet')
    plate = pd.read_parquet(FEAT / 'plate_features.parquet')
    time = pd.read_parquet(FEAT / 'time_features.parquet')

    m = base
    for name, t in [('nbh', nbh), ('plate', plate), ('time', time)]:
        clash = (set(t.columns) & set(m.columns)) - {'stop_id'}
        assert not clash, f'{name}: duplicate column names {clash}'
        m = m.merge(t, on='stop_id', how='left', validate='one_to_one')
    assert len(m) == EXPECTED['stops'] and m['stop_id'].is_unique

    flags = ['post_raw_search_arrest', 'post_raw_search_warrant', 'post_raw_search_inventory',
             'post_raw_search_plain_view']
    m['consent_sample'] = (m['search_conducted'].fillna(False) & m['post_raw_search_consent'].fillna(False)
                           & ~m[flags].fillna(False).any(axis=1))
    assert int(m['consent_sample'].sum()) == EXPECTED['consent']

    roles = {c: role(c) for c in m.columns}
    missing_role = [c for c, (r, _) in roles.items() if r is None]
    assert not missing_role, f'columns without a role: {missing_role}'

    m.to_parquet(PROC / 'nashville_master.parquet', index=False)

    # modelling sample: drop columns that are constant inside it
    cs = m[m['consent_sample']].reset_index(drop=True)
    const = [c for c in cs.columns if cs[c].nunique(dropna=False) <= 1]
    cs = cs.drop(columns=const)
    assert cs['contraband_found'].notna().all()
    cs.to_parquet(DATA / 'nashville_consent_searches.parquet', index=False)

    dd = pd.DataFrame([{'column': c, 'role': roles[c][0], 'dtype': str(cs[c].dtype),
                        'missing_pct': round(100 * float(cs[c].isna().mean()), 2), 'note': roles[c][1]}
                       for c in cs.columns])
    dd.to_csv(DATA / 'nashville_columns.csv', index=False)

    size = (DATA / 'nashville_consent_searches.parquet').stat().st_size / 1e6
    summary = {
        'all_stops_rows': len(m), 'all_stops_columns': m.shape[1],
        'all_stops_file_MB': round((PROC / 'nashville_master.parquet').stat().st_size / 1e6, 1),
        'consent_rows': len(cs), 'consent_columns': cs.shape[1], 'consent_file_MB': round(size, 2),
        'dropped_as_constant_in_sample': const,
        'target_rate': round(float(cs['contraband_found'].mean()), 4),
        'target_rate_by_race': {str(k): [int(n), round(float(r), 3)] for k, (n, r) in
                                cs.groupby('subject_race', dropna=False)['contraband_found']
                                .agg(['size', 'mean']).iterrows()},
        'columns_by_role': dd.groupby('role')['column'].count().to_dict(),
    }
    (DATA / 'nashville_consent_searches_summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    if size > 50:
        print(f'WARNING: {size:.0f} MB - GitHub warns above 50 MB and rejects above 100 MB')


if __name__ == '__main__':
    main()
