"""Officer track-record features for every Nashville stop.

Idea: what predicts a successful search may be WHO searches rather than who is searched.
Officer identity alone lifts AUC 0.566 -> 0.600 (FINDINGS.md, finding 4), but an identity
column cannot generalise to officers never seen in training. A track record can.

Every feature is computed STRICTLY FROM THE PAST of the stop being described:
  * an event counts only if its timestamp is strictly earlier than the stop's timestamp,
    so stops recorded at the same minute never see each other;
  * a stop with no recorded time (0.18%) is queried at 00:00 of its day (it sees only
    previous days) and becomes visible to others only from 00:00 the NEXT day, since
    we cannot know when during its day it happened;
  * the stop's own outcome is never included.
A brute-force recomputation on a random sample checks every count exactly.

Features (officer = officer_id_hash)
  window: previous 365 days
    off_stops_365d, off_searches_365d, off_hits_365d            counts
    off_consent_365d, off_consent_hits_365d                      consent-only searches
    off_search_rate_365d                                         searches / stops
    off_hit_rate_365d_shrunk                                     hit rate of all searches, shrunk
    off_consent_hit_rate_365d_shrunk                             hit rate of consent searches, shrunk
      shrinkage: (hits + K * p0) / (searches + K), K = 20 pseudo-searches, p0 = hit rate of
      ALL officers over the same past 365 days (so the prior uses no future data either)
  same day, before this stop (missing when the stop has no recorded time)
    off_stops_today_before, off_searches_today_before, off_hits_today_before
    off_minutes_since_first_stop_today                           proxy for shift progress
  last search by this officer, before this stop
    off_minutes_since_last_search, off_last_search_hit (1 / 0 / missing)
  career
    off_experience_days                                          days since first stop in the data
    off_experience_censored                                      officer already active in Jan 2010
    off_history_warmup                                           stop in 2010: < 365 days of history
  disparity - all of the officer's past since 2010 (more data than 365 days), shrunk
    off_consent_white_past, off_consent_black_past, off_consent_hisp_past     support counts
    off_log_search_ratio_black_white    log( consent-search rate per stop, Black drivers / white drivers )
    off_log_search_ratio_hisp_white     same for Hispanic drivers. > 0: this officer asks minority drivers more
    off_hit_gap_black_white             hit rate on Black drivers - hit rate on white drivers (outcome test)
    off_hit_gap_hisp_white              same for Hispanic drivers. < 0: lower evidentiary bar for that group
    off_hit_rate_same_race_past         the officer's past hit rate on drivers of THIS driver's race
                                        (race-derived: never in a race-blind model)
      rates per group are shrunk toward the officer's own overall rate (M = 50 stops for search
      rates, K = 20 searches for hit rates), so officers with few minority stops get ~0 disparity

Outputs
  opp_data/features/officer_features.parquet        all 3,088,286 stops (local)
  opp_data/features/officer_features_consent.parquet  consent-only searches (local; build_master.py puts the
                                                     features into data/nashville_consent_searches.parquet)
  data/officer_features_manifest.json               definitions, checks, descriptive tables (git)
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STOPS = ROOT / 'opp_data' / 'processed' / 'nashville_clean.parquet'
OUT_ALL = ROOT / 'opp_data' / 'features' / 'officer_features.parquet'
OUT_CONSENT = ROOT / 'opp_data' / 'features' / 'officer_features_consent.parquet'
MANIFEST = ROOT / 'data' / 'officer_features_manifest.json'

WINDOW = pd.Timedelta(days=365)
K_SHRINK = 20
M_SHRINK = 50
DATA_START = pd.Timestamp('2010-01-01')
GROUPS = {'w': 'white', 'b': 'black', 'h': 'hispanic'}
EV = (['ev_stop', 'ev_search', 'ev_hit', 'ev_consent', 'ev_consent_hit']
      + [f'ev_{k}_{g}' for g in GROUPS for k in ('stop', 'cons', 'chit')])
OFFICER = 'officer'
N_CHECK = 400


def load():
    cols = ['stop_id', 'raw_row_number', 'date', 'time', 'officer_id_hash', 'subject_race',
            'search_conducted', 'contraband_found', 'raw_search_consent', 'raw_search_arrest',
            'raw_search_warrant', 'raw_search_inventory', 'raw_search_plain_view']
    s = pd.read_parquet(STOPS, columns=cols)
    mins = (pd.to_numeric(s['time'].str.slice(0, 2), errors='coerce') * 60
            + pd.to_numeric(s['time'].str.slice(3, 5), errors='coerce'))
    s['time_missing'] = mins.isna()
    s['day'] = s['date'].dt.normalize()
    s['query_ts'] = s['day'] + pd.to_timedelta(mins.fillna(0), unit='min')
    # when does this stop become known to later stops?
    s['event_ts'] = s['query_ts'].where(~s['time_missing'], s['day'] + pd.Timedelta(days=1))
    s[OFFICER] = s['officer_id_hash'].astype('object')

    srch = s['search_conducted'].fillna(False).astype(bool)
    hit = s['contraband_found'].fillna(False).astype(bool)
    others = ['raw_search_arrest', 'raw_search_warrant', 'raw_search_inventory', 'raw_search_plain_view']
    cons = srch & s['raw_search_consent'].fillna(False).astype(bool) & ~s[others].fillna(False).astype(bool).any(axis=1)
    s['consent_sample'] = cons
    s['ev_stop'] = 1
    s['ev_search'] = srch.astype('int64')
    s['ev_hit'] = (srch & hit).astype('int64')
    s['ev_consent'] = cons.astype('int64')
    s['ev_consent_hit'] = (cons & hit).astype('int64')
    for g, race in GROUPS.items():
        is_g = s['subject_race'].eq(race).fillna(False).astype(bool)
        s[f'ev_stop_{g}'] = is_g.astype('int64')
        s[f'ev_cons_{g}'] = (is_g & cons).astype('int64')
        s[f'ev_chit_{g}'] = (is_g & cons & hit).astype('int64')
    return s


def cumulative(ev, by_officer):
    """Cumulative event counts at each distinct event time (ties collapsed first)."""
    keys = [OFFICER, 'event_ts'] if by_officer else ['event_ts']
    agg = ev.groupby(keys, sort=True)[EV].sum()
    cum = agg.groupby(level=OFFICER).cumsum() if by_officer else agg.cumsum()
    return cum.reset_index().rename(columns={'event_ts': 't'}).sort_values('t', kind='mergesort')


def counts_before(times, officers, cum, by_officer=True):
    """For each query time, cumulative counts of events with event_ts STRICTLY before it."""
    left = pd.DataFrame({'t': times.to_numpy(), '_row': np.arange(len(times))})
    if by_officer:
        left[OFFICER] = officers.to_numpy()
    left = left.sort_values('t', kind='mergesort')
    m = pd.merge_asof(left, cum, on='t', by=OFFICER if by_officer else None,
                      direction='backward', allow_exact_matches=False)
    return m.sort_values('_row')[EV].fillna(0).to_numpy(dtype='float64')


def shrink(hits, searches, p0):
    with np.errstate(invalid='ignore', divide='ignore'):
        return (hits + K_SHRINK * p0) / (searches + K_SHRINK)


def main():
    s = load()
    q = s[s[OFFICER].notna()].copy()                    # 10 stops have no officer id
    ev = q[[OFFICER, 'event_ts'] + EV]
    cum_o = cumulative(ev, by_officer=True)
    cum_g = cumulative(ev, by_officer=False)
    print(f'stops {len(s):,}; with officer {len(q):,}; officers {q[OFFICER].nunique():,}', flush=True)

    ix = {c: i for i, c in enumerate(EV)}
    now = counts_before(q['query_ts'], q[OFFICER], cum_o)
    ago = counts_before(q['query_ts'] - WINDOW, q[OFFICER], cum_o)
    w = now - ago                                        # events in [t - 365d, t)
    dstart = counts_before(q['day'], q[OFFICER], cum_o)
    today = now - dstart                                 # events in [day start, t)
    g = counts_before(q['query_ts'], None, cum_g, by_officer=False) \
        - counts_before(q['query_ts'] - WINDOW, None, cum_g, by_officer=False)
    with np.errstate(invalid='ignore', divide='ignore'):
        p0_all = g[:, ix['ev_hit']] / g[:, ix['ev_search']]
        p0_cons = g[:, ix['ev_consent_hit']] / g[:, ix['ev_consent']]

    f = pd.DataFrame(index=q.index)
    f['off_stops_365d'] = w[:, ix['ev_stop']]
    f['off_searches_365d'] = w[:, ix['ev_search']]
    f['off_hits_365d'] = w[:, ix['ev_hit']]
    f['off_consent_365d'] = w[:, ix['ev_consent']]
    f['off_consent_hits_365d'] = w[:, ix['ev_consent_hit']]
    with np.errstate(invalid='ignore', divide='ignore'):
        f['off_search_rate_365d'] = np.where(f['off_stops_365d'] > 0,
                                             f['off_searches_365d'] / f['off_stops_365d'], np.nan)
    f['off_hit_rate_365d_shrunk'] = shrink(f['off_hits_365d'], f['off_searches_365d'], p0_all)
    f['off_consent_hit_rate_365d_shrunk'] = shrink(f['off_consent_hits_365d'], f['off_consent_365d'], p0_cons)

    tm = q['time_missing'].to_numpy()
    for name, col in [('off_stops_today_before', 'ev_stop'), ('off_searches_today_before', 'ev_search'),
                      ('off_hits_today_before', 'ev_hit')]:
        f[name] = np.where(tm, np.nan, today[:, ix[col]])

    # first stop of the day (timed stops only), for shift progress
    timed = q.loc[~q['time_missing'], [OFFICER, 'day', 'query_ts']]
    first_today = timed.groupby([OFFICER, 'day'])['query_ts'].transform('min')
    f['off_minutes_since_first_stop_today'] = np.nan
    f.loc[timed.index, 'off_minutes_since_first_stop_today'] = \
        (timed['query_ts'] - first_today).dt.total_seconds() / 60

    # last search strictly before the stop
    srch_ev = q.loc[q['ev_search'] == 1, [OFFICER, 'event_ts', 'ev_hit']].rename(columns={'event_ts': 't'})
    srch_ev = srch_ev.groupby([OFFICER, 't'], as_index=False)['ev_hit'].max().sort_values('t', kind='mergesort')
    srch_ev['last_search_t'] = srch_ev['t']
    left = pd.DataFrame({'t': q['query_ts'].to_numpy(), OFFICER: q[OFFICER].to_numpy(),
                         '_row': np.arange(len(q))}).sort_values('t', kind='mergesort')
    m = pd.merge_asof(left, srch_ev, on='t', by=OFFICER, direction='backward',
                      allow_exact_matches=False).sort_values('_row')
    f['off_minutes_since_last_search'] = ((q['query_ts'].to_numpy() - m['last_search_t'].to_numpy())
                                          / np.timedelta64(1, 'm'))
    f['off_last_search_hit'] = m['ev_hit'].to_numpy(dtype='float64')

    first_seen = q.groupby(OFFICER)['event_ts'].transform('min')
    f['off_experience_days'] = ((q['query_ts'] - first_seen).dt.total_seconds() / 86400).clip(lower=0)
    f['off_experience_censored'] = first_seen < DATA_START + pd.Timedelta(days=30)
    f['off_history_warmup'] = q['query_ts'] < DATA_START + WINDOW

    # ---- disparity: all of the officer's past (lifetime since 2010), strictly before the stop
    life = now                                           # counts of events strictly before t
    g_life = counts_before(q['query_ts'], None, cum_g, by_officer=False)
    with np.errstate(invalid='ignore', divide='ignore'):
        p0_life = g_life[:, ix['ev_consent_hit']] / g_life[:, ix['ev_consent']]
        r_o = life[:, ix['ev_consent']] / life[:, ix['ev_stop']]             # officer's consent rate
    h_o = shrink(life[:, ix['ev_consent_hit']], life[:, ix['ev_consent']], p0_life)   # officer's hit rate
    rate, hitr = {}, {}
    for g in GROUPS:
        st, cs, ch = life[:, ix[f'ev_stop_{g}']], life[:, ix[f'ev_cons_{g}']], life[:, ix[f'ev_chit_{g}']]
        rate[g] = (cs + M_SHRINK * r_o) / (st + M_SHRINK)
        hitr[g] = (ch + K_SHRINK * h_o) / (cs + K_SHRINK)
    f['off_consent_white_past'] = life[:, ix['ev_cons_w']]
    f['off_consent_black_past'] = life[:, ix['ev_cons_b']]
    f['off_consent_hisp_past'] = life[:, ix['ev_cons_h']]
    with np.errstate(invalid='ignore', divide='ignore'):
        f['off_log_search_ratio_black_white'] = np.log(rate['b'] / rate['w'])
        f['off_log_search_ratio_hisp_white'] = np.log(rate['h'] / rate['w'])
    f['off_hit_gap_black_white'] = hitr['b'] - hitr['w']
    f['off_hit_gap_hisp_white'] = hitr['h'] - hitr['w']
    race = q['subject_race'].fillna('').astype(str).to_numpy()
    f['off_hit_rate_same_race_past'] = np.select(
        [race == 'white', race == 'black', race == 'hispanic'], [hitr['w'], hitr['b'], hitr['h']], default=h_o)
    for c in ['off_log_search_ratio_black_white', 'off_log_search_ratio_hisp_white']:
        f[c] = f[c].replace([np.inf, -np.inf], np.nan)

    # ---------------------------------------------------------------- exact brute-force check
    rng = np.random.default_rng(0)
    sample = rng.choice(q.index.to_numpy(), size=N_CHECK, replace=False)
    bad = []
    for i in sample:
        o, t, d = q.at[i, OFFICER], q.at[i, 'query_ts'], q.at[i, 'day']
        e = ev[ev[OFFICER] == o]
        win = e[(e['event_ts'] < t) & (e['event_ts'] >= t - WINDOW)]
        exp = {'off_stops_365d': len(win), 'off_searches_365d': win['ev_search'].sum(),
               'off_hits_365d': win['ev_hit'].sum(), 'off_consent_365d': win['ev_consent'].sum(),
               'off_consent_hits_365d': win['ev_consent_hit'].sum()}
        if not q.at[i, 'time_missing']:
            day_ev = e[(e['event_ts'] < t) & (e['event_ts'] >= d)]
            exp.update({'off_stops_today_before': len(day_ev), 'off_searches_today_before': day_ev['ev_search'].sum(),
                        'off_hits_today_before': day_ev['ev_hit'].sum()})
        past = e[e['event_ts'] < t]
        exp.update({'off_consent_white_past': past['ev_cons_w'].sum(),
                    'off_consent_black_past': past['ev_cons_b'].sum(),
                    'off_consent_hisp_past': past['ev_cons_h'].sum()})
        prev = e[(e['event_ts'] < t) & (e['ev_search'] == 1)]
        exp['off_last_search_hit'] = (float(prev.loc[prev['event_ts'] == prev['event_ts'].max(), 'ev_hit'].max())
                                      if len(prev) else np.nan)
        for k, v in exp.items():
            got = f.at[i, k]
            if not ((pd.isna(v) and pd.isna(got)) or (not pd.isna(v) and float(got) == float(v))):
                bad.append((int(i), k, float(v) if not pd.isna(v) else None, got))
    assert not bad, f'brute-force check failed on {len(bad)} values, e.g. {bad[:5]}'
    print(f'brute-force check: {N_CHECK} random stops, every count identical', flush=True)

    # ---------------------------------------------------------------- outputs
    out = s[['stop_id', 'raw_row_number', 'date', 'consent_sample']].join(f, how='left')
    for c in ['off_experience_censored', 'off_history_warmup']:
        out[c] = out[c].astype('boolean')
    float_cols = [c for c in f.columns if f[c].dtype.kind == 'f']
    out[float_cols] = out[float_cols].astype('float32')
    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_ALL, index=False)
    cons_out = out[out['consent_sample']].drop(columns=['consent_sample']).reset_index(drop=True)
    cons_out.to_parquet(OUT_CONSENT, index=False)

    # ---------------------------------------------------------------- descriptive report (no model)
    c = s.loc[s['consent_sample']].join(f, how='left')
    c = c[(c['date'] >= '2011-01-01') & (c['date'] < '2019-01-01')]     # past warm-up, 2010-2018 window
    y = c['ev_consent_hit']

    def table(key):
        t = c.assign(k=key).groupby('k', observed=True).agg(n=('ev_consent_hit', 'size'),
                                                              hit_rate=('ev_consent_hit', 'mean'))
        return {str(k): [int(r['n']), round(100 * float(r['hit_rate']), 1)] for k, r in t.iterrows()}

    man = {
        'built_utc': datetime.now(timezone.utc).isoformat(),
        'rows_all': int(len(out)), 'rows_consent': int(len(cons_out)),
        'officers': int(q[OFFICER].nunique()), 'k_shrink': K_SHRINK, 'window_days': 365,
        'brute_force_check': f'{N_CHECK} random stops, all counts identical',
        'descriptive_scope': 'consent-only searches 2011-2018 (2010 is the warm-up year)',
        'n_descriptive': int(len(c)), 'base_hit_rate_pct': round(100 * float(y.mean()), 1),
        'hit_rate_by_decile_of_officer_consent_hit_rate': table(
            pd.qcut(c['off_consent_hit_rate_365d_shrunk'], 10, labels=False, duplicates='drop')),
        'hit_rate_by_officer_last_search_hit': table(c['off_last_search_hit'].map({1.0: 'hit', 0.0: 'miss'}).fillna('none')),
        'hit_rate_by_searches_already_today': table(pd.cut(c['off_searches_today_before'], [-1, 0, 1, 1e9],
                                                           labels=['0', '1', '2+'])),
        'hit_rate_by_decile_of_officer_search_rate': table(
            pd.qcut(c['off_search_rate_365d'], 10, labels=False, duplicates='drop')),
        'coverage': {
            'share_with_prior_consent_search_365d_pct': round(100 * float((c['off_consent_365d'] > 0).mean()), 1),
            'median_prior_consent_searches_365d': float(c['off_consent_365d'].median()),
            'share_time_missing_pct': round(100 * float(c['time_missing'].mean()), 2),
        },
        'officer_consent_hit_rate_by_driver_race_median': {
            str(k): round(float(v), 4) for k, v in
            c.groupby('subject_race', observed=True)['off_consent_hit_rate_365d_shrunk'].median().items()
            if k in ('white', 'black', 'hispanic')},
        'features': [col for col in f.columns],
    }
    MANIFEST.write_text(json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ['rows_consent', 'n_descriptive', 'base_hit_rate_pct', 'coverage',
                                           'hit_rate_by_decile_of_officer_consent_hit_rate',
                                           'hit_rate_by_officer_last_search_hit',
                                           'hit_rate_by_searches_already_today',
                                           'hit_rate_by_decile_of_officer_search_rate',
                                           'officer_consent_hit_rate_by_driver_race_median']}, indent=1))


if __name__ == '__main__':
    main()
