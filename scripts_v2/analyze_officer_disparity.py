"""Officer-level disparity analysis: do officers who search minority drivers more find more, or less?

Descriptive, no model. Consent-only searches 2010-2018, per officer, comparing each officer
with himself (his own stops), which removes the officer's patrol area from the comparison.

For each officer with enough data (>= MIN_W white and >= MIN_B Black consent searches):
  search_ratio_bw  = consent searches per stop on Black drivers / same on white drivers
                     > 1: this officer asks Black drivers for consent more often than white drivers
  hit_b, hit_w     = share of those searches that found contraband
  hit_gap_bw       = hit_b - hit_w. The outcome test: if an officer searches Black drivers on
                     weaker evidence, his searches of Black drivers succeed LESS often (gap < 0)

Questions answered
  Q1  Are officers' search skew and their outcome gap related? (Spearman, bootstrap CI over officers)
  Q2  Officers grouped into quintiles of search skew: pooled hit rates by group, with CIs
  Q3  Outcome test officer by officer: how many show a significant negative / positive gap,
      against the 2.5% per side expected by chance
  Q4  Concentration: what share of all consent searches of Black drivers do the most skewed
      20% of officers carry out?
  Q5  The same for Hispanic drivers (fewer officers qualify)

Caveats written into the output: small samples per officer (shrinkage is NOT applied here -
these are the raw rates, restricted to officers with enough searches), the outcome test's
infra-marginality problem, contraband_found filled with FALSE when blank, and officer IDs are
hashed - results are reported by group of officers, never as a list of individuals.

Output: data/officer_disparity_findings.json (git) + printed summary.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
STOPS = ROOT / 'opp_data' / 'processed' / 'nashville_clean.parquet'
OUT = ROOT / 'data' / 'officer_disparity_findings.json'
MIN_W, MIN_B, MIN_H = 30, 30, 15
N_BOOT = 2000
SEED = 42


def officer_table(s, minority, min_minority):
    g = s.groupby('officer_id_hash')
    t = pd.DataFrame({
        'stops_w': g['is_w'].sum(), 'stops_m': g[f'is_{minority}'].sum(),
        'cons_w': g['cons_w'].sum(), 'cons_m': g[f'cons_{minority}'].sum(),
        'hits_w': g['hit_w'].sum(), 'hits_m': g[f'hit_{minority}'].sum(),
        'cons_all': g['cons'].sum(), 'hits_all': g['hit'].sum()})
    t = t[(t['cons_w'] >= MIN_W) & (t['cons_m'] >= min_minority)].copy()
    t['search_ratio'] = (t['cons_m'] / t['stops_m']) / (t['cons_w'] / t['stops_w'])
    t['hit_w'] = t['hits_w'] / t['cons_w']
    t['hit_m'] = t['hits_m'] / t['cons_m']
    t['hit_gap'] = t['hit_m'] - t['hit_w']
    t['hit_all'] = t['hits_all'] / t['cons_all']
    # two-proportion z-test per officer
    p = (t['hits_w'] + t['hits_m']) / (t['cons_w'] + t['cons_m'])
    se = np.sqrt(p * (1 - p) * (1 / t['cons_w'] + 1 / t['cons_m']))
    t['z'] = t['hit_gap'] / se.replace(0, np.nan)
    t['p'] = 2 * stats.norm.sf(np.abs(t['z']))
    return t


def boot_ci(values_fn, n, rng):
    draws = [values_fn(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def analyse(s, minority, label, min_minority, rng):
    t = officer_table(s, minority, min_minority)
    n = len(t)
    res = {'officers': n, 'min_searches': {'white': MIN_W, label: min_minority},
           'searches_covered': int(t['cons_all'].sum())}
    if n < 20:
        res['note'] = 'too few officers qualify'
        return res, t

    # Q1: relation between search skew and outcome gap / overall success
    x = np.log(t['search_ratio'].to_numpy()); gap = t['hit_gap'].to_numpy(); hit = t['hit_all'].to_numpy()
    res['median_search_ratio'] = round(float(t['search_ratio'].median()), 3)
    res['share_officers_ratio_above_1'] = round(float((t['search_ratio'] > 1).mean()), 3)
    res['spearman_skew_vs_gap'] = {
        'rho': round(float(stats.spearmanr(x, gap).statistic), 3),
        'ci95': boot_ci(lambda i: stats.spearmanr(x[i], gap[i]).statistic, n, rng)}
    res['spearman_skew_vs_overall_hit_rate'] = {
        'rho': round(float(stats.spearmanr(x, hit).statistic), 3),
        'ci95': boot_ci(lambda i: stats.spearmanr(x[i], hit[i]).statistic, n, rng)}

    # Q2: quintiles of search skew, pooled rates with bootstrap CI over officers
    t['quintile'] = pd.qcut(t['search_ratio'], 5, labels=[1, 2, 3, 4, 5])
    q2 = {}
    for qn, sub in t.groupby('quintile', observed=True):
        a = sub[['hits_all', 'cons_all', 'hits_w', 'cons_w', 'hits_m', 'cons_m']].to_numpy()
        pooled = lambda i, num, den: a[i, num].sum() / a[i, den].sum()
        m = len(sub)
        q2[int(qn)] = {
            'officers': m, 'search_ratio_range': [round(float(sub['search_ratio'].min()), 2),
                                                  round(float(sub['search_ratio'].max()), 2)],
            'consent_searches': int(sub['cons_all'].sum()),
            'hit_rate_all': [round(float(pooled(np.arange(m), 0, 1)), 4), boot_ci(lambda i: pooled(i, 0, 1), m, rng)],
            'hit_rate_white': [round(float(pooled(np.arange(m), 2, 3)), 4), boot_ci(lambda i: pooled(i, 2, 3), m, rng)],
            f'hit_rate_{label}': [round(float(pooled(np.arange(m), 4, 5)), 4), boot_ci(lambda i: pooled(i, 4, 5), m, rng)]}
    res['by_quintile_of_search_skew'] = q2

    # Q3: officer-by-officer outcome test
    res['outcome_test'] = {
        'share_gap_negative': round(float((t['hit_gap'] < 0).mean()), 3),
        'significant_negative_p05': int(((t['p'] < 0.05) & (t['hit_gap'] < 0)).sum()),
        'significant_positive_p05': int(((t['p'] < 0.05) & (t['hit_gap'] > 0)).sum()),
        'expected_by_chance_each_side': round(0.025 * n, 1),
        'median_gap': round(float(t['hit_gap'].median()), 4)}

    # Q4: concentration of minority searches among the most skewed officers
    top = t['quintile'] == 5
    res['concentration'] = {
        'top_quintile_share_of_officers': round(float(top.mean()), 3),
        f'top_quintile_share_of_{label}_consent_searches': round(float(t.loc[top, 'cons_m'].sum() / t['cons_m'].sum()), 3),
        'top_quintile_share_of_white_consent_searches': round(float(t.loc[top, 'cons_w'].sum() / t['cons_w'].sum()), 3)}
    return res, t


def main():
    rng = np.random.default_rng(SEED)
    cols = ['date', 'officer_id_hash', 'subject_race', 'search_conducted', 'contraband_found',
            'raw_search_consent', 'raw_search_arrest', 'raw_search_warrant', 'raw_search_inventory',
            'raw_search_plain_view']
    s = pd.read_parquet(STOPS, columns=cols)
    s = s[(s['date'] < '2019-01-01') & s['officer_id_hash'].notna()].copy()
    srch = s['search_conducted'].fillna(False).astype(bool)
    others = ['raw_search_arrest', 'raw_search_warrant', 'raw_search_inventory', 'raw_search_plain_view']
    cons = srch & s['raw_search_consent'].fillna(False).astype(bool) & ~s[others].fillna(False).astype(bool).any(axis=1)
    hit = cons & s['contraband_found'].fillna(False).astype(bool)
    s['cons'], s['hit'] = cons.astype(int), hit.astype(int)
    for g, race in {'w': 'white', 'b': 'black', 'h': 'hispanic'}.items():
        is_g = s['subject_race'].eq(race).fillna(False).astype(bool)
        s[f'is_{g}'] = is_g.astype(int)
        s[f'cons_{g}'] = (is_g & cons).astype(int)
        s[f'hit_{g}'] = (is_g & hit).astype(int)

    black, tb = analyse(s, 'b', 'black', MIN_B, rng)
    hisp, th = analyse(s, 'h', 'hispanic', MIN_H, rng)
    out = {
        'scope': 'consent-only searches 2010-2018, officers compared with their own stops',
        'black_vs_white': black, 'hispanic_vs_white': hisp,
        'caveats': [
            'Raw per-officer rates (no shrinkage); only officers with enough searches are included, '
            'so busy officers are over-represented.',
            'Outcome tests can mislead when the risk distribution differs by group (infra-marginality); '
            'a lower hit rate is consistent with, not proof of, a lower evidentiary bar.',
            'contraband_found was filled with FALSE when blank in the source; officers who record less '
            'look less successful.',
            'Within-officer comparison controls for the officer and roughly for his area, not for '
            'what he observed at the roadside.',
            'Officer IDs are hashed. Report groups of officers, never a list of individuals.'],
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
