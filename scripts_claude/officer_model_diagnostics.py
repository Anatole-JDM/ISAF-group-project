"""Diagnostics on the saved test predictions of evaluate_officer_models.py. Fits no model.

  A. stability by year      AUC per test year (2016, 2017, 2018), temporal split, bootstrap CI
  B. PLTR term stability    are the same terms selected on the temporal split and the officer holdout?
                            (exact terms, and variable combinations - thresholds move, variables should not)
  C. economics              if officers searched only the top X% of these searches by score: fruitless
                            searches avoided, finds lost, hit rate; against keeping X% at random; and the
                            best X for a given break-even hit rate
  D. officer counterfactual test searches grouped by the officer's PAST consent hit rate (known before the
                            search): what if the bottom 20% had the hit rate of the median group?
  E. calibration by race    at the same score, is the hit rate the same for Black, Hispanic and white
                            drivers? (predictive parity / sufficiency)

Selective labels apply to everything: only searches that were made are observed, so "avoided" means
"searches among those made that would not have been made"; nothing is known about stops not searched.

Usage   python scripts_claude/officer_model_diagnostics.py
Input   data/officer_model_predictions.parquet, data/pltr_terms.json, data/nashville_consent_searches.parquet
Output  data/officer_model_diagnostics.json + printed summary
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / 'data' / 'officer_model_predictions.parquet'
TABLE = ROOT / 'data' / 'nashville_consent_searches.parquet'
TERMS = ROOT / 'data' / 'pltr_terms.json'
OUT = ROOT / 'data' / 'officer_model_diagnostics.json'

SEED, N_BOOT = 42, 500
RACES = ['white', 'black', 'hispanic']
SHARES = [0.1, 0.2, 0.22, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BREAK_EVEN = [0.15, 0.20, 0.25, 0.30]      # hit rate at which a search is worth its cost
MAIN = [('full_time', 'scorecard'), ('full_time', 'xgboost'),
        ('full_time_officer', 'scorecard'), ('full_time_officer', 'xgboost'),
        ('full_time_officer', 'pltr'), ('full_time_officer', 'pltr_sparse'),
        ('full_time_officer_blind', 'scorecard'), ('full_time_officer_blind', 'xgboost'),
        ('full_time_officer_blind', 'pltr'), ('full_time_officer_blind', 'pltr_sparse'),
        ('officer_only', 'scorecard'), ('officer_only', 'xgboost')]


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def ci(draws):
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return [r4(lo), r4(hi)]


def load():
    pr = pd.read_parquet(PRED)
    t = pd.read_parquet(TABLE, columns=['stop_id', 'date', 'contraband_found', 'subject_race', 'officer_id_hash',
                                        'off_consent_hit_rate_365d_shrunk', 'off_log_search_ratio_black_white',
                                        'off_log_search_ratio_hisp_white'])
    t['y'] = t['contraband_found'].astype(int)
    t['year'] = t['date'].dt.year
    t['subject_race'] = t['subject_race'].astype('object').where(t['subject_race'].notna(), 'missing')
    d = pr.merge(t, on='stop_id', how='left', validate='many_to_one')
    assert d['y'].notna().all()
    return d


def get(d, split, mode, model):
    x = d[(d['split'] == split) & (d['mode'] == mode) & (d['model'] == model)]
    return x.reset_index(drop=True)


# --------------------------------------------------------------------------- A
def by_year(d, rng):
    out = {}
    for mode, model in MAIN:
        x = get(d, 'temporal', mode, model)
        if x.empty:
            continue
        rows = {}
        for yr, g in x.groupby('year'):
            y, p = g['y'].to_numpy(), g['p'].to_numpy()
            draws = []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(y), len(y))
                if 0 < y[i].sum() < len(i):
                    draws.append(roc_auc_score(y[i], p[i]))
            rows[int(yr)] = {'n': int(len(g)), 'base_rate': r4(y.mean()), 'auc': r4(roc_auc_score(y, p)), 'ci95': ci(draws)}
        out[f'{mode}/{model}'] = rows
    return out


# --------------------------------------------------------------------------- B
def term_stability():
    if not TERMS.exists():
        return {'note': 'no PLTR terms file'}
    T = json.loads(TERMS.read_text())
    out = {}
    for key in [k for k in T if k.startswith('temporal/')]:
        other = key.replace('temporal/', 'officer_holdout/', 1)
        if other not in T:
            continue
        a, b = pd.DataFrame(T[key]), pd.DataFrame(T[other])
        if a.empty or b.empty:
            continue
        ta, tb = set(a['term']), set(b['term'])
        va, vb = set(a['variables']), set(b['variables'])
        in_b = a['variables'].isin(vb)
        out[key.split('/', 1)[1]] = {
            'terms_temporal': len(a), 'terms_officer_holdout': len(b),
            'exact_terms_shared': len(ta & tb), 'exact_jaccard': r4(len(ta & tb) / len(ta | tb)),
            'variable_combos_shared': len(va & vb), 'variable_combo_jaccard': r4(len(va & vb) / len(va | vb)),
            'importance_share_of_temporal_terms_whose_variables_reappear': r4(a.loc[in_b, 'importance'].sum() / a['importance'].sum()),
            'top10_temporal': [{'term': r.term, 'odds_ratio': r4(r.odds_ratio), 'importance_share': r4(r.importance_share),
                                'variables_also_in_holdout': bool(v)}
                               for r, v in zip(a.head(10).itertuples(), in_b.head(10))],
        }
    return out


# --------------------------------------------------------------------------- C
def keep_top(p, share):
    k = int(round(share * len(p)))
    sel = np.zeros(len(p), bool)
    sel[np.argsort(-p, kind='mergesort')[:k]] = True
    return sel


def economics(d, rng):
    out = {}
    for mode, model in MAIN:
        x = get(d, 'temporal', mode, model)
        if x.empty:
            continue
        y, p = x['y'].to_numpy(), x['p'].to_numpy()
        H, M = int(y.sum()), int((1 - y).sum())
        curve = {}
        for s in SHARES:
            sel = keep_top(p, s)
            hk, mk = int(y[sel].sum()), int((1 - y[sel]).sum())
            avoided, lost = M - mk, H - hk
            curve[str(s)] = {
                'searches_kept': int(sel.sum()), 'hit_rate_kept': r4(hk / max(sel.sum(), 1)),
                'fruitless_avoided': avoided, 'fruitless_avoided_share': r4(avoided / M),
                'finds_lost': lost, 'finds_lost_share': r4(lost / H),
                'finds_lost_per_100_fruitless_avoided': r4(100 * lost / avoided) if avoided else None}
        # finds lost per 100 fruitless searches avoided when keeping half, with bootstrap CI
        draws = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            sel = keep_top(p[i], 0.5)
            yy = y[i]
            av = int((1 - yy).sum() - (1 - yy[sel]).sum())
            draws.append(100 * (yy.sum() - yy[sel].sum()) / av if av else np.nan)
        # best share for a given break-even hit rate: value = finds - c * fruitless, c = b / (1 - b)
        order = np.argsort(-p, kind='mergesort')
        cum_h = np.concatenate([[0], np.cumsum(y[order])])
        cum_m = np.concatenate([[0], np.cumsum(1 - y[order])])
        be = {}
        for b in BREAK_EVEN:
            c = b / (1 - b)
            val = cum_h - c * cum_m
            k = int(np.argmax(val))
            be[str(b)] = {'best_share_searched': r4(k / len(y)), 'value_vs_search_all_in_finds': r4(val[k] - val[-1]),
                          'finds_lost': int(H - cum_h[k]), 'fruitless_avoided': int(M - cum_m[k])}
        out[f'{mode}/{model}'] = {
            'n': int(len(y)), 'hits': H, 'fruitless': M, 'hit_rate_all': r4(H / len(y)),
            'random_finds_lost_per_100_fruitless_avoided': r4(100 * H / M),
            'keep_top_share': curve,
            'keep_half_finds_lost_per_100_avoided_ci95': ci(draws),
            'best_share_by_break_even_hit_rate': be}
    return out


# --------------------------------------------------------------------------- D
def officer_counterfactual(d, rng):
    x = get(d, 'temporal', 'full_time', 'scorecard')           # any model: we only need the test rows
    x = x[x['officer_id_hash'].notna()].reset_index(drop=True)
    x['q'] = pd.qcut(x['off_consent_hit_rate_365d_shrunk'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    M_all = int((1 - x['y']).sum())

    def table(x):
        g = x.groupby('q')
        return pd.DataFrame({'searches': g.size(), 'hits': g['y'].sum()})

    def cf(tb):
        low, med = tb.loc[1], tb.loc[3]
        h_low, h_med = low['hits'] / low['searches'], med['hits'] / med['searches']
        needed = low['hits'] / h_med                          # same finds at the median group's hit rate
        return h_low, h_med, low['searches'] - needed, low['searches'] * (h_med - h_low)

    tb = table(x)
    h_low, h_med, avoided, extra = cf(tb)
    # bootstrap over OFFICERS (searches by the same officer are not independent)
    officers = x['officer_id_hash'].unique()
    idx = {o: i for o, i in x.groupby('officer_id_hash').indices.items()}
    dr = {'h_low': [], 'h_med': [], 'avoided': []}
    for _ in range(N_BOOT):
        pick = rng.choice(officers, len(officers))
        b = x.iloc[np.concatenate([idx[o] for o in pick])]
        tl, tm, av, _ = cf(table(b))
        dr['h_low'].append(tl); dr['h_med'].append(tm); dr['avoided'].append(av)

    groups = {}
    for q, g in x.groupby('q'):
        rc = g['subject_race'].value_counts(normalize=True)
        groups[int(q)] = {
            'searches': int(len(g)), 'officers': int(g['officer_id_hash'].nunique()),
            'past_consent_hit_rate_range': [r4(g['off_consent_hit_rate_365d_shrunk'].min()),
                                            r4(g['off_consent_hit_rate_365d_shrunk'].max())],
            'hit_rate': r4(g['y'].mean()),
            'share_of_searched_drivers': {r: r4(rc.get(r, 0.0)) for r in RACES},
            'median_officer_search_skew_black_white': r4(np.exp(g['off_log_search_ratio_black_white'].median())),
            'median_officer_search_skew_hisp_white': r4(np.exp(g['off_log_search_ratio_hisp_white'].median()))}
    low = x[x['q'] == 1]
    minority = float(low['subject_race'].isin(['black', 'hispanic']).mean())
    return {
        'grouping': "test searches (2016-2018) in fifths of the officer's past consent hit rate, known before the search",
        'groups': groups,
        'bottom_fifth_hit_rate': [r4(h_low), ci(dr['h_low'])],
        'median_fifth_hit_rate': [r4(h_med), ci(dr['h_med'])],
        'if_bottom_fifth_had_median_hit_rate': {
            'same_finds_with_fewer_searches__fruitless_avoided': [int(round(avoided)), ci(dr['avoided'])],
            'as_share_of_all_fruitless_test_searches': r4(avoided / M_all),
            'of_which_black_or_hispanic_drivers_if_proportional': int(round(avoided * minority)),
            'or_same_searches__extra_finds': int(round(extra))},
        'assumption': 'illustrative: the bottom group keeps its finds and drops searches until it reaches the '
                      'median group hit rate; bootstrap over officers'}


# --------------------------------------------------------------------------- E
def calibration(d, rng):
    out = {}
    for mode, model in MAIN:
        if mode == 'officer_only':
            continue
        x = get(d, 'temporal', mode, model)
        if x.empty:
            continue
        x['bin'] = pd.qcut(x['p'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        large = {}
        for r in RACES:
            g = x[x['subject_race'] == r]
            large[r] = {'n': int(len(g)), 'mean_predicted': r4(g['p'].mean()), 'observed': r4(g['y'].mean()),
                        'observed_over_predicted': r4(g['y'].mean() / g['p'].mean())}
        bins = {}
        for b, g in x.groupby('bin'):
            bins[int(b)] = {r: {'n': int((g['subject_race'] == r).sum()),
                                'observed': r4(g.loc[g['subject_race'] == r, 'y'].mean()) if (g['subject_race'] == r).any() else None,
                                'mean_predicted': r4(g.loc[g['subject_race'] == r, 'p'].mean()) if (g['subject_race'] == r).any() else None}
                            for r in RACES}

        def gap(xx, grp):
            # at equal score: weighted mean over score fifths of (hit rate of grp - hit rate of white)
            num = den = 0.0
            for _, g in xx.groupby('bin'):
                a, w = g[g['subject_race'] == grp], g[g['subject_race'] == 'white']
                if len(a) and len(w):
                    num += len(a) * (a['y'].mean() - w['y'].mean())
                    den += len(a)
            return num / den if den else np.nan

        gaps = {}
        for grp in ['black', 'hispanic']:
            draws = [gap(x.iloc[rng.integers(0, len(x), len(x))], grp) for _ in range(N_BOOT)]
            gaps[f'{grp}_minus_white'] = [r4(gap(x, grp)), ci(draws)]
        out[f'{mode}/{model}'] = {'calibration_in_the_large': large, 'by_score_fifth': bins,
                                  'hit_rate_gap_at_equal_score': gaps}
    return out


def main():
    rng = np.random.default_rng(SEED)
    d = load()
    res = {'A_auc_by_test_year': by_year(d, rng), 'B_pltr_term_stability': term_stability(),
           'C_economics': economics(d, rng), 'D_officer_counterfactual': officer_counterfactual(d, rng),
           'E_calibration_by_race': calibration(d, rng),
           'caveats': ['Selective labels: only searches that were made are observed.',
                       'Economics treat each search as independent and ignore deterrence and the value of '
                       'the contraband; the break-even hit rate is an assumption, shown over a range.',
                       'Calibration: all models were trained on 2010-2015 (16.1% hit rate) and tested on '
                       '2016-2018 (21.4%), so every group is under-predicted; compare groups, not levels.',
                       'Officer IDs are hashed; results are reported for groups of officers only.']}
    OUT.write_text(json.dumps(res, indent=1))

    # ---- printed summary
    print('\nA. AUC by test year (temporal split)')
    for k, v in res['A_auc_by_test_year'].items():
        print(f'  {k:38}' + '  '.join(f'{y}: {r["auc"]:.3f} {r["ci95"]}' for y, r in v.items()))
    print('\nB. PLTR term stability, temporal vs officer holdout')
    for k, v in res['B_pltr_term_stability'].items():
        if isinstance(v, dict):
            print(f'  {k:38} terms {v["terms_temporal"]}/{v["terms_officer_holdout"]}  exact shared {v["exact_terms_shared"]}'
                  f'  variable combos shared {v["variable_combos_shared"]} (Jaccard {v["variable_combo_jaccard"]})'
                  f'  importance reappearing {v["importance_share_of_temporal_terms_whose_variables_reappear"]}')
    print('\nC. Economics (temporal split): keep the top half of searches by score')
    for k, v in res['C_economics'].items():
        h = v['keep_top_share']['0.5']
        print(f'  {k:38} avoided {h["fruitless_avoided"]:5} fruitless ({h["fruitless_avoided_share"]:.0%}), '
              f'lost {h["finds_lost"]:4} finds ({h["finds_lost_share"]:.0%}); finds lost per 100 avoided '
              f'{h["finds_lost_per_100_fruitless_avoided"]} {v["keep_half_finds_lost_per_100_avoided_ci95"]} '
              f'(random {v["random_finds_lost_per_100_fruitless_avoided"]})')
    oc = res['D_officer_counterfactual']
    print('\nD. Officer counterfactual (test searches by fifth of the officer\'s past consent hit rate)')
    for q, g in oc['groups'].items():
        print(f'  fifth {q}: {g["searches"]:5} searches, {g["officers"]:4} officers, hit rate {g["hit_rate"]:.3f}, '
              f'drivers W/B/H {g["share_of_searched_drivers"]}, skew B/W {g["median_officer_search_skew_black_white"]}')
    print('  ', json.dumps(oc['if_bottom_fifth_had_median_hit_rate']))
    print('\nE. Hit-rate gap at equal score (minority - white), temporal split')
    for k, v in res['E_calibration_by_race'].items():
        g = v['hit_rate_gap_at_equal_score']
        print(f'  {k:38} Black {g["black_minus_white"]}  Hispanic {g["hispanic_minus_white"]}')
    print(f'\nwrote {OUT.name}')


if __name__ == '__main__':
    main()
