"""Fairness, calibration and within-officer AUC for the three arms of the MATCHED run. Fits no model.

The matched run (src/train.py, mode "matched") is the only run where all three families - logistic
scorecard, XGBoost, TabPFN - were trained on the same rows (2,000) and scored on the same 9,113 test
searches (2016-2018). Its test scores are saved in outputs/scores__consent__matched.parquet with the
driver's race, the officer and the outcome, so these checks need no refit and no TabPFN token.

The app computes the fairness rates live on its Fairness page; this script writes them down, with
bootstrap CIs, for the report, and adds what the app does not have for TabPFN:
  * within-officer AUC: AUC inside each officer's own test searches (officers with >= 10 test searches
    and both outcomes, weighted by searches), CI by resampling officers - the same measure as
    officer_model_diagnostics.py section F
  * calibration by race (mean predicted - observed) and the hit-rate gap vs white drivers at equal score

Input   outputs/scores__consent__matched.parquet (on the `release` branch; git-ignored on main):
            git show origin/release:outputs/scores__consent__matched.parquet > outputs/scores__consent__matched.parquet
Output  data/matched_arms_diagnostics.json + printed summary
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / 'outputs' / 'scores__consent__matched.parquet'
OUT = ROOT / 'data' / 'matched_arms_diagnostics.json'
K, SEED, N_BOOT, MIN_OFFICER_SEARCHES = 2000, 42, 500, 10          # K as in FINDINGS / the deck
RACES = ['white', 'black', 'hispanic']
ARMS = {'scorecard': 'white box (logistic)', 'gbm': 'XGBoost', 'tabpfn': 'TabPFN'}


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def ci(draws):
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return [r4(lo), r4(hi)]


def within_officer(d, p, rng):
    per = np.array([(len(h), roc_auc_score(h['y'], p[h.index])) for _, h in d.groupby('officer_id_hash')
                    if len(h) >= MIN_OFFICER_SEARCHES and 0 < h['y'].sum() < len(h)])
    w, a = per[:, 0], per[:, 1]
    draws = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(w), len(w))
        draws.append((w[i] * a[i]).sum() / w[i].sum())
    officer_mean = pd.Series(p, index=d.index)[d['officer_id_hash'].notna()].groupby(d['officer_id_hash']).transform('mean')
    pp = pd.Series(p, index=d.index)[officer_mean.index]
    return {'within_officer_auc': r4((w * a).sum() / w.sum()), 'ci95_resampling_officers': ci(draws),
            'officers': int(len(w)), 'searches': int(w.sum()),
            'share_of_score_variance_between_officers': r4(((officer_mean - pp.mean()) ** 2).mean() / pp.var(ddof=0))}


def fairness(d, p, rng):
    sel = np.zeros(len(p), bool)
    sel[np.argsort(-p, kind='mergesort')[:K]] = True
    y, race = d['y'].to_numpy(), d['race'].to_numpy()

    def rates(yy, ss, pp):
        return np.array([ss.mean(), ss[yy == 1].mean() if (yy == 1).any() else np.nan,
                         ss[yy == 0].mean() if (yy == 0).any() else np.nan,
                         yy[ss].mean() if ss.any() else np.nan, pp.mean() - yy.mean()])

    names = ['share_flagged', 'true_positive_rate', 'false_positive_rate', 'hit_rate_flagged',
             'calibration_pred_minus_obs']
    idx = {g: np.flatnonzero(race == g) for g in RACES}
    base = {g: rates(y[i], sel[i], p[i]) for g, i in idx.items()}
    boot = {g: np.array([rates(y[j], sel[j], p[j]) for j in (rng.choice(i, len(i)) for _ in range(N_BOOT))])
            for g, i in idx.items()}
    out = {g: {nm: [r4(base[g][c]), ci(boot[g][:, c])] for c, nm in enumerate(names)} | {'n': int(len(idx[g]))}
           for g in RACES}
    for g in ['black', 'hispanic']:
        out[f'{g}_minus_white'] = {nm: [r4(base[g][c] - base['white'][c]), ci(boot[g][:, c] - boot['white'][:, c])]
                                   for c, nm in enumerate(names)}
    bw = np.isin(race, ['white', 'black'])
    tab = pd.crosstab(race[bw], sel[bw])
    chi2 = stats.chi2_contingency(tab, correction=False)
    out['chi2_selection_black_vs_white'] = [r4(chi2.statistic), float(chi2.pvalue)]
    # hit-rate gap vs white at equal score (weighted over score fifths)
    bins = pd.qcut(pd.Series(p).rank(method='first'), 5, labels=False).to_numpy()

    def gap(yy, rr, bb, grp):
        num = den = 0.0
        for b in range(5):
            a, w = (bb == b) & (rr == grp), (bb == b) & (rr == 'white')
            if a.any() and w.any():
                num += a.sum() * (yy[a].mean() - yy[w].mean())
                den += a.sum()
        return num / den if den else np.nan

    for grp in ['black', 'hispanic']:
        draws = []
        for _ in range(N_BOOT):
            j = rng.integers(0, len(y), len(y))
            draws.append(gap(y[j], race[j], bins[j], grp))
        out[f'hit_rate_gap_at_equal_score_{grp}_minus_white'] = [r4(gap(y, race, bins, grp)), ci(draws)]
    return out


def main():
    if not SCORES.exists():
        raise SystemExit(f'missing {SCORES.relative_to(ROOT)} - extract it from the release branch, see the docstring')
    d = pd.read_parquet(SCORES).reset_index(drop=True)
    d['race'] = d['subject_race'].astype('object').where(d['subject_race'].notna(), 'missing')
    rng = np.random.default_rng(SEED)
    res = {'source': 'outputs/scores__consent__matched.parquet (src matched run: every arm trained on the same '
                     '2,000 rows of 2010-2015, scored on the 9,113 consent searches of 2016-2018)',
           'K': K, 'test_n': int(len(d)), 'base_rate': r4(d['y'].mean()), 'arms': {}}
    for arm in ARMS:
        p = d[f'score_{arm}'].to_numpy()
        res['arms'][arm] = {'auc': r4(roc_auc_score(d['y'], p)), 'score_sd': r4(p.std()),
                            'within_officer': within_officer(d, p, rng), 'fairness_top_k': fairness(d, p, rng)}
    OUT.write_text(json.dumps(res, indent=1))

    print(f'Matched run: {len(d):,} test searches, K = {K:,} flagged\n')
    for arm, lab in ARMS.items():
        a = res['arms'][arm]; w = a['within_officer']; f = a['fairness_top_k']
        print(f'{lab:22} pooled AUC {a["auc"]:.3f} | within officer {w["within_officer_auc"]:.3f} '
              f'{w["ci95_resampling_officers"]} | {w["share_of_score_variance_between_officers"]:.0%} of score '
              f'variance between officers | score sd {a["score_sd"]:.3f}')
        for key, lab2 in [('share_flagged', 'flagged'), ('false_positive_rate', 'FPR'),
                          ('true_positive_rate', 'TPR'), ('hit_rate_flagged', 'hit rate of flagged'),
                          ('calibration_pred_minus_obs', 'predicted - observed')]:
            print(f'   {lab2:22} ' + '  '.join(f'{g} {f[g][key][0]:.3f}' if f[g][key][0] is not None else f'{g} n/a'
                                               for g in RACES))
        print(f'   chi2 selection B vs W {f["chi2_selection_black_vs_white"][0]:.0f}; hit-rate gap at equal score: '
              f'Black {f["hit_rate_gap_at_equal_score_black_minus_white"]}, '
              f'Hispanic {f["hit_rate_gap_at_equal_score_hispanic_minus_white"]}')
    print(f'\nwrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
