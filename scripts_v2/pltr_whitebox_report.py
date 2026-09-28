"""White-box report: everything section 2 of the team document needs for PLTR.

The white box is the SPARSE PLTR on `full_time_officer` (19 terms, penalty = weakest keeping <= 30 terms,
chosen on training data), temporal split - the same features and split as the XGBoost arm. Its race-blind
version (`full_time_officer_blind`) is the mitigation; the plain logistic scorecard is the baseline.

Two parts
  default (no model is fitted)  reads data/officer_model_predictions.parquet and data/pltr_terms.json:
      AUC / PR-AUC / Brier with bootstrap CIs; top-22% economics (the team's budget, K = 2,005 of 9,113:
      searches avoided, finds lost, searches per find); fairness by race at that budget (share flagged,
      TPR, FPR, hit rate, calibration) with bootstrap CIs of the gaps vs white; figures
  --fit  refits the white box (~20 min) for what the predictions cannot give:
      * local explanations of 3 test stops (a hit, a miss, the borderline case at the cut-off): PLTR is
        additive in log-odds, so each term's contribution is exact (coefficient x value)
      * coefficient stability: 50 bootstrap resamples of the training set, rules FIXED, adaptive weights and
        lasso re-estimated at the chosen penalty: selection frequency, sign changes, rank correlation of
        importance, test AUC spread
      * random split: stratified 80/20 over 2010-2018, same procedure, vs the time split

Usage   python scripts_claude/pltr_whitebox_report.py            # no fitting
        python scripts_claude/pltr_whitebox_report.py --fit      # + local cases, bootstrap, random split
Output  data/pltr_whitebox_report.json, reports/pltr/*.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_officer_models as E  # noqa: E402
from pltr import PLTR  # noqa: E402

ROOT = E.ROOT
OUT = ROOT / 'data' / 'pltr_whitebox_report.json'
FIG = ROOT / 'reports' / 'pltr'
MODE, BLIND, VARIANT = 'full_time_officer', 'full_time_officer_blind', 'pltr_sparse'
N_BOOT, N_BOOT_FAIR, N_BOOT_COEF = 1000, 500, 50
RACES = E.RACES


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def ci(draws):
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return [r4(lo), r4(hi)]


def preds():
    pr = pd.read_parquet(E.OUT_PRED)
    t = pd.read_parquet(E.TABLE, columns=['stop_id', 'contraband_found', 'subject_race'])
    t['y'] = t['contraband_found'].astype(int)
    t['race'] = t['subject_race'].astype('object').where(t['subject_race'].notna(), 'missing')
    return pr[pr['split'] == 'temporal'].merge(t[['stop_id', 'y', 'race']], on='stop_id', validate='many_to_one')


def get(pr, mode, model):
    x = pr[(pr['mode'] == mode) & (pr['model'] == model)].reset_index(drop=True)
    return x['y'].to_numpy(), x['p'].to_numpy().astype('float64'), x['race'].to_numpy()


def flagged(p):
    k = int(round(E.TOPK_SHARE * len(p)))
    sel = np.zeros(len(p), bool)
    sel[np.argsort(-p, kind='mergesort')[:k]] = True
    return sel


# --------------------------------------------------------------------------- from predictions
def performance(y, p, rng):
    out = {'auc': r4(roc_auc_score(y, p)), 'pr_auc': r4(average_precision_score(y, p)),
           'brier': r4(brier_score_loss(y, p)), 'brier_no_skill': r4(brier_score_loss(y, np.full(len(y), y.mean())))}
    draws = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            draws.append((roc_auc_score(y[i], p[i]), average_precision_score(y[i], p[i]), brier_score_loss(y[i], p[i])))
    draws = np.array(draws)
    out.update({'auc_ci95': ci(draws[:, 0]), 'pr_auc_ci95': ci(draws[:, 1]), 'brier_ci95': ci(draws[:, 2])})
    return out


def economics(y, p):
    sel = flagged(p)
    n, H, k, hk = len(y), int(y.sum()), int(sel.sum()), int(y[sel].sum())
    return {'flagged': k, 'of': n, 'searches_avoided': n - k, 'searches_avoided_share': r4((n - k) / n),
            'finds_kept': hk, 'finds_lost': H - hk, 'finds_lost_share': r4((H - hk) / H),
            'hit_rate_flagged': r4(hk / k), 'hit_rate_all': r4(H / n),
            'searches_per_find_flagged': r4(k / hk), 'searches_per_find_all': r4(n / H)}


def fairness(y, p, race, rng):
    sel = flagged(p)

    def stats(yy, ss, pp):
        return np.array([ss.mean(),
                         ss[yy == 1].mean() if (yy == 1).any() else np.nan,
                         ss[yy == 0].mean() if (yy == 0).any() else np.nan,
                         yy[ss].mean() if ss.any() else np.nan,
                         pp.mean() - yy.mean()])

    names = ['share_flagged', 'true_positive_rate', 'false_positive_rate', 'hit_rate_flagged',
             'calibration_pred_minus_obs']
    groups = {g: np.flatnonzero(race == g) for g in RACES}
    base = {g: stats(y[i], sel[i], p[i]) for g, i in groups.items()}
    boot = {g: [] for g in RACES}
    for _ in range(N_BOOT_FAIR):
        for g, i in groups.items():
            j = rng.choice(i, len(i))
            boot[g].append(stats(y[j], sel[j], p[j]))
    boot = {g: np.array(v) for g, v in boot.items()}
    out = {}
    for g in RACES:
        out[g] = {nm: [r4(base[g][c]), ci(boot[g][:, c])] for c, nm in enumerate(names)}
        out[g]['n'] = int(len(groups[g]))
    for g in ['black', 'hispanic']:
        out[f'{g}_minus_white'] = {nm: [r4(base[g][c] - base['white'][c]), ci(boot[g][:, c] - boot['white'][:, c])]
                                   for c, nm in enumerate(names)}
        out[f'{g}_over_white_share_flagged'] = r4(base[g][0] / base['white'][0])
    return out


def fig_gains(pr):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot([0, 1], [0, 1], color='grey', ls='--', lw=1, label='random')
    for (mode, model), lab in {('full_time', 'scorecard'): "team's scorecard (no officer features)",
                               (MODE, 'xgboost'): 'XGBoost + officer',
                               (MODE, VARIANT): 'PLTR 19 terms + officer',
                               (BLIND, VARIANT): 'PLTR race-blind + officer'}.items():
        y, p, _ = get(pr, mode, model)
        o = np.argsort(-p, kind='mergesort')
        ax.plot(np.arange(1, len(y) + 1) / len(y), np.cumsum(y[o]) / y.sum(), lw=1.8, label=lab)
    ax.axvline(E.TOPK_SHARE, color='k', lw=0.8, alpha=0.5)
    ax.text(E.TOPK_SHARE + 0.01, 0.05, 'top 22%', fontsize=8)
    ax.set(xlabel='share of searches made (highest score first)', ylabel='share of all finds kept',
           title='Gains curve, test 2016-2018 (9,113 consent searches)')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(FIG / 'gains_curve.png', dpi=160)
    plt.close(fig)


READABLE = [('num__', ''), ('cat__subject_race_', 'driver race = '), ('subject_sex in', 'driver sex in'),
            ('off_consent_hit_rate_365d_shrunk', 'officer consent hit rate (past year)'),
            ('off_hit_rate_same_race_past', "officer hit rate on driver's race"),
            ('off_hit_rate_365d_shrunk', 'officer hit rate (past year)'),
            ('off_experience_days', 'officer experience (days)'),
            ('off_minutes_since_first_stop_today', 'min since first stop today'),
            ('off_log_minutes_since_last_search <= 4.7', 'last search < ~110 min ago'),
            ('off_consent_365d', 'officer consent searches (past year)'),
            ('off_stops_365d <= 1.71e+03', 'officer stops (past year) <= 1,710'),
            ('hour_sin > 0.921', 'hour ~4:30-7:30'),
            ('off_hit_gap_hisp_white', 'officer hit gap Hisp-white'),
            ('off_log_search_ratio_hisp_white > 0.475', 'officer Hisp/white search skew > 1.6x'),
            ('off_search_rate_365d', 'officer search rate'), ('subject_age', 'driver age')]


def readable(term):
    for a, b in READABLE:
        term = term.replace(a, b)
    return term if len(term) <= 80 else term[:77] + '...'


def fig_terms(terms):
    t = terms.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#c0392b' if o < 1 else '#2471a3' for o in t['odds_ratio']]
    lo = np.log(t['odds_ratio'])
    ax.barh(range(len(t)), lo, color=colors)
    ax.set_yticks(range(len(t)), [readable(s) for s in t['term']], fontsize=7.5)
    for i, (o, sh) in enumerate(zip(t['odds_ratio'], t['importance_share'])):
        ax.text(np.log(o), i, f'  OR {o:.2f} ({sh:.0%})  ', va='center', fontsize=7,
                ha='left' if o >= 1 else 'right')
    ax.set_xlim(lo.min() - 0.3, lo.max() + 0.3)
    ax.axvline(0, color='k', lw=0.8)
    ax.set(xlabel='log odds ratio (linear terms: per standard deviation)',
           title='PLTR white box: 10 most important terms (share of importance)')
    fig.tight_layout()
    fig.savefig(FIG / 'terms_top10.png', dpi=160)
    plt.close(fig)


def fig_fairness(fair_main, fair_blind):
    metrics = [('share_flagged', 'share flagged'), ('true_positive_rate', 'true positive rate'),
               ('false_positive_rate', 'false positive rate'), ('hit_rate_flagged', 'hit rate of flagged')]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=False)
    w = 0.38
    for ax, (key, lab) in zip(axes, metrics):
        for j, (f, name) in enumerate([(fair_main, 'race-aware'), (fair_blind, 'race-blind')]):
            vals = [f[g][key][0] for g in RACES]
            err = np.array([[f[g][key][0] - f[g][key][1][0], f[g][key][1][1] - f[g][key][0]] for g in RACES]).T
            ax.bar(np.arange(3) + (j - 0.5) * w, vals, w, yerr=err, capsize=3, label=name)
        ax.set_xticks(range(3), RACES)
        ax.set_title(lab, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle('PLTR 19 terms at the top-22% budget, by driver race (95% CI)', fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / 'fairness_by_race.png', dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- refits
def local_cases(ms, d, te, p_te):
    X_te = d.loc[te, ms.num_cols + ms.cat_cols]
    Z = ms._design(X_te).astype('float64')
    contrib = Z * ms.coef_
    names = ms.lin_names_ + [ms._describe(ms.rules_[j]) for j in ms.keep_rules_]
    y = d.loc[te, 'y'].to_numpy()
    cut = np.sort(p_te)[::-1][int(round(E.TOPK_SHARE * len(p_te))) - 1]
    picks = {'hit (highest-scored search that found contraband)': int(np.argmax(np.where(y == 1, p_te, -1))),
             'miss (highest-scored search that found nothing)': int(np.argmax(np.where(y == 0, p_te, -1))),
             'borderline (score closest to the top-22% cut-off)': int(np.argmin(np.abs(p_te - cut)))}
    rows = d.loc[te].reset_index(drop=True)
    out = {'cutoff_score': r4(cut), 'intercept_log_odds': r4(ms.intercept_), 'cases': {}}
    for label, i in picks.items():
        c = contrib[i]
        top = np.argsort(-np.abs(c))[:6]
        out['cases'][label] = {
            'score': r4(p_te[i]), 'found': int(y[i]),
            'driver': {'race': str(rows.loc[i, 'subject_race']), 'sex': str(rows.loc[i, 'subject_sex']),
                       'age': r4(rows.loc[i, 'subject_age'])},
            'officer_past_consent_hit_rate': r4(rows.loc[i, 'off_consent_hit_rate_365d_shrunk']),
            'log_odds': r4(ms.intercept_ + c.sum()),
            'top_contributions_log_odds': [{'term': names[j], 'contribution': r4(c[j])} for j in top if c[j] != 0]}
    return out


def bootstrap_coefs(ms, d, te, rng):
    Z, y = ms._Z_train, np.asarray(ms._y_train)
    Z_te = ms._design(d.loc[te, ms.num_cols + ms.cat_cols]).astype('float64')
    y_te = d.loc[te, 'y'].to_numpy()
    full = ms.coef_
    sel_idx = np.flatnonzero(full != 0)
    B = np.zeros((N_BOOT_COEF, Z.shape[1]))
    aucs = []
    for b in range(N_BOOT_COEF):
        i = rng.integers(0, len(y), len(y))
        scale, adapt = ms._stage2_weights(Z[i], y[i])
        las = ms._lasso((Z[i] / scale * adapt).astype('float32'), y[i], ms.C_)
        B[b] = las.coef_.ravel() * adapt / scale
        aucs.append(roc_auc_score(y_te, Z_te @ B[b] + float(las.intercept_[0])))
    sd = Z.std(axis=0)
    imp = np.abs(B) * sd
    rho = [spearmanr(imp[a], imp[c]).statistic for a in range(N_BOOT_COEF) for c in range(a + 1, N_BOOT_COEF)]
    names = ms.lin_names_ + [ms._describe(ms.rules_[j]) for j in ms.keep_rules_]
    per_term = []
    for j in sel_idx[np.argsort(-np.abs(full[sel_idx]) * sd[sel_idx])]:
        nz = B[:, j] != 0
        per_term.append({'term': names[j], 'odds_ratio_full': r4(np.exp(full[j])),
                         'selected_share': r4(nz.mean()),
                         'same_sign_share_when_selected': r4((np.sign(B[nz, j]) == np.sign(full[j])).mean()) if nz.any() else None,
                         'odds_ratio_ci95_over_resamples': ci(np.exp(B[:, j]))})
    others = np.setdiff1d(np.arange(Z.shape[1]), sel_idx)
    return {'resamples': N_BOOT_COEF,
            'note': 'rules fixed; adaptive weights and lasso re-estimated at the chosen penalty on each resample',
            'terms_selected_per_resample': [int(k) for k in (B != 0).sum(axis=1)],
            'selected_terms_kept_in_90pct_of_resamples': int(sum(t['selected_share'] >= 0.9 for t in per_term)),
            'selected_terms_with_any_sign_change': int(sum(t['same_sign_share_when_selected'] is not None
                                                          and t['same_sign_share_when_selected'] < 1 for t in per_term)),
            'unselected_terms_entering_in_over_half': int(((B[:, others] != 0).mean(axis=0) > 0.5).sum()),
            'importance_rank_correlation_mean_min': [r4(np.mean(rho)), r4(np.min(rho))],
            'test_auc_mean_sd_min_max': [r4(np.mean(aucs)), r4(np.std(aucs, ddof=1)), r4(np.min(aucs)), r4(np.max(aucs))],
            'per_term': per_term}


def random_split(d, num, cat):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=E.SEED)
    tr, te = next(sss.split(d, d['y']))
    X = d[num + cat]
    m = PLTR(num_cols=num, cat_cols=cat, random_state=E.SEED).fit(X.iloc[tr], d['y'].iloc[tr].to_numpy())
    ms = m.at_most(E.PLTR_MAX_TERMS)
    sc = E.make('scorecard', num, cat).fit(X.iloc[tr], d['y'].iloc[tr])
    y_te = d['y'].iloc[te].to_numpy()
    return {'n_train': int(len(tr)), 'n_test': int(len(te)),
            'pltr_sparse_auc': r4(roc_auc_score(y_te, ms.predict_proba(X.iloc[te])[:, 1])),
            'pltr_sparse_terms': int(np.count_nonzero(ms.coef_)),
            'scorecard_auc': r4(roc_auc_score(y_te, sc.predict_proba(X.iloc[te])[:, 1]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', action='store_true', help='refit for local cases, bootstrap and random split (~20 min)')
    args = ap.parse_args()
    rng = np.random.default_rng(E.SEED)
    FIG.mkdir(parents=True, exist_ok=True)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}

    pr = preds()
    terms = pd.DataFrame(json.loads(E.OUT_TERMS.read_text())[f'temporal/{MODE}/{VARIANT}'])
    models = {'white_box': (MODE, VARIANT), 'white_box_race_blind': (BLIND, VARIANT),
              'white_box_1se_61_terms': (MODE, 'pltr'), 'baseline_scorecard_officer': (MODE, 'scorecard'),
              'team_scorecard_no_officer': ('full_time', 'scorecard')}
    res['from_predictions'] = {}
    for key, (mode, model) in models.items():
        y, p, race = get(pr, mode, model)
        res['from_predictions'][key] = {'mode': mode, 'model': model, 'performance': performance(y, p, rng),
                                        'economics_top22': economics(y, p), 'fairness_top22': fairness(y, p, race, rng)}
        print(f'{key:28} AUC {res["from_predictions"][key]["performance"]["auc"]}', flush=True)
    res['top10_terms'] = terms.head(10)[['term', 'odds_ratio', 'avg_marginal_effect', 'support_share',
                                         'importance_share']].round(4).to_dict(orient='records')
    fig_gains(pr)
    fig_terms(terms)
    fig_fairness(res['from_predictions']['white_box']['fairness_top22'],
                 res['from_predictions']['white_box_race_blind']['fairness_top22'])

    if args.fit:
        d = E.load()
        tr, te = E.splits(d)['temporal']
        num, cat = E.MODES[MODE]
        t0 = time.time()
        m = PLTR(num_cols=num, cat_cols=cat, random_state=E.SEED).fit(d.loc[tr, num + cat], d.loc[tr, 'y'].to_numpy())
        ms = m.at_most(E.PLTR_MAX_TERMS)
        ms._y_train = d.loc[tr, 'y'].to_numpy()
        p_te = ms.predict_proba(d.loc[te, num + cat])[:, 1]
        saved = get(pr, MODE, VARIANT)[1]
        E.log(f'refit {time.time() - t0:.0f}s; {np.count_nonzero(ms.coef_)} terms; AUC {roc_auc_score(d.loc[te, "y"], p_te):.4f}'
              f' (saved run: {roc_auc_score(get(pr, MODE, VARIANT)[0], saved):.4f})')
        res['local_explanations'] = local_cases(ms, d, te, p_te)
        E.log('local explanations done')
        res['coefficient_bootstrap'] = bootstrap_coefs(ms, d, te, rng)
        E.log('bootstrap done')
        res['random_split'] = random_split(d, num, cat)
        res['random_split']['time_split_pltr_sparse_auc'] = r4(roc_auc_score(d.loc[te, 'y'], p_te))
        E.log(f'random split done: {res["random_split"]}')

    OUT.write_text(json.dumps(res, indent=1))
    print(f'wrote {OUT.name} and figures in {FIG.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
