"""XGBoost with the officer features: stability, fairness, interpretation.

Same sample, preprocessing, hyperparameters and splits as evaluate_officer_models.py (imported from it),
so every number here is comparable to data/officer_model_results.json.

Models (all XGBoost)
  full_time                  the team's baseline, no officer features
  full_time_officer          + officer track record and disparity features (the model under study)
  full_time_officer_blind    without driver race, sex and the race-derived officer feature

1. STABILITY (course definition: distance among models + slow variation in feature contributions)
   a. AUC by test year (2016, 2017, 2018)
   b. seed stability: 10 seeds, AUC spread and pairwise Spearman of the scores
   c. bootstrap of the training set (N_BOOT_FIT refits): AUC spread, TreeSHAP importance per feature,
      Spearman between importance rankings, top-10 overlap
   d. training period: fit on 2010-2012 and on 2013-2015, compare the two importance rankings
   e. officers: GroupKFold by officer vs plain KFold on the training years
   f. PSI of every officer feature and of the score, train vs test
   g. distance among models: Spearman of scores and overlap of the top-22% selected rows
2. FAIRNESS at the team's search budget (top 22% of test rows), per race, with bootstrap CIs:
   selection rate, false positive rate (innocent drivers searched), hit rate; calibration by race;
   AUC within each race
3. INTERPRETATION with exact TreeSHAP (xgboost pred_contribs), collapsed back to the original columns:
   global mean |SHAP|, direction, share carried by officer features, mean SHAP by driver race,
   dependence of the top officer features, three local explanations

Usage
  python scripts_claude/xgboost_officer_analysis.py
Outputs: data/xgboost_officer_analysis.json, reports/xgboost_officer/*.png
"""
import json
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_officer_models as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'xgboost_officer_analysis.json'
FIG = ROOT / 'reports' / 'xgboost_officer'

SEED = E.SEED
N_SEEDS, N_BOOT_FIT, N_BOOT_CI = 10, 20, 1000
MAIN = 'full_time_officer'
COMPARE = ['full_time', 'full_time_officer', 'full_time_officer_blind']
OFFICER_COLS = E.OFF + E.OFF_RACE


def log(msg):
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def fit(mode, X, y, seed=SEED):
    num, cat = E.MODES[mode]
    m = E.make('xgboost', num, cat)
    m.set_params(clf__random_state=seed)
    return m.fit(X[num + cat], y)


def predict(m, mode, X):
    num, cat = E.MODES[mode]
    return m.predict_proba(X[num + cat])[:, 1]


# ------------------------------------------------------------------ TreeSHAP collapsed to raw columns
def shap_raw(m, mode, X):
    """Exact TreeSHAP in log-odds, one column per ORIGINAL feature (one-hot levels summed)."""
    num, cat = E.MODES[mode]
    prep, clf = m.named_steps['prep'], m.named_steps['clf']
    Z = prep.transform(X[num + cat])
    names = list(prep.get_feature_names_out())
    raw = clf.get_booster().predict(xgb.DMatrix(Z, feature_names=names), pred_contribs=True)
    contrib, base = raw[:, :-1], raw[:, -1]
    owner = []
    for n in names:
        kind, rest = n.split('__', 1)
        owner.append(rest if kind == 'num' else next(c for c in sorted(cat, key=len, reverse=True)
                                                      if rest.startswith(c + '_')))
    S = pd.DataFrame(contrib, columns=names, index=X.index).T.groupby(owner).sum().T
    return S[num + cat], float(base[0])


def importance(S):
    return S.abs().mean().sort_values(ascending=False)


def rank_stats(imps):
    """Spearman between every pair of importance vectors and overlap of their top-10 sets."""
    rhos, overlaps = [], []
    for a, b in combinations(imps, 2):
        a, b = a.align(b, fill_value=0)
        rhos.append(spearmanr(a, b).statistic)
        overlaps.append(len(set(a.nlargest(10).index) & set(b.nlargest(10).index)) / 10)
    return {'spearman_mean': r4(np.mean(rhos)), 'spearman_min': r4(np.min(rhos)),
            'top10_overlap_mean': r4(np.mean(overlaps)), 'top10_overlap_min': r4(np.min(overlaps))}


def topk_mask(p, share=E.TOPK_SHARE):
    sel = np.zeros(len(p), bool)
    sel[np.argsort(-p, kind='mergesort')[:int(round(share * len(p)))]] = True
    return sel


# ------------------------------------------------------------------ 1. stability
def stability(d, tr, te, rng):
    Xtr, ytr, Xte, yte = d[tr], d.loc[tr, 'y'].to_numpy(), d[te], d.loc[te, 'y'].to_numpy()
    out = {}

    # a. AUC by test year, every model
    log('stability: AUC by test year')
    models = {mode: fit(mode, Xtr, ytr) for mode in COMPARE}
    preds = {mode: predict(m, mode, Xte) for mode, m in models.items()}
    years = Xte['year'].to_numpy()
    out['auc_by_test_year'] = {mode: {int(yr): {'n': int((years == yr).sum()),
                                                 'auc': r4(roc_auc_score(yte[years == yr], p[years == yr]))}
                                       for yr in sorted(set(years))}
                               for mode, p in preds.items()}

    # b. seed stability
    log(f'stability: {N_SEEDS} seeds')
    ps = [predict(fit(MAIN, Xtr, ytr, seed=s), MAIN, Xte) for s in range(N_SEEDS)]
    aucs = [roc_auc_score(yte, p) for p in ps]
    rho = [spearmanr(a, b).statistic for a, b in combinations(ps, 2)]
    sel = [topk_mask(p) for p in ps]
    ov = [(a & b).sum() / a.sum() for a, b in combinations(sel, 2)]
    out['seeds'] = {'n': N_SEEDS, 'auc_mean': r4(np.mean(aucs)), 'auc_std': r4(np.std(aucs)),
                    'auc_min': r4(min(aucs)), 'auc_max': r4(max(aucs)),
                    'score_spearman_min': r4(min(rho)), 'topk_overlap_min': r4(min(ov))}

    # c. bootstrap of the training set: performance AND explanation
    log(f'stability: {N_BOOT_FIT} bootstrap refits with TreeSHAP')
    n = len(Xtr)
    boot_auc, boot_imp = [], []
    shap_sample = Xte.sample(n=min(3000, len(Xte)), random_state=SEED)
    for b in range(N_BOOT_FIT):
        i = rng.integers(0, n, n)
        m = fit(MAIN, Xtr.iloc[i], ytr[i], seed=b)
        boot_auc.append(roc_auc_score(yte, predict(m, MAIN, Xte)))
        boot_imp.append(importance(shap_raw(m, MAIN, shap_sample)[0]))
    I = pd.DataFrame(boot_imp)
    ranks = I.rank(axis=1, ascending=False)
    feat = pd.DataFrame({'mean_abs_shap': I.mean(), 'std': I.std(), 'cv': I.std() / I.mean(),
                         'rank_median': ranks.median(), 'rank_min': ranks.min(), 'rank_max': ranks.max(),
                         'in_top10_share': (ranks <= 10).mean()}).sort_values('mean_abs_shap', ascending=False)
    out['bootstrap'] = {'n': N_BOOT_FIT, 'auc_mean': r4(np.mean(boot_auc)), 'auc_std': r4(np.std(boot_auc)),
                        'auc_p2.5': r4(np.percentile(boot_auc, 2.5)), 'auc_p97.5': r4(np.percentile(boot_auc, 97.5)),
                        'importance_ranking': rank_stats(boot_imp),
                        'features': {k: {c: r4(v) for c, v in row.items()} for k, row in feat.iterrows()}}
    _plot_boot(I)

    # d. training period
    log('stability: 2010-2012 vs 2013-2015 training')
    per = {}
    for label, (lo, hi) in {'2010-2012': (2010, 2012), '2013-2015': (2013, 2015)}.items():
        mk = tr & d['year'].between(lo, hi)
        m = fit(MAIN, d[mk], d.loc[mk, 'y'].to_numpy())
        per[label] = {'n_train': int(mk.sum()), 'auc_on_2016_2018': r4(roc_auc_score(yte, predict(m, MAIN, Xte))),
                      'imp': importance(shap_raw(m, MAIN, shap_sample)[0])}
    a, b = per['2010-2012']['imp'], per['2013-2015']['imp']
    out['training_period'] = {k: {'n_train': v['n_train'], 'auc_on_2016_2018': v['auc_on_2016_2018'],
                                  'top10': list(v['imp'].index[:10])} for k, v in per.items()}
    out['training_period']['importance_ranking'] = rank_stats([a, b])
    out['training_period']['biggest_rank_moves'] = (
        (a.rank(ascending=False) - b.rank(ascending=False)).abs().sort_values(ascending=False).head(8)
        .astype(int).to_dict())

    # e. officers: unseen officers vs random folds, on the training years
    log('stability: GroupKFold by officer vs KFold')
    groups = Xtr['officer_id_hash'].fillna('__none__').to_numpy()
    cv = {}
    for label, splitter in {'random_kfold': KFold(5, shuffle=True, random_state=SEED),
                            'officer_groupkfold': GroupKFold(5)}.items():
        res = []
        for a_, b_ in splitter.split(Xtr, ytr, groups):
            m = fit(MAIN, Xtr.iloc[a_], ytr[a_])
            res.append(roc_auc_score(ytr[b_], predict(m, MAIN, Xtr.iloc[b_])))
        cv[label] = {'auc_mean': r4(np.mean(res)), 'auc_folds': [r4(x) for x in res]}
    cv['gap'] = r4(cv['random_kfold']['auc_mean'] - cv['officer_groupkfold']['auc_mean'])
    out['officer_cv'] = cv

    # f. drift train -> test
    log('stability: PSI')
    psi = {c: r4(E_psi(Xtr[c].dropna(), Xte[c].dropna())) for c in OFFICER_COLS}
    psi['__score__'] = r4(E_psi(predict(models[MAIN], MAIN, Xtr), preds[MAIN]))
    out['psi_train_vs_test'] = dict(sorted(psi.items(), key=lambda kv: -(kv[1] or 0)))

    # g. distance among models
    dist = {}
    for a_, b_ in combinations(COMPARE, 2):
        sa, sb = topk_mask(preds[a_]), topk_mask(preds[b_])
        dist[f'{a_} vs {b_}'] = {'score_spearman': r4(spearmanr(preds[a_], preds[b_]).statistic),
                                  'topk_overlap': r4((sa & sb).sum() / sa.sum())}
    out['distance_among_models'] = dist
    return out, models, preds


def E_psi(expected, actual, bins=10):
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    cuts = np.unique(np.nanquantile(e, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return np.nan
    cuts[0], cuts[-1] = -np.inf, np.inf
    ep = np.clip(np.histogram(e, cuts)[0] / len(e), 1e-6, None)
    ap = np.clip(np.histogram(a, cuts)[0] / len(a), 1e-6, None)
    return float(np.sum((ap - ep) * np.log(ap / ep)))


# ------------------------------------------------------------------ 2. fairness
def fairness(d, te, preds, rng):
    yte, race = d.loc[te, 'y'].to_numpy(), d.loc[te, 'subject_race'].to_numpy()
    out = {}
    for mode, p in preds.items():
        r = E.fairness(yte, p, race, rng)
        w = r['white']
        for g in ['black', 'hispanic']:
            r[g]['vs_white'] = {
                'selection_rate_ratio': r4(r[g]['selection_rate'][0] / w['selection_rate'][0]),
                'fpr_diff_pp': r4(100 * (r[g]['false_positive_rate'][0] - w['false_positive_rate'][0])),
                'hit_rate_diff_pp': r4(100 * (r[g]['hit_rate_of_selected'][0] - w['hit_rate_of_selected'][0]))}
        for g in E.RACES:
            m = race == g
            r[g]['auc_within_group'] = r4(roc_auc_score(yte[m], p[m]))
            r[g]['mean_score'] = r4(p[m].mean())
            r[g]['observed_hit_rate'] = r4(yte[m].mean())
            r[g]['calibration_gap_pp'] = r4(100 * (p[m].mean() - yte[m].mean()))
        out[mode] = r
    _plot_fairness(out)
    return out


# ------------------------------------------------------------------ 3. interpretation
def interpretation(d, te, model):
    Xte = d[te]
    S, base = shap_raw(model, MAIN, Xte)
    imp = importance(S)
    num, cat = E.MODES[MAIN]
    direction = {}
    for c in num:
        x = Xte[c].to_numpy(float)
        ok = ~np.isnan(x)
        direction[c] = r4(spearmanr(x[ok], S[c].to_numpy()[ok]).statistic) if ok.sum() > 50 else None
    officer_share = S[OFFICER_COLS].abs().sum(axis=1).sum() / S.abs().sum(axis=1).sum()
    race = Xte['subject_race'].to_numpy()
    by_race = {g: S[race == g].mean().loc[imp.index[:12]].round(4).to_dict() for g in E.RACES}

    dep = {}
    for c in [c for c in imp.index if c in OFFICER_COLS][:4]:
        q = pd.qcut(Xte[c], 5, duplicates='drop')
        g = pd.DataFrame({'bin': q, 'shap': S[c], 'y': d.loc[te, 'y']}).groupby('bin', observed=True)
        dep[c] = [{'bin': str(k), 'n': int(len(v)), 'mean_shap': r4(v['shap'].mean()),
                   'observed_hit_rate': r4(v['y'].mean())} for k, v in g]

    p = model.predict_proba(Xte[num + cat])[:, 1]
    local = {}
    for label, i in {'highest_score': int(np.argmax(p)), 'median_score': int(np.argsort(p)[len(p) // 2]),
                     'lowest_score': int(np.argmin(p))}.items():
        row = S.iloc[i]
        top = row.reindex(row.abs().sort_values(ascending=False).index[:8])
        local[label] = {'score': r4(p[i]), 'contraband_found': int(d.loc[te, 'y'].iloc[i]),
                        'base_logodds': r4(base),
                        'top_contributions_logodds': {k: {'value': _fmt(Xte[k].iloc[i]), 'shap': r4(v)}
                                                      for k, v in top.items()}}
    _plot_shap(S, imp, Xte)
    return {'method': 'exact TreeSHAP (xgboost pred_contribs), log-odds, one-hot levels summed per column',
            'n_rows': int(len(Xte)),
            'global_mean_abs_shap': {k: r4(v) for k, v in imp.items()},
            'officer_features_share_of_attribution': r4(officer_share),
            'direction_spearman_value_vs_shap': dict(sorted(direction.items(), key=lambda kv: -abs(kv[1] or 0))),
            'mean_shap_by_race_top12': by_race,
            'dependence_top_officer_features': dep,
            'local_examples': local}


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 4)
    return v if isinstance(v, str) else (None if pd.isna(v) else str(v))


# ------------------------------------------------------------------ figures
def _label(c):
    return c + ('  [officer]' if c in OFFICER_COLS else '')


def _plot_boot(I):
    top = I.mean().sort_values(ascending=False).index[:15][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([I[c] for c in top], orientation="horizontal", tick_labels=[_label(c) for c in top])
    ax.set_xlabel('mean |SHAP| on the test sample (log-odds)')
    ax.set_title(f'Importance across {len(I)} bootstrap refits')
    fig.tight_layout()
    fig.savefig(FIG / 'stability_importance_bootstrap.png', dpi=130)
    plt.close(fig)


def _plot_shap(S, imp, X):
    top = imp.index[:15][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([_label(c) for c in top], imp[top],
            color=['#c0504d' if c in OFFICER_COLS else '#4f81bd' for c in top])
    ax.set_xlabel('mean |SHAP| (log-odds)')
    ax.set_title('Global importance (red = officer feature)')
    fig.tight_layout()
    fig.savefig(FIG / 'interpretation_global_importance.png', dpi=130)
    plt.close(fig)

    nums = [c for c in imp.index if c in E.MODES[MAIN][0]][:12][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(SEED)
    for k, c in enumerate(nums):
        x = X[c].to_numpy(float)
        z = (pd.Series(x).rank(pct=True)).to_numpy()
        ax.scatter(S[c], k + rng.uniform(-0.3, 0.3, len(x)), c=z, cmap='coolwarm', s=2, alpha=0.4)
    ax.set_yticks(range(len(nums)), [_label(c) for c in nums])
    ax.axvline(0, color='grey', lw=0.8)
    ax.set_xlabel('SHAP (log-odds); colour = feature value percentile (blue low, red high)')
    ax.set_title('Direction of effect, numeric features')
    fig.tight_layout()
    fig.savefig(FIG / 'interpretation_beeswarm.png', dpi=130)
    plt.close(fig)


def _plot_fairness(res):
    metrics = [('selection_rate', 'Selection rate'), ('false_positive_rate', 'False positive rate'),
               ('hit_rate_of_selected', 'Hit rate of selected')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    w = 0.26
    for ax, (key, title) in zip(axes, metrics):
        for j, mode in enumerate(COMPARE):
            vals = [res[mode][g][key][0] for g in E.RACES]
            lo = [res[mode][g][key][0] - res[mode][g][key][1][0] for g in E.RACES]
            hi = [res[mode][g][key][1][1] - res[mode][g][key][0] for g in E.RACES]
            ax.bar(np.arange(3) + (j - 1) * w, vals, w, yerr=[lo, hi], capsize=3, label=mode)
        ax.set_xticks(range(3), E.RACES)
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Fairness at a fixed budget (top {int(E.TOPK_SHARE * 100)}% of test searches), 95% CIs')
    fig.tight_layout()
    fig.savefig(FIG / 'fairness_by_race.png', dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = E.load()
    tr, te = (pd.Series(a, index=d.index) for a in E.splits(d)['temporal'])
    log(f'rows {len(d):,}; train {tr.sum():,}; test {te.sum():,}')

    stab, models, preds = stability(d, tr, te, rng)
    res = {'built_utc': datetime.now(timezone.utc).isoformat(), 'split': 'temporal: train < 2016, test >= 2016',
           'n_train': int(tr.sum()), 'n_test': int(te.sum()), 'topk_share': E.TOPK_SHARE,
           'auc_test': {m: r4(roc_auc_score(d.loc[te, 'y'], p)) for m, p in preds.items()},
           'stability': stab}
    OUT.write_text(json.dumps(res, indent=1))
    log('fairness')
    res['fairness_topk'] = fairness(d, te, preds, rng)
    OUT.write_text(json.dumps(res, indent=1))
    log('interpretation')
    res['interpretation'] = interpretation(d, te, models[MAIN])
    res['seconds'] = round(time.time() - t0)
    OUT.write_text(json.dumps(res, indent=1))
    log(f'wrote {OUT.relative_to(ROOT)} and figures in {FIG.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
