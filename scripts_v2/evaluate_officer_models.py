"""Does the officer track record improve prediction? Logistic, XGBoost and PLTR, two splits.

Replicates the team's setup in src/ exactly so numbers are comparable:
  * sample: consent-only searches 2010-2018 (58,865 rows)
  * preprocessing: median-impute + scale numerics, most-frequent-impute + one-hot (min 25) categoricals
  * scorecard = LogisticRegression(max_iter=2000); XGBoost = 400 trees, depth 5, lr 0.05,
    subsample 0.8, colsample 0.8; seed 42
  * base features = src 'full_time': age, sex, race, precinct, zone, reason for stop, plate state,
    hour, weekday, month + the 7 time extras

Feature sets
  full_time                  the team's current best
  full_time_officer          + officer track record + disparity (+ officer hit rate on this driver's race)
  full_time_officer_blind    without driver race, sex and every race-derived officer feature
  officer_only               the officer features alone: how much does WHO searches say by itself?

Splits
  temporal         train < 2016, test >= 2016                                  (the team's split)
  officer_holdout  same, but test officers are 20% of officers NEVER seen in training

Reported: AUC with bootstrap 95% CI, PR-AUC, Brier; the AUC gain over full_time with a PAIRED
bootstrap CI (same resampled rows for both models); fairness at a fixed search budget (top 22% of
test rows, the team's K=2,000/9,113): selection rate, false positive rate, hit rate by race, with CIs;
PLTR's selected terms with odds ratios, average marginal effects and importance (|coef| x sd).

PLTR is reported twice from the same fit: `pltr` (penalty by the one-standard-error rule) and
`pltr_sparse` (the weakest penalty keeping <= PLTR_MAX_TERMS terms, chosen on the training data).
Every test prediction is saved for officer_model_diagnostics.py (year-by-year AUC, economics,
calibration by race).

Usage
  python scripts_claude/evaluate_officer_models.py            # everything (PLTR is the slow part)
  python scripts_claude/evaluate_officer_models.py --quick    # temporal split only
Outputs: data/officer_model_results.json, data/pltr_terms.json, data/officer_model_predictions.parquet
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pltr import PLTR, linear_preprocessor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / 'data' / 'nashville_consent_searches.parquet'
OUT = ROOT / 'data' / 'officer_model_results.json'
OUT_TERMS = ROOT / 'data' / 'pltr_terms.json'
OUT_PRED = ROOT / 'data' / 'officer_model_predictions.parquet'

SEED, SPLIT_YEAR, TOPK_SHARE = 42, 2016, 0.22
N_BOOT, N_BOOT_FAIR = 1000, 500
RACES = ['white', 'black', 'hispanic']

BASE_NUM = ['subject_age', 'hour', 'day_of_week', 'month']
BASE_CAT = ['subject_sex', 'subject_race', 'precinct', 'zone', 'reason_for_stop', 'plate_state']
TIME = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'time_heaped', 'is_federal_holiday', 'is_holiday_window']
OFF = ['off_stops_365d', 'off_search_rate_365d', 'off_hit_rate_365d_shrunk', 'off_consent_365d',
       'off_consent_hit_rate_365d_shrunk', 'off_searches_today_before', 'off_hits_today_before',
       'off_minutes_since_first_stop_today', 'off_log_minutes_since_last_search', 'off_last_search_hit',
       'off_experience_days', 'off_log_search_ratio_black_white', 'off_log_search_ratio_hisp_white',
       'off_hit_gap_black_white', 'off_hit_gap_hisp_white']
OFF_RACE = ['off_hit_rate_same_race_past']

MODES = {
    'full_time': (BASE_NUM + TIME, BASE_CAT),
    'full_time_officer': (BASE_NUM + TIME + OFF + OFF_RACE, BASE_CAT),
    'full_time_officer_blind': (BASE_NUM + TIME + OFF, [c for c in BASE_CAT if c not in ('subject_race', 'subject_sex')]),
    'officer_only': (OFF, []),
}
PLTR_MODES = ['full_time_officer', 'full_time_officer_blind']
PLTR_MAX_TERMS = 30


def log(msg):
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)


def load():
    d = pd.read_parquet(TABLE)                                  # officer features are in the master table
    d = d[d['date'] < '2019-01-01'].reset_index(drop=True)
    d['y'] = d['contraband_found'].astype(int)
    d['year'] = d['date'].dt.year
    d['hour'] = d['hour'].where(d['hour'] >= 0)                  # -1 means time missing
    d['off_log_minutes_since_last_search'] = np.log1p(d['off_minutes_since_last_search'])
    for c in TIME + OFF + OFF_RACE + BASE_NUM:
        d[c] = pd.to_numeric(d[c].astype('float64'), errors='coerce')
    for c in BASE_CAT:
        # string columns hold pd.NA for blanks; sklearn's imputers only recognise np.nan
        d[c] = d[c].astype('object').where(d[c].notna(), np.nan)
    return d


def splits(d, seed=SEED):
    train_t, test_t = d['year'] < SPLIT_YEAR, d['year'] >= SPLIT_YEAR
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    officers = d['officer_id_hash'].fillna('__none__').to_numpy()
    tr_idx, te_idx = next(gss.split(d, groups=officers))
    held = np.zeros(len(d), bool)
    held[te_idx] = True
    return {'temporal': (train_t.to_numpy(), test_t.to_numpy()),
            'officer_holdout': ((train_t & ~held).to_numpy(), (test_t & held).to_numpy())}


def make(model, num, cat):
    if model == 'scorecard':
        return Pipeline([('prep', linear_preprocessor(num, cat)),
                         ('clf', LogisticRegression(max_iter=2000, random_state=SEED))])
    if model == 'xgboost':
        prep = linear_preprocessor(num, cat)
        prep.transformers[0][1].steps.pop()             # the team's GBM does not scale numerics
        return Pipeline([('prep', prep), ('clf', XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=SEED, n_jobs=-1))])
    return PLTR(num_cols=num, cat_cols=cat, random_state=SEED)


def auc_ci(y, p, rng):
    n = len(y)
    idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    draws = [roc_auc_score(y[i], p[i]) for i in idx if 0 < y[i].sum() < n]
    return [round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)]


def paired_delta(y, p_new, p_base, rng):
    n = len(y)
    draws = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n:
            draws.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_base[i]))
    return {'delta_auc': round(float(roc_auc_score(y, p_new) - roc_auc_score(y, p_base)), 4),
            'ci95': [round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)]}


def fairness(y, p, race, rng):
    k = int(round(TOPK_SHARE * len(p)))
    sel = np.zeros(len(p), bool)
    sel[np.argsort(-p, kind='mergesort')[:k]] = True
    out = {'k': k, 'share_selected': TOPK_SHARE}
    for g in RACES:
        m = race == g
        yy, ss = y[m], sel[m]
        def stats(ii):
            y_, s_ = yy[ii], ss[ii]
            fpr = s_[y_ == 0].mean() if (y_ == 0).any() else np.nan
            hit = y_[s_].mean() if s_.any() else np.nan
            return s_.mean(), fpr, hit
        base = stats(np.arange(len(yy)))
        draws = np.array([stats(rng.integers(0, len(yy), len(yy))) for _ in range(N_BOOT_FAIR)])
        ci = np.nanpercentile(draws, [2.5, 97.5], axis=0)
        out[g] = {'n': int(m.sum()),
                  'selection_rate': [round(float(base[0]), 4), [round(float(ci[0, 0]), 4), round(float(ci[1, 0]), 4)]],
                  'false_positive_rate': [round(float(base[1]), 4), [round(float(ci[0, 1]), 4), round(float(ci[1, 1]), 4)]],
                  'hit_rate_of_selected': [round(float(base[2]), 4), [round(float(ci[0, 2]), 4), round(float(ci[1, 2]), 4)]]}
    return out


def score(y_te, p, race_te, rng, n_features, seconds):
    return {'auc': round(float(roc_auc_score(y_te, p)), 4), 'auc_ci95': auc_ci(y_te, p, rng),
            'pr_auc': round(float(average_precision_score(y_te, p)), 4),
            'brier': round(float(brier_score_loss(y_te, p)), 4),
            'n_features': n_features, 'seconds': seconds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='temporal split only')
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    d = load()
    log(f'rows {len(d):,}; base rate {d["y"].mean():.3f}; officers {d["officer_id_hash"].nunique():,}')
    sp = splits(d)
    if args.quick:
        sp = {'temporal': sp['temporal']}

    results = {'built_utc': datetime.now(timezone.utc).isoformat(), 'rows': int(len(d)),
               'split_year': SPLIT_YEAR, 'topk_share': TOPK_SHARE, 'pltr_sparse_max_terms': PLTR_MAX_TERMS,
               'splits': {}}
    terms_out, pred_frames = {}, []
    for sname, (tr, te) in sp.items():
        y_tr, y_te = d.loc[tr, 'y'].to_numpy(), d.loc[te, 'y'].to_numpy()
        race_te = d.loc[te, 'subject_race'].to_numpy()
        res = {'n_train': int(tr.sum()), 'n_test': int(te.sum()),
               'base_rate_train': round(float(y_tr.mean()), 4), 'base_rate_test': round(float(y_te.mean()), 4),
               'models': {}}
        if sname == 'officer_holdout':
            res['test_officers_seen_in_training'] = int(len(set(d.loc[te, 'officer_id_hash'].dropna())
                                                            & set(d.loc[tr, 'officer_id_hash'].dropna())))
        preds = {}
        for mode, (num, cat) in MODES.items():
            for model in ['scorecard', 'xgboost', 'pltr']:
                if model == 'pltr' and mode not in PLTR_MODES:
                    continue
                t0 = time.time()
                X_tr, X_te = d.loc[tr, num + cat], d.loc[te, num + cat]
                m = make(model, num, cat).fit(X_tr, y_tr)
                seconds = round(time.time() - t0, 1)
                # PLTR: the 1-SE model and a <= PLTR_MAX_TERMS variant of the same fit (penalty chosen
                # on the training data from the number of terms, never from test results)
                variants = [(model, m)] + ([('pltr_sparse', m.at_most(PLTR_MAX_TERMS))] if model == 'pltr' else [])
                for vname, mv in variants:
                    p = mv.predict_proba(X_te)[:, 1]
                    preds[(mode, vname)] = p
                    pred_frames.append(pd.DataFrame({'stop_id': d.loc[te, 'stop_id'].to_numpy(), 'split': sname,
                                                     'mode': mode, 'model': vname, 'p': p.astype('float32')}))
                    r = score(y_te, p, race_te, rng, len(num) + len(cat), seconds)
                    if ('full_time', model) in preds and mode not in ('full_time', 'officer_only'):
                        r['vs_full_time'] = paired_delta(y_te, p, preds[('full_time', model)], rng)
                    if mode != 'officer_only':
                        r['fairness_topk'] = fairness(y_te, p, race_te, rng)
                    if model == 'pltr':
                        # the new white box against the team's current white box, same test rows
                        r['vs_scorecard_full_time'] = paired_delta(y_te, p, preds[('full_time', 'scorecard')], rng)
                        r['pltr'] = mv.summary()
                        terms_out[f'{sname}/{mode}/{vname}'] = mv.terms().round(5).to_dict(orient='records')
                    if model == 'xgboost' and mode == 'full_time_officer':
                        names = mv.named_steps['prep'].get_feature_names_out()
                        imp = mv.named_steps['clf'].feature_importances_
                        top = np.argsort(-imp)[:15]
                        r['top_features_gain'] = {str(names[i]): round(float(imp[i]), 4) for i in top}
                    res['models'].setdefault(mode, {})[vname] = r
                    cmp = r.get('vs_full_time') or r.get('vs_scorecard_full_time')
                    cmp_label = 'vs full_time' if 'vs_full_time' in r else 'vs current white box'
                    extra = f'  [{r["pltr"]["selected_linear"] + r["pltr"]["selected_rules"]} terms]' if 'pltr' in r else ''
                    log(f'{sname:15} {mode:25} {vname:11} AUC {r["auc"]:.4f} {r["auc_ci95"]}'
                        + (f'  {cmp_label} {cmp["delta_auc"]:+.4f} {cmp["ci95"]}' if cmp else '')
                        + extra + f'  ({seconds}s)')
        results['splits'][sname] = res
        OUT.write_text(json.dumps(results, indent=1))            # save as we go
        OUT_TERMS.write_text(json.dumps(terms_out, indent=1))
        pd.concat(pred_frames, ignore_index=True).to_parquet(OUT_PRED, index=False)
    log(f'wrote {OUT.name}, {OUT_TERMS.name} and {OUT_PRED.name}')


if __name__ == '__main__':
    main()
