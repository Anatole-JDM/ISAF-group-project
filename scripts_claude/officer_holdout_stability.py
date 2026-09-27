"""Is the officer-holdout result a lucky draw? Repeat it over 5 random draws of held-out officers.

evaluate_officer_models.py holds out one draw of 20% of officers (seed 42); its confidence intervals
cover the test searches, not the choice of officers. Here the draw itself varies: seeds 0-4, same
temporal cut (train < 2016, test >= 2016 searches of officers never seen in training), same models and
settings (imported from evaluate_officer_models.py).

Reported per feature set and model: AUC and gain over full_time on each draw (paired bootstrap CI),
then mean / min / max / sd across draws and in how many draws the gain's CI excludes 0.
With --with-pltr: PLTR (1-SE and <= 30 terms) on the two officer feature sets, and which variable
combinations PLTR selects in every draw (the stable core of the white box). Adds ~40 minutes.

Usage   python scripts_claude/officer_holdout_stability.py [--with-pltr]
Output  data/officer_holdout_stability.json + printed summary
"""
import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_officer_models as E  # noqa: E402

OUT = E.ROOT / 'data' / 'officer_holdout_stability.json'
SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 300


def paired(y, a, b, rng):
    draws = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            draws.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return [round(float(roc_auc_score(y, a) - roc_auc_score(y, b)), 4),
            [round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-pltr', action='store_true')
    args = ap.parse_args()
    rng = np.random.default_rng(E.SEED)
    d = E.load()
    per_seed, pltr_vars = {}, {}
    for seed in SEEDS:
        tr, te = E.splits(d, seed)['officer_holdout']
        y_tr, y_te = d.loc[tr, 'y'].to_numpy(), d.loc[te, 'y'].to_numpy()
        info = {'n_train': int(tr.sum()), 'n_test': int(te.sum()), 'base_rate_test': round(float(y_te.mean()), 4),
                'test_officers': int(d.loc[te, 'officer_id_hash'].nunique()),
                'test_officers_seen_in_training': int(len(set(d.loc[te, 'officer_id_hash'].dropna())
                                                          & set(d.loc[tr, 'officer_id_hash'].dropna()))),
                'models': {}}
        preds = {}
        for mode, (num, cat) in E.MODES.items():
            models = ['scorecard', 'xgboost'] + (['pltr'] if args.with_pltr and mode in E.PLTR_MODES else [])
            for model in models:
                t0 = time.time()
                m = E.make(model, num, cat).fit(d.loc[tr, num + cat], y_tr)
                variants = [(model, m)] + ([('pltr_sparse', m.at_most(E.PLTR_MAX_TERMS))] if model == 'pltr' else [])
                for vname, mv in variants:
                    p = mv.predict_proba(d.loc[te, num + cat])[:, 1]
                    preds[(mode, vname)] = p
                    r = {'auc': round(float(roc_auc_score(y_te, p)), 4)}
                    base = ('full_time', model if model != 'pltr' else 'scorecard')
                    if mode != 'full_time' and base in preds:
                        r['gain_vs_' + '_'.join(base)] = paired(y_te, p, preds[base], rng)
                    if model == 'pltr':
                        t = mv.terms()
                        r['terms'] = int(len(t))
                        pltr_vars.setdefault(f'{mode}/{vname}', {})[seed] = set(t['variables'])
                    info['models'][f'{mode}/{vname}'] = r
                    g = next((v for k, v in r.items() if k.startswith('gain_vs_')), None)
                    E.log(f'seed {seed}  {mode:25} {vname:11} AUC {r["auc"]:.4f}'
                          + (f'  gain {g[0]:+.4f} {g[1]}' if g else '') + f'  ({time.time() - t0:.0f}s)')
        per_seed[seed] = info

    summary = {}
    for key in per_seed[SEEDS[0]]['models']:
        rows = [per_seed[s]['models'][key] for s in SEEDS]
        auc = np.array([r['auc'] for r in rows])
        s = {'auc_mean': round(float(auc.mean()), 4), 'auc_sd': round(float(auc.std(ddof=1)), 4),
             'auc_min': round(float(auc.min()), 4), 'auc_max': round(float(auc.max()), 4)}
        gk = next((k for k in rows[0] if k.startswith('gain_vs_')), None)
        if gk:
            gains = np.array([r[gk][0] for r in rows])
            s.update({'gain_vs': gk[len('gain_vs_'):], 'gain_mean': round(float(gains.mean()), 4),
                      'gain_min': round(float(gains.min()), 4), 'gain_max': round(float(gains.max()), 4),
                      'draws_with_gain_ci_above_0': int(sum(r[gk][1][0] > 0 for r in rows))})
        summary[key] = s

    core = {}
    for key, by_seed in pltr_vars.items():
        sets = list(by_seed.values())
        core[key] = {'variable_combos_in_every_draw': sorted(set.intersection(*sets)),
                     'mean_pairwise_jaccard': round(float(np.mean([len(a & b) / len(a | b)
                                                                    for a, b in combinations(sets, 2)])), 3),
                     'combos_per_draw': [len(x) for x in sets]}

    out = {'seeds': SEEDS, 'note': 'officer holdout repeated over draws of held-out officers; the main run uses seed 42',
           'per_seed': {str(k): v for k, v in per_seed.items()}, 'summary_across_draws': summary,
           'pltr_stable_core': core}
    OUT.write_text(json.dumps(out, indent=1))
    print('\nAcross draws of held-out officers:')
    for k, s in summary.items():
        line = f'  {k:38} AUC {s["auc_mean"]:.4f} (min {s["auc_min"]:.4f}, max {s["auc_max"]:.4f}, sd {s["auc_sd"]:.4f})'
        if 'gain_mean' in s:
            line += (f'  gain vs {s["gain_vs"]} {s["gain_mean"]:+.4f} (min {s["gain_min"]:+.4f}, max {s["gain_max"]:+.4f}),'
                     f' CI above 0 in {s["draws_with_gain_ci_above_0"]}/{len(SEEDS)} draws')
        print(line)
    for k, c in core.items():
        print(f'  PLTR {k}: {len(c["variable_combos_in_every_draw"])} variable combinations selected in every draw; '
              f'mean pairwise Jaccard {c["mean_pairwise_jaccard"]}')
    print(f'wrote {OUT.name}')


if __name__ == '__main__':
    main()
