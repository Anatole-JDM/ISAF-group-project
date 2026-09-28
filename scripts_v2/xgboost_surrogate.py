"""Global surrogate of the XGBoost + officer-features model.

A surrogate is a white-box model fitted to the BLACK BOX'S SCORES, not to the true outcome. It explains
what the black box does, and is only as trustworthy as its fidelity (how closely it reproduces them).

Black box: XGBoost `full_time_officer`, same setup as evaluate_officer_models.py (imported).
Surrogates, fitted on the training rows to the black box's predicted PROBABILITY:
  tree_depth_{2..6}   DecisionTreeRegressor, min 200 rows per leaf
  linear              LinearRegression on the scorecard's preprocessing (standardised numerics + one-hot)

Why probability and not log-odds: on the log-odds scale the squared error is dominated by a few hundred
rows the black box scores near zero (log-odds -5 to -6), so a shallow tree spends its splits isolating
them. On the probability scale the same depth-3 tree ranks drivers closer to the black box (Spearman
0.60 vs 0.56, top-22% overlap 0.52 vs 0.50), which is what a search policy uses.

Fidelity is measured on the TEST rows (2016-2018), which neither model was fitted on:
  r2                  share of the black box's score variance the surrogate reproduces
  spearman            rank agreement of the two scores
  topk_overlap        share of the black box's top-22% selection the surrogate also selects (a tree has
                      few distinct scores; ties at the cut-off are broken by row order)
  auc_vs_truth        the surrogate's own AUC on contraband_found, next to the black box's

Usage
  python scripts_claude/xgboost_surrogate.py
Outputs: data/xgboost_surrogate.json, reports/xgboost_officer/surrogate_tree.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import r2_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_officer_models as E  # noqa: E402
from pltr import linear_preprocessor  # noqa: E402
from xgboost_officer_analysis import FIG, MAIN, OFFICER_COLS, fit, predict, r4, topk_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'xgboost_surrogate.json'
DEPTHS, SHOW_DEPTH, MIN_LEAF = [2, 3, 4, 5, 6], 3, 200


def clean(name):
    return name.split('__', 1)[1]


def surrogate(kind, num, cat, depth=None):
    prep = linear_preprocessor(num, cat)
    if kind == 'tree':
        prep.transformers[0][1].steps.pop()          # trees read raw numeric values: no scaling
        return Pipeline([('prep', prep), ('reg', DecisionTreeRegressor(
            max_depth=depth, min_samples_leaf=MIN_LEAF, random_state=E.SEED))])
    return Pipeline([('prep', prep), ('reg', LinearRegression())])


def fidelity(p_bb, p_s, y):
    sb, ss = topk_mask(p_bb), topk_mask(p_s)
    return {'r2': r4(r2_score(p_bb, p_s)), 'spearman': r4(spearmanr(p_bb, p_s).statistic),
            'topk_overlap': r4((sb & ss).sum() / sb.sum()), 'auc_vs_truth': r4(roc_auc_score(y, p_s))}


def main():
    d = E.load()
    tr, te = E.splits(d)['temporal']
    num, cat = E.MODES[MAIN]
    Xtr, Xte = d.loc[tr, num + cat], d.loc[te, num + cat]
    ytr, yte = d.loc[tr, 'y'].to_numpy(), d.loc[te, 'y'].to_numpy()

    bb = fit(MAIN, d[tr], ytr)
    p_tr, p_te = predict(bb, MAIN, d[tr]), predict(bb, MAIN, d[te])
    res = {'black_box': f'xgboost {MAIN}', 'target': 'black-box probability', 'fitted_on': 'train (< 2016)',
           'evaluated_on': 'test (>= 2016)', 'black_box_auc_vs_truth': r4(roc_auc_score(yte, p_te)),
           'surrogates': {}}

    fitted = {}
    for name, kind, depth in [(f'tree_depth_{k}', 'tree', k) for k in DEPTHS] + [('linear', 'linear', None)]:
        m = surrogate(kind, num, cat, depth).fit(Xtr, p_tr)
        r = fidelity(p_te, m.predict(Xte), yte)
        if kind == 'tree':
            r['n_leaves'] = int(m.named_steps['reg'].get_n_leaves())
        res['surrogates'][name] = r
        fitted[name] = m
        print(f'{name:14} ' + '  '.join(f'{k} {v}' for k, v in r.items()), flush=True)

    # the tree shown in the report
    m = fitted[f'tree_depth_{SHOW_DEPTH}']
    names = [clean(n) for n in m.named_steps['prep'].get_feature_names_out()]
    reg = m.named_steps['reg']
    res['shown_tree'] = {'depth': SHOW_DEPTH, 'rules': export_text(reg, feature_names=names, decimals=3)}
    used = pd.Series(reg.feature_importances_, index=names)
    used = used[used > 0].sort_values(ascending=False)
    res['shown_tree']['split_importance'] = {k: r4(v) for k, v in used.items()}
    res['shown_tree']['officer_share_of_split_importance'] = r4(
        used[[n for n in used.index if n in OFFICER_COLS]].sum())

    # leaves as plain-language segments: rows, mean black-box score, observed hit rate on test
    leaf_te = reg.apply(m.named_steps['prep'].transform(Xte))
    seg = (pd.DataFrame({'leaf': leaf_te, 'score': p_te, 'y': yte}).groupby('leaf')
           .agg(n=('y', 'size'), black_box_score=('score', 'mean'), observed_hit_rate=('y', 'mean'))
           .sort_values('black_box_score', ascending=False))
    res['shown_tree']['test_leaves'] = [{k: (int(v) if k == 'n' else r4(v)) for k, v in row.items()}
                                        for _, row in seg.iterrows()]

    lin = fitted['linear']
    coef = pd.Series(lin.named_steps['reg'].coef_,
                     index=[clean(n) for n in lin.named_steps['prep'].get_feature_names_out()])
    res['linear_top_coefficients'] = {k: r4(v) for k, v in coef.reindex(
        coef.abs().sort_values(ascending=False).index[:15]).items()}

    fig, ax = plt.subplots(figsize=(18, 7))
    plot_tree(reg, feature_names=names, filled=True, impurity=False, precision=2, fontsize=9, ax=ax)
    s = res['surrogates'][f'tree_depth_{SHOW_DEPTH}']
    ax.set_title(f'Depth-{SHOW_DEPTH} surrogate of XGBoost (value = black-box probability). '
                 f'Test fidelity R² {s["r2"]}, top-22% overlap {s["topk_overlap"]}')
    fig.tight_layout()
    fig.savefig(FIG / 'surrogate_tree.png', dpi=120)
    plt.close(fig)

    OUT.write_text(json.dumps(res, indent=1))
    print(res['shown_tree']['rules'])
    print(json.dumps(res['shown_tree']['test_leaves'], indent=0))
    print(res['shown_tree']['split_importance'], res['shown_tree']['officer_share_of_split_importance'])
    print(res['linear_top_coefficients'])


if __name__ == '__main__':
    main()
