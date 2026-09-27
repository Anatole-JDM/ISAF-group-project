"""Known-answer tests for pltr.py, on synthetic data only.

1. The rescaling trick really solves the adaptive lasso: compare against a direct
   proximal-gradient minimisation of  -loglik + lambda * sum w_j |beta_j|.
2. PLTR recovers a planted two-variable threshold effect that plain logistic regression
   cannot represent, and beats it out of sample.
3. PLTR runs with categorical columns and translates their rules back into category lists.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pltr import PLTR, linear_preprocessor  # noqa: E402


def test_rescaling_equals_weighted_lasso():
    rng = np.random.default_rng(0)
    n, p = 3000, 8
    X = rng.normal(size=(n, p))
    beta = np.array([1.2, -0.8, 0.5, 0, 0, 0, 0, 0])
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ beta - 0.3)))).astype(int)
    w = np.array([0.5, 1, 2, 3, 1, 4, 2, 1.5])
    lam = 0.01                                          # penalty per observation

    # rescaled problem, solved by sklearn (saga: unpenalised intercept)
    Xs = X / w
    C = 1 / (lam * n)
    m = LogisticRegression(l1_ratio=1.0, solver='saga', C=C, max_iter=20000, tol=1e-10).fit(Xs, y)
    b_rescaled = m.coef_.ravel() / w

    # direct proximal gradient (ISTA) on the weighted objective
    b, b0, step = np.zeros(p), 0.0, 0.5
    for _ in range(20000):
        pr = 1 / (1 + np.exp(-(X @ b + b0)))
        g, g0 = X.T @ (pr - y) / n, np.mean(pr - y)
        z = b - step * g
        b = np.sign(z) * np.maximum(np.abs(z) - step * lam * w, 0)
        b0 -= step * g0
    diff = np.max(np.abs(b - b_rescaled))
    assert diff < 1e-3, f'rescaling trick differs from direct solution by {diff}'
    print(f'[1] adaptive lasso: rescaled sklearn vs direct solver, max coef diff = {diff:.2e}  OK')


def test_recovers_interaction():
    rng = np.random.default_rng(1)
    n = 12000
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f'x{i}' for i in range(6)])
    # an XOR-type effect: risk is high when x0 and x1 are on the SAME side of their thresholds.
    # No linear combination of x0 and x1 can express it; a single step (x0>a AND x1<b) could be
    # approximated linearly almost perfectly, so it would not test anything.
    planted = (X['x0'] > 0.3) == (X['x1'] > -0.2)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + 2.5 * planted)))).astype(int)
    tr, te = slice(0, 8000), slice(8000, None)
    m = PLTR(num_cols=list(X.columns), cat_cols=[], min_leaf=100).fit(X.iloc[tr], y[tr])
    auc_pltr = roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1])
    lin = linear_preprocessor(list(X.columns), [])
    lr = LogisticRegression(max_iter=2000).fit(lin.fit_transform(X.iloc[tr]), y[tr])
    auc_lr = roc_auc_score(y[te], lr.predict_proba(lin.transform(X.iloc[te]))[:, 1])
    terms = m.terms()
    top = terms.iloc[0]['term']
    assert 'x0' in top and 'x1' in top, f'top term should involve x0 and x1, got: {top}'
    assert auc_pltr > auc_lr + 0.05, f'PLTR {auc_pltr:.3f} should beat logit {auc_lr:.3f}'
    print(f'[2] planted rule recovered: "{top}"; AUC PLTR {auc_pltr:.3f} vs logit {auc_lr:.3f}  OK')
    print('    summary:', m.summary())


def test_categoricals():
    rng = np.random.default_rng(2)
    n = 8000
    zone = rng.choice([f'Z{i}' for i in range(12)], size=n)
    hot = np.isin(zone, ['Z1', 'Z5', 'Z7'])
    age = rng.integers(16, 80, size=n).astype(float)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2 + 1.5 * hot + 0.8 * (age < 25))))).astype(int)
    X = pd.DataFrame({'zone': zone, 'age': age})
    m = PLTR(num_cols=['age'], cat_cols=['zone'], min_leaf=100).fit(X, y)
    terms = m.terms()
    zone_rules = terms[terms['term'].str.contains('zone in')]
    assert len(zone_rules) > 0, 'expected a rule on zone'
    txt = zone_rules.iloc[0]['term']
    print(f'[3] categorical rule translated back: "{txt}"  OK')


def test_honest_cv_on_noise():
    """Regression test for the leak: rules learned on all rows made the CV look good on noise."""
    rng = np.random.default_rng(3)
    n, p = 5000, 12
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f'x{i}' for i in range(p)])
    X['zone'] = rng.choice([f'z{i}' for i in range(30)], n)
    y = (rng.random(n) < 0.2).astype(int)               # no signal at all
    m = PLTR(num_cols=[f'x{i}' for i in range(p)], cat_cols=['zone'], min_leaf=100).fit(X.iloc[:4000], y[:4000])
    cv = float(m.cv_auc_.max())
    test = roc_auc_score(y[4000:], m.predict_proba(X.iloc[4000:])[:, 1])
    assert cv < 0.56, f'CV AUC on pure noise should be ~0.5, got {cv:.3f} (leak?)'
    print(f'[4] pure noise: honest CV AUC {cv:.3f}, test AUC {test:.3f}, terms kept '
          f'{m.summary()["selected_linear"] + m.summary()["selected_rules"]}  OK')


def test_sparsity_controls():
    """1-SE rule is never larger than the best-CV model; at_most(N) respects N; a category seen in
    fewer than min_leaf training rows is removed from the linear part, however extreme its outcome."""
    rng = np.random.default_rng(4)
    n = 8000
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f'x{i}' for i in range(8)])
    X['zone'] = rng.choice([f'z{i}' for i in range(6)], n)
    X.loc[:59, 'zone'] = 'small'           # 60 rows: own one-hot column, below min_leaf -> dropped
    X.loc[60:70, 'zone'] = 'tiny'          # 11 rows: pooled into the encoder's 'infrequent' column -> dropped
    eta = -1.5 + 0.6 * X['x0'] - 0.4 * X['x1'] + 0.8 * ((X['x2'] > 0) & (X['x3'] > 0))
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    y[:71] = 1                                                    # extreme outcome on both
    num = [f'x{i}' for i in range(8)]
    m1 = PLTR(num_cols=num, cat_cols=['zone'], min_leaf=100).fit(X, y)
    mb = PLTR(num_cols=num, cat_cols=['zone'], min_leaf=100, select='best').fit(X, y)
    k1, kb = np.count_nonzero(m1.coef_), np.count_nonzero(mb.coef_)
    assert k1 <= kb, f'1-SE model has {k1} terms, best-CV model {kb}'
    assert not any(('small' in t) or ('infrequent' in t) for t in m1.lin_names_), 'rare categories should be dropped'
    assert {'cat__zone_small', 'cat__zone_infrequent_sklearn'} <= set(m1.lin_dropped_), m1.lin_dropped_
    small = m1.at_most(5)
    assert 0 < np.count_nonzero(small.coef_) <= 5 and np.count_nonzero(m1.coef_) == k1   # original untouched
    assert np.count_nonzero(m1.at_most(1000).coef_) == k1, 'a cap above the model size must not enlarge it'
    t = small.terms()
    assert t['importance'].is_monotonic_decreasing and abs(t['importance_share'].sum() - 1) < 1e-9
    print(f'[5] sparsity: 1-SE {k1} terms vs best-CV {kb}; at_most(5) -> {np.count_nonzero(small.coef_)} terms; '
          f'rare categories dropped; top term "{t.iloc[0]["term"]}"  OK')


if __name__ == '__main__':
    test_rescaling_equals_weighted_lasso()
    test_recovers_interaction()
    test_categoricals()
    test_honest_cv_on_noise()
    test_sparsity_controls()
    print('all PLTR tests passed')
