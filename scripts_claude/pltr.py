"""Penalised Logistic Tree Regression (PLTR) - course slides 66-71.

Dumitrescu, Hue, Hurlin & Tokpavi (2022), "Machine learning for credit scoring: Improving
logistic regression with non-linear decision-tree effects", EJOR 297(3).

Stage 1 - threshold effects extracted from short trees, fitted on the TRAINING data only:
  * univariate: for each variable, a depth-1 tree gives one rule  1{x_j <= c_j}
  * bivariate:  for each pair, a depth-2 tree gives the leaves of a two-step rule
                1{x_j <= c_j and x_k > c_k} ...; each leaf except the largest is a rule
Stage 2 - logistic regression on [original features, rules] with an ADAPTIVE LASSO:
  * pilot: ridge logistic regression -> beta_pilot
  * weights w_v = 1 / |beta_pilot_v|^gamma
  * minimise  -loglik + lambda * sum_v w_v |theta_v|
    solved exactly with a standard lasso after rescaling each column by 1/w_v
    (theta_v = w_v * beta_v turns the weighted penalty into a uniform one)
  pilot ridge at C = 1 (the team's scorecard setting); lambda chosen by stratified 5-fold CV on
  the training data (AUC), with the rules, encodings and weights RE-LEARNED INSIDE EACH FOLD -
  otherwise the rules have already seen the validation labels and the CV is optimistic.

Categorical variables enter the TREES through an out-of-fold smoothed target encoding (so a
rule reads "zone in {zones with a high past hit rate}") and enter the LINEAR part one-hot
encoded, exactly like the team's scorecard. Every threshold on an encoded categorical is
translated back into the list of categories it selects.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def linear_preprocessor(num, cat):
    """Same preprocessing as src/models.py::_preprocessor (the team's scorecard)."""
    return ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore', min_frequency=25, sparse_output=False))]), cat),
    ])


class PLTR:
    def __init__(self, num_cols, cat_cols, gamma=1.0, min_leaf=200, te_smoothing=50,
                 Cs=np.logspace(-3, 1, 9), pilot_C=1.0, cv=5, random_state=42):
        self.num_cols, self.cat_cols = list(num_cols), list(cat_cols)
        self.gamma, self.min_leaf, self.te_smoothing = gamma, min_leaf, te_smoothing
        self.Cs, self.pilot_C, self.cv, self.random_state = np.asarray(Cs), pilot_C, cv, random_state

    # ------------------------------------------------------------------ tree inputs
    def _te_fit(self, X, y):
        """Out-of-fold smoothed target encoding for the trees; full-train mapping for new data."""
        prior = float(y.mean())
        self._te_maps, oof = {}, pd.DataFrame(index=X.index)
        folds = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        for c in self.cat_cols:
            col = X[c].astype('object').fillna('__missing__')
            enc = pd.Series(np.nan, index=X.index)
            for tr, va in folds.split(X, y):
                st = pd.DataFrame({'c': col.iloc[tr].to_numpy(), 'y': y[tr]}).groupby('c')['y'].agg(['sum', 'count'])
                m = (st['sum'] + self.te_smoothing * prior) / (st['count'] + self.te_smoothing)
                enc.iloc[va] = col.iloc[va].map(m).fillna(prior).to_numpy()
            st = pd.DataFrame({'c': col.to_numpy(), 'y': y}).groupby('c')['y'].agg(['sum', 'count'])
            self._te_maps[c] = ((st['sum'] + self.te_smoothing * prior) / (st['count'] + self.te_smoothing), prior)
            oof[c] = enc.to_numpy()
        return oof

    def _tree_inputs(self, X, oof=None):
        T = pd.DataFrame(index=X.index)
        for c in self.num_cols:
            T[c] = pd.to_numeric(X[c], errors='coerce').astype('float64').fillna(self._medians[c])
        for c in self.cat_cols:
            if oof is not None:
                T[c] = oof[c].to_numpy()
            else:
                m, prior = self._te_maps[c]
                T[c] = X[c].astype('object').fillna('__missing__').map(m).fillna(prior).astype('float64').to_numpy()
        return T[self.num_cols + self.cat_cols]

    # ------------------------------------------------------------------ rules
    def _leaf_rules(self, tree, feats):
        """Human-readable conjunctions for every leaf of a fitted tree."""
        t = tree.tree_
        rules = {}

        def walk(node, conds):
            if t.children_left[node] == -1:
                rules[node] = conds
                return
            f, thr = feats[t.feature[node]], t.threshold[node]
            walk(t.children_left[node], conds + [(f, '<=', thr)])
            walk(t.children_right[node], conds + [(f, '>', thr)])

        walk(0, [])
        return rules

    def _extract_rules(self, T, y):
        self.rules_ = []                                 # list of (features, leaf conditions)
        cols = list(T.columns)
        for c in cols:
            tr = DecisionTreeClassifier(max_depth=1, min_samples_leaf=self.min_leaf,
                                        random_state=self.random_state).fit(T[[c]], y)
            if tr.tree_.node_count == 3:
                self.rules_.append([(c, '<=', float(tr.tree_.threshold[0]))])
        for a, b in combinations(cols, 2):
            tr = DecisionTreeClassifier(max_depth=2, min_samples_leaf=self.min_leaf,
                                        random_state=self.random_state).fit(T[[a, b]], y)
            leaves = self._leaf_rules(tr, [a, b])
            if len(leaves) < 3:                          # no genuine two-variable structure
                continue
            sizes = {n: tr.tree_.n_node_samples[n] for n in leaves}
            ref = max(sizes, key=sizes.get)
            for n, conds in leaves.items():
                if n != ref and len({f for f, _, _ in conds}) == 2:
                    self.rules_.append([(f, op, float(thr)) for f, op, thr in conds])

    def _rule_matrix(self, T):
        cols = []
        for conds in self.rules_:
            m = np.ones(len(T), dtype=bool)
            for f, op, thr in conds:
                v = T[f].to_numpy()
                m &= (v <= thr) if op == '<=' else (v > thr)
            cols.append(m)
        return np.column_stack(cols).astype('float64') if cols else np.zeros((len(T), 0))

    # ------------------------------------------------------------------ stage 1 (data-dependent)
    def _stage1(self, X, y):
        """Learn encodings and rules from (X, y) and return the training design matrix."""
        self._medians = {c: float(pd.to_numeric(X[c], errors='coerce').median()) for c in self.num_cols}
        oof = self._te_fit(X, y)
        T_oof = self._tree_inputs(X, oof=oof)
        self._extract_rules(T_oof, y)
        self.lin_ = linear_preprocessor(self.num_cols, self.cat_cols).fit(X)
        self.lin_names_ = list(self.lin_.get_feature_names_out())
        R = self._rule_matrix(T_oof)
        keep, seen = [], set()                           # drop constant and duplicate rule columns
        for j in range(R.shape[1]):
            col = R[:, j]
            if col.min() == col.max():
                continue
            key = hash(col.tobytes())
            if key in seen:
                continue
            seen.add(key)
            keep.append(j)
        self.keep_rules_ = np.array(keep, dtype=int)
        self.n_linear_, self.n_rules_ = len(self.lin_names_), len(self.keep_rules_)
        return np.hstack([self.lin_.transform(X), R[:, self.keep_rules_]]).astype('float32')

    def _design(self, X):
        """Design matrix for new data, using what _stage1 learned."""
        T = self._tree_inputs(X)
        R = self._rule_matrix(T)[:, self.keep_rules_]
        return np.hstack([self.lin_.transform(X), R]).astype('float32')

    def _stage2_weights(self, Z, y):
        """Column scale and adaptive weights (pilot ridge at a fixed C, like the team's scorecard)."""
        scale = Z.std(axis=0).astype('float64')
        scale[scale == 0] = 1.0
        pilot = LogisticRegression(C=self.pilot_C, max_iter=3000).fit((Z / scale).astype('float32'), y)
        return scale, np.abs(pilot.coef_.ravel()) ** self.gamma      # adapt = 1 / w_v

    def _lasso(self, Zw, y, C):
        # liblinear: fast coordinate descent for L1. It penalises the intercept slightly;
        # a large intercept_scaling makes that penalty negligible.
        return LogisticRegression(l1_ratio=1.0, solver='liblinear', C=C, max_iter=2000, tol=1e-4,
                                  intercept_scaling=100.0, random_state=self.random_state).fit(Zw, y)

    # ------------------------------------------------------------------ fit / predict
    def fit(self, X, y):
        """Choose lambda by an HONEST cross-validation: the rules, encodings and adaptive weights are
        re-learned inside every fold from that fold's training rows only. Learning the rules once on
        all training rows and then cross-validating only the lasso leaks labels into the validation
        folds (tested: CV AUC 0.755 vs test AUC 0.566 on synthetic data) and picks too weak a penalty."""
        y = np.asarray(y).astype(int)
        X = X.reset_index(drop=True)
        folds = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        cv_auc = np.zeros((self.cv, len(self.Cs)))
        for k, (tr, va) in enumerate(folds.split(X, y)):
            fold = PLTR(self.num_cols, self.cat_cols, gamma=self.gamma, min_leaf=self.min_leaf,
                        te_smoothing=self.te_smoothing, pilot_C=self.pilot_C, cv=self.cv,
                        random_state=self.random_state)
            Z_tr = fold._stage1(X.iloc[tr], y[tr])
            Z_va = fold._design(X.iloc[va])
            scale, adapt = fold._stage2_weights(Z_tr, y[tr])
            Zw_tr = (Z_tr / scale * adapt).astype('float32')
            Zw_va = (Z_va / scale * adapt).astype('float32')
            for j, C in enumerate(self.Cs):
                m = self._lasso(Zw_tr, y[tr], C)
                cv_auc[k, j] = roc_auc_score(y[va], m.decision_function(Zw_va))
        self.cv_auc_ = cv_auc.mean(axis=0)
        self.C_ = float(self.Cs[int(np.argmax(self.cv_auc_))])

        Z = self._stage1(X, y)                           # final model on all training rows
        self.scale_, self.adapt_ = self._stage2_weights(Z, y)
        self.lasso_ = self._lasso((Z / self.scale_ * self.adapt_).astype('float32'), y, self.C_)
        theta = self.lasso_.coef_.ravel()
        self.coef_ = theta * self.adapt_ / self.scale_                # back on the Z scale
        self.intercept_ = float(self.lasso_.intercept_[0])
        self._Z_train = Z
        return self

    def decision_function(self, X):
        return self._design(X).astype('float64') @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    # ------------------------------------------------------------------ interpretation
    def _describe(self, conds):
        parts = []
        for f, op, thr in conds:
            if f in self.cat_cols:
                m, _ = self._te_maps[f]
                sel = sorted(map(str, m.index[(m <= thr) if op == '<=' else (m > thr)]))
                shown = ', '.join(sel[:6]) + (f', ... ({len(sel)} values)' if len(sel) > 6 else '')
                parts.append(f'{f} in {{{shown}}}')
            else:
                parts.append(f'{f} {op} {thr:.3g}')
        return ' AND '.join(parts)

    def terms(self):
        """Every selected term with its coefficient, odds ratio and average marginal effect."""
        names = self.lin_names_ + [self._describe(self.rules_[j]) for j in self.keep_rules_]
        kinds = ['linear'] * self.n_linear_ + [
            'rule (1 variable)' if len(self.rules_[j]) == 1 else 'rule (2 variables)' for j in self.keep_rules_]
        Z, b, b0 = self._Z_train, self.coef_, self.intercept_
        eta = Z @ b + b0
        rows = []
        for v in np.flatnonzero(b != 0):
            if np.all(np.isin(Z[:, v], [0.0, 1.0])):    # binary: slide 36 formula
                e1 = eta + (1 - Z[:, v]) * b[v]
                e0 = eta - Z[:, v] * b[v]
                ame = float(np.mean(1 / (1 + np.exp(-e1)) - 1 / (1 + np.exp(-e0))))
            else:                                        # continuous: slide 35 formula, per unit
                p = 1 / (1 + np.exp(-eta))
                ame = float(np.mean(p * (1 - p)) * b[v])
            rows.append({'term': names[v], 'kind': kinds[v], 'coef': float(b[v]),
                         'odds_ratio': float(np.exp(b[v])), 'avg_marginal_effect': ame,
                         'support_share': float(np.mean(Z[:, v] != 0))})
        out = pd.DataFrame(rows)
        return out.reindex(out['avg_marginal_effect'].abs().sort_values(ascending=False).index).reset_index(drop=True)

    def summary(self):
        b = self.coef_
        return {'linear_terms': self.n_linear_, 'rules_extracted': len(self.rules_),
                'rules_after_dedup': self.n_rules_,
                'selected_linear': int(np.count_nonzero(b[:self.n_linear_])),
                'selected_rules': int(np.count_nonzero(b[self.n_linear_:])),
                'lasso_C': self.C_, 'cv_auc_at_C': round(float(self.cv_auc_.max()), 4),
                'cv_auc_by_C': {f'{c:.4g}': round(float(a), 4) for c, a in zip(self.Cs, self.cv_auc_)},
                'gamma': self.gamma, 'pilot_C': self.pilot_C}
