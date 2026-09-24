"""The three model arms required by the brief, behind one interface.

    1. white-box            : WOE scorecard (logistic regression)
    2. machine learning     : XGBoost
    3. tabular foundation   : TabPFN

TabPFN trains on at most ~5,000 samples on CPU, so it is subsampled. Use the
SAME subsample for every arm when comparing, or the comparison confounds model
class with training-set size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TABPFN_MAX_TRAIN = 5000
SEED = 42


def _split_cols(X: pd.DataFrame):
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return num, cat


def _preprocessor(X: pd.DataFrame, scale: bool = True) -> ColumnTransformer:
    num, cat = _split_cols(X)
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=False)),
        ]), cat),
    ])


# --------------------------------------------------------------------------- arms
def make_scorecard(X: pd.DataFrame, class_weight=None) -> Pipeline:
    """White-box arm. Readable to a caseworker: coefficients are the explanation.

    class_weight defaults to None, NOT "balanced". Balanced weighting improves
    ranking on an imbalanced target but destroys calibration — it inflates
    predicted probabilities, which showed up here as a Brier of 0.300 against
    the GBM's 0.168. Calibration IS the sufficiency dimension of the fairness
    taxonomy, so a miscalibrated white-box arm would fail the audit for a
    reason that has nothing to do with fairness. Pass class_weight="balanced"
    only if you are reporting ranking metrics alone, and say so.

    For the full WOE/IV scorecard banks actually deploy, swap the preprocessor
    for optbinning's BinningProcess — it bins monotonically and yields points
    per attribute, which presents far better than raw coefficients.
    """
    return Pipeline([
        ("prep", _preprocessor(X, scale=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight,
                                   random_state=SEED)),
    ])


def make_gbm(X: pd.DataFrame, prefer_xgboost: bool = True, **kw) -> Pipeline:
    """ML arm. Uses XGBoost when installed, else sklearn's HistGradientBoosting.

    Both are gradient-boosted trees and perform comparably here. The fallback
    exists so the pipeline produces real numbers before XGBoost is installed;
    install it (`pip install xgboost`) for the version named in the report.
    """
    if prefer_xgboost:
        try:
            from xgboost import XGBClassifier

            params = dict(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=SEED, n_jobs=-1,
            )
            params.update(kw)
            return Pipeline([("prep", _preprocessor(X, scale=False)),
                             ("clf", XGBClassifier(**params))])
        except ImportError:
            pass

    from sklearn.ensemble import HistGradientBoostingClassifier

    params = dict(max_iter=400, max_depth=5, learning_rate=0.05, random_state=SEED)
    params.update({k: v for k, v in kw.items()
                   if k in HistGradientBoostingClassifier().get_params()})
    return Pipeline([("prep", _preprocessor(X, scale=False)),
                     ("clf", HistGradientBoostingClassifier(**params))])


def gbm_backend() -> str:
    try:
        import xgboost  # noqa: F401
        return "xgboost"
    except Exception:                                    # noqa: BLE001
        return "sklearn-histgb"


# Back-compat alias
make_xgboost = make_gbm


class TabPFNArm:
    """Foundation-model arm. Subsamples to TABPFN_MAX_TRAIN on fit.

    TabPFN has no native global explanation, so this is the honest black-box arm
    of the comparison. tabpfn-extensions provides SHAP-based local explanations.
    """

    def __init__(self, max_train: int = TABPFN_MAX_TRAIN, seed: int = SEED):
        self.max_train, self.seed = max_train, seed
        self.prep = self.clf = None

    def fit(self, X: pd.DataFrame, y):
        from tabpfn import TabPFNClassifier

        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_train:
            idx = rng.choice(len(X), self.max_train, replace=False)
            X, y = X.iloc[idx], y[idx]
        self.prep = _preprocessor(X, scale=False).fit(X)
        self.clf = TabPFNClassifier().fit(self.prep.transform(X), y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self.prep.transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def tabpfn_available() -> bool:
    """True only if TabPFN can actually be imported.

    Catches Exception, not just ImportError: TabPFN pulls in skrub, which tries
    to create ~/skrub_data at import time and raises PermissionError in a
    sandbox. A licence/token problem surfaces later, at fit() -- see
    train.fit_stratum, which isolates each arm so one failure does not kill the
    run.
    """
    try:
        import tabpfn  # noqa: F401
        return True
    except Exception:                                    # noqa: BLE001
        return False


def build_all(X: pd.DataFrame, include_tabpfn: bool = True) -> dict:
    arms = {"scorecard": make_scorecard(X), "gbm": make_gbm(X)}
    if include_tabpfn and tabpfn_available():
        arms["tabpfn"] = TabPFNArm()
    return arms
