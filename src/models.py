"""The three model arms required by the brief, behind one interface.

    1. white-box            : WOE scorecard (logistic regression)
    2. machine learning     : XGBoost
    3. tabular foundation   : TabPFN

TabPFN trains on at most ~5,000 samples on CPU, so it is subsampled. Use the
SAME subsample for every arm when comparing, or the comparison confounds model
class with training-set size.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# TabPFN's CPU ceiling is ~5,000 samples, but cost scales roughly QUADRATICALLY
# with context length: ~400 rows fits in 10s, 5,000 rows took >10 minutes on
# CPU without finishing. Lower this to get a tractable run.
#
#     export TABPFN_MAX_TRAIN=2000     # ~6x faster than 5000
#
# This does NOT weaken the comparison: `matched` mode trains EVERY arm on the
# same subsample, so the three-way contrast stays fair at any size. State the
# number you used in the report and move on.
TABPFN_MAX_TRAIN = int(os.environ.get("TABPFN_MAX_TRAIN", "5000"))

# Where TabPFN runs.
#   local  (default) -- weights on this machine. On CPU this is SLOW, and the
#           cost is driven by the TEST set, not the training set: measured 209s
#           to predict 9,113 rows from a 500-row context, and roughly 4x that
#           from a 2,000-row context. A full run is ~1 hour on CPU.
#   client -- Prior Labs' hosted inference (pip install tabpfn-client). Runs on
#           their GPUs, uses the same TABPFN_TOKEN, and turns that hour into
#           under a minute. NOTE: this uploads the feature matrix to a third
#           party. Fine here -- the Stanford Open Policing data is already
#           public and de-identified under an open licence -- but it is a
#           deliberate choice, not an incidental one.
#
#     pip install --upgrade tabpfn-client
#     export TABPFN_BACKEND=client TABPFN_TOKEN=...
TABPFN_BACKEND = os.environ.get("TABPFN_BACKEND", "local").strip().lower() or "local"
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


# Which TabPFN checkpoint the foundation-model arm uses.
#
# Default V2 because it is UNGATED: no PriorLabs token, no licence acceptance,
# no account of any kind. Everyone on the team can run it immediately, and it
# is the version in Hollmann et al., Nature 637 (2025), under the Prior Labs
# License (Apache 2.0 + attribution).
#
# V3_5 is newer and scores better, but its weights are non-commercial and it
# requires a PriorLabs account, a licence acceptance at ux.priorlabs.ai, and an
# 876 MB download. If you have already done that, opt in per-machine without
# touching the code or making teammates follow you:
#
#     export TABPFN_VERSION=V3_5
#
# Valid: V2, V2_5, V2_6, V3, V3_5, V3_5_FAST (tabpfn.constants.ModelVersion).
# The version actually used is recorded in every metrics JSON, so the report
# cannot silently misstate which model was fitted.
TABPFN_VERSION = os.environ.get("TABPFN_VERSION", "V2").strip() or "V2"


class TabPFNArm:
    """Foundation-model arm. Subsamples to TABPFN_MAX_TRAIN on fit.

    TabPFN has no native global explanation, so this is the honest black-box arm
    of the comparison. tabpfn-extensions provides SHAP-based local explanations.

    DEFAULTS TO V2, DELIBERATELY. The weights for v3/v3.5 live in GATED
    HuggingFace repos (`Prior-Labs/tabpfn-v3.5` returns 401), which need three
    separate approvals: a PriorLabs API token, a PriorLabs licence acceptance,
    AND a HuggingFace account that has been granted access to the gated repo.
    `Prior-Labs/TabPFN-v2-clf` returns 200 -- ungated, no token, no licence gate.

    v2 is also the better citation for a course report: it is the version in
    Hollmann et al., "Accurate predictions on small data with a tabular
    foundation model", Nature 637 (2025). The v2 weights are under the Prior
    Labs License (Apache 2.0 plus an attribution requirement); v3.5 weights are
    non-commercial.

    Pass version="V3_5" to opt in, once you have HF access. Valid values are the
    names of tabpfn.constants.ModelVersion: V2, V2_5, V2_6, V3, V3_5, V3_5_FAST.
    """

    def __init__(self, max_train: int = TABPFN_MAX_TRAIN, seed: int = SEED,
                 version: str | None = TABPFN_VERSION,
                 backend: str = TABPFN_BACKEND):
        self.max_train, self.seed, self.version = max_train, seed, version
        self.backend = backend
        self.prep = self.clf = None

    def _make(self):
        if self.backend == "client":
            # Hosted inference. `version` does not apply -- the server picks the
            # checkpoint -- so it is recorded as "client" in the metrics JSON
            # rather than silently reported as whatever TABPFN_VERSION said.
            from tabpfn_client import TabPFNClassifier as ClientClassifier

            return ClientClassifier()

        from tabpfn import TabPFNClassifier

        if not self.version:
            return TabPFNClassifier()                 # package default (currently 3.5, gated)
        from tabpfn.constants import ModelVersion

        try:
            mv = ModelVersion[self.version.upper()]
        except KeyError as exc:
            valid = [m.name for m in ModelVersion]
            raise ValueError(f"unknown TabPFN version {self.version!r}; valid: {valid}") from exc
        return TabPFNClassifier.create_default_for_version(mv)

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_train:
            idx = rng.choice(len(X), self.max_train, replace=False)
            X, y = X.iloc[idx], y[idx]
        self.prep = _preprocessor(X, scale=False).fit(X)
        self.clf = self._make().fit(self.prep.transform(X), y)
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
