"""Local explanations: why did the model score THIS stop the way it did?

The course's own description names this as aim (1): "identify the variables that
play the most important role ... **or understand individual decisions**".

THE THREE ARMS SUPPORT THREE DIFFERENT GRADES OF EXPLANATION, and that asymmetry
is the finding, not an inconvenience:

  scorecard  EXACT, closed form.   For a linear model the Shapley value of feature
                                   j is coef_j * (x_j - E[x_j]). No approximation,
                                   no library, no sampling.
  gbm        EXACT (TreeSHAP).     xgboost computes it natively via
                                   predict(pred_contribs=True). Exact for trees.
  tabpfn     APPROXIMATE only.     No native attribution exists. We use occlusion:
                                   replace one feature with its background value and
                                   measure the change in predicted probability. This
                                   is NOT a Shapley value -- it ignores interactions
                                   and depends on the background choice. It is
                                   labelled as such everywhere it is reported.

So interpretability is not a property you either have or lack: the white box gives
you an exact answer for free, the tree gives you an exact answer with a special
algorithm, and the foundation model gives you an estimate whose error you cannot
bound. That is a real cost of the black box, and it is measurable rather than
rhetorical.

All three return contributions in LOG-ODDS for the scorecard/gbm and in probability
for tabpfn occlusion; do not add them across arms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import models as M


# --------------------------------------------------------------------------- helpers
def _prep_and_names(pipe, X: pd.DataFrame):
    """Transformed matrix plus the expanded (post one-hot) feature names."""
    prep = pipe[:-1]
    Z = prep.transform(X)
    try:
        names = list(prep.get_feature_names_out())
    except Exception:                                     # noqa: BLE001
        names = [f"f{i}" for i in range(Z.shape[1])]
    return np.asarray(Z), names


def _collapse(contrib: pd.Series, raw_cols) -> pd.Series:
    """Sum one-hot columns back to their source feature.

    A 17-level `zone` becomes 17 columns; the caseworker wants one number for
    "zone", not seventeen. Sums are valid because Shapley values are additive.
    """
    out = {}
    for name, v in contrib.items():
        base = name.split("__", 1)[-1]
        hit = next((c for c in raw_cols if base == c or base.startswith(c + "_")), base)
        out[hit] = out.get(hit, 0.0) + float(v)
    return pd.Series(out).sort_values(key=np.abs, ascending=False)


# --------------------------------------------------------------------------- scorecard
def explain_scorecard(pipe, X_train: pd.DataFrame, X_row: pd.DataFrame) -> pd.Series:
    """EXACT Shapley values for the linear arm: coef_j * (x_j - E[x_j])."""
    Ztr, names = _prep_and_names(pipe, X_train)
    Zr, _ = _prep_and_names(pipe, X_row)
    coef = pipe[-1].coef_.ravel()
    contrib = pd.Series(coef * (Zr[0] - Ztr.mean(axis=0)), index=names)
    return _collapse(contrib, list(X_train.columns))


# --------------------------------------------------------------------------- gbm
def explain_gbm(pipe, X_train: pd.DataFrame, X_row: pd.DataFrame) -> pd.Series:
    """EXACT TreeSHAP via xgboost's native pred_contribs. Falls back to occlusion."""
    Zr, names = _prep_and_names(pipe, X_row)
    clf = pipe[-1]
    try:
        import xgboost as xgb

        raw = clf.get_booster().predict(xgb.DMatrix(Zr, feature_names=names),
                                        pred_contribs=True)[0]
        contrib = pd.Series(raw[:-1], index=names)        # last entry is the base value
        return _collapse(contrib, list(X_train.columns))
    except Exception:                                     # noqa: BLE001
        return explain_occlusion(pipe, X_train, X_row, label="gbm")


# --------------------------------------------------------------------------- model-agnostic
def explain_occlusion(model, X_train: pd.DataFrame, X_row: pd.DataFrame,
                      label: str = "tabpfn") -> pd.Series:
    """APPROXIMATE. Replace each feature with its background value, measure the change.

    NOT a Shapley value: it ignores interactions and is sensitive to the background
    (median for numerics, mode for categoricals). Reported as "occlusion" wherever
    it is shown, never as SHAP. This is the only option for TabPFN, which exposes
    no attribution of its own -- which is precisely the point.
    """
    base_p = float(model.predict_proba(X_row)[0, 1])
    bg = {}
    for c in X_train.columns:
        s = X_train[c]
        bg[c] = s.median() if pd.api.types.is_numeric_dtype(s) else (
            s.mode().iloc[0] if len(s.mode()) else np.nan)

    out = {}
    for c in X_train.columns:
        pert = X_row.copy()
        pert[c] = bg[c]
        out[c] = base_p - float(model.predict_proba(pert)[0, 1])
    return pd.Series(out).sort_values(key=np.abs, ascending=False)


# --------------------------------------------------------------------------- api
def explain_row(arm: str, model, X_train: pd.DataFrame, X_row: pd.DataFrame) -> dict:
    """{'method': ..., 'exact': bool, 'units': ..., 'contributions': Series}"""
    if arm == "scorecard":
        return {"method": "exact Shapley (linear closed form)", "exact": True,
                "units": "log-odds",
                "contributions": explain_scorecard(model, X_train, X_row)}
    if arm == "gbm":
        return {"method": "exact TreeSHAP (xgboost pred_contribs)", "exact": True,
                "units": "log-odds",
                "contributions": explain_gbm(model, X_train, X_row)}
    return {"method": "occlusion (approximate, NOT Shapley)", "exact": False,
            "units": "probability",
            "contributions": explain_occlusion(model, X_train, X_row, label=arm)}
