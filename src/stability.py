"""Structural stability, in the sense the course uses it:

    - distance among models
    - slow variation in feature contributions

So this is not just "does AUC hold up on a later year". It is whether the
*explanation* is stable: if the drivers of the score churn between resamples,
between years, or between officers, the model is not trustworthy even when its
AUC is flat.

Three axes for Nashville:
  1. TEMPORAL  — stops fall 444k (2012) to 204k (2018) while the search rate
                 holds near 4%. The 2016 Driving While Black report triggered
                 documented MNPD practice changes, so pre/post-2016 is a real
                 regime split rather than an arbitrary cut.
  2. OFFICER   — officer_id_hash. If performance collapses when officers are
                 held out, the model learned discretion, not contraband risk.
                 That is the selective-labels claim, made measurable.
  3. GEOGRAPHIC— leave-one-precinct-out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

SEED = 42
REGIME_SPLIT_YEAR = 2016     # Driving While Black report


# --------------------------------------------------------------------------- splits
def temporal_split(df: pd.DataFrame, split_year: int = REGIME_SPLIT_YEAR):
    return df["year"] < split_year, df["year"] >= split_year


def officer_cv(X, y, groups, model, n_splits: int = 5) -> pd.DataFrame:
    """Grouped CV by officer: no officer appears in both train and test.

    Compare against a random split. A large gap is the headline stability
    finding — it means the model does not transfer to unseen officers.
    """
    y = np.asarray(y)
    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        m = clone(model) if hasattr(model, "get_params") else model
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        rows.append({"fold": fold, "n_test": len(te),
                     "n_officers_test": pd.Series(groups).iloc[te].nunique(),
                     "auc": roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- drift
def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index. >0.25 is the usual 'material shift' threshold."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    cuts = np.unique(np.nanquantile(e, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return np.nan
    e_pct = np.histogram(e, bins=cuts)[0] / len(e)
    a_pct = np.histogram(a, bins=cuts)[0] / len(a)
    eps = 1e-6
    e_pct, a_pct = np.clip(e_pct, eps, None), np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_table(df_ref: pd.DataFrame, df_new: pd.DataFrame, cols=None) -> pd.DataFrame:
    cols = cols or df_ref.select_dtypes("number").columns.tolist()
    return (pd.DataFrame([{"feature": c, "psi": psi(df_ref[c].dropna(), df_new[c].dropna())}
                          for c in cols if c in df_new.columns])
            .sort_values("psi", ascending=False, ignore_index=True))


# --------------------------------------------------------------------------- contributions
def coefficient_stability(model, X, y, n_boot: int = 50, seed: int = SEED) -> pd.DataFrame:
    """Bootstrap the model and measure how much each coefficient moves.

    'Slow variation in feature contributions' operationalised: a feature whose
    coefficient changes sign across resamples is not a finding, it is noise.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    coefs = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        m = clone(model).fit(X.iloc[idx], y[idx])
        clf = m[-1] if hasattr(m, "steps") else m
        names = (m[:-1].get_feature_names_out() if hasattr(m, "steps")
                 else np.arange(clf.coef_.shape[1]))
        coefs.append(pd.Series(clf.coef_.ravel(), index=names))
    C_ = pd.DataFrame(coefs)
    out = pd.DataFrame({"mean": C_.mean(), "std": C_.std(),
                        "sign_flip_rate": ((C_ > 0).mean()).apply(lambda p: min(p, 1 - p))})
    out["cv"] = (out["std"] / out["mean"].abs()).replace([np.inf, -np.inf], np.nan)
    return out.sort_values("sign_flip_rate", ascending=False)


def model_distance(preds: dict) -> pd.DataFrame:
    """Pairwise distance among model score vectors: 1 - Spearman correlation.

    'Distance among models' from the slides. Two models with the same AUC that
    rank people differently are not interchangeable for a top-K policy.
    """
    names = list(preds)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            M.loc[a, b] = 1 - pd.Series(preds[a]).corr(pd.Series(preds[b]), method="spearman")
    return M


def contribution_drift(model, X, y, year: pd.Series, top_n: int = 15) -> pd.DataFrame:
    """Refit per year and track each feature's coefficient over time."""
    y = np.asarray(y)
    # POSITIONAL, not label-based. groupby().groups yields index LABELS; feeding
    # those to y[i] / X.iloc[i] silently mis-aligns whenever the frame's index
    # is not a clean RangeIndex (e.g. after any filtering upstream).
    yr_s = pd.Series(np.asarray(year))
    rows = {}
    for yr, pos in yr_s.groupby(yr_s).indices.items():
        i = np.asarray(pos)
        if len(i) < 200 or len(np.unique(y[i])) < 2:
            continue
        m = clone(model).fit(X.iloc[i], y[i])
        clf = m[-1] if hasattr(m, "steps") else m
        names = m[:-1].get_feature_names_out() if hasattr(m, "steps") else None
        rows[yr] = pd.Series(clf.coef_.ravel(), index=names)
    out = pd.DataFrame(rows)
    return out.loc[out.abs().mean(axis=1).nlargest(top_n).index]
