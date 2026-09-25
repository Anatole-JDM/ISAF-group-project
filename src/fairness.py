"""Fairness metrics, organised by the course taxonomy (slide 242).

    independence : statistical parity, conditional parity
    separation   : equal opportunity (TPR), predictive equality (FPR)
    sufficiency  : calibration, predictive parity (PPV)

OBJECT DEFINITIONS for this project — state these on a slide before any metric:

    Y   = 1 if contraband is found          (what the officer is trying to predict)
    Yhat= 1 if the model recommends a search
    D   = protected attribute (driver race)
    favorable outcome for the individual = NOT being searched

The course convention is that Y = 1 is favorable to the individual (slide 238,
the bank/applicant setting). Here it is the opposite: Y = 1 is favorable to the
police. Every metric therefore reads backwards unless the inversion is declared.
`both_codings()` computes each metric under both so the report can show that the
same model is fair under one reading and unfair under the other.

Two metrics carry the argument:

    FPR by race  = share of INNOCENT drivers the model would search.
                   This is predictive equality, and it is the harm that matters.
    PPV by race  = hit rate = the "outcome test" from the economics of
                   discrimination. This is sufficiency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- core
def group_rates(y_true, y_pred, groups, y_score=None) -> pd.DataFrame:
    """Per-group confusion-derived rates. One row per protected group."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = pd.Series(groups).reset_index(drop=True)

    rows = []
    for g, idx in groups.groupby(groups).groups.items():
        i = np.asarray(idx)
        yt, yp = y_true[i], y_pred[i]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        rows.append({
            "group": g,
            "n": len(i),
            "base_rate": yt.mean() if len(i) else np.nan,
            "selection_rate": yp.mean() if len(i) else np.nan,        # independence
            "TPR": tp / (tp + fn) if (tp + fn) else np.nan,           # separation
            "FPR": fp / (fp + tn) if (fp + tn) else np.nan,           # separation  <- the harm
            "PPV": tp / (tp + fp) if (tp + fp) else np.nan,           # sufficiency <- outcome test
            "NPV": tn / (tn + fn) if (tn + fn) else np.nan,
            "mean_score": np.asarray(y_score)[i].mean() if y_score is not None else np.nan,
        })
    return pd.DataFrame(rows).set_index("group")


def disparities(rates: pd.DataFrame, reference: str = "white") -> pd.DataFrame:
    """Differences and ratios against a reference group.

    Report BOTH. A difference of 4pp on a 20% base is a ratio of 0.8; on a 2%
    base it is a ratio of 3.0. Presenting only one is how audits get gamed.
    """
    if reference not in rates.index:
        raise KeyError(f"reference group {reference!r} not in {list(rates.index)}")
    ref = rates.loc[reference]
    cols = ["selection_rate", "TPR", "FPR", "PPV", "base_rate"]
    out = pd.DataFrame(index=rates.index)
    for c in cols:
        out[f"{c}_diff"] = rates[c] - ref[c]
        out[f"{c}_ratio"] = rates[c] / ref[c]
    out["n"] = rates["n"]
    return out


# --------------------------------------------------------------------------- taxonomy
def independence(rates: pd.DataFrame, reference: str = "white") -> pd.Series:
    """Statistical parity: P(Yhat=1 | D) should not depend on D."""
    return rates["selection_rate"] - rates.loc[reference, "selection_rate"]


def separation(rates: pd.DataFrame, reference: str = "white") -> pd.DataFrame:
    """Equal opportunity (TPR gap) and predictive equality (FPR gap)."""
    return pd.DataFrame({
        "equal_opportunity_gap": rates["TPR"] - rates.loc[reference, "TPR"],
        "predictive_equality_gap": rates["FPR"] - rates.loc[reference, "FPR"],
    })


def sufficiency(rates: pd.DataFrame, reference: str = "white") -> pd.Series:
    """Predictive parity: P(Y=1 | Yhat=1, D) should not depend on D.

    With Yhat = "searched", this IS the outcome test.
    """
    return rates["PPV"] - rates.loc[reference, "PPV"]


def group_rates_ci(y_true, y_pred, groups, n_boot: int = 2000,
                   alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Percentile bootstrap CIs on the per-group rates.

    WHY THIS IS NOT OPTIONAL HERE. Subgroup sizes are wildly uneven -- on the
    consent test set, white n=3,368 and black n=4,947 but hispanic n=689, and at
    a top-K of 2,000 only ~12 hispanic drivers are selected. A TPR or PPV built
    on a dozen cases is not a measurement, and reporting it as a bare point
    estimate invites a question you cannot answer.

    Resampling is done WITHIN each group, with the decision rule held fixed
    (y_pred is passed in already thresholded), so the interval reflects sampling
    variability in the group's rates and not variability in where the cutoff
    happened to fall.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    g = pd.Series(groups).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)

    rows = []
    for grp, idx in g.groupby(g).groups.items():
        i = np.asarray(idx)
        yt, yp = y_true[i], y_pred[i]
        draws = {k: [] for k in ("selection_rate", "TPR", "FPR", "PPV", "base_rate")}
        for _ in range(n_boot):
            b = rng.integers(0, len(i), len(i))
            t, p = yt[b], yp[b]
            tp = int(((t == 1) & (p == 1)).sum()); fp = int(((t == 0) & (p == 1)).sum())
            fn = int(((t == 1) & (p == 0)).sum()); tn = int(((t == 0) & (p == 0)).sum())
            draws["selection_rate"].append(p.mean())
            draws["base_rate"].append(t.mean())
            draws["TPR"].append(tp / (tp + fn) if (tp + fn) else np.nan)
            draws["FPR"].append(fp / (fp + tn) if (fp + tn) else np.nan)
            draws["PPV"].append(tp / (tp + fp) if (tp + fp) else np.nan)
        row = {"group": grp, "n": len(i), "n_selected": int(yp.sum())}
        for k, v in draws.items():
            a = np.asarray(v, dtype=float)
            row[k] = np.nanmean(a)
            row[f"{k}_lo"] = np.nanpercentile(a, lo_q)
            row[f"{k}_hi"] = np.nanpercentile(a, hi_q)
            row[f"{k}_width"] = row[f"{k}_hi"] - row[f"{k}_lo"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def calibration_by_group(y_true, y_score, groups, bins: int = 10) -> pd.DataFrame:
    """Observed frequency vs predicted score, per group. Sufficiency, continuous form."""
    d = pd.DataFrame({
        "y": np.asarray(y_true).astype(int),
        "s": np.asarray(y_score, dtype=float),
        "g": pd.Series(groups).reset_index(drop=True),
    })
    d["bin"] = pd.qcut(d["s"], q=bins, duplicates="drop")
    out = (d.groupby(["g", "bin"], observed=True)
             .agg(n=("y", "size"), predicted=("s", "mean"), observed=("y", "mean"))
             .reset_index())
    out["gap"] = out["observed"] - out["predicted"]
    return out


# --------------------------------------------------------------------------- convention
def both_codings(y_true, y_pred, groups, reference: str = "white") -> dict:
    """Every metric under both Y codings, to demonstrate the convention inversion.

    native : Y = 1 contraband found   (favorable to the police)
    course : Y = 1 no contraband      (favorable to the driver, matching slide 238)

    Note the mapping: TPR in one coding is TNR in the other, and the group that
    looks advantaged flips with it. That is the slide.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        C.Y_NATIVE: group_rates(y_true, y_pred, groups),
        C.Y_COURSE: group_rates(1 - y_true, 1 - y_pred, groups),
    }


# --------------------------------------------------------------------------- headline
def pooled_vs_stratified(
    df: pd.DataFrame,
    target: str = C.TARGET,
    group_col: str = C.PRIMARY_PROTECTED,
    stratum_col: str = "search_type",
    discretionary: str = C.SEARCH_TYPE_DISCRETIONARY,
    reference: str = "white",
    groups=None,
) -> pd.DataFrame:
    """The headline table: hit rate pooled vs. split by discretion.

    Reproduces the verified Nashville result (2026-09-24, n=127,122, 2010-2018,
    resolved search_type, zero unresolved):

        stratum        white    black    b-w        hispanic  h-w
        consent        20.84%   15.68%   -5.16pp     7.88%   -12.96pp
        non-consent    20.83%   27.12%   +6.30pp    15.63%    -5.20pp
        POOLED         20.83%   21.66%   +0.82pp    12.07%    -8.77pp

    Not a composition artifact: consent is 44.6% of white searches, 47.8% of
    black, 46.0% of hispanic.

    The two strata have large gaps in OPPOSITE directions, so pooling cancels
    them and reports "no disparity". This is effect modification, not Simpson's
    paradox -- say so precisely; a jury may push on it.

    Replicates in Illinois (pooled -0.07pp vs consent -7.25pp) and North
    Carolina (pooled -1.15pp vs consent -6.30pp), both computed on raw
    search_basis rather than the resolved type.
    """
    groups = groups or C.RACE_REPORTABLE
    d = df[df[group_col].isin(groups)].copy()
    hit = d[target].astype("string").str.strip().isin({"TRUE", "True", "1"}).astype(int)
    d = d.assign(_hit=hit, _disc=d[stratum_col].eq(discretionary))

    rows = []
    for label, sub in [
        (discretionary, d[d["_disc"]]),
        (f"non-{discretionary}", d[~d["_disc"]]),
        ("POOLED", d),
    ]:
        r = sub.groupby(group_col)["_hit"].agg(["size", "sum", "mean"])
        r.columns = ["searches", "hits", "hit_rate"]
        r["stratum"] = label
        r["gap_vs_ref_pp"] = (r["hit_rate"] - r.loc[reference, "hit_rate"]) * 100
        rows.append(r.reset_index())
    return pd.concat(rows, ignore_index=True)[
        [group_col, "stratum", "searches", "hits", "hit_rate", "gap_vs_ref_pp"]
    ]
