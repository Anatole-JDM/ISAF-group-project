"""Economic performance under a budget constraint.

The decision is not "classify at 0.5". It is: with capacity for K searches,
WHICH stops do we search? So the operating metric is precision@K, and models
with identical AUC can differ sharply in net benefit.

THE HONEST PART, and it belongs in the report rather than a footnote:
the cost of a false positive here is the cost of searching an INNOCENT driver.
That is not in the data and it is not a technical quantity — it is a policy
judgement about how much an unjustified search costs a person. Every number
below is a stated assumption, so always ship the sensitivity table. Choosing
cost_fp is itself the trustworthy-AI argument: as cost_fp rises, the optimal
policy searches fewer people and the fairness gaps shrink.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Stated assumptions — sensitivity-test all of these, do not treat as facts.
DEFAULT_BENEFIT_TP = 1.0     # normalised: one justified search = 1 unit
DEFAULT_COST_FP = 1.0        # one innocent driver searched = 1 unit (i.e. parity)


def net_benefit_at_k(y_true, y_score, k: int,
                     benefit_tp: float = DEFAULT_BENEFIT_TP,
                     cost_fp: float = DEFAULT_COST_FP) -> dict:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    k = min(int(k), len(s))
    top = np.argsort(-s)[:k]
    tp = int(y[top].sum())
    fp = k - tp
    return {
        "k": k, "tp": tp, "fp": fp,
        "precision_at_k": tp / k if k else np.nan,
        "recall_at_k": tp / y.sum() if y.sum() else np.nan,
        "net_benefit": benefit_tp * tp - cost_fp * fp,
    }


def top_k_curve(y_true, y_score, ks=None, **kw) -> pd.DataFrame:
    n = len(np.asarray(y_score))
    ks = ks if ks is not None else np.unique(np.linspace(50, n, 40).astype(int))
    return pd.DataFrame([net_benefit_at_k(y_true, y_score, k, **kw) for k in ks])


def sensitivity(y_true, y_score, k: int, cost_ratios=(0.25, 0.5, 1, 2, 4, 8)) -> pd.DataFrame:
    """Net benefit as the price of searching an innocent driver varies.

    The row where net_benefit turns negative is the point at which the model
    should not be deployed at all. Name that number in the recommendation.
    """
    return pd.DataFrame([
        {"cost_fp": c, **net_benefit_at_k(y_true, y_score, k, cost_fp=c)}
        for c in cost_ratios
    ])


def equity_efficiency_frontier(y_true, y_score, groups, ks,
                               reference: str = "white", **kw) -> pd.DataFrame:
    """Net benefit against the FPR disparity it produces, as K varies.

    Plot this. It is the clearest single picture of the trade-off the client
    is actually being asked to make, and it beats any AUC table.
    """
    from .fairness import group_rates

    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    g = pd.Series(groups).reset_index(drop=True)

    rows = []
    for k in ks:
        kk = min(int(k), len(s))
        thr_idx = np.argsort(-s)[:kk]
        yhat = np.zeros(len(s), dtype=int)
        yhat[thr_idx] = 1
        r = group_rates(y, yhat, g)
        nb = net_benefit_at_k(y, s, kk, **kw)
        if reference in r.index:
            gap = (r["FPR"] - r.loc[reference, "FPR"]).drop(reference)
            rows.append({"k": kk, "net_benefit": nb["net_benefit"],
                         "precision_at_k": nb["precision_at_k"],
                         "max_FPR_gap": gap.abs().max(),
                         **{f"FPR_gap_{i}": v for i, v in gap.items()}})
    return pd.DataFrame(rows)
