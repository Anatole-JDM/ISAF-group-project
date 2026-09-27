"""Are the AUC differences real? DeLong tests and officer-clustered bootstrap.

Two problems with how AUCs get compared in this project so far:

1. **No test.** We report 0.6403 / 0.6428 / 0.6459 as if the ordering means
   something, with no standard error on any DIFFERENCE. The models are scored on
   the SAME test rows, so their AUCs are correlated and an unpaired comparison is
   wrong. DeLong (1988) is the standard paired test.

2. **Clustering.** Observations cluster by officer -- 1,477 officers, the top 10
   accounting for 10.2% of searches. Every interval reported elsewhere assumes
   independent rows, which understates the variance. Resampling OFFICERS rather
   than rows fixes this.

Both matter here because this project's own finding 10.2 measured a context
variance of 0.024, which is an order of magnitude larger than most of the gaps
being compared.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- DeLong
def _midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def delong_auc_cov(y_true, scores: dict):
    """Fast DeLong: returns (auc per model, covariance matrix) for the SAME rows."""
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y)                     # positives first
    y = y[order]
    m = int(y.sum())
    n = len(y) - m
    names = list(scores)
    P = np.vstack([np.asarray(scores[k], dtype=float)[order] for k in names])
    k = P.shape[0]

    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(P[r, :m])
        ty[r] = _midrank(P[r, m:])
        tz[r] = _midrank(P[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    cov = np.atleast_2d(cov)
    return dict(zip(names, aucs)), cov, names


def delong_test(y_true, scores: dict) -> list[dict]:
    """Pairwise DeLong tests. H0: the two AUCs are equal."""
    aucs, cov, names = delong_auc_cov(y_true, scores)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = aucs[names[i]] - aucs[names[j]]
            var = cov[i, i] + cov[j, j] - 2 * cov[i, j]
            se = float(np.sqrt(max(var, 1e-300)))
            z = d / se
            out.append({"a": names[i], "b": names[j],
                        "auc_a": float(aucs[names[i]]), "auc_b": float(aucs[names[j]]),
                        "diff": float(d), "se": se, "z": float(z),
                        "p_value": float(2 * stats.norm.sf(abs(z))),
                        "ci95": [float(d - 1.96 * se), float(d + 1.96 * se)]})
    return out


# --------------------------------------------------------------------------- clustered
def cluster_bootstrap_auc_diff(y_true, score_a, score_b, clusters,
                               n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap the AUC difference by resampling CLUSTERS (officers), not rows.

    Rows within an officer are not independent -- the officer's own track record,
    shift patterns and habits are shared. Resampling rows treats 58,865 searches as
    58,865 independent draws when they are closer to 1,477. This widens the interval
    to something honest.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true).astype(int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    cl = np.asarray(clusters)
    uniq = np.unique(cl)
    idx_by = {c: np.flatnonzero(cl == c) for c in uniq}
    rng = np.random.default_rng(seed)

    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        rows = np.concatenate([idx_by[c] for c in pick])
        yy = y[rows]
        if len(np.unique(yy)) < 2:
            continue
        diffs.append(roc_auc_score(yy, a[rows]) - roc_auc_score(yy, b[rows]))
    d = np.asarray(diffs, dtype=float)
    return {"diff_mean": float(d.mean()), "se": float(d.std()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
            "n_clusters": int(len(uniq)), "n_boot": int(len(d))}
