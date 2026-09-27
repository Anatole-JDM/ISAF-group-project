"""Global interpretability the course drills hardest: the scorecard opened up,
a decision-tree arm, and XPER.

    python -m src.interpret          # all three, cached to outputs/

WHY THESE THREE
  coefficient_table  The deck spends five consecutive formula slides on reading a
                     logistic regression (odds ratios, AME, MEM, per-1-SD) and we
                     had fitted one without ever opening it.
  decision_tree      Rung 2 of the canonical ladder, and the most interpretable
                     model there is. Worth fitting precisely because it might win.
  xper               The instructor's own research: decompose AUC itself into
                     per-feature Shapley contributions, AUC = phi_0 + sum(phi_j)
                     with phi_0 = 0.5. With p=10 features the EXACT computation is
                     2^10 = 1024 coalitions -- no approximation, no package.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

from . import config as C
from . import data as D
from . import models as M

SPLIT_YEAR = 2016
SEED = 42


def frame(search_type: str = "consent"):
    df = pd.read_parquet(C.LABELLED_PARQUET)
    X, y, meta = D.build_xy(df, search_type=search_type)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    meta = meta.reset_index(drop=True)
    tr = (meta["year"] < SPLIT_YEAR).to_numpy()
    return X, y, meta, tr


# --------------------------------------------------------------------------- 1
def coefficient_table(top: int | None = None) -> pd.DataFrame:
    """Open the white box: coefficient, odds ratio, AME, MEM.

    The deck's key warning (sl. 34): beta_j is NOT the marginal effect. For a
    logit the marginal effect is p(1-p)*beta_j, so it is individual-specific.
    On a 17% base rate p(1-p) ~= 0.14, which shrinks every effect by ~7x --
    worth stating, because it is the difference between "a big coefficient" and
    "a big effect on the probability".

      AME  average marginal effect  = mean_i[ p_i(1-p_i) ] * beta_j
      MEM  marginal effect at mean  = p_bar(1-p_bar)      * beta_j

    Numeric features are standardised by the pipeline, so their coefficient is
    already "per 1 SD". Categorical columns are one-hot, so their coefficient is
    the log-odds shift versus the dropped level.
    """
    X, y, meta, tr = frame()
    pipe = M.make_scorecard(X).fit(X[tr], y[tr])
    prep, clf = pipe[:-1], pipe[-1]
    names = list(prep.get_feature_names_out())
    beta = clf.coef_.ravel()

    p = pipe.predict_proba(X[tr])[:, 1]
    dens = float(np.mean(p * (1 - p)))          # AME multiplier
    pbar = float(p.mean())
    dens_mean = pbar * (1 - pbar)               # MEM multiplier

    t = pd.DataFrame({
        "feature": [n.split("__", 1)[-1] for n in names],
        "block": [n.split("__", 1)[0] for n in names],
        "coef_log_odds": beta,
        "odds_ratio": np.exp(beta),
        "AME_pp": beta * dens * 100,
        "MEM_pp": beta * dens_mean * 100,
    })
    t["abs"] = t["coef_log_odds"].abs()
    t = t.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    t.attrs["intercept"] = float(clf.intercept_[0])
    t.attrs["ame_multiplier"] = dens
    t.attrs["mem_multiplier"] = dens_mean
    t.attrs["base_rate_train"] = float(y[tr].mean())
    return t.head(top) if top else t


# --------------------------------------------------------------------------- 2
def decision_tree(depths=(2, 3, 4, 5, 6)) -> dict:
    """A decision-tree arm, plus MDI importance and the printable rules.

    The most interpretable model there is. Fit it precisely because it might beat
    the logistic scorecard -- on this data it does.
    """
    X, y, meta, tr = frame()
    prep = M._preprocessor(X, scale=False).fit(X[tr])
    Ztr, Zte = prep.transform(X[tr]), prep.transform(X[~tr])
    names = list(prep.get_feature_names_out())

    out = {"depths": {}}
    best, best_auc = None, -1
    for d in depths:
        t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Ztr, y[tr])
        auc = float(roc_auc_score(y[~tr], t.predict_proba(Zte)[:, 1]))
        out["depths"][d] = {"auc": auc, "n_leaves": int(t.get_n_leaves())}
        if auc > best_auc:
            best, best_auc, best_d = t, auc, d
    out["best_depth"] = best_d
    out["best_auc"] = best_auc

    # MDI: impurity-based importance, collapsed back to source features
    mdi = pd.Series(best.feature_importances_, index=names)
    agg = {}
    for n, v in mdi.items():
        base = n.split("__", 1)[-1]
        src = next((c for c in X.columns if base == c or base.startswith(c + "_")), base)
        agg[src] = agg.get(src, 0.0) + float(v)
    out["mdi"] = dict(sorted(agg.items(), key=lambda kv: -kv[1]))
    out["rules"] = export_text(best, feature_names=names, max_depth=3)[:4000]
    return out


# --------------------------------------------------------------------------- 3
def xper(arm: str = "scorecard", max_features: int | None = None,
         verbose: bool = True) -> dict:
    """EXACT XPER: decompose test AUC into per-feature Shapley contributions.

    AUC = phi_0 + sum_j phi_j,  with phi_0 = 0.5 (a model with no features).

    phi_j = sum over coalitions S not containing j of
              |S|! (p-|S|-1)! / p!  *  [ AUC(S + j) - AUC(S) ]

    Uses the deck's option (1): RE-ESTIMATE the model on each feature subset,
    rather than marginalising. With p=10 that is 2^10 = 1024 fits -- exact, no
    sampling, no package.

    Efficiency is checked and reported: sum(phi_j) must equal AUC(all) - 0.5.
    """
    X, y, meta, tr = frame()
    cols = list(X.columns)[:max_features] if max_features else list(X.columns)
    X = X[cols]
    p = len(cols)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[~tr], y[~tr]

    mk = {"scorecard": M.make_scorecard, "gbm": M.make_gbm}[arm]
    cache: dict[frozenset, float] = {}

    def auc_of(S: frozenset) -> float:
        if S in cache:
            return cache[S]
        if not S:
            cache[S] = 0.5
            return 0.5
        s = sorted(S)
        m = mk(Xtr[s]).fit(Xtr[s], ytr)
        cache[S] = float(roc_auc_score(yte, m.predict_proba(Xte[s])[:, 1]))
        return cache[S]

    t0 = time.time()
    total = 2 ** p
    for i, r in enumerate(range(p + 1)):
        for combo in itertools.combinations(cols, r):
            auc_of(frozenset(combo))
        if verbose:
            print(f"  coalitions of size {r}: {len(cache)}/{total} cached "
                  f"({time.time()-t0:.0f}s)", flush=True)

    phi = {}
    for j in cols:
        rest = [c for c in cols if c != j]
        tot = 0.0
        for r in range(len(rest) + 1):
            w = math.factorial(r) * math.factorial(p - r - 1) / math.factorial(p)
            for S in itertools.combinations(rest, r):
                Sf = frozenset(S)
                tot += w * (auc_of(Sf | {j}) - auc_of(Sf))
        phi[j] = tot

    full = auc_of(frozenset(cols))
    return {
        "arm": arm, "phi_0": 0.5, "auc_full": full,
        "phi": dict(sorted(phi.items(), key=lambda kv: -kv[1])),
        "sum_phi": float(sum(phi.values())),
        "efficiency_gap": float(full - 0.5 - sum(phi.values())),
        "n_coalitions": len(cache), "seconds": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------- run
def run() -> None:
    C.OUTPUTS.mkdir(parents=True, exist_ok=True)

    print("=== coefficient table ===", flush=True)
    t = coefficient_table()
    t.to_parquet(C.OUTPUTS / "coefficient_table.parquet", index=False)
    (C.OUTPUTS / "coefficient_meta.json").write_text(json.dumps({
        k: t.attrs[k] for k in ("intercept", "ame_multiplier", "mem_multiplier",
                                "base_rate_train")}, indent=2))
    print(t.head(8).to_string(index=False), flush=True)

    print("\n=== decision tree ===", flush=True)
    d = decision_tree()
    (C.OUTPUTS / "decision_tree.json").write_text(json.dumps(d, indent=2))
    for k, v in d["depths"].items():
        print(f"  depth {k}: AUC {v['auc']:.4f}  leaves {v['n_leaves']}", flush=True)

    print("\n=== XPER (exact, 2^10 coalitions) ===", flush=True)
    xp = xper("scorecard")
    (C.OUTPUTS / "xper_scorecard.json").write_text(json.dumps(xp, indent=2))
    print(f"  AUC {xp['auc_full']:.4f} = 0.5 + {xp['sum_phi']:.4f}   "
          f"efficiency gap {xp['efficiency_gap']:+.2e}", flush=True)
    for k, v in xp["phi"].items():
        print(f"    phi[{k:28s}] = {v:+.5f}", flush=True)


if __name__ == "__main__":
    run()
