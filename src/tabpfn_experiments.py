"""TabPFN-specific experiments: the things only the foundation-model arm can do.

    python -m src.tabpfn_experiments            # all five, cached to outputs/
    python -m src.tabpfn_experiments curve      # just one

Run with the hosted backend or this is impractical:

    export TABPFN_BACKEND=client TABPFN_TOKEN=...

WHY THESE FIVE. TabPFN differs from the other two arms in ways that map onto all
four graded dimensions:

  1. learning_curve      performance  -- TabPFN's claim is small-data efficiency.
                                        Test it on OUR data rather than citing it.
  2. context_stability   stability    -- TabPFN conditions on training rows AS
                                        CONTEXT, so predictions depend on WHICH
                                        rows were drawn. No other arm has this
                                        failure mode.
  3. calibration_by_race fairness     -- calibration is the sufficiency leg of the
                                        taxonomy, and TabPFN has the best Brier
                                        of the three arms (0.1690 vs 0.1824).
  4. model_agreement     interpretab. -- all three score ~0.55. Do they agree on
                                        WHY? Equal accuracy with incompatible
                                        explanations is a real finding.
  5. thinking_mode       performance  -- extra fit-time compute. If it does not
                                        move AUC, that is more evidence of no signal.

COST NOTE. Hosted inference is billed per call and TabPFN re-processes the
training context for every test row. The experiments therefore evaluate on a
FIXED random subsample of the test set (EVAL_N), identical across every arm and
every configuration, so comparisons stay valid while cost stays bounded.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import config as C
from . import data as D
from . import models as M

SEED = 42
SPLIT_YEAR = 2016
EVAL_N = 3000        # fixed test subsample for every experiment
PERM_EVAL_N = 1500   # smaller still for permutation importance (many refits)
CURVE_SIZES = (100, 250, 500, 1000, 2000, 5000)


# --------------------------------------------------------------------------- frame
def frame(include_nbh: bool = False):
    """(X_train_pool, y_train_pool, X_eval, y_eval, meta_eval) on consent searches."""
    df = pd.read_parquet(C.LABELLED_PARQUET)
    X, y, meta = D.build_xy(df, search_type="consent", include_nbh=include_nbh)
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    meta = meta.reset_index(drop=True)

    tr = (meta["year"] < SPLIT_YEAR).to_numpy()
    rng = np.random.default_rng(SEED)
    te_idx = np.flatnonzero(~tr)
    if len(te_idx) > EVAL_N:
        te_idx = rng.choice(te_idx, EVAL_N, replace=False)
    return (X[tr].reset_index(drop=True), y[tr].to_numpy(),
            X.iloc[te_idx].reset_index(drop=True), y.iloc[te_idx].to_numpy(),
            meta.iloc[te_idx].reset_index(drop=True))


def _fit_predict(arm: str, Xtr, ytr, Xte, **kw):
    if arm == "scorecard":
        m = M.make_scorecard(Xtr).fit(Xtr, ytr)
    elif arm == "gbm":
        m = M.make_gbm(Xtr).fit(Xtr, ytr)
    else:
        m = M.TabPFNArm(max_train=len(Xtr), **kw).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


# --------------------------------------------------------------------------- 1
def learning_curve(sizes=CURVE_SIZES, seed: int = SEED) -> pd.DataFrame:
    """AUC vs training size for all three arms, on one fixed evaluation set.

    TabPFN's headline claim is small-data efficiency, so the expected shape is
    TabPFN leading at small n and converging with the GBM as n grows. On a
    problem with no signal it may not replicate -- which is itself a result.
    """
    Xtr, ytr, Xte, yte, _ = frame()
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        n = min(n, len(Xtr))
        idx = rng.choice(len(Xtr), n, replace=False)
        Xs, ys = Xtr.iloc[idx], ytr[idx]
        if len(np.unique(ys)) < 2:
            continue
        for arm in ("scorecard", "gbm", "tabpfn"):
            t = time.time()
            try:
                p = _fit_predict(arm, Xs, ys, Xte)
                rows.append({"n_train": n, "arm": arm,
                             "auc": roc_auc_score(yte, p),
                             "brier": brier_score_loss(yte, p),
                             "base_rate": float(ys.mean()),
                             "seconds": round(time.time() - t, 1)})
            except Exception as exc:                       # noqa: BLE001
                rows.append({"n_train": n, "arm": arm, "auc": np.nan,
                             "error": f"{type(exc).__name__}: {exc}"[:160]})
            print(f"  n={n:>5} {arm:10s} {rows[-1].get('auc', float('nan')):.4f}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2
def context_stability(n: int = 2000, n_contexts: int = 5, seed: int = SEED) -> dict:
    """How much do TabPFN's predictions depend on WHICH rows are in the context?

    TabPFN trains no weights -- it conditions on the training rows. Draw several
    different contexts of the same size and measure the spread in AUC, the
    per-row standard deviation of predicted probability, and how much the top-K
    ranking churns. A model whose top-200 list reshuffles when you swap the
    context is not deployable, whatever its AUC.

    The GBM is run as a control: it also has sampling variance, so the question
    is whether TabPFN's is materially larger.
    """
    Xtr, ytr, Xte, yte, _ = frame()
    rng = np.random.default_rng(seed)
    out = {}
    for arm in ("tabpfn", "gbm"):
        preds, aucs = [], []
        for i in range(n_contexts):
            idx = rng.choice(len(Xtr), min(n, len(Xtr)), replace=False)
            try:
                p = _fit_predict(arm, Xtr.iloc[idx], ytr[idx], Xte)
            except Exception as exc:                       # noqa: BLE001
                out[arm] = {"error": f"{type(exc).__name__}: {exc}"[:160]}
                break
            preds.append(p)
            aucs.append(roc_auc_score(yte, p))
            print(f"  {arm} context {i + 1}/{n_contexts}: AUC {aucs[-1]:.4f}", flush=True)
        if arm in out:
            continue
        P = np.vstack(preds)
        K = 200
        tops = [set(np.argsort(-p)[:K]) for p in P]
        jac = [len(a & b) / len(a | b) for i, a in enumerate(tops) for b in tops[i + 1:]]
        out[arm] = {
            "n_contexts": len(preds), "n_train": int(n),
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "auc_min": float(np.min(aucs)), "auc_max": float(np.max(aucs)),
            "per_row_pred_std_mean": float(P.std(axis=0).mean()),
            "per_row_pred_std_max": float(P.std(axis=0).max()),
            "top200_jaccard_mean": float(np.mean(jac)) if jac else None,
        }
    return out


# --------------------------------------------------------------------------- 3
def calibration_by_race(n: int = 2000, bins: int = 5) -> pd.DataFrame:
    """Calibration within each racial group, for all three arms.

    Calibration IS the sufficiency leg of the fairness taxonomy. TabPFN has the
    best Brier overall; the question is whether that holds WITHIN groups, which
    is what sufficiency actually requires.
    """
    from .fairness import calibration_by_group

    Xtr, ytr, Xte, yte, meta = frame()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xtr), min(n, len(Xtr)), replace=False)
    keep = meta["subject_race"].isin(C.RACE_REPORTABLE).to_numpy()

    frames = []
    for arm in ("scorecard", "gbm", "tabpfn"):
        try:
            p = _fit_predict(arm, Xtr.iloc[idx], ytr[idx], Xte)
        except Exception as exc:                           # noqa: BLE001
            print(f"  {arm}: {type(exc).__name__}: {exc}"[:160], flush=True)
            continue
        cal = calibration_by_group(yte[keep], p[keep], meta["subject_race"][keep], bins=bins)
        cal["arm"] = arm
        for g in cal["g"].unique():
            m = keep & (meta["subject_race"] == g).to_numpy()
            cal.loc[cal["g"] == g, "group_brier"] = brier_score_loss(yte[m], p[m])
        frames.append(cal)
        print(f"  {arm} calibrated", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- 4
def model_agreement(n: int = 2000, n_repeats: int = 2, seed: int = SEED) -> pd.DataFrame:
    """Permutation importance for all three arms: do they agree on WHY?

    All three score ~0.55. If they rank drivers differently while scoring
    identically, that is a finding about interpretability -- equal accuracy,
    incompatible explanations -- and it speaks directly to the course's
    "distance among models" framing.

    Model-agnostic on purpose: TabPFN has no native global explanation, so this
    is the only way to put all three on the same footing.
    """
    Xtr, ytr, Xte, yte, _ = frame()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), min(n, len(Xtr)), replace=False)
    Xs, ys = Xtr.iloc[idx], ytr[idx]
    Xe, ye = Xte.iloc[:PERM_EVAL_N], yte[:PERM_EVAL_N]

    rows = []
    for arm in ("scorecard", "gbm", "tabpfn"):
        if arm == "scorecard":
            m = M.make_scorecard(Xs).fit(Xs, ys)
        elif arm == "gbm":
            m = M.make_gbm(Xs).fit(Xs, ys)
        else:
            m = M.TabPFNArm(max_train=len(Xs)).fit(Xs, ys)
        base = roc_auc_score(ye, m.predict_proba(Xe)[:, 1])
        for col in Xe.columns:
            drops = []
            for r in range(n_repeats):
                Xp = Xe.copy()
                Xp[col] = rng.permutation(Xp[col].to_numpy())
                drops.append(base - roc_auc_score(ye, m.predict_proba(Xp)[:, 1]))
            rows.append({"arm": arm, "feature": col, "base_auc": base,
                         "auc_drop": float(np.mean(drops))})
        print(f"  {arm}: base AUC {base:.4f}, {len(Xe.columns)} features permuted", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 5
def thinking_mode_check(n: int = 2000) -> pd.DataFrame:
    """Does extra fit-time compute help? Hosted backend only."""
    if M.TABPFN_BACKEND != "client":
        return pd.DataFrame([{"note": "thinking_mode requires TABPFN_BACKEND=client"}])
    Xtr, ytr, Xte, yte, _ = frame()
    idx = np.random.default_rng(SEED).choice(len(Xtr), min(n, len(Xtr)), replace=False)
    Xs, ys = Xtr.iloc[idx], ytr[idx]

    rows = []
    for label, kw in [("off", {}), ("medium", {"thinking_effort": "medium"}),
                      ("high", {"thinking_effort": "high"})]:
        t = time.time()
        try:
            from tabpfn_client import TabPFNClassifier as Clf

            prep = M._preprocessor(Xs, scale=False).fit(Xs)
            clf = Clf(**kw).fit(prep.transform(Xs), ys)
            p = clf.predict_proba(prep.transform(Xte))[:, 1]
            rows.append({"thinking": label, "auc": roc_auc_score(yte, p),
                         "brier": brier_score_loss(yte, p),
                         "seconds": round(time.time() - t, 1)})
        except Exception as exc:                           # noqa: BLE001
            rows.append({"thinking": label, "auc": np.nan,
                         "error": f"{type(exc).__name__}: {exc}"[:160]})
        print(f"  thinking={label}: {rows[-1].get('auc', float('nan'))}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- run
EXPERIMENTS = {
    "curve": (learning_curve, "tabpfn_learning_curve.parquet"),
    "stability": (context_stability, "tabpfn_context_stability.json"),
    "calibration": (calibration_by_race, "tabpfn_calibration_by_race.parquet"),
    "agreement": (model_agreement, "tabpfn_model_agreement.parquet"),
    "thinking": (thinking_mode_check, "tabpfn_thinking_mode.parquet"),
}


def run(which=None) -> None:
    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    names = [which] if which else list(EXPERIMENTS)
    print(f"backend={M.TABPFN_BACKEND} version={M.TABPFN_VERSION} eval_n={EVAL_N}")
    for name in names:
        fn, out = EXPERIMENTS[name]
        print(f"\n=== {name} ===", flush=True)
        t = time.time()
        res = fn()
        path = C.OUTPUTS / out
        if isinstance(res, pd.DataFrame):
            res.to_parquet(path, index=False)
        else:
            path.write_text(json.dumps(res, indent=2))
        print(f"  -> {path.name}  ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
