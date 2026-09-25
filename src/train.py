"""Fit the arms, score a held-out period, and cache artifacts for the app.

The app must not fit models on page load. This module writes:

    outputs/scores__<stratum>__<mode>.parquet   one row per test search
    outputs/metrics__<stratum>__<mode>.json     headline metrics per arm

SPLIT. The primary split is TEMPORAL at 2016, the year the Driving While Black
report triggered documented MNPD practice changes. That is a regime change, not
an arbitrary cut, so it is the honest test of stability.

MODES.
  matched : every arm trains on the SAME subsample (default 5,000 rows, the
            TabPFN CPU ceiling). This is the only fair three-way comparison --
            otherwise model class is confounded with training-set size.
  full    : scorecard and GBM train on all available rows, to show what the
            subsample costs. TabPFN cannot participate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

from . import config as C
from . import data as D
from . import models as M

SPLIT_YEAR = 2016
MATCHED_N = M.TABPFN_MAX_TRAIN
SEED = 42


@dataclass
class ArmMetrics:
    arm: str
    n_train: int
    n_test: int
    base_rate_train: float
    base_rate_test: float
    auc: float
    pr_auc: float
    brier: float


def _metrics(arm, ytr, yte, p) -> ArmMetrics:
    return ArmMetrics(
        arm=arm,
        n_train=int(len(ytr)), n_test=int(len(yte)),
        base_rate_train=float(np.mean(ytr)), base_rate_test=float(np.mean(yte)),
        auc=float(roc_auc_score(yte, p)),
        pr_auc=float(average_precision_score(yte, p)),
        brier=float(brier_score_loss(yte, p)),
    )


def fit_stratum(df: pd.DataFrame, search_type: str | None = "consent",
                mode: str = "matched", matched_n: int = MATCHED_N,
                split_year: int = SPLIT_YEAR, seed: int = SEED):
    """Fit every available arm on one stratum. Returns (scores_df, metrics_list)."""
    X, y, meta = D.build_xy(df, search_type=search_type,
                            include_protected=not mode.endswith("blind"))
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    meta = meta.reset_index(drop=True)

    tr = meta["year"] < split_year
    te = ~tr
    if tr.sum() < 100 or te.sum() < 100:
        raise ValueError(f"degenerate split for {search_type!r}: {tr.sum()} / {te.sum()}")

    Xtr_all, ytr_all = X[tr], y[tr].to_numpy()
    Xte, yte = X[te], y[te].to_numpy()

    if mode.startswith("matched") and len(Xtr_all) > matched_n:
        idx = np.random.default_rng(seed).choice(len(Xtr_all), matched_n, replace=False)
        Xtr, ytr = Xtr_all.iloc[idx], ytr_all[idx]
    else:
        Xtr, ytr = Xtr_all, ytr_all

    arms = M.build_all(Xtr, include_tabpfn=mode.startswith("matched"))

    out = meta[te].reset_index(drop=True)[
        ["subject_race", "officer_id_hash", "precinct", "year", "search_type"]
    ].copy()
    out["y"] = yte

    metrics = []
    failed = {}
    for name, model in arms.items():
        # Isolate each arm. TabPFN in particular can fail at fit() on a licence
        # or token problem (TabPFN >= 9 requires a one-time licence acceptance
        # and TABPFN_TOKEN), and that must not cost us the other two arms.
        try:
            model.fit(Xtr, ytr)
            p = model.predict_proba(Xte)[:, 1]
        except Exception as exc:                          # noqa: BLE001
            failed[name] = f"{type(exc).__name__}: {exc}".split("\n")[0][:200]
            continue
        out[f"score_{name}"] = p
        metrics.append(_metrics(name, ytr, yte, p))
    return out, metrics, failed


def officer_signal(df: pd.DataFrame, search_type: str = "consent",
                   split_year: int = SPLIT_YEAR) -> dict:
    """THE selective-labels experiment, quantified.

    Refit the ML arm with officer identity added as a feature. If AUC jumps,
    the model is learning WHO STOPPED YOU rather than anything about the driver
    or the situation — which is the claim the whole project rests on.

    Verified on Nashville consent searches, 2026-09-24:
        baseline (pre-search features)  AUC = 0.5649
        + officer identity              AUC = 0.5955
    Against a baseline only 0.0649 above chance, officer identity raises the
    above-chance signal by ~47%, i.e. roughly a third of the model's total
    discriminative power is officer identity.
    """
    from sklearn.metrics import roc_auc_score

    X, y, meta = D.build_xy(df, search_type=search_type)
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    tr = meta["year"] < split_year
    te = ~tr

    out = {}
    for label, frame in [("baseline", X),
                         ("with_officer", X.assign(officer=meta["officer_id_hash"].astype(str)))]:
        m = M.make_gbm(frame).fit(frame[tr], y[tr].to_numpy())
        p = m.predict_proba(frame[te])[:, 1]
        out[label] = float(roc_auc_score(y[te].to_numpy(), p))
    out["delta"] = out["with_officer"] - out["baseline"]
    out["above_chance_uplift"] = out["delta"] / (out["baseline"] - 0.5)
    return out


# CONSENT ONLY -- the group's decision, and the right one: consent searches are
# the purely discretionary ones, so they are where officer judgement (and
# therefore bias) actually operates. The other strata are mechanical.
#
# This does NOT retire the pooling comparison. fairness.pooled_vs_stratified()
# reads the full searches frame and is what JUSTIFIES this scope: pooled shows
# no black-white disparity (+0.82pp) while consent-only shows -5.16pp. Keep that
# table in the deck as the reason the scope is defensible; it costs seconds and
# needs no model.
#
# Practical effect: drops the `pooled` stratum, whose 29,185 test rows were ~45
# minutes of TabPFN prediction on CPU.
def run(search_types=("consent",),
        modes=("matched", "full", "full_blind"), cache: bool = True) -> dict:
    df = D.load_searches()
    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    results = {}
    for st in search_types:
        for mode in modes:
            key = f"{st or 'pooled'}__{mode}".replace(" ", "_")
            try:
                scores, metrics, failed = fit_stratum(df, st, mode=mode)
            except Exception as exc:                      # noqa: BLE001
                print(f"  [skip] {key}: {exc}")
                continue
            for arm, why in failed.items():
                print(f"  [warn] {key}: arm {arm!r} failed -> {why}")
            if cache:
                scores.to_parquet(C.OUTPUTS / f"scores__{key}.parquet", index=False)
                (C.OUTPUTS / f"metrics__{key}.json").write_text(
                    json.dumps({"stratum": st or "pooled", "mode": mode,
                                "split_year": SPLIT_YEAR,
                                "matched_n": MATCHED_N,
                                "gbm_backend": M.gbm_backend(),
                                "tabpfn_backend": M.TABPFN_BACKEND,
                                "tabpfn_version": ("client" if M.TABPFN_BACKEND == "client"
                                                   else M.TABPFN_VERSION),
                                "failed_arms": failed,
                                "arms": [asdict(m) for m in metrics]}, indent=2)
                )
            results[key] = (scores, metrics)
            print(f"  [ok]   {key}: " + " | ".join(
                f"{m.arm} AUC={m.auc:.4f} PR={m.pr_auc:.4f}" for m in metrics))

    if cache:
        sig = officer_signal(df)
        (C.OUTPUTS / "officer_signal.json").write_text(json.dumps(sig, indent=2))
        print(f"  [ok]   officer signal: baseline={sig['baseline']:.4f} "
              f"+officer={sig['with_officer']:.4f} "
              f"(uplift {sig['above_chance_uplift']:.0%} of above-chance signal)")
    return results


def load_cached(stratum: str = "consent", mode: str = "matched"):
    key = f"{stratum}__{mode}".replace(" ", "_")
    sp = C.OUTPUTS / f"scores__{key}.parquet"
    mp = C.OUTPUTS / f"metrics__{key}.json"
    if not sp.exists():
        return None, None
    return pd.read_parquet(sp), json.loads(mp.read_text())


def available_runs() -> list[str]:
    return sorted(p.stem.replace("scores__", "") for p in C.OUTPUTS.glob("scores__*.parquet"))


if __name__ == "__main__":
    print(f"gbm backend: {M.gbm_backend()} | tabpfn: {M.tabpfn_available()}")
    run()
