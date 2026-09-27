"""Shared layout of the three model-family pages: white box, black box, tabular foundation.

Each page shows the family's arm (as named in src/models.py and the outputs/ artifacts):
what it is, its metrics in every cached run, and, for the run picked in the sidebar,
its score distribution and calibration by race.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import common
from src import config as C
from src import fairness as F

# arm name in outputs/ -> label used across the app
ARMS = {
    "scorecard": "White box · scorecard",
    "gbm": "Black box · gradient boosting",
    "tabpfn": "Tabular foundation · TabPFN",
}


MIN_BIN = 50  # calibration bins with fewer searches are too noisy to plot


def _concerns(meta_key: str, arm: str) -> bool:
    """Run settings recorded for one arm only (gbm_backend, tabpfn_*) are shown on that arm's page only."""
    if meta_key.startswith("gbm"):
        return arm == "gbm"
    if meta_key.startswith("tabpfn"):
        return arm == "tabpfn"
    return True


def runs_for_arm(arm: str) -> pd.DataFrame:
    """One row per cached run (stratum × training mode) that contains this arm."""
    from src import train as T

    rows = []
    for key in T.available_runs():
        stratum, mode = key.split("__", 1)
        _, meta = common.load_run(stratum, mode)
        for a in (meta or {}).get("arms", []):
            if a["arm"] == arm:
                extra = {k: v for k, v in meta.items() if k not in ("arms", "stratum", "mode", "split_year")
                         and _concerns(k, arm)}
                rows.append({"stratum": stratum, "mode": mode, **{k: a[k] for k in
                             ("n_train", "n_test", "auc", "pr_auc", "brier")}, **extra})
    return pd.DataFrame(rows)


def _local_explanation(arm: str, scores: pd.DataFrame) -> None:
    """Why did the model score THIS stop the way it did?

    The three arms support three different grades of explanation, and saying which
    is which is the point: exact closed form for the linear arm, exact TreeSHAP for
    the trees, and approximate occlusion for TabPFN because it has no native
    attribution at all.
    """
    import numpy as np
    from src import data as D
    from src import explain as E
    from src import models as M

    st.subheader("Why this stop? — local explanation")
    grade = {
        "scorecard": ("EXACT — Shapley in closed form", "success",
                      "For a linear model the Shapley value of feature j is "
                      "`coef_j x (x_j - E[x_j])`. No sampling, no approximation."),
        "gbm": ("EXACT — TreeSHAP", "success",
                "xgboost computes exact Shapley values for trees natively via "
                "`predict(pred_contribs=True)`.") if M.gbm_backend() == "xgboost" else
               ("APPROXIMATE — occlusion (xgboost not installed)", "warning",
                "Without xgboost this arm falls back to scikit-learn's HistGradientBoosting, "
                "which has no native Shapley. The attribution below is occlusion, not TreeSHAP."),
        "tabpfn": ("APPROXIMATE — occlusion, NOT Shapley", "warning",
                   "TabPFN exposes no attribution of its own. We replace each feature "
                   "with its background value and measure the change. This ignores "
                   "interactions and its error cannot be bounded — which is a real "
                   "cost of the black box, not a detail."),
    }[arm]
    (st.success if grade[1] == "success" else st.warning)(f"**{grade[0]}** — {grade[2]}")

    i = st.number_input("Test-set row", 0, len(scores) - 1, 0, step=1, key=f"exp_{arm}")
    row_meta = scores.iloc[int(i)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Race", str(row_meta["subject_race"]))
    c2.metric("Contraband found", "yes" if row_meta["y"] == 1 else "no")
    c3.metric(f"{arm} score", f"{row_meta.get('score_' + arm, float('nan')):.4f}")

    # The cached scores come from outputs/; attributing a row refits the arm here, so
    # this is the one place the app needs the modelling package rather than a parquet.
    if arm == "tabpfn" and not M.tabpfn_available():
        st.info("Per-stop attribution for TabPFN needs the `tabpfn` package and a token, "
                "which the deployed app does not carry. The scores and calibration above are "
                "the real cached results; run the app locally to attribute an individual stop.")
        return
    if not st.button("Explain this stop", key=f"btn_{arm}"):
        st.caption("Fitting the arm and attributing one prediction takes a few seconds.")
        return
    with st.spinner("Fitting and attributing…"):
        stratum = st.session_state.get("run_stratum", "consent")
        df = D.load_searches()
        X, y, meta = D.build_xy(df, search_type=None if stratum == "pooled" else stratum)
        X = X.reset_index(drop=True); y = np.asarray(y); meta = meta.reset_index(drop=True)
        tr = (meta["year"] < 2016).to_numpy()
        idx = np.random.default_rng(42).choice(int(tr.sum()), 2000, replace=False)
        Xtr, ytr = X[tr].iloc[idx], y[tr][idx]
        model = {"scorecard": M.make_scorecard, "gbm": M.make_gbm}.get(arm)
        model = M.TabPFNArm(max_train=2000).fit(Xtr, ytr) if model is None else model(Xtr).fit(Xtr, ytr)
        res = E.explain_row(arm, model, Xtr, X[~tr].iloc[[int(i)]])
    c = res["contributions"].rename("contribution").to_frame().reset_index(names="feature")
    fig = px.bar(c, x="contribution", y="feature", orientation="h",
                 color=c["contribution"] > 0,
                 color_discrete_map={True: "#c0392b", False: "#2c7fb8"})
    fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"},
                      xaxis_title=f"contribution ({res['units']})")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Method: {res['method']}. One-hot columns are summed back to their source "
               "feature — valid because Shapley values are additive. Red pushes the score up.")


def render(arm: str, title: str, description: str, missing_hint: str, explain_todo: str = "") -> None:
    st.header(title)
    st.markdown(description)

    st.subheader("Results in every cached run")
    runs = runs_for_arm(arm)
    if runs.empty:
        st.info(missing_hint)
    else:
        st.dataframe(runs.style.format({"n_train": "{:,}", "n_test": "{:,}", "auc": "{:.4f}",
                                        "pr_auc": "{:.4f}", "brier": "{:.4f}"}),
                     hide_index=True, width="stretch")
        st.caption("Train < 2016, test ≥ 2016 (the regime split). `full_blind` removes race from the "
                   "features; the gap to `full` is what race contributes.")

    scores, meta = common.run_selector(require_arm=arm)
    col = f"score_{arm}"
    st.subheader(f"Selected run: {st.session_state['run_stratum']} / {st.session_state['run_mode']}")
    if col not in scores:
        st.info(f"This arm is not in the selected run. {missing_hint}")
    else:
        sub = scores[scores["subject_race"].isin(C.RACE_REPORTABLE)]
        tab_dist, tab_cal = st.tabs(["Scores by race", "Calibration by race"])
        with tab_dist:
            fig = px.histogram(sub, x=col, color="subject_race", histnorm="probability", nbins=40,
                               barmode="overlay", opacity=0.55, color_discrete_map=common.RACE_COLORS,
                               labels={col: "Predicted probability of contraband", "subject_race": "Race"})
            fig.update_yaxes(title="Share of the group's test searches", tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
            st.caption("If the distributions differ, a top-K policy on this score selects the groups at "
                       "different rates (independence).")
        with tab_cal:
            cal = F.calibration_by_group(sub["y"], sub[col], sub["subject_race"])
            dropped = int((cal["n"] < MIN_BIN).sum())
            cal = cal[cal["n"] >= MIN_BIN]
            fig = px.line(cal, x="predicted", y="observed", color="g", markers=True,
                          color_discrete_map=common.RACE_COLORS, hover_data={"n": ":,"},
                          labels={"predicted": "Mean predicted probability (decile)",
                                  "observed": "Observed hit rate", "g": "Race"})
            lim = float(max(cal["predicted"].max(), cal["observed"].max()))
            fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="perfect calibration",
                                     line=dict(color="#8a8984", dash="dot", width=1)))
            fig.update_xaxes(tickformat=".0%")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
            st.caption("Sufficiency in its continuous form: at the same score, is contraband found at the "
                       "same rate for every group? Lines off the diagonal are miscalibrated. "
                       f"Score deciles are cut on all groups together; {dropped} group-decile cells with "
                       f"fewer than {MIN_BIN} searches are left out as too noisy.")
        _local_explanation(arm, scores)
