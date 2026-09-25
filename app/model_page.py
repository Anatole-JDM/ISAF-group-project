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


def render(arm: str, title: str, description: str, missing_hint: str, explain_todo: str) -> None:
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

    scores, meta = common.run_selector()
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
    st.info(explain_todo)
