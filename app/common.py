"""Shared by the pages: repo root on sys.path (for `src`), cached data and model runs, the map component.

The app never fits models. The model pages read the artifacts `python -m src.train` writes
to outputs/, so every control is instant.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from src import config as C  # noqa: E402
from src import data as D  # noqa: E402

# Declared here because Streamlit can only declare components from an imported module, not a page script.
stops_map = components.declare_component("stops_map", path=str(Path(__file__).parent / "stops_map"))


# --------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading searches…")
def load_df() -> pd.DataFrame:
    return D.load_searches()


def searches() -> pd.DataFrame:
    """The labelled search sample, or a readable error if the data has not been fetched."""
    try:
        return load_df()
    except FileNotFoundError:
        st.error("Data not found. Run `python scripts/download_data.py` first.")
        st.stop()


# --------------------------------------------------------------------------- model runs
# src.train imports scikit-learn, so it is imported only by the pages that read model runs.
@st.cache_data
def load_run(stratum: str, mode: str):
    from src import train as T
    return T.load_cached(stratum, mode)


@st.cache_data
def load_officer_signal():
    p = C.OUTPUTS / "officer_signal.json"
    return json.loads(p.read_text()) if p.exists() else None


def score_cols(scores: pd.DataFrame) -> list[str]:
    return [c for c in scores.columns if c.startswith("score_")]


def arm_name(col: str) -> str:
    return col.replace("score_", "")


WITHIN_OFFICER_MODELS = {
    "full_time/xgboost": "XGBoost, stop & driver only",
    "full_time_officer/pltr_sparse": "White box (PLTR, 19 rules) + officer record",
    "full_time_officer/xgboost": "XGBoost + officer record",
    "full_time_officer_blind/pltr_sparse": "White box (PLTR), race-blind + officer record",
    "full_time_officer_blind/xgboost": "XGBoost, race-blind + officer record",
}


@st.cache_data
def load_diag():
    p = ROOT / "data" / "officer_model_diagnostics.json"
    return json.loads(p.read_text()) if p.exists() else None


def within_officer_chart(f: dict) -> None:
    """Slide 13: pooled AUC vs AUC inside each officer's own test searches (officer-resampled CI)."""
    import plotly.graph_objects as go

    rows = [{"model": lab, "pooled": f[k]["pooled_auc"], "within": f[k]["within_officer_auc"],
             "lo": f[k]["ci95_resampling_officers"][0], "hi": f[k]["ci95_resampling_officers"][1],
             "between": f[k]["share_of_score_variance_between_officers"]}
            for k, lab in WITHIN_OFFICER_MODELS.items() if k in f]
    t = pd.DataFrame(rows).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t["pooled"], y=t["model"], mode="markers+text", name="all officers pooled",
                             marker=dict(symbol="diamond", size=12, color="#b0b7bf"),
                             text=[f"{v:.3f}" for v in t["pooled"]], textposition="top center"))
    fig.add_trace(go.Scatter(x=t["within"], y=t["model"], mode="markers+text",
                             name="inside each officer's own searches (95% CI)",
                             marker=dict(size=12, color="#1f77d0"),
                             error_x=dict(type="data", symmetric=False, array=t["hi"] - t["within"],
                                          arrayminus=t["within"] - t["lo"]),
                             text=[f"{v:.3f}" for v in t["within"]], textposition="bottom center"))
    fig.add_vline(x=0.5, line_dash="dot", line_color="#8a8984", annotation_text="chance")
    fig.update_layout(xaxis_title="AUC, test 2016–2018", legend=dict(orientation="h", y=-0.2),
                      height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    n = next(iter(f.values()))
    between = [v["share_of_score_variance_between_officers"] for k, v in f.items() if "officer" in k]
    st.caption(
        f"Within-officer AUC: computed inside each officer's own test searches and averaged, weighted by "
        f"searches ({n['officers']} officers with enough searches and both outcomes, {n['searches']:,} "
        f"searches); CI from resampling officers. With the officer record, {min(between):.0%}–"
        f"{max(between):.0%} of the score variance lies between officers: the gain identifies which "
        "officer, not which driver."
    )


# Colors are fixed per entity so a filter never repaints the survivors (the map uses the same ones).
RACE_COLORS = {
    "black": "#2a78d6",
    "white": "#eb6834",
    "hispanic": "#1baf7a",
    "asian/pacific islander": "#eda100",
    "other": "#e87ba4",
    "unknown": "#4a3aa7",
}


# Per-column, because `n` is a COUNT and mean_score can be NaN: a blanket
# "{:.2%}" rendered group sizes as 494700.00% and NaN as "nan%".
RATE_FMT = {
    "n": "{:,.0f}",
    "base_rate": "{:.2%}", "selection_rate": "{:.2%}",
    "TPR": "{:.2%}", "FPR": "{:.2%}", "PPV": "{:.2%}", "NPV": "{:.2%}",
    "mean_score": "{:.4f}",
}


STRATA = ["consent", "probable_cause", "pooled"]
MODES = ["matched", "full", "full_blind"]


@st.cache_data
def modes_with_arm(stratum: str, arm: str) -> list[str]:
    """The training modes whose cached run for this stratum actually fitted `arm`."""
    out = []
    for m in MODES:
        _, meta = load_run(stratum, m)
        if meta and any(a["arm"] == arm for a in meta.get("arms", [])):
            out.append(m)
    return out


def run_selector(require_arm: str | None = None) -> tuple[pd.DataFrame, dict]:
    """The sidebar 'Run' picker shared by the model pages. Returns (scores, meta), or stops the page.

    `require_arm` opens the page on a mode that actually contains its arm. TabPFN is fitted only
    in `matched`, so on the default `full` the foundation page used to greet a first-time visitor
    with "this arm is not in the selected run" — an empty page for the one model family the brief
    asks us to compare. The steer reads what is cached rather than hardcoding a mode, fires once
    per arm per session, and only when the current mode genuinely lacks the arm, so it never
    overrides a mode the visitor chose.
    """
    from src import train as T

    if not T.available_runs():
        st.error("No cached model runs. Run `python -m src.train` first.")
        st.stop()
    # Defaults, and re-assigning them keeps the choice alive when switching between model pages.
    st.session_state.setdefault("run_stratum", "consent")
    st.session_state.setdefault("run_mode", "full")
    for key in ("run_stratum", "run_mode"):
        st.session_state[key] = st.session_state[key]

    # Assigning a widget's key BEFORE the widget is created sets its default; after, it raises.
    if require_arm is not None:
        steered = st.session_state.setdefault("_steered_arms", set())
        if require_arm not in steered:
            steered.add(require_arm)
            fitted_in = modes_with_arm(st.session_state["run_stratum"], require_arm)
            if fitted_in and st.session_state["run_mode"] not in fitted_in:
                st.session_state["run_mode"] = fitted_in[0]

    with st.sidebar:
        st.header("Run")
        stratum = st.selectbox("Stratum", STRATA, key="run_stratum")
        mode = st.radio(
            "Training mode", MODES, key="run_mode",
            help="matched = every arm, TabPFN included, trains on the same subsample, the "
                 "only fair three-way comparison (the slides' Act I numbers). full = all "
                 "rows, scorecard and XGBoost only. full_blind = all rows with race removed "
                 "from the features.",
        )
        scores, meta = load_run(stratum, mode)
        if scores is None:
            st.error(f"No cached run for {stratum}/{mode}.")
            st.stop()
        n_train = sorted({a["n_train"] for a in meta.get("arms", [])})
        st.caption(f"train n = {', '.join(f'{n:,}' for n in n_train)} · test n = {len(scores):,} · "
                   f"split at {meta['split_year']}")
        st.caption(f"GBM backend: `{meta['gbm_backend']}`")
    return scores, meta
