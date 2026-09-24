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


def run_selector() -> tuple[pd.DataFrame, dict]:
    """The sidebar 'Run' picker shared by the model pages. Returns (scores, meta), or stops the page."""
    from src import train as T

    if not T.available_runs():
        st.error("No cached model runs. Run `python -m src.train` first.")
        st.stop()
    # Defaults, and re-assigning them keeps the choice alive when switching between model pages.
    st.session_state.setdefault("run_stratum", "consent")
    st.session_state.setdefault("run_mode", "full")
    for key in ("run_stratum", "run_mode"):
        st.session_state[key] = st.session_state[key]

    with st.sidebar:
        st.header("Run")
        stratum = st.selectbox("Stratum", ["consent", "probable_cause", "pooled"], key="run_stratum")
        mode = st.radio(
            "Training mode", ["matched", "full", "full_blind"], key="run_mode",
            help="matched = every arm trains on the same 5,000 rows (the TabPFN "
                 "ceiling), the only fair three-way comparison. full = all rows. "
                 "full_blind = all rows with race removed from the features.",
        )
        scores, meta = load_run(stratum, mode)
        if scores is None:
            st.error(f"No cached run for {stratum}/{mode}.")
            st.stop()
        st.caption(f"test n = {len(scores):,} · split at {meta['split_year']}")
        st.caption(f"GBM backend: `{meta['gbm_backend']}`")
    return scores, meta
