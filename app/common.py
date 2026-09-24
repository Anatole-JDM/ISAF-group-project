"""Shared by the pages: repo root on sys.path (for `src`), the cached search sample, the map component."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from src import data as D  # noqa: E402

# Declared here because Streamlit can only declare components from an imported module, not a page script.
stops_map = components.declare_component("stops_map", path=str(Path(__file__).parent / "stops_map"))


@st.cache_data(show_spinner="Loading searches…")
def load() -> pd.DataFrame:
    return D.load_searches()


def searches() -> pd.DataFrame:
    """The labelled search sample, or a readable error if the data has not been fetched."""
    try:
        return load()
    except FileNotFoundError:
        st.error("Data not found. Run `python scripts/download_data.py` first.")
        st.stop()
