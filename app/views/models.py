"""4 · Models — the three arms compared."""
from __future__ import annotations

import streamlit as st


st.header("Model comparison")
stratum = st.selectbox(
    "Stratum", ["consent", "probable cause", "plain view", "arrest", None],
    format_func=lambda s: "ALL (pooled — for contrast only)" if s is None else s,
)
st.caption(
    "Fit the three arms on this stratum and compare. Use the SAME TabPFN "
    "subsample for every arm, or the comparison confounds model class with "
    "training-set size."
)
st.code(
    "from src import data, models\n"
    f"X, y, meta = data.build_xy(df, search_type={stratum!r})\n"
    "arms = models.build_all(X)\n"
    "# fit each, then stability.model_distance({name: scores})",
    language="python",
)
st.warning("TODO: fit models, cache scores, render AUC / PR-AUC / calibration here.")
