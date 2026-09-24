"""3 · Pooling — the headline table."""
from __future__ import annotations

import streamlit as st

import common
from src import fairness as F

st.header("Why pooling hides the disparity")
st.markdown(
    "Consent searches are fully discretionary. Incident-to-arrest, warrant, "
    "inventory and plain-view searches are largely **mechanical** — contraband "
    "is likely because something already happened."
)
tbl = F.pooled_vs_stratified(common.searches())
piv = tbl.pivot(index="stratum", columns="subject_race", values="hit_rate")
st.dataframe(piv.style.format("{:.2%}"), width="stretch")
st.dataframe(
    tbl.style.format({"hit_rate": "{:.2%}", "gap_vs_ref_pp": "{:+.2f}"}),
    width="stretch",
)
st.info(
    "The Black–white gap runs in **opposite directions** across strata, so "
    "pooling cancels them and reports 'no disparity'. This is effect "
    "modification, not Simpson's paradox."
)
