"""3 · Pooling — the headline table."""
from __future__ import annotations

import streamlit as st

import common  # noqa: F401  (puts the repo root on sys.path for `src`)
from src import fairness as F

df = common.searches()

st.header("Why pooling hides the disparity")
st.markdown(
    "Consent searches are fully discretionary. Incident-to-arrest, warrant, "
    "inventory and plain-view searches are largely **mechanical** — contraband "
    "is likely because something already happened. Averaging a discretionary "
    "decision with mechanical ones does not average comparable things."
)
tbl = F.pooled_vs_stratified(df)
st.dataframe(
    tbl.style.format({"hit_rate": "{:.2%}", "gap_vs_ref_pp": "{:+.2f}"}),
    width="stretch",
)
st.info(
    "The gap runs in **opposite directions** in the two strata, so pooling "
    "cancels them and reports 'no disparity'. This is effect modification, "
    "not Simpson's paradox — the distinction is worth stating precisely."
)
