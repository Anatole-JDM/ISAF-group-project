"""6 · Budget — top-K economics and the equity/efficiency frontier."""
from __future__ import annotations

import streamlit as st


st.header("Budget-constrained deployment")
st.markdown(
    "The client cannot search everyone. Given capacity for **K** searches, "
    "which stops? The operating metric is precision@K, not accuracy at 0.5."
)
k = st.slider("Search capacity K", 100, 20_000, 5_000, step=100)
cost_fp = st.slider(
    "Cost of searching an innocent driver (relative to one justified search)",
    0.25, 8.0, 1.0, step=0.25,
)
st.caption(
    "This number is **not in the data**. It is a policy judgement about what "
    "an unjustified search costs a person. As it rises, the optimal policy "
    "searches fewer people and the fairness gaps shrink — which is the "
    "trustworthy-AI argument, quantified."
)
st.metric("Assumed FP cost", f"{cost_fp:.2f}×")
st.warning("TODO: E.top_k_curve + E.equity_efficiency_frontier on fitted scores.")
