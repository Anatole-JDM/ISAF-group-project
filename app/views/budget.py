"""6 · Budget — top-K economics and the equity/efficiency frontier."""
from __future__ import annotations

import numpy as np
import streamlit as st

import common
from src import config as C
from src import economics as E

scores, meta = common.run_selector()

st.header("Budget-constrained deployment")
arm5 = st.selectbox("Arm ", [common.arm_name(c) for c in common.score_cols(scores)], key="arm5")
cost_fp = st.slider(
    "Cost of searching an innocent driver (relative to one justified search)",
    0.25, 8.0, 1.0, step=0.25,
)
st.caption(
    "This number is **not in the data**. It is a policy judgement about what "
    "an unjustified search costs a person. As it rises the optimal policy "
    "searches fewer people and the fairness gaps shrink — that is the "
    "trustworthy-AI argument, quantified."
)
s5 = scores[f"score_{arm5}"].to_numpy()
y5 = scores["y"].to_numpy()
curve = E.top_k_curve(y5, s5, cost_fp=cost_fp)
st.line_chart(curve.set_index("k")[["net_benefit"]])
st.line_chart(curve.set_index("k")[["precision_at_k"]])

st.subheader("Sensitivity — where deployment stops paying")
kk = st.slider("K for sensitivity", 100, len(scores),
               min(2000, len(scores)), step=100, key="k5")
st.dataframe(E.sensitivity(y5, s5, kk).style.format({
    "precision_at_k": "{:.2%}", "recall_at_k": "{:.2%}",
    "net_benefit": "{:,.0f}"}), width="stretch")

st.subheader("Equity / efficiency frontier")
st.caption(
    "Top-K is ranked over the **whole** test set, the same definition the "
    "Fairness page uses, so the two agree for the same K. Gaps are reported "
    "only for groups with enough volume to be stable."
)
ks = np.unique(np.linspace(200, len(scores), 25).astype(int))
fr = E.equity_efficiency_frontier(y5, s5, scores["subject_race"], ks,
                                  report_groups=C.RACE_REPORTABLE,
                                  cost_fp=cost_fp)
if not fr.empty:
    st.scatter_chart(fr, x="max_FPR_gap", y="net_benefit")
    st.caption("Each point is a search capacity K. Up is more benefit; "
               "left is more equal false-positive rates.")
