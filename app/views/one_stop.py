"""7 · One stop — every arm's score for a single test-set search."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import common

scores, meta = common.run_selector()

st.header("Explain one stop")
i = st.number_input("Row in the test set", 0, len(scores) - 1, 0, step=1)
row = scores.iloc[int(i)]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Race", str(row["subject_race"]))
c2.metric("Year", int(row["year"]))
c3.metric("Precinct", str(row["precinct"]))
c4.metric("Contraband found", "yes" if row["y"] == 1 else "no")
st.subheader("Score by arm")
st.dataframe(
    pd.DataFrame({"score": {common.arm_name(c): float(row[c]) for c in common.score_cols(scores)}})
    .style.format("{:.4f}"), width="stretch",
)
st.info(
    "TODO for the group: attach SHAP local explanations here. The scorecard "
    "arm can show signed coefficient contributions directly; the GBM needs "
    "`shap.TreeExplainer`; TabPFN needs `tabpfn-extensions`."
)
