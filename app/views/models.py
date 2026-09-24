"""4 · Models — the arms compared."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import common

scores, meta = common.run_selector()

st.header("Model comparison")
m = pd.DataFrame(meta["arms"]).set_index("arm")
st.dataframe(
    m[["n_train", "n_test", "base_rate_train", "base_rate_test",
       "auc", "pr_auc", "brier"]].style.format({
           "base_rate_train": "{:.2%}", "base_rate_test": "{:.2%}",
           "auc": "{:.4f}", "pr_auc": "{:.4f}", "brier": "{:.4f}"}),
    width="stretch",
)
best = m["auc"].max()
if best < 0.65:
    st.error(
        f"**Best AUC is {best:.3f}.** Contraband presence is close to "
        "unpredictable from pre-search observables. This is the finding, not "
        "a bug — and it is the basis of the recommendation. A near-random "
        "score that also carries measurable racial disparity should not be "
        "deployed, whatever its interpretability properties."
    )
st.subheader("Do the models rank people the same way?")
from src import stability as S
st.caption(
    "1 − Spearman correlation between score vectors. Two models with equal "
    "AUC that rank differently are not interchangeable under a top-K policy."
)
st.dataframe(
    S.model_distance({common.arm_name(c): scores[c] for c in common.score_cols(scores)})
     .style.format("{:.4f}"),
    width="stretch",
)
st.caption(f"Base-rate drift across the split: "
           f"{m['base_rate_train'].iloc[0]:.1%} train → "
           f"{m['base_rate_test'].iloc[0]:.1%} test.")
