"""5 · Fairness — the course taxonomy, both Y codings."""
from __future__ import annotations

import numpy as np
import streamlit as st

import common
from src import config as C
from src import fairness as F

scores, meta = common.run_selector()

st.header("Fairness")
st.markdown(
    f"""
`Y = 1` contraband found · `Ŷ = 1` model recommends a search · `D` driver race.
Favourable for the individual is **{C.FAVORABLE_FOR_INDIVIDUAL.replace('_', ' ')}**
— the opposite of the course convention that `Y = 1` is favourable, so metrics
are shown under both codings.
    """
)
arm = st.selectbox("Arm", [common.arm_name(c) for c in common.score_cols(scores)])
k = st.slider("Search capacity K (top-K by score)", 100,
              len(scores), min(2000, len(scores)), step=100)

s = scores[f"score_{arm}"].to_numpy()
yhat = np.zeros(len(s), dtype=int)
yhat[np.argsort(-s)[:k]] = 1
sub = scores["subject_race"].isin(C.RACE_REPORTABLE)

rates = F.group_rates(scores["y"][sub], yhat[sub.to_numpy()],
                      scores["subject_race"][sub], y_score=s[sub.to_numpy()])
st.subheader("Per-group rates")
st.dataframe(
    rates.style.format(common.RATE_FMT, na_rep="—"),
    width="stretch",
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Independence** — statistical parity")
    st.dataframe(independence := F.independence(rates).to_frame("gap")
                 .style.format("{:+.2%}"), width="stretch")
with c2:
    st.markdown("**Separation** — equal opportunity / predictive equality")
    st.dataframe(F.separation(rates).style.format("{:+.2%}"),
                 width="stretch")
with c3:
    st.markdown("**Sufficiency** — predictive parity (outcome test)")
    st.dataframe(F.sufficiency(rates).to_frame("gap").style.format("{:+.2%}"),
                 width="stretch")
st.error(
    "**FPR gap is the harm metric.** It is the share of *innocent* drivers "
    "the model would search, by race. Read it before AUC."
)

with st.expander("Both Y codings — the convention inversion"):
    both = F.both_codings(scores["y"][sub], yhat[sub.to_numpy()],
                          scores["subject_race"][sub])
    cc1, cc2 = st.columns(2)
    cc1.markdown(f"`{C.Y_NATIVE}` — Y=1 contraband found")
    cc1.dataframe(both[C.Y_NATIVE].style.format(common.RATE_FMT, na_rep="—"))
    cc2.markdown(f"`{C.Y_COURSE}` — Y=1 innocent (course convention)")
    cc2.dataframe(both[C.Y_COURSE].style.format(common.RATE_FMT, na_rep="—"))
