"""Fairness testing — the course taxonomy for each model, both Y codings."""
from __future__ import annotations

import numpy as np
import streamlit as st

import common
from model_page import ARMS
from src import config as C
from src import fairness as F
import pandas as pd
import plotly.express as px

scores, meta = common.run_selector()

st.header("Fairness testing")
st.markdown(
    f"""
`Y = 1` contraband found · `Ŷ = 1` model recommends a search · `D` driver race.
Favourable for the individual is **{C.FAVORABLE_FOR_INDIVIDUAL.replace('_', ' ')}**
— the opposite of the course convention that `Y = 1` is favourable, so metrics
are shown under both codings.
    """
)
k = st.slider("Search capacity K (top-K by score), the same for every model", 100,
              len(scores), min(2000, len(scores)), step=100, key="fair_k")
st.error(
    "**FPR gap is the harm metric.** It is the share of *innocent* drivers "
    "the model would search, by race. Read it before AUC."
)


def fairness_report(arm: str) -> None:
    s = scores[f"score_{arm}"].to_numpy()
    yhat = np.zeros(len(s), dtype=int)
    yhat[np.argsort(-s)[:k]] = 1
    sub = scores["subject_race"].isin(C.RACE_REPORTABLE)

    rates = F.group_rates(scores["y"][sub], yhat[sub.to_numpy()],
                          scores["subject_race"][sub], y_score=s[sub.to_numpy()])
    st.subheader("Per-group rates")
    st.dataframe(rates.style.format(common.RATE_FMT, na_rep="—"), width="stretch")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Independence** — statistical parity")
        st.dataframe(F.independence(rates).to_frame("gap").style.format("{:+.2%}"),
                     width="stretch")
    with c2:
        st.markdown("**Separation** — equal opportunity / predictive equality")
        st.dataframe(F.separation(rates).style.format("{:+.2%}"), width="stretch")
    with c3:
        st.markdown("**Sufficiency** — predictive parity (outcome test)")
        st.dataframe(F.sufficiency(rates).to_frame("gap").style.format("{:+.2%}"),
                     width="stretch")

    with st.expander("Both Y codings — the convention inversion"):
        both = F.both_codings(scores["y"][sub], yhat[sub.to_numpy()],
                              scores["subject_race"][sub])
        cc1, cc2 = st.columns(2)
        cc1.markdown(f"`{C.Y_NATIVE}` — Y=1 contraband found")
        cc1.dataframe(both[C.Y_NATIVE].style.format(common.RATE_FMT, na_rep="—"))
        cc2.markdown(f"`{C.Y_COURSE}` — Y=1 innocent (course convention)")
        cc2.dataframe(both[C.Y_COURSE].style.format(common.RATE_FMT, na_rep="—"))


arms = [common.arm_name(c) for c in common.score_cols(scores)]
for arm, tab in zip(arms, st.tabs([ARMS.get(a, a) for a in arms])):
    with tab:
        fairness_report(arm)
if "tabpfn" not in arms:
    st.caption("No tab for the tabular foundation model (TabPFN): it is fitted only in the `matched` "
               "training mode, where the `tabpfn` package is installed. See Tabular foundation models.")


# --------------------------------------------------------------------------- hypothesis tests
st.divider()
st.subheader("Is the disparity statistically significant? χ² and equivalence testing")

_arms = [common.arm_name(c) for c in common.score_cols(scores)]
_arm = st.selectbox("Arm to test", _arms, key="fair_arm")
_K = int(k)   # the slider above; this section used to read a key that was never set
_s = scores[f"score_{_arm}"].to_numpy()
_yhat = np.zeros(len(_s), int)
_yhat[np.argsort(-_s)[: min(_K, len(_s))]] = 1

_c1, _c2 = st.columns(2)
with _c1:
    st.markdown("**χ² test of independence** — H₀: selection is independent of race")
    r = F.chi2_parity_test(_yhat, scores["subject_race"])
    st.metric("χ²", f"{r['chi2']:.1f}", help=f"dof {r['dof']}, n = {r['n']:,}")
    st.metric("p-value", f"{r['p_value']:.2e}")
    st.caption(
        f"White {r['selection_rate']['white']:.1%} vs black "
        f"{r['selection_rate']['black']:.1%}, gap {r['gap_pp']:+.1f}pp. "
        "Statistical parity is rejected outright."
    )
with _c2:
    st.markdown("**TOST equivalence** — how much unfairness would you have to accept?")
    sw = F.tost_delta_sweep(_yhat, scores["subject_race"])
    ok = sw[sw["equivalent"]]
    d_star = float(ok["delta"].iloc[0]) if len(ok) else float("nan")
    st.metric("Smallest δ certifying fairness", f"{d_star:.2f}" if d_star == d_star else "never")
    st.caption(
        "χ² on large n rejects almost anything, so rejection alone is weak. Equivalence "
        "testing inverts the burden: how wide a tolerance δ would you need before this "
        "model passes? The deck never assigns δ a value, so we sweep it."
    )

fig = px.line(sw, x="delta", y=["z_lower", "z_upper"],
              labels={"value": "z", "delta": "equivalence margin δ"})
fig.add_hline(y=float(sw["z_crit"].iloc[0]), line_dash="dot", line_color="#8a8984",
              annotation_text="critical value")
if d_star == d_star:
    fig.add_vline(x=d_star, line_color="#c0392b",
                  annotation_text=f"certifies at δ={d_star:.2f}")
st.plotly_chart(fig, width="stretch")
st.error(
    f"**You would have to declare a {d_star:.0%} selection gap acceptable before this model "
    "certifies as fair.** That is the magnitude of the unfairness, not merely its "
    "significance." if d_star == d_star else
    "The model does not certify as equivalent at any δ up to 0.50."
)
