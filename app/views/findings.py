"""Findings and conclusive thoughts — what the models use, and the recommendation to the client."""
from __future__ import annotations

import streamlit as st

import common
from src import fairness as F

st.header("Findings and conclusive thoughts")

st.subheader("1. Pooling hides a disparity in discretionary searches")
tbl = F.pooled_vs_stratified(common.searches())
gaps = tbl.pivot(index="stratum", columns="subject_race", values="gap_vs_ref_pp").drop(columns="white")
st.dataframe(gaps.style.format("{:+.2f} pp"), width="stretch")
st.caption("Hit-rate gap vs white drivers, in percentage points. Opposite signs in the two strata "
           "cancel when pooled: see Problem definition.")

st.subheader("2. What is the model actually using? Measured, not asserted.")
sig = common.load_officer_signal()
if not sig:
    st.info("Run `python -m src.train` to measure it.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Pre-search features only", f"{sig['baseline']:.4f}", help="AUC")
    c2.metric("+ officer identity", f"{sig['with_officer']:.4f}",
              delta=f"{sig['delta']:+.4f}")
    c3.metric("Officer share of above-chance signal",
              f"{sig['above_chance_uplift']:.0%}")

    full_m = common.load_run("consent", "full")[1]
    blind_m = common.load_run("consent", "full_blind")[1]
    if full_m and blind_m:
        a_full = max(a["auc"] for a in full_m["arms"])
        a_blind = max(a["auc"] for a in blind_m["arms"])
        race_share = (a_full - a_blind) / (a_full - 0.5)
        d1, d2, d3 = st.columns(3)
        d1.metric("With race", f"{a_full:.4f}")
        d2.metric("Race removed", f"{a_blind:.4f}", delta=f"{a_blind - a_full:+.4f}")
        d3.metric("Race share of above-chance signal", f"{race_share:.0%}")
        st.error(
            f"**This is the recommendation.** Baseline AUC is only "
            f"{a_full - 0.5:.4f} above chance. Of that sliver, "
            f"**{sig['above_chance_uplift']:.0%} is officer identity** and "
            f"**{race_share:.0%} is driver race**. The model has almost no "
            "signal, and what little it has comes from who stopped you and "
            "what you look like — not from anything about the situation. "
            "Removing race does not fix it: precinct and zone are strong "
            "proxies in a segregated city."
        )

st.subheader("3. The selection inverts")
st.markdown(
    "A contraband-optimising model searches *white* drivers far more, because white drivers "
    "have the higher hit rate in discretionary searches — the reverse of actual officer "
    "behaviour. That is the outcome test expressed as a model. See **Fairness testing** at "
    "your chosen search capacity K."
)

st.subheader("Recommendation to the client: do not deploy")
st.markdown(
    "Not because it is unfair *or* because it is inaccurate, but because all four dimensions "
    "point the same way — near-random discrimination, signal dominated by officer identity and "
    "race, large false-positive disparities, and negative net benefit under any defensible cost "
    "of searching an innocent driver."
)
