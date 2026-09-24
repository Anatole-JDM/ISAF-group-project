"""1 · The problem — selective labels, stated before any model."""
from __future__ import annotations

import streamlit as st

import common

st.header("Read this before any number")
st.markdown(
    """
**Contraband is only observed for drivers who were searched — and officers chose
that sample.** About 4% of stops end in a search. The model is therefore trained
on officer-selected stops, not on stops.

1. If officers search some groups on weaker evidence, that group's recorded hit
   rate is depressed **by the selection itself**, so a model can learn officer
   discretion rather than contraband risk.
2. Deployed on *all* stops, it would extrapolate far outside its training support.
    """
)
sig = common.load_officer_signal()
if sig:
    st.subheader("What is the model actually using? Measured, not asserted.")
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

st.page_link(
    "views/explore.py", icon="🗺️",
    label="Where and when do stops and searches happen? → 2 · Explore the stops",
)
