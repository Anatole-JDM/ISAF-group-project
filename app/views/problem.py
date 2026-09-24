"""Problem definition — the decision, the objects, selective labels, and why we stratify by search type."""
from __future__ import annotations

import streamlit as st

import common
from src import fairness as F

st.header("Problem definition")
st.markdown(
    """
**Client:** a police department (or its oversight board) deciding which traffic stops warrant a
search, under a fixed search capacity. **Target:** `contraband_found`, observed only for the
stops where a search occurred.

| Object | Definition |
|---|---|
| `Y` | 1 if contraband is found: what the officer is trying to predict |
| `Ŷ` | 1 if the model recommends a search |
| `D` | protected attribute: driver race |
| Favourable for the individual | **not being searched** |

The course convention is that `Y = 1` is the *favourable* outcome for the individual. Here it is
the opposite (`Y = 1` is favourable to the police), so every fairness metric is reported under
both codings.
    """
)

st.subheader("Read this before any number: selective labels")
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

st.subheader("Why pooling hides the disparity")
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
    "modification, not Simpson's paradox. The models are therefore fitted and "
    "judged **per search type** (the Stratum picker on the model pages)."
)
