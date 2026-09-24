"""1 · The problem — selective labels, stated before any model."""
from __future__ import annotations

import streamlit as st

import common  # noqa: F401  (puts the repo root on sys.path for `src`)

df = common.searches()

st.header("Read this before any number")
st.markdown(
    """
**Contraband is only observed for drivers who were searched — and officers chose
that sample.** Roughly 4% of stops result in a search. The model is therefore
trained on officer-selected stops, not on stops.

Two consequences the client must understand:

1. If officers search some groups on weaker evidence, the recorded hit rate for
   those groups is depressed **by the selection itself**. A model fit to this
   data can learn officer discretion rather than contraband risk.
2. A model deployed on *all* stops would extrapolate far outside its training
   support.

This is why the recommendation rests on trustworthiness rather than AUC.
    """
)
c1, c2, c3 = st.columns(3)
c1.metric("Searches (labelled)", f"{len(df):,}")
c2.metric("Discretionary (consent)", f"{int(df['is_discretionary'].sum()):,}")
c3.metric("Unresolved search type", f"{int(df['search_type'].eq('unresolved').sum()):,}")
st.subheader("Search types")
st.dataframe(df["search_type"].value_counts().rename("searches"), width="stretch")

st.page_link(
    "views/explore.py", icon="🗺️",
    label="Where and when do stops and searches happen? → 2 · Explore the stops",
)
