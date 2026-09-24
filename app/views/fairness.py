"""5 · Fairness — the course taxonomy, both Y codings."""
from __future__ import annotations

import streamlit as st

import common  # noqa: F401  (puts the repo root on sys.path for `src`)
from src import config as C

st.header("Fairness")
st.markdown(
    f"""
**Object definitions.** `Y = 1` contraband found · `Ŷ = 1` model recommends a
search · `D` driver race. The favourable outcome for the individual is
**{C.FAVORABLE_FOR_INDIVIDUAL.replace('_', ' ')}** — the opposite of the course's
convention that `Y = 1` is favourable (slide 238), so metrics are reported under
both codings.

- **False positive rate by race** — innocent drivers the model would search.
  *Predictive equality* (separation). This is the harm that matters.
- **Hit rate (PPV) by race** — the *outcome test* from the economics of
  discrimination. *Sufficiency*.
    """
)
st.warning("TODO: wire fitted predictions into F.group_rates / F.both_codings.")
