"""Client-facing app: the 'interactive application' deliverable.

    streamlit run app/streamlit_app.py

Five tabs, in the order the argument should be made to a client:
    1. The Problem      - selective labels, stated before any model
    2. Pooling          - the headline table
    3. Models           - the three arms compared
    4. Fairness         - the course taxonomy, both Y codings
    5. Budget           - top-K economics and the equity/efficiency frontier
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C          # noqa: E402
from src import data as D            # noqa: E402
from src import economics as E       # noqa: E402
from src import fairness as F        # noqa: E402

st.set_page_config(page_title="Search Decision Support", layout="wide")


@st.cache_data(show_spinner="Loading searches…")
def load() -> pd.DataFrame:
    return D.load_searches()


st.title("Traffic-stop search decision support")
st.caption(
    "Nashville MPD, 2010–2018 · Stanford Open Policing Project · "
    "Pierson et al., *Nature Human Behaviour* 4 (2020)"
)

try:
    df = load()
except FileNotFoundError:
    st.error("Data not found. Run `python scripts/download_data.py` first.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1 · The problem", "2 · Pooling", "3 · Models", "4 · Fairness", "5 · Budget"]
)

# --------------------------------------------------------------------------- 1
with tab1:
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
    st.dataframe(df["search_type"].value_counts().rename("searches"), use_container_width=True)

# --------------------------------------------------------------------------- 2
with tab2:
    st.header("Why pooling hides the disparity")
    st.markdown(
        "Consent searches are fully discretionary. Incident-to-arrest, warrant, "
        "inventory and plain-view searches are largely **mechanical** — contraband "
        "is likely because something already happened. Averaging a discretionary "
        "decision with mechanical ones does not average comparable things."
    )
    tbl = F.pooled_vs_stratified(df)
    st.dataframe(
        tbl.style.format({"hit_rate": "{:.2%}", "gap_vs_ref_pp": "{:+.2f}"}),
        use_container_width=True,
    )
    st.info(
        "The gap runs in **opposite directions** in the two strata, so pooling "
        "cancels them and reports 'no disparity'. This is effect modification, "
        "not Simpson's paradox — the distinction is worth stating precisely."
    )

# --------------------------------------------------------------------------- 3
with tab3:
    st.header("Model comparison")
    stratum = st.selectbox(
        "Stratum", ["consent", "probable cause", "plain view", "arrest", None],
        format_func=lambda s: "ALL (pooled — for contrast only)" if s is None else s,
    )
    st.caption(
        "Fit the three arms on this stratum and compare. Use the SAME TabPFN "
        "subsample for every arm, or the comparison confounds model class with "
        "training-set size."
    )
    st.code(
        "from src import data, models\n"
        f"X, y, meta = data.build_xy(df, search_type={stratum!r})\n"
        "arms = models.build_all(X)\n"
        "# fit each, then stability.model_distance({name: scores})",
        language="python",
    )
    st.warning("TODO: fit models, cache scores, render AUC / PR-AUC / calibration here.")

# --------------------------------------------------------------------------- 4
with tab4:
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

# --------------------------------------------------------------------------- 5
with tab5:
    st.header("Budget-constrained deployment")
    st.markdown(
        "The client cannot search everyone. Given capacity for **K** searches, "
        "which stops? The operating metric is precision@K, not accuracy at 0.5."
    )
    k = st.slider("Search capacity K", 100, 20_000, 5_000, step=100)
    cost_fp = st.slider(
        "Cost of searching an innocent driver (relative to one justified search)",
        0.25, 8.0, 1.0, step=0.25,
    )
    st.caption(
        "This number is **not in the data**. It is a policy judgement about what "
        "an unjustified search costs a person. As it rises, the optimal policy "
        "searches fewer people and the fairness gaps shrink — which is the "
        "trustworthy-AI argument, quantified."
    )
    st.metric("Assumed FP cost", f"{cost_fp:.2f}×")
    st.warning("TODO: E.top_k_curve + E.equity_efficiency_frontier on fitted scores.")
