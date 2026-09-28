"""Problem definition — the decision, the objects, selective labels, and why we stratify by search type."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
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

st.subheader("Why consent searches only: the gap by legal basis")
st.markdown(
    "Consent and probable-cause searches rest on the officer's judgement of *this* driver. "
    "Arrest, warrant and inventory searches are **mechanical**: the legal authority exists "
    "before the officer weighs any evidence. Plain view is discretionary, but it is recorded "
    "*because* something was already seen."
)

BASES = {"Probable cause": ["probable cause"], "Consent (our sample)": ["consent"],
         "Plain view": ["plain view"], "Arrest, warrant, inventory": ["arrest", "warrant", "inventory"]}


@st.cache_data
def gap_by_basis(df: pd.DataFrame) -> pd.DataFrame:
    """Black − white hit rate per legal basis (resolved search_type), plus all searches."""
    d = df[df["subject_race"].isin(["white", "black"])]
    hit = d["contraband_found"].astype("string").str.strip().isin({"TRUE", "True", "1"})
    d = d.assign(_hit=hit.astype(int))
    rows = []
    for label, types in [*BASES.items(), ("All searches", None)]:
        s = d if types is None else d[d["search_type"].isin(types)]
        r = s.groupby("subject_race")["_hit"].mean()
        rows.append({"legal basis": label, "searches": len(s), "white hit rate": r["white"],
                     "Black hit rate": r["black"], "gap (pp)": 100 * (r["black"] - r["white"])})
    return pd.DataFrame(rows)


gb = gap_by_basis(common.searches())
fig = px.bar(gb, x="gap (pp)", y="legal basis", orientation="h", text_auto="+.1f",
             color=gb["legal basis"].eq("Consent (our sample)"),
             color_discrete_map={True: "#1f77d0", False: "#b0b7bf"})
fig.update_layout(showlegend=False, yaxis={"categoryorder": "array",
                                           "categoryarray": list(gb["legal basis"])[::-1]},
                  xaxis_title="Hit rate, Black minus white drivers (percentage points)")
fig.add_vline(x=0, line_color="#555")
st.plotly_chart(fig, width="stretch")
st.dataframe(gb.style.format({"searches": "{:,}", "white hit rate": "{:.2%}", "Black hit rate": "{:.2%}",
                              "gap (pp)": "{:+.2f}"}), hide_index=True, width="stretch")
st.caption(
    "Below zero: searches of Black drivers find contraband less often, consistent with a lower bar for "
    "searching them (the outcome test; it cannot prove intent, see infra-marginality). Pooled over all "
    "searches the gap has the wrong sign. Hit rate = searches of the group that found contraband / "
    "all searches of the group."
)

st.subheader("Why pooling hides the disparity")
st.markdown(
    "Split in two, consent against everything else. Note that \"non-consent\" is not purely "
    "mechanical: about half of it is probable cause and plain view (above)."
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
