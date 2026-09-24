"""Client-facing app: the 'interactive application' deliverable.

    python -m src.train          # fit once, writes outputs/
    streamlit run app/streamlit_app.py

The app never fits models. It reads cached artifacts from outputs/ so every
control is instant.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C          # noqa: E402
from src import data as D            # noqa: E402
from src import economics as E       # noqa: E402
from src import fairness as F        # noqa: E402
from src import train as T           # noqa: E402

st.set_page_config(page_title="Search Decision Support", layout="wide")


@st.cache_data(show_spinner="Loading searches…")
def load_df() -> pd.DataFrame:
    return D.load_searches()


@st.cache_data
def load_run(stratum: str, mode: str):
    return T.load_cached(stratum, mode)


@st.cache_data
def load_officer_signal():
    p = C.OUTPUTS / "officer_signal.json"
    return json.loads(p.read_text()) if p.exists() else None


def score_cols(scores: pd.DataFrame) -> list[str]:
    return [c for c in scores.columns if c.startswith("score_")]


def arm_name(col: str) -> str:
    return col.replace("score_", "")


st.title("Traffic-stop search decision support")
st.caption(
    "Nashville MPD, 2010–2018 · Stanford Open Policing Project · "
    "Pierson et al., *Nature Human Behaviour* 4 (2020)"
)

runs = T.available_runs()
if not runs:
    st.error("No cached model runs. Run `python -m src.train` first.")
    st.stop()

with st.sidebar:
    st.header("Run")
    stratum = st.selectbox("Stratum", ["consent", "probable_cause", "pooled"], index=0)
    mode = st.radio(
        "Training mode", ["matched", "full", "full_blind"], index=1,
        help="matched = every arm trains on the same 5,000 rows (the TabPFN "
             "ceiling), the only fair three-way comparison. full = all rows. "
             "full_blind = all rows with race removed from the features.",
    )
    scores, meta = load_run(stratum, mode)
    if scores is None:
        st.error(f"No cached run for {stratum}/{mode}.")
        st.stop()
    st.caption(f"test n = {len(scores):,} · split at {meta['split_year']}")
    st.caption(f"GBM backend: `{meta['gbm_backend']}`")

tabs = st.tabs(["1 · The problem", "2 · Pooling", "3 · Models",
                "4 · Fairness", "5 · Budget", "6 · One stop"])

# --------------------------------------------------------------------------- 1
with tabs[0]:
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
    sig = load_officer_signal()
    if sig:
        st.subheader("What is the model actually using? Measured, not asserted.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pre-search features only", f"{sig['baseline']:.4f}", help="AUC")
        c2.metric("+ officer identity", f"{sig['with_officer']:.4f}",
                  delta=f"{sig['delta']:+.4f}")
        c3.metric("Officer share of above-chance signal",
                  f"{sig['above_chance_uplift']:.0%}")

        full_m = load_run("consent", "full")[1]
        blind_m = load_run("consent", "full_blind")[1]
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

# --------------------------------------------------------------------------- 2
with tabs[1]:
    st.header("Why pooling hides the disparity")
    st.markdown(
        "Consent searches are fully discretionary. Incident-to-arrest, warrant, "
        "inventory and plain-view searches are largely **mechanical** — contraband "
        "is likely because something already happened."
    )
    tbl = F.pooled_vs_stratified(load_df())
    piv = tbl.pivot(index="stratum", columns="subject_race", values="hit_rate")
    st.dataframe(piv.style.format("{:.2%}"), width="stretch")
    st.dataframe(
        tbl.style.format({"hit_rate": "{:.2%}", "gap_vs_ref_pp": "{:+.2f}"}),
        width="stretch",
    )
    st.info(
        "The Black–white gap runs in **opposite directions** across strata, so "
        "pooling cancels them and reports 'no disparity'. This is effect "
        "modification, not Simpson's paradox."
    )

# --------------------------------------------------------------------------- 3
with tabs[2]:
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
        S.model_distance({arm_name(c): scores[c] for c in score_cols(scores)})
         .style.format("{:.4f}"),
        width="stretch",
    )
    st.caption(f"Base-rate drift across the split: "
               f"{m['base_rate_train'].iloc[0]:.1%} train → "
               f"{m['base_rate_test'].iloc[0]:.1%} test.")

# --------------------------------------------------------------------------- 4
with tabs[3]:
    st.header("Fairness")
    st.markdown(
        f"""
`Y = 1` contraband found · `Ŷ = 1` model recommends a search · `D` driver race.
Favourable for the individual is **{C.FAVORABLE_FOR_INDIVIDUAL.replace('_', ' ')}**
— the opposite of the course convention that `Y = 1` is favourable, so metrics
are shown under both codings.
        """
    )
    arm = st.selectbox("Arm", [arm_name(c) for c in score_cols(scores)])
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
        rates.style.format({"base_rate": "{:.2%}", "selection_rate": "{:.2%}",
                            "TPR": "{:.2%}", "FPR": "{:.2%}", "PPV": "{:.2%}",
                            "NPV": "{:.2%}", "mean_score": "{:.4f}"}),
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
        cc1.dataframe(both[C.Y_NATIVE].style.format("{:.2%}"))
        cc2.markdown(f"`{C.Y_COURSE}` — Y=1 innocent (course convention)")
        cc2.dataframe(both[C.Y_COURSE].style.format("{:.2%}"))

# --------------------------------------------------------------------------- 5
with tabs[4]:
    st.header("Budget-constrained deployment")
    arm5 = st.selectbox("Arm ", [arm_name(c) for c in score_cols(scores)], key="arm5")
    cost_fp = st.slider(
        "Cost of searching an innocent driver (relative to one justified search)",
        0.25, 8.0, 1.0, step=0.25,
    )
    st.caption(
        "This number is **not in the data**. It is a policy judgement about what "
        "an unjustified search costs a person. As it rises the optimal policy "
        "searches fewer people and the fairness gaps shrink — that is the "
        "trustworthy-AI argument, quantified."
    )
    s5 = scores[f"score_{arm5}"].to_numpy()
    y5 = scores["y"].to_numpy()
    curve = E.top_k_curve(y5, s5, cost_fp=cost_fp)
    st.line_chart(curve.set_index("k")[["net_benefit"]])
    st.line_chart(curve.set_index("k")[["precision_at_k"]])

    st.subheader("Sensitivity — where deployment stops paying")
    kk = st.slider("K for sensitivity", 100, len(scores),
                   min(2000, len(scores)), step=100, key="k5")
    st.dataframe(E.sensitivity(y5, s5, kk).style.format({
        "precision_at_k": "{:.2%}", "recall_at_k": "{:.2%}",
        "net_benefit": "{:,.0f}"}), width="stretch")

    st.subheader("Equity / efficiency frontier")
    sub5 = scores["subject_race"].isin(C.RACE_REPORTABLE).to_numpy()
    ks = np.unique(np.linspace(200, int(sub5.sum()), 25).astype(int))
    fr = E.equity_efficiency_frontier(y5[sub5], s5[sub5],
                                      scores["subject_race"][sub5], ks,
                                      cost_fp=cost_fp)
    if not fr.empty:
        st.scatter_chart(fr, x="max_FPR_gap", y="net_benefit")
        st.caption("Each point is a search capacity K. Up is more benefit; "
                   "left is more equal false-positive rates.")

# --------------------------------------------------------------------------- 6
with tabs[5]:
    st.header("Explain one stop")
    i = st.number_input("Row in the test set", 0, len(scores) - 1, 0, step=1)
    row = scores.iloc[int(i)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Race", str(row["subject_race"]))
    c2.metric("Year", int(row["year"]))
    c3.metric("Precinct", str(row["precinct"]))
    c4.metric("Contraband found", "yes" if row["y"] == 1 else "no")
    st.subheader("Score by arm")
    st.dataframe(
        pd.DataFrame({"score": {arm_name(c): float(row[c]) for c in score_cols(scores)}})
        .style.format("{:.4f}"), width="stretch",
    )
    st.info(
        "TODO for the group: attach SHAP local explanations here. The scorecard "
        "arm can show signed coefficient contributions directly; the GBM needs "
        "`shap.TreeExplainer`; TabPFN needs `tabpfn-extensions`."
    )
