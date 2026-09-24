"""Performance comparisons — the arms side by side: accuracy, ranking agreement, value under a search budget."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import common
from model_page import ARMS
from src import config as C
from src import economics as E
from src import stability as S

scores, meta = common.run_selector()

st.header("Performance comparisons")
tab_metrics, tab_budget, tab_one = st.tabs(["Accuracy and ranking", "Under a search budget", "One stop"])

with tab_metrics:
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
    st.caption(
        "1 − Spearman correlation between score vectors. Two models with equal "
        "AUC that rank differently are not interchangeable under a top-K policy."
    )
    st.dataframe(
        S.model_distance({common.arm_name(c): scores[c] for c in common.score_cols(scores)})
         .style.format("{:.4f}"),
        width="stretch",
    )
    st.caption(f"Base-rate drift across the split: "
               f"{m['base_rate_train'].iloc[0]:.1%} train → "
               f"{m['base_rate_test'].iloc[0]:.1%} test.")

with tab_budget:
    arm5 = st.selectbox("Arm", [common.arm_name(c) for c in common.score_cols(scores)], key="arm5",
                        format_func=lambda a: ARMS.get(a, a))
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
    st.caption(
        "Top-K is ranked over the **whole** test set, the same definition the "
        "Fairness page uses, so the two agree for the same K. Gaps are reported "
        "only for groups with enough volume to be stable."
    )
    ks = np.unique(np.linspace(200, len(scores), 25).astype(int))
    fr = E.equity_efficiency_frontier(y5, s5, scores["subject_race"], ks,
                                      report_groups=C.RACE_REPORTABLE,
                                      cost_fp=cost_fp)
    if not fr.empty:
        st.scatter_chart(fr, x="max_FPR_gap", y="net_benefit")
        st.caption("Each point is a search capacity K. Up is more benefit; "
                   "left is more equal false-positive rates.")

with tab_one:
    i = st.number_input("Row in the test set", 0, len(scores) - 1, 0, step=1)
    row = scores.iloc[int(i)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Race", str(row["subject_race"]))
    c2.metric("Year", int(row["year"]))
    c3.metric("Precinct", str(row["precinct"]))
    c4.metric("Contraband found", "yes" if row["y"] == 1 else "no")
    st.subheader("Score by arm")
    st.dataframe(
        pd.DataFrame({"score": {common.arm_name(c): float(row[c]) for c in common.score_cols(scores)}})
        .style.format("{:.4f}"), width="stretch",
    )
    st.info(
        "TODO for the group: attach SHAP local explanations here. The scorecard "
        "arm can show signed coefficient contributions directly; the GBM needs "
        "`shap.TreeExplainer`; TabPFN needs `tabpfn-extensions`."
    )
