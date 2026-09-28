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
import plotly.express as px

# Open on a run that contains all three arms (`matched`), the comparison the slides report.
scores, meta = common.run_selector(require_arm="tabpfn")

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
        "**Per-stop explanations live on each arm's own page** — White box, Black box "
        "and Tabular foundation — because the three support different grades of "
        "attribution: exact closed-form Shapley for the linear arm, exact TreeSHAP for "
        "the trees, and approximate occlusion for TabPFN, which has no native "
        "attribution at all."
    )


# --------------------------------------------------------------------------- XPER
st.divider()
st.subheader("XPER — decomposing AUC itself")
import json as _json
_xp = C.OUTPUTS / "xper_scorecard.json"
if not _xp.exists():
    st.info("Run `python -m src.interpret` to compute it (2^10 = 1,024 exact coalitions, ~5 min).")
else:
    d = _json.loads(_xp.read_text())
    st.markdown(
        f"""
Performance itself, split by feature: **AUC = φ₀ + Σⱼ φⱼ** with φ₀ = 0.5 (a model with
no features). Computed **exactly** — all 2^10 = {d['n_coalitions']:,} coalitions, each one a
re-estimation on that feature subset. No sampling, no approximation.
"""
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("AUC", f"{d['auc_full']:.4f}")
    c2.metric("Σφⱼ (above chance)", f"{d['sum_phi']:.4f}")
    c3.metric("Efficiency gap", f"{d['efficiency_gap']:.1e}",
              help="φ₀ + Σφⱼ − AUC must be zero. This is machine precision.")

    t = (pd.Series(d["phi"], name="phi").to_frame().reset_index(names="feature"))
    t["share_of_signal"] = t["phi"] / d["sum_phi"]
    fig = px.bar(t.iloc[::-1], x="phi", y="feature", orientation="h",
                 color=t.iloc[::-1]["phi"] > 0,
                 color_discrete_map={True: "#c0392b", False: "#2c7fb8"},
                 labels={"phi": "φⱼ — contribution to AUC"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(t.style.format({"phi": "{:+.5f}", "share_of_signal": "{:+.1%}"}),
                 hide_index=True, width="stretch")
    top = t.iloc[0]
    st.error(
        f"**`{top['feature']}` alone is {top['share_of_signal']:.1%} of everything the model "
        f"knows.** The whole above-chance signal is {d['sum_phi']:.4f} AUC; race accounts for "
        f"{top['phi']:.5f} of it. The model is close to a race detector with noise attached."
    )


# --------------------------------------------------------------------------- significance
st.divider()
st.subheader("Are the AUC differences real? DeLong tests")
_sig_p = C.OUTPUTS / "significance__consent__matched.json"
if not _sig_p.exists():
    st.info("Run `python scripts/run_significance.py` to compute it.")
else:
    _sig = _json.loads(_sig_p.read_text())
    st.markdown(
        f"Paired DeLong test on the same {_sig['n_test']:,} test searches "
        f"({_sig['stratum']} / `{_sig['mode']}`, every model trained on the same rows). "
        "H₀: the two models have the same AUC."
    )
    _pairs = pd.DataFrame([{
        "comparison": f"{ARMS.get(p['a'], p['a'])}  vs  {ARMS.get(p['b'], p['b'])}",
        "AUC a": p["auc_a"], "AUC b": p["auc_b"], "difference": p["diff"],
        "95% CI": f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]",
        "z": p["z"], "p-value": p["p_value"],
        "significant at 5%": "yes" if p["p_value"] < 0.05 else "no",
    } for p in _sig["pairs"]])
    st.dataframe(_pairs.style.format({"AUC a": "{:.4f}", "AUC b": "{:.4f}", "difference": "{:+.4f}",
                                      "z": "{:+.2f}", "p-value": "{:.3g}"}),
                 hide_index=True, width="stretch")
    _n_sig = int((_pairs["p-value"] < 0.05).sum())
    _max_d = float(_pairs["difference"].abs().max())
    st.caption(
        f"{_n_sig} of {len(_pairs)} pairwise differences are significant at 5%. The largest "
        f"difference is {_max_d:.3f} AUC, between models that sit only a few points above "
        "chance, and of the same order as the AUC spread from simply redrawing the training "
        "rows (see Stability). Statistically distinguishable is not the same as practically "
        "better."
    )
