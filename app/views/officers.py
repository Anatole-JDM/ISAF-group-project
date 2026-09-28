"""Officer features (comparison) — what the officer's own track record would add.

A benchmark, not the main result: the features help AUC but cannot be used to decide
whether to search a given driver (see the page text).

Reads the committed results of the officer-feature track (scripts_claude/ and
reports/xgboost_officer/); nothing is fitted here. Reproduce:

    python scripts_claude/build_officer_features.py && python scripts_claude/build_master.py
    python scripts_claude/evaluate_officer_models.py      # AUCs, PLTR terms, predictions
    python scripts_claude/officer_model_diagnostics.py    # AUC by year, economics, calibration
    python scripts_claude/officer_holdout_stability.py    # 5 draws of held-out officers
    python scripts_claude/analyze_officer_disparity.py    # officer outcome test, no model
    python scripts_claude/xgboost_officer_analysis.py     # XGBoost stability / fairness / SHAP
    python scripts_claude/xgboost_surrogate.py            # global surrogate
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

import common

DATA = common.ROOT / "data"
FIGS = common.ROOT / "reports" / "xgboost_officer"

FEATURE_SETS = {
    "full_time": "Driver, place, time (team baseline)",
    "full_time_officer": "+ officer track record",
    "full_time_officer_blind": "+ officer track record, race-blind",
    "officer_only": "Officer track record only",
}
MODELS = {"scorecard": "Scorecard (logistic)", "pltr": "PLTR", "pltr_sparse": "PLTR sparse",
          "xgboost": "XGBoost"}
RACES = ["white", "black", "hispanic"]


@st.cache_data
def load(name: str):
    p = DATA / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ci(v) -> str:
    """[point, [lo, hi]] -> '12.3% [11.0, 13.5]'."""
    return f"{v[0]:.1%} [{v[1][0]:.1%}, {v[1][1]:.1%}]"


def figure(name: str, caption: str) -> None:
    p = FIGS / name
    if p.exists():
        st.image(str(p), caption=caption)


results = load("officer_model_results")
diag = load("officer_model_diagnostics")
holdout = load("officer_holdout_stability")
xgb = load("xgboost_officer_analysis")
surrogate = load("xgboost_surrogate")
pltr = load("pltr_terms")
disparity = load("officer_disparity_findings")

st.header("Officer features — a comparison, not the deployable model")
if results is None:
    st.error("Officer results not found in `data/`. Run `python scripts_claude/evaluate_officer_models.py`.")
    st.stop()

st.warning(
    "**Not for scoring drivers at the roadside.** When an officer decides whether to search *this* "
    "driver, the officer is the same for every driver in front of them, so a feature describing the "
    "officer cannot rank those drivers: it raises test-set AUC but has no decision value there. It "
    "would also score the officer's past hit rate rather than the driver's risk. **What these "
    "features are for:** reviewing officers' consent-search practice, the recommendation in "
    "Findings. The white box used for that review is the race-blind PLTR (tab *White box: PLTR*)."
)
st.markdown(
    """
**What is added.** 16 features describing the searching officer, computed **strictly from that
officer's earlier stops**: past-year hit rate and consent hit rate (shrunk toward the force-wide
rate), searches already made today, experience, and how much more often the officer asks minority
drivers for consent. The stop's own outcome is never used.

**Why include them at all.** They answer a question the main models cannot: is the ceiling of
~0.57 AUC a limit of the models, or of the information available about the driver? Same sample and
split as the rest of the app (consent searches 2010–2018, train before 2016, test 2016–2018), plus a
second split that holds out 20% of **officers** entirely.
"""
)
st.info(
    "**Reading.** Officer history lifts AUC by about 0.07 in every model family tested, which "
    "confirms that the driver-level features, not the algorithms, set the ceiling. The signal "
    "is about *who searches*, not *who is searched*."
)

tabs = st.tabs(["Performance", "White box: PLTR", "Black box: XGBoost", "Stability",
                "Fairness", "Economics", "Officer outcome test"])

# --------------------------------------------------------------------------- performance
with tabs[0]:
    rows = []
    for split, lab in (("temporal", "time split"), ("officer_holdout", "unseen officers")):
        for fs, models in results["splits"][split]["models"].items():
            for m, r in models.items():
                rows.append({"features": FEATURE_SETS.get(fs, fs), "model": MODELS.get(m, m),
                             "split": lab, "AUC": r["auc"],
                             "95% CI": f"[{r['auc_ci95'][0]:.3f}, {r['auc_ci95'][1]:.3f}]",
                             "PR-AUC": r.get("pr_auc"), "Brier": r.get("brier"),
                             "n features": r.get("n_features")})
    perf = pd.DataFrame(rows)
    split_pick = st.radio("Split", ["time split", "unseen officers"], horizontal=True)
    st.dataframe(perf[perf["split"] == split_pick].drop(columns="split")
                 .style.format({"AUC": "{:.3f}", "PR-AUC": "{:.3f}", "Brier": "{:.4f}"}),
                 width="stretch", hide_index=True)
    t = results["splits"]["temporal"]
    st.caption(f"Time split: train n = {t['n_train']:,} (hit rate {t['base_rate_train']:.1%}), "
               f"test n = {t['n_test']:,} ({t['base_rate_test']:.1%}). AUC CIs are bootstrap 95%.")

    gains = [{"model": MODELS.get(m, m), "features": FEATURE_SETS.get(fs, fs),
              "AUC gain vs baseline": r["vs_full_time"]["delta_auc"],
              "paired 95% CI": f"[{r['vs_full_time']['ci95'][0]:+.3f}, {r['vs_full_time']['ci95'][1]:+.3f}]"}
             for fs, models in t["models"].items() for m, r in models.items() if "vs_full_time" in r]
    if gains:
        st.markdown("**Gain from the officer features** (paired bootstrap: the same resampled rows for both models)")
        st.dataframe(pd.DataFrame(gains).style.format({"AUC gain vs baseline": "{:+.3f}"}),
                     width="stretch", hide_index=True)

    if diag and "F_within_officer_auc" in diag:
        st.markdown("**Pooled AUC vs AUC inside each officer's own searches** — the decision an officer "
                    "actually faces is which of *their own* stops to search")
        common.within_officer_chart(diag["F_within_officer_auc"])

    if diag:
        st.markdown("**AUC by test year**")
        by_year = pd.DataFrame({k: {y: v["auc"] for y, v in d.items()}
                                for k, d in diag["A_auc_by_test_year"].items()}).T
        st.dataframe(by_year.style.format("{:.3f}"), width="stretch")

# --------------------------------------------------------------------------- PLTR
with tabs[1]:
    st.info(
        "**This is the white box from slide 12 on** — not the logistic scorecard of Act I (White box "
        "models page). Race-aware PLTR: 19 rules, AUC 0.630, 81% of its weight about the officer "
        "(slide 12). **Race-blind PLTR: AUC 0.628, the version recommended** for the officer-level "
        "review (slides 16–18). It opens on the race-blind version below."
    )
    st.markdown(
        """
**Penalised Logistic Tree Regression** (Dumitrescu, Hué, Hurlin & Tokpavi 2022; course section 2.4).
Short trees generate threshold rules on one or two variables, an adaptive lasso keeps a few of them,
and the result is still a logistic regression: every term has an odds ratio and a marginal effect.
"""
    )
    if pltr:
        variant = st.selectbox("Model", list(pltr), index=list(pltr).index(
            "temporal/full_time_officer_blind/pltr_sparse") if
            "temporal/full_time_officer_blind/pltr_sparse" in pltr else 0)
        terms = pd.DataFrame(pltr[variant])
        cols = [c for c in ["term", "kind", "coef", "odds_ratio", "avg_marginal_effect",
                            "support_share", "importance_share"] if c in terms]
        st.dataframe(terms[cols].style.format({
            "coef": "{:+.3f}", "odds_ratio": "{:.3f}", "avg_marginal_effect": "{:+.2%}",
            "support_share": "{:.1%}", "importance_share": "{:.1%}"}),
            width="stretch", hide_index=True)
        st.caption(f"{len(terms)} terms. AME = average marginal effect on the probability of finding "
                   "contraband; support = share of test searches the rule applies to.")
    if holdout:
        core = holdout["pltr_stable_core"]
        st.markdown("**Terms that survive every draw of held-out officers** (5 draws)")
        st.dataframe(pd.DataFrame([{"model": k, "variables in every draw": ", ".join(v["variable_combos_in_every_draw"]),
                                    "mean pairwise Jaccard": v["mean_pairwise_jaccard"]}
                                   for k, v in core.items()]).style.format({"mean pairwise Jaccard": "{:.2f}"}),
                     width="stretch", hide_index=True)
        st.caption("The exact rules move from draw to draw (low Jaccard); the officer's track record "
                   "and Hispanic search skew are always selected.")

# --------------------------------------------------------------------------- XGBoost
with tabs[2]:
    if xgb:
        a = xgb["auc_test"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline", f"{a['full_time']:.3f}", help="AUC, test 2016-2018")
        c2.metric("+ officer features", f"{a['full_time_officer']:.3f}",
                  delta=f"{a['full_time_officer'] - a['full_time']:+.3f}")
        c3.metric("race-blind", f"{a['full_time_officer_blind']:.3f}",
                  delta=f"{a['full_time_officer_blind'] - a['full_time']:+.3f}")
        st.markdown(f"Global explanation: {xgb['interpretation']['method']}.")
        c1, c2 = st.columns(2)
        with c1:
            figure("interpretation_global_importance.png", "Mean |SHAP| per feature")
        with c2:
            figure("interpretation_beeswarm.png", "SHAP beeswarm: direction and spread")
        fs_chk = (diag or {}).get("G_first_stop_check")
        if fs_chk:
            st.markdown("**Checked: searches at the officer's first stop of the calendar day hit more often.** "
                        "`off_minutes_since_first_stop_today` ranks high in both XGBoost and PLTR, so it was "
                        "tested for artefacts:")
            hb = pd.DataFrame(fs_chk["by_hour_band"]).T
            long = pd.concat([hb[["hit_first"]].rename(columns={"hit_first": "hit rate"}).assign(stop="first stop of the day"),
                              hb[["hit_later"]].rename(columns={"hit_later": "hit rate"}).assign(stop="later stops")])
            fig = px.bar(long.reset_index(names="hour band"), x="hour band", y="hit rate", color="stop",
                         barmode="group", text_auto=".0%",
                         color_discrete_map={"first stop of the day": "#1f77d0", "later stops": "#b0b7bf"})
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
            first, later = fs_chk["hit_rate_first_vs_later"]
            st.caption(
                f"{fs_chk['share_at_first_stop']:.0%} of {fs_chk['searches']:,} consent searches happen at the "
                f"officer's first stop of the calendar day; they hit {first:.1%} against {later:.1%}, in every "
                f"hour band and for low-, mid- and high-volume officers. "
                f"{fs_chk['share_of_first_stop_searches_between_00_and_04']:.0%} of them fall between 00:00 and "
                "03:59, so this is not \"start of shift\". The feature uses only earlier stops (no leakage). "
                "Reading: later, repeated searches look more speculative — a hypothesis for training, not a "
                "causal claim."
            )
    if surrogate:
        st.subheader("Global surrogate")
        st.markdown("A decision tree fitted to **XGBoost's predicted probability** on the training "
                    "years, then checked against XGBoost on the test years.")
        sg = pd.DataFrame(surrogate["surrogates"]).T
        st.dataframe(sg.style.format({"r2": "{:.2f}", "spearman": "{:.2f}", "topk_overlap": "{:.0%}",
                                      "auc_vs_truth": "{:.3f}", "n_leaves": "{:.0f}"}, na_rep="—"),
                     width="stretch")
        st.caption(f"XGBoost itself: AUC {surrogate['black_box_auc_vs_truth']:.3f}. Officer features carry "
                   f"{surrogate['shown_tree']['officer_share_of_split_importance']:.0%} of the depth-3 "
                   "tree's split importance. Readable, but it reproduces only ~40% of the score variance: "
                   "fine to describe the black box, not to replace it.")
        figure("surrogate_tree.png", "Depth-3 surrogate tree")

# --------------------------------------------------------------------------- stability
with tabs[3]:
    if xgb:
        s = xgb["stability"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("10 seeds: AUC", f"{s['seeds']['auc_mean']:.3f} ± {s['seeds']['auc_std']:.3f}")
        c2.metric("10 seeds: top-22% overlap (min)", f"{s['seeds']['topk_overlap_min']:.0%}")
        c3.metric("20 bootstraps: AUC 95% range",
                  f"{s['bootstrap']['auc_p2.5']:.3f}–{s['bootstrap']['auc_p97.5']:.3f}")
        c4.metric("Unseen-officer gap", f"{s['officer_cv']['gap']:.3f}",
                  help=f"Random KFold {s['officer_cv']['random_kfold']['auc_mean']:.3f} vs "
                       f"GroupKFold by officer {s['officer_cv']['officer_groupkfold']['auc_mean']:.3f}")
        ir = s["bootstrap"]["importance_ranking"]
        st.markdown(f"**Explanation stability.** Across bootstrap refits the SHAP importance ranking "
                    f"correlates at {ir['spearman_mean']:.2f} on average (min {ir['spearman_min']:.2f}). "
                    "Training on 2010–2012 vs 2013–2015 moves it more.")
        figure("stability_importance_bootstrap.png", "SHAP importance across 20 bootstrap refits")

        st.markdown("**Distance among models** (same test rows)")
        st.dataframe(pd.DataFrame(s["distance_among_models"]).T
                     .style.format({"score_spearman": "{:.2f}", "topk_overlap": "{:.0%}"}), width="stretch")
        st.caption("Adding the officer features changes who is flagged far more than it changes AUC.")

        st.markdown("**Drift, PSI train → test** (> 0.25 = material shift)")
        psi = pd.Series(s["psi_train_vs_test"], name="PSI").sort_values(ascending=False).to_frame()
        st.dataframe(psi.style.format("{:.3f}").highlight_between(left=0.25, right=99, color="#f4cccc"),
                     width="stretch")
        st.caption("The most important features are also the ones that drift most (partly mechanical: "
                   "experience grows with time). A deployed model would need monitoring.")
    if holdout:
        st.markdown("**Officer holdout over 5 random draws of officers**")
        st.dataframe(pd.DataFrame(holdout["summary_across_draws"]).T[["auc_mean", "auc_sd", "auc_min", "auc_max"]]
                     .style.format("{:.3f}"), width="stretch")

# --------------------------------------------------------------------------- fairness
with tabs[4]:
    t = results["splits"]["temporal"]["models"]
    c1, c2 = st.columns(2)
    fs = c1.selectbox("Features", list(t), index=list(t).index("full_time_officer_blind")
                      if "full_time_officer_blind" in t else 0, format_func=lambda k: FEATURE_SETS.get(k, k))
    m = c2.selectbox("Model", list(t[fs]), format_func=lambda k: MODELS.get(k, k))
    ft = t[fs][m].get("fairness_topk")
    if ft:
        st.markdown(f"Top {ft['share_selected']:.0%} of test searches flagged (K = {ft['k']:,}); 95% bootstrap CIs.")
        tbl = pd.DataFrame([{"race": g, "n": ft[g]["n"],
                             "selection rate": ci(ft[g]["selection_rate"]),
                             "false positive rate": ci(ft[g]["false_positive_rate"]),
                             "hit rate of selected": ci(ft[g]["hit_rate_of_selected"])}
                            for g in RACES if g in ft])
        st.dataframe(tbl, width="stretch", hide_index=True)
    key = f"{fs}/{m}"
    if diag and key in diag["E_calibration_by_race"]:
        gap = diag["E_calibration_by_race"][key]["hit_rate_gap_at_equal_score"]
        st.markdown("**Sufficiency: hit-rate gap vs white drivers at the same score**")
        st.dataframe(pd.DataFrame([{"comparison": k.replace("_", " "),
                                    "gap (pp)": f"{v[0] * 100:+.1f} [{v[1][0] * 100:+.1f}, {v[1][1] * 100:+.1f}]"}
                                   for k, v in gap.items()]),
                     width="stretch", hide_index=True)
    figure("fairness_by_race.png", "XGBoost: selection rate, FPR and hit rate by race, three feature sets")
    st.markdown(
        """
- **Race-blind models select the three groups far more equally** (independence and separation improve),
  but at the same score minority drivers are less likely to carry contraband: sufficiency fails.
  Blinding reproduces the officers' lower bar. Both criteria cannot hold together here.
- **Removing race does not remove race.** `off_hit_rate_same_race_past` is computed from the driver's
  race, so the race-blind models drop it too.
"""
    )

# --------------------------------------------------------------------------- economics
with tabs[5]:
    if diag:
        keys = list(diag["C_economics"])
        key = st.selectbox("Model", keys, index=keys.index("full_time_officer_blind/pltr")
                           if "full_time_officer_blind/pltr" in keys else 0)
        e = diag["C_economics"][key]
        st.markdown(f"Test searches: {e['n']:,}, of which {e['hits']:,} found contraband "
                    f"({e['hit_rate_all']:.1%}). What if only the top-scored share had been carried out?")
        kt = pd.DataFrame(e["keep_top_share"]).T
        kt.index = [f"keep top {float(i):.0%}" for i in kt.index]
        st.dataframe(kt.style.format({"searches_kept": "{:,.0f}", "hit_rate_kept": "{:.1%}",
                                      "fruitless_avoided": "{:,.0f}", "fruitless_avoided_share": "{:.1%}",
                                      "finds_lost": "{:,.0f}", "finds_lost_share": "{:.1%}",
                                      "finds_lost_per_100_fruitless_avoided": "{:.1f}"}), width="stretch")
        st.caption(f"Random selection loses {e['random_finds_lost_per_100_fruitless_avoided']:.1f} finds per "
                   "100 fruitless searches avoided; lower is better.")
        cf = diag["D_officer_counterfactual"]
        w = cf["if_bottom_fifth_had_median_hit_rate"]
        st.markdown(
            f"**Officer counterfactual.** The fifth of test searches made by officers with the weakest "
            f"past record hit {cf['bottom_fifth_hit_rate'][0]:.1%}, against {cf['median_fifth_hit_rate'][0]:.1%} "
            f"for the middle fifth. At the median rate they would have made the same finds with "
            f"~{w['same_finds_with_fewer_searches__fruitless_avoided'][0]:,} fewer fruitless searches "
            f"({w['as_share_of_all_fruitless_test_searches']:.1%} of all fruitless test searches)."
        )
        st.caption(cf["assumption"])

# --------------------------------------------------------------------------- outcome test
with tabs[6]:
    if disparity:
        st.markdown(
            """
**No model here.** Each officer is compared **with himself**: how often he asks minority drivers for
consent relative to white drivers (search skew), and his hit rate on each group (the outcome test).
That holds the officer and roughly his area fixed.
"""
        )
        grp = st.radio("Comparison", ["black_vs_white", "hispanic_vs_white"], horizontal=True,
                       format_func=lambda k: k.replace("_", " "))
        d = disparity[grp]
        other = grp.split("_")[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Officers compared", f"{d['officers']}")
        c2.metric("Median search skew", f"{d['median_search_ratio']:.2f}×",
                  help=f"{d['share_officers_ratio_above_1']:.0%} of officers are above 1")
        rho = d["spearman_skew_vs_overall_hit_rate"]
        c3.metric("Skew vs overall hit rate", f"ρ = {rho['rho']:.2f}",
                  help=f"95% CI [{rho['ci95'][0]:.2f}, {rho['ci95'][1]:.2f}]")
        q = pd.DataFrame([{"quintile of search skew": k, "officers": v["officers"],
                           "skew range": f"{v['search_ratio_range'][0]:.2f}–{v['search_ratio_range'][1]:.2f}×",
                           "searches": v["consent_searches"], "hit rate, all": ci(v["hit_rate_all"]),
                           "on white": ci(v["hit_rate_white"]), f"on {other}": ci(v[f"hit_rate_{other}"])}
                          for k, v in d["by_quintile_of_search_skew"].items()])
        st.dataframe(q, width="stretch", hide_index=True)
        ot, conc = d["outcome_test"], d["concentration"]
        st.markdown(
            f"- **{ot['share_gap_negative']:.0%}** of officers find less on {other} drivers than on white "
            f"drivers (median gap {ot['median_gap'] * 100:+.1f} pp); {ot['significant_negative_p05']} are "
            f"significantly negative vs ~{ot['expected_by_chance_each_side']:.0f} expected by chance.\n"
            f"- The most skewed fifth of officers make "
            f"**{conc[f'top_quintile_share_of_{other}_consent_searches']:.0%}** of consent searches of {other} "
            f"drivers, but {conc['top_quintile_share_of_white_consent_searches']:.0%} of those of white drivers."
        )
        st.warning("**How to say it:** consistent with a lower evidentiary bar, not proof of intent.\n\n"
                   + "\n".join(f"- {c}" for c in disparity["caveats"]))
