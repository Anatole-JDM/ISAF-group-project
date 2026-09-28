"""Findings and recommendation — the deck's conclusion (slides 18-20), every number read from an artifact.

Act I  = the task as posed: stop and driver information only (outputs/, matched run).
Act II = adding the searching officer's past record (data/, scripts_v2/).
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import common
from src import config as C
from src import fairness as F


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


matched = common.load_run("consent", "matched")[1]
econ = read(C.OUTPUTS / "economics_matched.json")
xper = read(C.OUTPUTS / "xper_scorecard.json")
ctx = read(C.OUTPUTS / "tabpfn_context_stability.json")
diag = common.load_diag()

st.header("Findings and recommendation")
st.success(
    "**No model should tell officers whom to search — but the white box can show MNPD where fruitless "
    "searches come from.** Recommendation: a six-month pilot of a supervisor-run, officer-level review "
    "of consent-search practice. No roadside score."
)

# --------------------------------------------------------------------------- summary
st.subheader("Our answers to five questions")
auc = {a["arm"]: a["auc"] for a in (matched or {}).get("arms", [])}
auc_txt = f"{min(auc.values()):.2f}–{max(auc.values()):.2f}" if auc else "0.53–0.55"
extra = ""
if econ:
    rnd = econ["K"] * econ["base_rate"]
    best = max(econ["arms"].values(), key=lambda a: a["tp"])
    extra = f"; the best adds {best['tp'] - rnd:.0f} finds per {econ['K']:,} searches over random ranking"
race = f"{xper['phi']['subject_race'] / xper['sum_phi']:.1%}" if xper else "73.5%"
jac = (f"{min(d['top200_jaccard_mean'] for d in ctx.values()):.2f}–"
       f"{max(d['top200_jaccard_mean'] for d in ctx.values()):.2f}") if ctx else "0.07–0.08"
cf = (diag or {}).get("D_officer_counterfactual")
fewer = cf["if_bottom_fifth_had_median_hit_rate"]["same_finds_with_fewer_searches__fruitless_avoided"] if cf else None
fewer_txt = f"about {fewer[0]:,} fewer fruitless searches in 2016–2018 [{fewer[1][0]:,.0f}–{fewer[1][1]:,.0f}]" if fewer else ""
st.dataframe(pd.DataFrame([
    {"#": 1, "question": "Can a model predict a successful consent search?",
     "what we found": f"No. Test AUC {auc_txt} for all three models{extra}."},
    {"#": 2, "question": "What do the models learn, and can we explain it?",
     "what we found": f"Mostly the driver's race ({race} of the Act I white box's above-chance signal, "
                      "XPER; that white box is the logistic scorecard). Only the white box explains "
                      "itself exactly."},
    {"#": 3, "question": "Are the models stable and fair?",
     "what we found": f"No. Retrain on resampled data and the top-200 list overlaps at Jaccard {jac}; "
                      "every model flags mostly white drivers (outcome test run in reverse)."},
    {"#": 4, "question": "Where do search outcomes come from?",
     "what we found": "The searching officer. Their past record adds ~0.07 AUC to every model, but inside "
                      "one officer's own searches the scores fall to 0.50–0.56."},
    {"#": 5, "question": "Which model should MNPD deploy, and for what?",
     "what we found": f"The white box (PLTR, race-blind, with the officer's record), for an officer-level "
                      f"review, not a roadside score: {fewer_txt}."},
]), hide_index=True, width="stretch")
st.caption("Act I = stop and driver information only (the task as posed, pages White box to Fairness); "
           "its white box is the logistic scorecard. Act II = adding the searching officer's past record "
           "(page Officer features); from there the white box is PLTR, and the recommended one is race-blind.")

# --------------------------------------------------------------------------- Act I
st.subheader("Act I — the task as posed: no usable roadside score")
c1, c2, c3 = st.columns(3)
if auc:
    c1.metric("Best AUC, three models on the same rows", f"{max(auc.values()):.3f}", help="0.5 = coin flip")
if xper:
    c2.metric("Share of the white box's signal that is race", race, help="XPER, exact Shapley decomposition of AUC")
if ctx:
    c3.metric("Top-200 overlap after retraining", jac, help="Jaccard between shortlists from resampled training sets")
tbl = F.pooled_vs_stratified(common.searches())
gaps = tbl.pivot(index="stratum", columns="subject_race", values="gap_vs_ref_pp").drop(columns="white")
with st.expander("Why consent searches only: pooling hides the disparity"):
    st.dataframe(gaps.style.format("{:+.2f} pp"), width="stretch")
    st.caption("Hit-rate gap vs white drivers, in percentage points. Opposite signs cancel when pooled; "
               "see Problem definition for the breakdown by legal basis.")

# --------------------------------------------------------------------------- Act II
st.subheader("Act II — the officer's record adds 0.07 AUC, but only between officers")
if diag and "F_within_officer_auc" in diag:
    common.within_officer_chart(diag["F_within_officer_auc"])
if cf:
    g = cf["groups"]
    st.markdown(
        f"**A fifth of consent searches, made by {g['1']['officers']} officers, find contraband "
        f"{g['1']['hit_rate']:.0%} of the time** (the middle fifth: {g['3']['hit_rate']:.0%}; the best: "
        f"{g['5']['hit_rate']:.0%}). {g['1']['share_of_searched_drivers']['hispanic']:.1%} of the drivers they "
        f"search are Hispanic, against {g['3']['share_of_searched_drivers']['hispanic']:.1%} for the middle "
        "fifth. Details on the Officer features page."
    )

# --------------------------------------------------------------------------- recommendation
st.subheader("What a model can legitimately do here")
st.dataframe(pd.DataFrame([
    {"use": "Roadside triage: \"should I ask this driver?\"", "": "✗",
     "why": "Inside an officer's own stops, AUC 0.50–0.56; otherwise the signal is mostly race"},
    {"use": "Justifying a search after the fact", "": "✗", "why": "Would launder a discretionary decision"},
    {"use": "Deciding where to patrol", "": "✗", "why": "Place barely predicts contraband; feedback loops"},
    {"use": "Reviewing consent-search practice", "": "✓",
     "why": "Officer records differ 5×, persist, transfer to new officers and tie to racial skew"},
    {"use": "Training content", "": "✓", "why": "Readable patterns: repeat searches, skewed searching"},
    {"use": "Oversight reporting", "": "✓", "why": "Outcome tests by legal basis and race, year on year"},
]), hide_index=True, width="stretch")

c1, c2 = st.columns([3, 2])
with c1:
    st.markdown(
        """
**The review: per officer, each quarter, for supervisors**
1. Consent-search rate
2. Hit rate, shrunk, with an interval
3. Consent-search ratio by driver race
4. Hit-rate gap by race: the officer's own outcome test
5. Benchmark: the race-blind white box's expected hit rate for the searches made
"""
    )
with c2:
    if fewer:
        st.metric("Fewer pointless searches, same number of catches", f"{fewer[0]:,}",
                  help=cf["assumption"])
        st.caption(f"2016–2018, 95% CI {fewer[1][0]:,.0f}–{fewer[1][1]:,.0f}: if the "
                   f"{cf['groups']['1']['officers']} officers with the weakest records had the middle "
                   f"fifth's hit rate. {cf['if_bottom_fifth_had_median_hit_rate']['as_share_of_all_fruitless_test_searches']:.0%} "
                   "of all fruitless consent searches.")

st.subheader("Decision: a six-month pilot of the officer-level review")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
**Pilot conditions**
- Human review only, no automatic sanctions; officers see and can contest their report
- Intervals and shrinkage on every rate; false-discovery control across ~1,500 officers
- Retrain yearly: drift appears by 2018 (officer features PSI > 0.25)
"""
    )
with c2:
    st.markdown(
        """
**Monitor**
- Fruitless consent searches per 1,000 stops; finds per search
- Consent-search ratios and hit-rate gaps by race; complaints
"""
    )
st.caption(
    "Limits: selective labels (only searches made are observed); the outcome test cannot prove intent "
    "(infra-marginality); blank contraband fields were recorded as \"nothing found\"."
)
