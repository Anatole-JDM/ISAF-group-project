"""Stability — does the model survive a change of training data, of size, or of settings?

Reads the cached TabPFN experiments (python -m src.tabpfn_experiments) and the matched run's
metrics; nothing is fitted here:

    outputs/tabpfn_context_stability.json    5 training contexts, same size, same test set
    outputs/tabpfn_learning_curve.parquet    AUC / Brier by training size
    outputs/tabpfn_model_agreement.parquet   permutation importance per arm
    outputs/tabpfn_thinking_mode.parquet     TabPFN fit-time compute settings
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import common
from model_page import ARMS
from src import config as C
from src.models import SEED

OUT = C.OUTPUTS


def read_json(name: str):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def read_parquet(name: str):
    p = OUT / name
    return pd.read_parquet(p) if p.exists() else None


def missing(name: str) -> None:
    st.info(f"Not computed yet: `outputs/{name}`. Run `python -m src.tabpfn_experiments`.")


st.header("Stability")
st.markdown(
    """
*If we obtain two datasets from the same population, the algorithm should induce approximately
the same model from both* (Turney 1995, course section 7.1). Four checks follow: resample the
training data, grow it, compare models of equal accuracy, and change the settings. Consent searches, trained before 2016 and scored
on the same 2016–2018 test set throughout.
"""
)

# --------------------------------------------------------------------------- 1. resampling
st.subheader("1. Resample the training data: does the ranked list survive?")
ctx = read_json("tabpfn_context_stability.json")
if ctx is None:
    missing("tabpfn_context_stability.json")
else:
    rows = [{"model": ARMS.get(a, a), "training draws": d["n_contexts"], "rows per draw": d["n_train"],
             "AUC mean": d["auc_mean"], "AUC sd": d["auc_std"],
             "AUC range": d["auc_max"] - d["auc_min"],
             "per-stop score sd": d["per_row_pred_std_mean"],
             "top-200 overlap (Jaccard)": d["top200_jaccard_mean"]} for a, d in ctx.items()]
    st.dataframe(pd.DataFrame(rows).style.format({
        "AUC mean": "{:.4f}", "AUC sd": "{:.4f}", "AUC range": "{:.4f}",
        "per-stop score sd": "{:.4f}", "top-200 overlap (Jaccard)": "{:.2f}"}),
        width="stretch", hide_index=True)
    worst = min(d["top200_jaccard_mean"] for d in ctx.values())
    best = max(d["top200_jaccard_mean"] for d in ctx.values())
    st.error(
        f"**The ranked list is close to arbitrary.** Redraw the training rows and the 200 stops "
        f"flagged as highest-risk overlap by only {worst:.2f}–{best:.2f} (Jaccard): roughly nine in "
        "ten of them change. Under a search budget the top-K list *is* the policy, so this matters "
        "more than any AUC."
    )
    spread = max(d["auc_max"] - d["auc_min"] for d in ctx.values())
    st.caption(f"AUC moves by up to {spread:.3f} from the choice of training rows alone, which is "
               "about as large as the gap between the model families. Single-seed comparisons "
               "between models are therefore not reliable (see the significance tests on Performance).")

# --------------------------------------------------------------------------- 2. learning curve
st.subheader("2. Grow the training data: is the ceiling data or signal?")
lc = read_parquet("tabpfn_learning_curve.parquet")
if lc is None:
    missing("tabpfn_learning_curve.parquet")
else:
    lc = lc.assign(model=lc["arm"].map(lambda a: ARMS.get(a, a)))
    metric = st.radio("Metric", ["auc", "brier"], horizontal=True,
                      format_func=lambda m: {"auc": "AUC", "brier": "Brier score"}[m])
    fig = px.line(lc, x="n_train", y=metric, color="model", markers=True, log_x=True,
                  labels={"n_train": "training rows (log scale)", "auc": "AUC", "brier": "Brier"})
    if metric == "auc":
        fig.add_hline(y=0.5, line_dash="dot", line_color="#8a8984", annotation_text="chance")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"{lc['n_train'].min():,} to {lc['n_train'].max():,} training rows, fixed evaluation "
        "subsample. A fifty-fold increase in data barely moves AUC: with real signal and a "
        "data-starved model the curve would climb. The ceiling is the information in the "
        "features, not the amount of data."
    )

# --------------------------------------------------------------------------- 3. explanations
st.subheader("3. Same accuracy, different reasons: distance among models")
ag = read_parquet("tabpfn_model_agreement.parquet")
if ag is None:
    missing("tabpfn_model_agreement.parquet")
else:
    imp = ag.pivot_table(index="feature", columns="arm", values="auc_drop", aggfunc="mean")
    imp = imp[[a for a in ARMS if a in imp.columns]]
    imp.columns = [ARMS.get(a, a) for a in imp.columns]
    order = imp.mean(axis=1).sort_values(ascending=False).index
    long = imp.loc[order].reset_index().melt(id_vars="feature", var_name="model",
                                             value_name="AUC drop when shuffled")
    fig = px.bar(long, x="AUC drop when shuffled", y="feature", color="model", barmode="group",
                 orientation="h")
    fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": list(order[::-1])})
    st.plotly_chart(fig, width="stretch")
    st.caption("Permutation importance: the drop in AUC when one feature is shuffled.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Rank agreement** (Spearman between importance vectors)")
        st.dataframe(imp.corr(method="spearman").style.format("{:+.2f}"), width="stretch")
    with c2:
        st.markdown("**Feature-importance distance** ‖φ(f₁) − φ(f₂)‖₂ (course sl. 193)")
        cols = list(imp.columns)
        dist = pd.DataFrame([[float(np.linalg.norm(imp[a].fillna(0) - imp[b].fillna(0))) for b in cols]
                             for a in cols], index=cols, columns=cols)
        st.dataframe(dist.style.format("{:.4f}"), width="stretch")
    st.error(
        "**Interpretability is not a property you can read off one model.** The three models are "
        "within a few hundredths of AUC of each other, yet they disagree on what drives the "
        "prediction, some pairs with a negative rank correlation. Choose a different model of "
        "equal accuracy and you get a different story about why a driver was flagged."
    )

# --------------------------------------------------------------------------- 4. settings
st.subheader("4. Change the settings: randomness in the foundation model")
tm = read_parquet("tabpfn_thinking_mode.parquet")
meta = read_json("metrics__consent__matched.json")
if tm is None:
    missing("tabpfn_thinking_mode.parquet")
else:
    st.dataframe(tm.rename(columns={"thinking": "fit-time compute", "seconds": "seconds"})
                 .style.format({"auc": "{:.4f}", "brier": "{:.4f}", "seconds": "{:.1f}"}),
                 width="stretch", hide_index=True)
    st.caption("Up to ~5× more compute at fit time leaves AUC unchanged: more evidence that there "
               "is little to find.")
if meta:
    tab = next((a for a in meta["arms"] if a["arm"] == "tabpfn"), None)
    st.markdown(
        "**What we report for the foundation model** (the course's checklist for randomness in "
        "large models, section 7.2: a result is only reproducible if its configuration is stated):"
    )
    st.dataframe(pd.DataFrame([
        {"item": "backend", "value": str(meta.get("tabpfn_backend", "—"))},
        {"item": "model version", "value": str(meta.get("tabpfn_version", "—"))},
        {"item": "training context (rows)", "value": f"{tab['n_train']:,}" if tab else "—"},
        {"item": "test rows", "value": f"{tab['n_test']:,}" if tab else "—"},
        {"item": "random seed", "value": str(SEED)},
        {"item": "context draws for the stability check", "value":
            str(ctx["tabpfn"]["n_contexts"]) if ctx and "tabpfn" in ctx else "—"},
    ]), width="stretch", hide_index=True)
    st.caption("With the hosted backend (`client`) the provider controls the weights and hardware, "
               "so the version can change without notice: another reason a deployed model would "
               "need its outputs monitored.")
