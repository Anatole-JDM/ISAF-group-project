"""White box models — the scorecard arm, its coefficients, and a decision tree."""
import json

import pandas as pd
import plotly.express as px
import streamlit as st

import common
import model_page
from src import config as C

model_page.render(
    "scorecard", "White box models",
    description="""
> **Two white boxes in this project.** This page is the **Act I white box: the logistic
> scorecard** (stop and driver features only, slides 7–10). It is the model whose signal is
> 73.5% the driver's race. From slide 12 on, the white box is **PLTR** (19 rules, with the
> officer's record), and the one we recommend is its **race-blind** version: see the
> **Officer features** page, tab *White box: PLTR*.

**Scorecard: logistic regression on one-hot-encoded pre-search features.** Readable to a
caseworker: the coefficients are the explanation.

- `class_weight` is left at None, not "balanced". Balanced weighting improves ranking on an
  imbalanced target but destroys calibration, and calibration *is* the sufficiency dimension
  of the fairness taxonomy.
- For the full WOE/IV scorecard banks deploy, swap the preprocessor for optbinning's
  `BinningProcess`: monotonic bins and points per attribute.
""",
    missing_hint="Run `python -m src.train` to fit it.",
)


# --------------------------------------------------------------------------- coefficients
st.divider()
st.subheader("The coefficients — opening the white box")

_ct = C.OUTPUTS / "coefficient_table.parquet"
_cm = C.OUTPUTS / "coefficient_meta.json"
if not _ct.exists():
    st.info("Run `python -m src.interpret` to build the coefficient table.")
else:
    t = pd.read_parquet(_ct)
    m = json.loads(_cm.read_text()) if _cm.exists() else {}
    st.markdown(
        f"""
A coefficient is **not** a marginal effect. For a logit the marginal effect is
`p(1-p) x beta`, so it is individual-specific. On this data the base rate is
**{m.get('base_rate_train', float('nan')):.1%}**, giving a multiplier of
**{m.get('ame_multiplier', float('nan')):.3f}** — every coefficient shrinks about
**{1/m.get('ame_multiplier', 1):.0f}x** when expressed in probability.

- **odds ratio** = `exp(beta)`
- **AME** = average marginal effect, in percentage points
- **MEM** = marginal effect at the mean
"""
    )
    top = t.head(15)
    st.dataframe(top.style.format({"coef_log_odds": "{:+.4f}", "odds_ratio": "{:.3f}",
                                   "AME_pp": "{:+.2f}", "MEM_pp": "{:+.2f}"}),
                 hide_index=True, width="stretch")
    fig = px.bar(top.iloc[::-1], x="AME_pp", y="feature", orientation="h",
                 color=top.iloc[::-1]["AME_pp"] > 0,
                 color_discrete_map={True: "#c0392b", False: "#2c7fb8"},
                 labels={"AME_pp": "average marginal effect (percentage points)"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.error(
        "**The single largest coefficient in a model built to predict contraband is a "
        "race indicator.** `subject_race_hispanic` carries an odds ratio of 0.43 and an "
        "average marginal effect of −11.3 percentage points."
    )

# --------------------------------------------------------------------------- decision tree
st.divider()
st.subheader("A decision tree — the most interpretable model there is")
_dt = C.OUTPUTS / "decision_tree.json"
if not _dt.exists():
    st.info("Run `python -m src.interpret` to fit it.")
else:
    d = json.loads(_dt.read_text())
    rows = [{"max_depth": int(k), "AUC": v["auc"], "leaves": v["n_leaves"]}
            for k, v in d["depths"].items()]
    st.dataframe(pd.DataFrame(rows).style.format({"AUC": "{:.4f}"}),
                 hide_index=True, width="stretch")
    st.success(
        f"A **{d['depths'][str(d['best_depth'])]['n_leaves']}-leaf tree scores "
        f"{d['best_auc']:.4f}**, beating the logistic scorecard (0.5516) and closing most "
        "of the gap to gradient boosting (0.5663). The most interpretable model available "
        "is competitive — the argument for the white box, made with numbers."
    )
    with st.expander("MDI (impurity-based) feature importance"):
        st.dataframe(pd.Series(d["mdi"], name="MDI").to_frame().head(12)
                     .style.format("{:.4f}"), width="stretch")
    with st.expander("The tree itself (first 3 levels)"):
        st.code(d["rules"], language="text")
