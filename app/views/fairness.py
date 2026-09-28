"""Fairness testing — the course taxonomy for each model, both Y codings."""
from __future__ import annotations

import numpy as np
import streamlit as st

import common
from model_page import ARMS
from src import config as C
from src import fairness as F
import pandas as pd
import plotly.express as px

# Open on a run that contains all three arms (`matched`), the comparison the slides report.
scores, meta = common.run_selector(require_arm="tabpfn")

st.header("Fairness testing")
st.markdown(
    f"""
`Y = 1` contraband found · `Ŷ = 1` model recommends a search · `D` driver race.
Favourable for the individual is **{C.FAVORABLE_FOR_INDIVIDUAL.replace('_', ' ')}**
— the opposite of the course convention that `Y = 1` is favourable, so metrics
are shown under both codings.
    """
)
k = st.slider("Search capacity K (top-K by score), the same for every model", 100,
              len(scores), min(2000, len(scores)), step=100, key="fair_k")
st.error(
    "**FPR gap is the harm metric.** It is the share of *innocent* drivers "
    "the model would search, by race. Read it before AUC."
)


def fairness_report(arm: str) -> None:
    s = scores[f"score_{arm}"].to_numpy()
    yhat = np.zeros(len(s), dtype=int)
    yhat[np.argsort(-s)[:k]] = 1
    sub = scores["subject_race"].isin(C.RACE_REPORTABLE)

    rates = F.group_rates(scores["y"][sub], yhat[sub.to_numpy()],
                          scores["subject_race"][sub], y_score=s[sub.to_numpy()])
    st.subheader("Per-group rates")
    st.dataframe(rates.style.format(common.RATE_FMT, na_rep="—"), width="stretch")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Independence** — statistical parity")
        st.dataframe(F.independence(rates).to_frame("gap").style.format("{:+.2%}"),
                     width="stretch")
    with c2:
        st.markdown("**Separation** — equal opportunity / predictive equality")
        st.dataframe(F.separation(rates).style.format("{:+.2%}"), width="stretch")
    with c3:
        st.markdown("**Sufficiency** — predictive parity (outcome test)")
        st.dataframe(F.sufficiency(rates).to_frame("gap").style.format("{:+.2%}"),
                     width="stretch")

    with st.expander("Both Y codings — the convention inversion"):
        both = F.both_codings(scores["y"][sub], yhat[sub.to_numpy()],
                              scores["subject_race"][sub])
        cc1, cc2 = st.columns(2)
        cc1.markdown(f"`{C.Y_NATIVE}` — Y=1 contraband found")
        cc1.dataframe(both[C.Y_NATIVE].style.format(common.RATE_FMT, na_rep="—"))
        cc2.markdown(f"`{C.Y_COURSE}` — Y=1 innocent (course convention)")
        cc2.dataframe(both[C.Y_COURSE].style.format(common.RATE_FMT, na_rep="—"))


arms = [common.arm_name(c) for c in common.score_cols(scores)]
for arm, tab in zip(arms, st.tabs([ARMS.get(a, a) for a in arms])):
    with tab:
        fairness_report(arm)
if "tabpfn" not in arms:
    st.caption("No tab for the tabular foundation model (TabPFN): it is fitted only in the `matched` "
               "training mode, where the `tabpfn` package is installed. See Tabular foundation models.")


# --------------------------------------------------------------------------- hypothesis tests
st.divider()
st.subheader("Is the disparity statistically significant? χ² and equivalence testing")

_arms = [common.arm_name(c) for c in common.score_cols(scores)]
_arm = st.selectbox("Arm to test", _arms, key="fair_arm")
_K = int(k)   # the slider above; this section used to read a key that was never set
_s = scores[f"score_{_arm}"].to_numpy()
_yhat = np.zeros(len(_s), int)
_yhat[np.argsort(-_s)[: min(_K, len(_s))]] = 1

_c1, _c2 = st.columns(2)
with _c1:
    st.markdown("**χ² test of independence** — H₀: selection is independent of race")
    r = F.chi2_parity_test(_yhat, scores["subject_race"])
    st.metric("χ²", f"{r['chi2']:.1f}", help=f"dof {r['dof']}, n = {r['n']:,}")
    st.metric("p-value", f"{r['p_value']:.2e}")
    st.caption(
        f"White {r['selection_rate']['white']:.1%} vs black "
        f"{r['selection_rate']['black']:.1%}, gap {r['gap_pp']:+.1f}pp. "
        "Statistical parity is rejected outright."
    )
with _c2:
    st.markdown("**TOST equivalence** — how much unfairness would you have to accept?")
    sw = F.tost_delta_sweep(_yhat, scores["subject_race"])
    ok = sw[sw["equivalent"]]
    d_star = float(ok["delta"].iloc[0]) if len(ok) else float("nan")
    st.metric("Smallest δ certifying fairness", f"{d_star:.2f}" if d_star == d_star else "never")
    st.caption(
        "χ² on large n rejects almost anything, so rejection alone is weak. Equivalence "
        "testing inverts the burden: how wide a tolerance δ would you need before this "
        "model passes? The deck never assigns δ a value, so we sweep it."
    )

fig = px.line(sw, x="delta", y=["z_lower", "z_upper"],
              labels={"value": "z", "delta": "equivalence margin δ"})
fig.add_hline(y=float(sw["z_crit"].iloc[0]), line_dash="dot", line_color="#8a8984",
              annotation_text="critical value")
if d_star == d_star:
    fig.add_vline(x=d_star, line_color="#c0392b",
                  annotation_text=f"certifies at δ={d_star:.2f}")
st.plotly_chart(fig, width="stretch")
st.error(
    f"**You would have to declare a {d_star:.0%} selection gap acceptable before this model "
    "certifies as fair.** That is the magnitude of the unfairness, not merely its "
    "significance." if d_star == d_star else
    "The model does not certify as equivalent at any δ up to 0.50."
)


# --------------------------------------------------------------------------- FPDP
st.divider()
st.subheader("Which variables drive the disparity? Fairness partial dependence (FPDP)")
_fp_p = C.OUTPUTS / "fpdp_pdp.json"
if not _fp_p.exists():
    st.info("Not computed yet: `outputs/fpdp_pdp.json`.")
else:
    import json as _json

    _fp = _json.loads(_fp_p.read_text())
    st.markdown(
        """
Step 2 of the course's fairness method: freeze one feature at a single value for every driver,
rescore with the already-fitted model, and recompute the χ² parity test. A **candidate variable** is
one whose freezing brings the test back above p = 0.05, i.e. a variable through which the disparity
flows. Each feature is frozen at its median, 10th and 90th percentile (mode for categories); the
best result is kept.
"""
    )
    st.warning(
        "**Model tested:** this was run on the race-blind model *with officer-behaviour features* "
        "(see Officer features (comparison)), not on the models in the tabs above, whose run is "
        "not available yet. Read it as a check on where a disparity comes from once race is "
        "removed, not as a result for the recommended model."
    )
    _t = pd.DataFrame(_fp["fpdp"]).sort_values("chi2")
    _n_rep = int(_t["repairs"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("χ² with nothing frozen", f"{_fp['base_chi2']:.1f}")
    c2.metric("Best single freeze", f"{_t['chi2'].iloc[0]:.1f}", help=f"freezing `{_t['feature'].iloc[0]}`")
    c3.metric("Candidate variables", f"{_n_rep} of {len(_t)}")
    _fig = px.bar(_t.iloc[::-1], x="chi2", y="feature", orientation="h",
                  color=_t.iloc[::-1]["repairs"].map({True: "repairs parity", False: "does not repair"}),
                  color_discrete_map={"repairs parity": "#2c7fb8", "does not repair": "#c0392b"},
                  labels={"chi2": "χ² after freezing the feature", "color": ""})
    _fig.add_vline(x=_fp["base_chi2"], line_dash="dot", line_color="#8a8984",
                   annotation_text="nothing frozen")
    _fig.update_layout(height=max(350, 22 * len(_t)))
    st.plotly_chart(_fig, width="stretch")
    st.dataframe(_t[["feature", "chi2", "p", "chi2_drop", "repairs"]]
                 .rename(columns={"chi2_drop": "χ² drop", "chi2": "χ²", "p": "p-value"})
                 .style.format({"χ²": "{:.1f}", "p-value": "{:.2e}", "χ² drop": "{:.1f}"}),
                 hide_index=True, width="stretch")
    st.error(
        f"**{_n_rep} of {len(_t)} features repair parity on their own.** The disparity is not "
        "carried by one variable that could be neutralised (as in the course's German Credit "
        "example); it is spread across the whole feature set, so no single-variable mitigation "
        "fixes it." if _n_rep == 0 else
        f"**{_n_rep} candidate variable(s)** carry the disparity: see the blue bars."
    )
