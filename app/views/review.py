"""Officer review — the recommended use of the white box (slides 18-19), as a tool a supervisor can use.

Per officer, over a rolling 12-month window ending at the chosen quarter, the five items of slide 18:
  1. consent-search rate                     past-year consents / stops (master table, as of the officer's latest search)
  2. hit rate, shrunk, with an interval      the officer's consent searches, shrunk toward the force rate
  3. consent-search ratio by driver race     off_log_search_ratio_* (per stop, shrunk; master table)
  4. hit-rate gap by race                    the officer's own outcome test
  5. benchmark                               the race-blind PLTR's expected hit rate for the searches made

Reads committed files only; nothing is fitted here:
  data/nashville_consent_searches.parquet    searches, outcomes and officer features
  data/officer_model_predictions.parquet     race-blind PLTR (19-30 terms) scores on the 2016-2018 test searches
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import common

DATA = common.ROOT / "data"
K_SHRINK = 20            # pseudo-searches at the force rate, as in the officer features
FDR = 0.10               # Benjamini-Hochberg level across officers
RACES = ["white", "black", "hispanic"]


@st.cache_data(show_spinner="Loading searches and white-box scores…")
def load() -> pd.DataFrame:
    p = pd.read_parquet(DATA / "officer_model_predictions.parquet")
    p = p[(p["split"] == "temporal") & (p["mode"] == "full_time_officer_blind") & (p["model"] == "pltr_sparse")]
    m = pd.read_parquet(DATA / "nashville_consent_searches.parquet", columns=[
        "stop_id", "officer_id_hash", "date", "subject_race", "contraband_found", "off_consent_365d",
        "off_stops_365d", "off_log_search_ratio_black_white", "off_log_search_ratio_hisp_white"])
    d = p[["stop_id", "p"]].merge(m, on="stop_id")
    d["y"] = d["contraband_found"].astype(int)
    d["officer"] = d["officer_id_hash"].astype(str)
    return d


def review(w: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """One row per officer in the window.

    The FLAG rests on item 2, the officer's hit rate against the force's (exact one-sided binomial test,
    Benjamini-Hochberg across the officers reviewed), as in slide 15. Item 5, the white-box benchmark,
    is shown beside it but does not drive the flag: the race-blind PLTR takes the officer's own past hit
    rate as an input, so its expectation already knows the officer's record, and a gap to it reads
    "worse than their own record predicts", not "worse than the force".
    """
    # Recalibrate the benchmark in the window so the force as a whole sits at observed = expected.
    w = w.assign(e=np.clip(w["p"] * w["y"].sum() / w["p"].sum(), 1e-6, 1 - 1e-6))
    force = w["y"].mean()
    g = w.groupby("officer").agg(searches=("y", "size"), hits=("y", "sum"), expected=("e", "sum"))
    g["hit_rate"] = g["hits"] / g["searches"]
    g["vs_force_pp"] = 100 * (g["hit_rate"] - force)
    a, b = g["hits"] + K_SHRINK * force, g["searches"] - g["hits"] + K_SHRINK * (1 - force)
    g["shrunk"] = a / (a + b)
    g["lo"], g["hi"] = stats.beta.ppf(0.025, a, b), stats.beta.ppf(0.975, a, b)
    g["benchmark"] = g["expected"] / g["searches"]
    g["vs_benchmark_pp"] = 100 * (g["hit_rate"] - g["benchmark"])
    g["p_low"] = stats.binom.cdf(g["hits"], g["searches"], force)   # fewer finds than the force rate
    g = g[g["searches"] >= min_n].copy()
    if g.empty:
        return g.assign(q=[], flag=[])
    pv = g["p_low"].to_numpy()
    o = np.argsort(pv)
    q = np.empty(len(pv))
    q[o] = np.minimum.accumulate((pv[o] * len(pv) / np.arange(1, len(pv) + 1))[::-1])[::-1]
    g["q"] = np.minimum(q, 1)
    g["flag"] = g["q"] < FDR
    return g.sort_values("p_low")


def latest(w: pd.DataFrame, officer: str) -> pd.Series:
    return w[w["officer"] == officer].sort_values("date").iloc[-1]


d = load()
st.header("Officer review — the recommended use of the white box")
st.info(
    "**What MNPD would use** (slides 18–19): per officer, the five items of the review. An officer is "
    "**flagged** when their consent searches find contraband significantly less often than the force's "
    "(item 2, with false-discovery control across officers), as in slide 15. Item 5 adds what the "
    "**race-blind white box (PLTR)** expects for the searches they made. The tool does **not** score "
    "drivers and does **not** decide anything: it tells a supervisor whose practice to look at first."
)

c1, c2 = st.columns([2, 1])
quarters = [str(q) for q in pd.period_range("2016Q4", "2018Q4", freq="Q")]
qe = c1.select_slider("Review quarter (each rate covers the 12 months ending with it)", quarters,
                      value=quarters[-1])
min_n = c2.number_input("Minimum consent searches in the 12 months", 5, 50, 10, step=5,
                        help="Below this, an officer is listed as 'not enough searches to assess'.")
end = pd.Period(qe, freq="Q").end_time
w = d[(d["date"] > end - pd.DateOffset(months=12)) & (d["date"] <= end)]
g = review(w, int(min_n))
n_off = w["officer"].nunique()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Consent searches in the window", f"{len(w):,}")
m2.metric("Officers reviewed", f"{len(g)}", help=f"{n_off - len(g)} more made fewer than {min_n} searches")
m3.metric("Flagged for review", f"{int(g['flag'].sum())}",
          help=f"Hit rate significantly below the force's (one-sided exact binomial test), "
               f"Benjamini–Hochberg FDR {FDR:.0%}")
fruitless = w.loc[w["officer"].isin(g.index[g["flag"]]), "y"].eq(0).sum()
m4.metric("Their share of fruitless searches", f"{fruitless / max(w['y'].eq(0).sum(), 1):.0%}")

tab_all, tab_one = st.tabs(["All officers (supervisor view)", "One officer (the review)"])

# --------------------------------------------------------------------------- all officers
with tab_all:
    force = w["y"].mean()
    n_max = g["searches"].max() if len(g) else int(min_n) + 1
    n = np.linspace(max(int(min_n), 1), max(n_max, int(min_n) + 1), 200)
    fig = go.Figure()
    for zz, dash, lab in [(1.96, "dash", "95% limits"), (3.09, "dot", "99.8% limits")]:
        half = 100 * zz * np.sqrt(force * (1 - force) / n)
        fig.add_trace(go.Scatter(x=np.r_[n, n[::-1]], y=np.r_[half, -half[::-1]], mode="lines",
                                 line=dict(color="#8a8984", dash=dash, width=1), name=lab, hoverinfo="skip"))
    for flag, color, lab in [(False, "#b0b7bf", "not flagged"), (True, "#c0392b", "flagged")]:
        s = g[g["flag"] == flag]
        fig.add_trace(go.Scatter(
            x=s["searches"], y=s["vs_force_pp"], mode="markers", name=lab, marker=dict(color=color, size=8),
            text=[f"officer {o[:8]}<br>{r.hits:.0f}/{r.searches:.0f} found · q {r.q:.3f}"
                  for o, r in s.iterrows()], hoverinfo="text"))
    fig.update_layout(xaxis_title="Consent searches in the 12 months",
                      yaxis_title=f"Hit rate minus force rate {force:.1%} (pp)", height=420,
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Funnel plot: officers with few searches scatter widely by chance, so the limits narrow as "
               "searches grow. Below the limits = fewer finds than the force's rate, consistent with "
               "searching on weaker evidence. The flag also corrects for testing every officer at once, "
               "so a point just outside the 95% limits is not necessarily flagged.")
    tbl = g.assign(officer=[o[:8] for o in g.index],
                   **{"hit rate (shrunk) [95%]": [f"{s:.0%} [{lo:.0%}, {hi:.0%}]"
                                                  for s, lo, hi in zip(g["shrunk"], g["lo"], g["hi"])]})
    st.dataframe(tbl[["officer", "searches", "hits", "hit rate (shrunk) [95%]", "vs_force_pp", "q", "flag",
                      "benchmark", "vs_benchmark_pp"]]
                 .rename(columns={"vs_force_pp": "vs force (pp)", "benchmark": "white-box benchmark",
                                  "vs_benchmark_pp": "vs benchmark (pp)"})
                 .style.format({"vs force (pp)": "{:+.1f}", "q": "{:.3f}", "white-box benchmark": "{:.0%}",
                                "vs benchmark (pp)": "{:+.1f}"}),
                 hide_index=True, width="stretch")
    st.caption(f"Sorted from the strongest evidence of a shortfall. q = Benjamini–Hochberg adjusted p-value "
               f"(one-sided exact binomial test against the force rate {force:.1%}); flagged if q < {FDR:.2f}. "
               f"Shrinkage: {K_SHRINK} pseudo-searches at the force rate. The white-box benchmark already "
               "includes the officer's own record, so 'vs benchmark' reads *worse than their own record "
               "predicts*. Officer IDs are hashed; only the first 8 characters are shown.")

# --------------------------------------------------------------------------- one officer
with tab_one:
    choices = list(g.index)
    if not choices:
        st.info("No officer reaches the minimum number of searches in this window.")
    else:
        off = st.selectbox("Officer (hashed ID)", choices, format_func=lambda o: f"{o[:8]}"
                           + ("  — flagged" if g.loc[o, "flag"] else ""))
        r, last, mine = g.loc[off], latest(w, off), w[w["officer"] == off]
        med = w.groupby("officer").tail(1)

        st.markdown(f"#### Review of officer `{off[:8]}` — 12 months to {end:%d %b %Y}")
        k1, k2, k3 = st.columns(3)
        rate = last["off_consent_365d"] / max(last["off_stops_365d"], 1)
        force_rate = (med["off_consent_365d"] / med["off_stops_365d"].clip(lower=1)).median()
        k1.metric("1 · Consent-search rate", f"{rate:.1%}", help="Consent searches per stop over the past year, "
                  "as of the officer's latest search", delta=f"{100 * (rate - force_rate):+.1f} pp vs median officer",
                  delta_color="off")
        k2.metric("2 · Hit rate, shrunk", f"{r['shrunk']:.0%}", delta=f"{r['vs_force_pp']:+.1f} pp vs force",
                  help=f"95% interval {r['lo']:.0%}–{r['hi']:.0%}; raw {r['hits']:.0f}/{r['searches']:.0f}")
        k2.caption(f"95% interval {r['lo']:.0%}–{r['hi']:.0%} · raw {r['hits']:.0f} of {r['searches']:.0f} · "
                   f"q = {r['q']:.3f}" + (" · **flagged**" if r["flag"] else " · not flagged"))
        k3.metric("5 · White-box benchmark", f"{r['benchmark']:.0%}", delta=f"{r['vs_benchmark_pp']:+.1f} pp",
                  help="Race-blind PLTR's expected hit rate for the searches this officer made. It includes "
                       "the officer's own past record, so the gap reads 'vs their own record'.")
        k3.caption("Compared with what their own record and these stops predict; "
                   "slide 18 marks this item for the pilot.")

        st.markdown("**3 · Consent-search ratio by driver race** (per stop, vs white drivers, shrunk)")
        c1, c2 = st.columns(2)
        for col, key, lab in [(c1, "off_log_search_ratio_black_white", "Black / white"),
                              (c2, "off_log_search_ratio_hisp_white", "Hispanic / white")]:
            v, fm = np.exp(last[key]), np.exp(med[key]).median()
            col.metric(lab, f"{v:.2f}×", delta=f"median officer {fm:.2f}×", delta_color="off")

        st.markdown("**4 · Hit rate by driver race** (the officer's own outcome test)")
        by = mine.groupby("subject_race").agg(searches=("y", "size"), hits=("y", "sum")).reindex(RACES).fillna(0)
        by["hit rate"] = np.where(by["searches"] > 0, by["hits"] / by["searches"].clip(lower=1), np.nan)
        by["benchmark"] = mine.assign(e=mine["p"]).groupby("subject_race")["e"].mean().reindex(RACES)
        st.dataframe(by.style.format({"searches": "{:.0f}", "hits": "{:.0f}", "hit rate": "{:.0%}",
                                      "benchmark": "{:.0%}"}, na_rep="—"), width="stretch")
        if (by["searches"] < 10).any():
            st.caption("Groups with fewer than 10 searches: read as an indication only, not a finding.")

        # trend over the quarters
        rows = []
        for q in quarters:
            e_ = pd.Period(q, freq="Q").end_time
            ww = d[(d["date"] > e_ - pd.DateOffset(months=12)) & (d["date"] <= e_)]
            gg = review(ww, 1)
            if off in gg.index:
                rows.append({"quarter": q, "hit rate (shrunk)": gg.loc[off, "shrunk"],
                             "benchmark": gg.loc[off, "benchmark"], "searches": gg.loc[off, "searches"]})
        if rows:
            tr = pd.DataFrame(rows)
            fig = go.Figure([go.Scatter(x=tr["quarter"], y=tr["hit rate (shrunk)"], name="hit rate (shrunk)",
                                        mode="lines+markers", line=dict(color="#1f77d0")),
                             go.Scatter(x=tr["quarter"], y=tr["benchmark"], name="white-box benchmark",
                                        mode="lines+markers", line=dict(color="#8a8984", dash="dash"))])
            fig.update_layout(yaxis_tickformat=".0%", height=300, margin=dict(t=10),
                              legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig, width="stretch")
            st.caption("Each point is a rolling 12-month window: one quarter's searches alone are too few "
                       "to assess most officers (median officer: 4 consent searches in 2016–2018).")

st.warning(
    "**How this may be used** (slide 19): human review only, no automatic sanctions; the officer sees and "
    "can contest their report. A shortfall is consistent with a lower evidentiary bar, not proof of it "
    "(infra-marginality), and blank contraband fields were recorded as \"nothing found\". Scores cover "
    "2016–2018, the test years the white box never saw."
)
