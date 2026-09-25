"""Dataset — every stop on a map, with filters (where, when, who, search type).

Data: data/processed/stops.parquet (scripts/build_stops_parquet.py), built with the same
definitions as the analysis: 2010-2018, and search_type resolved by src.data.add_search_type.

Speed:
- The map is a custom component (app/stops_map/index.html). At first start every mapped stop is
  exported as byte-packed columns to app/static/stops/, which the browser fetches once. After that,
  a filter change only sends the filter values to the map, which filters the rows itself and updates
  in place (no reload, pan/zoom kept).
- The metrics and charts use integer codes precomputed at load (a NumPy mask + `bincount` per
  change), cached per filter combination.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import common
from src import config as C
from src import fairness as F

APP = Path(__file__).resolve().parents[1]
DATA = common.ROOT / "data" / "processed" / "stops.parquet"
BROWSER_DATA = APP / "static" / "stops"  # served at /app/static/stops/ (see .streamlit/config.toml)
BBOX = {"lat0": 35.90, "lat1": 36.50, "lng0": -87.15, "lng1": -86.45}  # same box as build_stops_parquet.py
DAY0 = np.datetime64(C.DATE_MIN, "D")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CATEGORIES = ["subject_race", "subject_sex", "violation", "outcome", "precinct", "search_type"]
FLAGS = ["arrest_made", "citation_issued", "search_conducted", "frisk_performed", "contraband_found"]
SEARCH_TYPES = ["not searched"] + [t for t in C.SEARCH_TYPE_ORDER if t != "unresolved"]

RACE_CHART_COLORS = common.RACE_COLORS


# ---------------------------------------------------------------- Data + precomputed codes
@st.cache_resource(show_spinner="Loading 3M stops…")
def load_data() -> tuple[pd.DataFrame, dict, pd.DataFrame, str]:
    df = pd.read_parquet(DATA)
    df["search_type"] = df["search_type"].cat.set_categories(SEARCH_TYPES)
    arrays = {
        "day": df["date"].to_numpy().astype("datetime64[D]").astype(np.int32),
        "month": ((df["date"].dt.year - DAY0.astype(object).year) * 12
                  + df["date"].dt.month - 1).to_numpy(np.int16),
        "weekday": df["date"].dt.dayofweek.to_numpy(np.int8),
        "hour": df["hour"].fillna(-1).to_numpy(np.int8),
        "age": df["subject_age"].fillna(-1).to_numpy(np.float32),
    }
    for col in CATEGORIES:
        arrays[col] = df[col].cat.codes.to_numpy(np.int8)  # -1 = missing
    for col in FLAGS:
        arrays[col] = df[col].astype("Float32").to_numpy(np.float32, na_value=np.nan)
    # ~100m grid cells for the map; -1 = no usable coordinates (counted everywhere, just not drawn).
    lat, lng = df["lat"].to_numpy(np.float64), df["lng"].to_numpy(np.float64)
    mapped = ~np.isnan(lat) & ~np.isnan(lng)
    keys = np.round(lat[mapped] * 1000).astype(np.int64) * 1_000_000 + np.round(lng[mapped] * 1000).astype(np.int64) + 180_000
    codes, uniques = pd.factorize(keys)
    arrays["cell"] = np.full(len(df), -1, dtype=np.int32)
    arrays["cell"][mapped] = codes
    cells = pd.DataFrame({"lat": (uniques // 1_000_000) / 1000, "lng": (uniques % 1_000_000 - 180_000) / 1000})
    version = export_for_browser(df, arrays, cells, mapped)
    return df, arrays, cells, version


def export_for_browser(df: pd.DataFrame, arrays: dict, cells: pd.DataFrame, mapped: np.ndarray) -> str:
    """Write every mapped stop as byte-packed columns (one gzipped file) + meta.json for the map component.

    Rows are sorted by grid cell and date, which makes the file compress ~4x better (~50 MB -> ~12 MB).
    Skipped when the export is already up to date with the Parquet file.
    """
    version = f"{int(DATA.stat().st_mtime)}-3"  # bump the suffix when the export format changes
    meta_path = BROWSER_DATA / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("version") == version:
        return version

    assert len(cells) < 2**16, "cell ids must fit in uint16"
    rows = np.flatnonzero(mapped)
    order = rows[np.lexsort((arrays["day"][rows], arrays["cell"][rows]))]
    lat_step = (BBOX["lat1"] - BBOX["lat0"]) / 65535
    lng_step = (BBOX["lng1"] - BBOX["lng0"]) / 65535
    flags = ((arrays["arrest_made"] == 1).astype(np.uint8)
             | ((arrays["search_conducted"] == 1).astype(np.uint8) << 1)
             | ((arrays["contraband_found"] == 1).astype(np.uint8) << 2))
    lat = np.nan_to_num(df["lat"].to_numpy(np.float64), nan=BBOX["lat0"])
    lng = np.nan_to_num(df["lng"].to_numpy(np.float64), nan=BBOX["lng0"])
    columns = {  # uint16 columns first so every array stays 2-byte aligned
        "lat": np.round((lat - BBOX["lat0"]) / lat_step).astype(np.uint16),
        "lng": np.round((lng - BBOX["lng0"]) / lng_step).astype(np.uint16),
        "cell": arrays["cell"].astype(np.uint16),
        "day": (arrays["day"] - DAY0.astype(np.int32)).astype(np.uint16),
        "hour": arrays["hour"].astype(np.uint8),                       # 255 = missing
        "weekday": arrays["weekday"].astype(np.uint8),
        "age": np.where(arrays["age"] < 0, 0, arrays["age"]).astype(np.uint8),  # 0 = missing
        **{col: arrays[col].astype(np.uint8) for col in CATEGORIES},   # 255 = missing
        "flags": flags,  # bit 0 arrest, bit 1 search, bit 2 contraband
    }
    BROWSER_DATA.mkdir(parents=True, exist_ok=True)
    meta_columns, offset = [], 0
    with gzip.open(BROWSER_DATA / "stops.bin.gz", "wb", compresslevel=6) as out:
        for name, values in columns.items():
            out.write(values[order].tobytes())
            meta_columns.append({"name": name, "dtype": str(values.dtype), "offset": offset})
            offset += len(order) * values.itemsize
    meta = {
        "version": version, "n": len(order), "bytes": offset, "columns": meta_columns,
        "day0": int(DAY0.astype(np.int32)),
        "quant": {"lat0": BBOX["lat0"], "lng0": BBOX["lng0"], "lat_step": lat_step, "lng_step": lng_step},
        "categories": {col: list(df[col].cat.categories) for col in CATEGORIES},
        "cells": {"lat": cells["lat"].round(3).tolist(), "lng": cells["lng"].round(3).tolist()},
    }
    meta_path.write_text(json.dumps(meta))
    return version


def build_mask(f: dict) -> np.ndarray:
    a = ARRAYS
    mask = np.ones(len(a["day"]), dtype=bool)
    lo, hi = f["dates"]
    if (lo, hi) != (DAY_MIN, DAY_MAX):
        mask &= (a["day"] >= lo) & (a["day"] <= hi)
    if f["hours"] != (0, 23):
        mask &= (a["hour"] >= f["hours"][0]) & (a["hour"] <= f["hours"][1])
    if f["weekdays"] is not None:
        mask &= np.isin(a["weekday"], f["weekdays"])
    if f["ages"] != (10, 99):
        mask &= (a["age"] >= f["ages"][0]) & (a["age"] <= f["ages"][1])
    for col in CATEGORIES:
        if f[col] is not None:
            allowed = np.zeros(len(DF[col].cat.categories) + 1, dtype=bool)  # last slot = code -1 (NA)
            allowed[list(f[col])] = True
            mask &= allowed[a[col]]
    for col in ("arrest_made", "search_conducted", "contraband_found"):
        if f["only_" + col]:
            mask &= a[col] == 1
    return mask


def rate(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


@st.cache_data(max_entries=256, show_spinner=False)
def summarize(f: dict) -> dict:
    """Metrics and chart tables for one filter combination (small, so cheap to cache)."""
    a, mask = ARRAYS, build_mask(f)
    races = list(DF["subject_race"].cat.categories)
    race = a["subject_race"][mask].astype(np.int32)
    has_race = race >= 0
    searched = a["search_conducted"][mask] == 1

    hour = a["hour"][mask].astype(np.int32)
    ok = has_race & (hour >= 0)
    by_hour = np.bincount(hour[ok] * len(races) + race[ok], minlength=24 * len(races)).reshape(24, -1)
    by_month = np.bincount(a["month"][mask][has_race].astype(np.int32) * len(races) + race[has_race],
                           minlength=(ARRAYS["month"].max() + 1) * len(races)).reshape(-1, len(races))

    rates = pd.DataFrame({"subject_race": races, "stops": np.bincount(race[has_race], minlength=len(races))})
    for col in ("arrest_made", "citation_issued", "search_conducted", "frisk_performed"):
        v = a[col][mask]
        good = has_race & ~np.isnan(v)
        rates[col.split("_")[0] + "_rate"] = (np.bincount(race[good], weights=v[good], minlength=len(races))
                                             / np.bincount(race[good], minlength=len(races)).clip(1))

    # The headline table from Problem definition, recomputed on the filtered searches.
    search_rows = DF.loc[mask & (a["search_conducted"] == 1), ["subject_race", "search_type", C.TARGET]]
    try:
        pooling = F.pooled_vs_stratified(search_rows)
    except KeyError:  # the reference group (white) is missing from a stratum under these filters
        pooling = None

    return {
        "n": int(mask.sum()),
        "n_mapped": int((a["cell"][mask] >= 0).sum()),
        "arrest_rate": rate(a["arrest_made"][mask]),
        "citation_rate": rate(a["citation_issued"][mask]),
        "search_rate": rate(a["search_conducted"][mask]),
        "hit_rate": rate(a["contraband_found"][mask][searched]),
        "by_hour": pd.DataFrame(by_hour, columns=races).rename_axis("hour").reset_index()
        .melt(id_vars="hour", var_name="subject_race", value_name="stops"),
        "by_month": pd.DataFrame(by_month, columns=races,
                                 index=pd.date_range(C.DATE_MIN, periods=len(by_month), freq="MS"))
        .rename_axis("date").reset_index().melt(id_vars="date", var_name="subject_race", value_name="stops"),
        "rates": rates[rates["stops"] > 0].reset_index(drop=True),
        "pooling": pooling,
        "head": DF.iloc[np.flatnonzero(mask)[:1000]],
    }


# ---------------------------------------------------------------- Sidebar filters
if not DATA.exists():
    st.error(f"`{DATA.relative_to(common.ROOT)}` not found. Run `python scripts/download_data.py`, "
             "then `python scripts/build_stops_parquet.py`.")
    st.stop()

DF, ARRAYS, CELLS, DATA_VERSION = load_data()
DAY_MIN, DAY_MAX = int(ARRAYS["day"].min()), int(ARRAYS["day"].max())


def to_day(d) -> int:
    return int(np.datetime64(d, "D").astype(np.int32))


def multiselect_filter(label: str, column: str, help: str | None = None) -> tuple[int, ...] | None:
    """Selected category codes, or None when everything is selected (so rows with NA are kept)."""
    options = list(DF[column].cat.categories)
    selected = st.sidebar.multiselect(label, options, default=options, key=column, help=help)
    return None if len(selected) == len(options) else tuple(options.index(s) for s in selected)


st.sidebar.header("Filters")
min_date, max_date = DF["date"].min().date(), DF["date"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
hour_range = st.sidebar.slider("Time of stop (hour of day)", 0, 23, (0, 23), format="%d:00")
weekdays = st.sidebar.multiselect("Day of week", DAYS, default=DAYS)

st.sidebar.subheader("Driver")
races = multiselect_filter("Race", "subject_race")
sexes = multiselect_filter("Sex", "subject_sex")
age_range = st.sidebar.slider("Age", 10, 99, (10, 99))

st.sidebar.subheader("Stop")
search_types = multiselect_filter(
    "Search type", "search_type",
    help="Resolved as in Problem definition: the most mechanical basis wins, so 'consent' means "
         "a purely discretionary search.",
)
violations = multiselect_filter("Violation", "violation")
outcomes = multiselect_filter("Outcome", "outcome")
precincts = multiselect_filter("Precinct", "precinct")
only_arrests = st.sidebar.toggle("Arrests only")
only_searches = st.sidebar.toggle("Searches only")
only_contraband = st.sidebar.toggle("Contraband found only")

filters = {
    "dates": (to_day(date_range[0]), to_day(date_range[1])) if len(date_range) == 2 else (DAY_MIN, DAY_MAX),
    "hours": tuple(hour_range),
    "weekdays": None if len(weekdays) == 7 else tuple(DAYS.index(d) for d in weekdays),
    "ages": tuple(age_range),
    "subject_race": races, "subject_sex": sexes, "violation": violations,
    "outcome": outcomes, "precinct": precincts, "search_type": search_types,
    "only_arrest_made": only_arrests, "only_search_conducted": only_searches,
    "only_contraband_found": only_contraband,
}

# ---------------------------------------------------------------- Metrics + map
st.header("Dataset")
st.markdown(
    "**Stanford Open Policing Project — Nashville, TN (Metropolitan Nashville PD).** "
    f"{len(DF):,} traffic stops from 2010 to 2018 (the partial 2019 tail is dropped), of which "
    f"{int((ARRAYS['search_conducted'] == 1).sum()):,} ended in a search: the only stops where "
    "`contraband_found` is observed. Licence: Open Data Commons Attribution."
)
st.subheader("Where and when stops happen")


def show_metrics(s: dict, container) -> None:
    def pct(x: float, digits: int) -> str:
        return "–" if np.isnan(x) else f"{x:.{digits}%}"

    cols = container.columns(5)
    cols[0].metric("Stops", f"{s['n']:,}")
    cols[1].metric("Arrest rate", pct(s["arrest_rate"], 2))
    cols[2].metric("Citation rate", pct(s["citation_rate"], 1))
    cols[3].metric("Search rate", pct(s["search_rate"], 2))
    cols[4].metric("Hit rate, all searches pooled", pct(s["hit_rate"], 1),
                   help="Contraband found / searches. Pooled over search types, which hides the consent "
                        "disparity; see the 'Hit rate: consent vs. rest' tab and Problem definition.")


def map_filters(f: dict) -> dict:
    """The filters in the map component's terms (None = no filter on that field)."""
    day0 = int(DAY0.astype(np.int32))
    lo, hi = f["dates"]
    return {
        "days": None if (lo, hi) == (DAY_MIN, DAY_MAX) else [lo - day0, hi - day0],
        "hours": None if f["hours"] == (0, 23) else list(f["hours"]),
        "weekdays": f["weekdays"],
        "ages": None if f["ages"] == (10, 99) else list(f["ages"]),
        "cats": {col: f[col] for col in CATEGORIES},
        "flags": f["only_arrest_made"] | f["only_search_conducted"] << 1 | f["only_contraband_found"] << 2,
    }


# The map goes first so it gets the new filters before the metrics/charts are computed.
metrics_box = st.empty()
common.stops_map(filters=map_filters(filters), version=DATA_VERSION, key="stops_map", default=None)
map_note = st.empty()
summary = summarize(filters)
show_metrics(summary, metrics_box.container())
map_note.caption(
    "Layer, colors and the hour-by-hour playback are controlled on the map itself and apply instantly; "
    "the hour slider on the map overrides the sidebar's hour range, for the map only. "
    f"{summary['n'] - summary['n_mapped']:,} of these stops have no usable location: they are counted in "
    "the numbers but not drawn."
)

# ---------------------------------------------------------------- Charts
tab_pool, tab_hour, tab_trend, tab_groups, tab_table = st.tabs(
    ["Hit rate: consent vs. rest", "Time of day", "Over time", "Outcomes by race", "Data"]
)

with tab_pool:
    st.markdown(
        "The headline table from Problem definition, recomputed on the stops selected by the filters. "
        "Use it to check whether the reversal holds within a precinct, a period or a time of day."
    )
    pooling = summary["pooling"]
    if pooling is None or pooling.empty:
        st.info("Needs searches of white drivers (the reference group) in both the consent and non-consent "
                "strata. Widen the filters.")
    else:
        fig = px.bar(
            pooling, x="stratum", y="hit_rate", color="subject_race", barmode="group",
            color_discrete_map=RACE_CHART_COLORS, text_auto=".1%",
            hover_data={"searches": ":,", "hits": ":,", "gap_vs_ref_pp": ":+.2f"},
            labels={"stratum": "", "hit_rate": "Hit rate (contraband found / searches)",
                    "subject_race": "Race", "gap_vs_ref_pp": "gap vs white (pp)"},
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(bargap=0.3, bargroupgap=0.1)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(
            pooling.style.format({"searches": "{:,}", "hits": "{:,}", "hit_rate": "{:.2%}",
                                  "gap_vs_ref_pp": "{:+.2f}"}),
            hide_index=True, width="stretch",
        )
        if pooling["searches"].min() < 200:
            st.warning("Some cells have fewer than 200 searches: treat those hit rates as noisy.")
    st.page_link("views/problem.py", label="Why pooling hides the disparity → Problem definition", icon="📊")

with tab_hour:
    by_hour = summary["by_hour"].copy()
    by_hour = by_hour[by_hour.groupby("subject_race")["stops"].transform("sum") > 0]
    by_hour["share of group's stops"] = by_hour["stops"] / by_hour.groupby("subject_race")["stops"].transform("sum")
    normalize = st.toggle("Normalize within each race (compare daily patterns)", value=True)
    fig = px.line(
        by_hour, x="hour", y="share of group's stops" if normalize else "stops",
        color="subject_race", color_discrete_map=RACE_CHART_COLORS, markers=True,
        labels={"hour": "Hour of day", "subject_race": "Race"},
    )
    fig.update_traces(line_width=2)
    fig.update_layout(hovermode="x unified", xaxis=dict(dtick=2))
    if normalize:
        fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.caption("The dusk period is often used for the 'veil of darkness' test: if the share of Black "
               "drivers stopped drops after dark (when race is harder to see), that suggests bias in daylight stops.")

with tab_trend:
    freq = st.radio("Granularity", ["Month", "Year"], horizontal=True)
    trend = summary["by_month"]
    if freq == "Year":
        trend = (trend.assign(date=trend["date"].dt.to_period("Y").dt.to_timestamp())
                 .groupby(["date", "subject_race"], as_index=False)["stops"].sum())
    trend = trend[trend.groupby("subject_race")["stops"].transform("sum") > 0]
    fig = px.line(trend, x="date", y="stops", color="subject_race", color_discrete_map=RACE_CHART_COLORS,
                  labels={"date": "", "subject_race": "Race"})
    fig.update_traces(line_width=2)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

with tab_groups:
    rates = summary["rates"]
    metric = st.selectbox("Rate", ["arrest_rate", "search_rate", "citation_rate", "frisk_rate"],
                          format_func=lambda m: m.replace("_", " ").capitalize())
    fig = px.bar(rates.sort_values(metric), x=metric, y="subject_race", orientation="h",
                 color="subject_race", color_discrete_map=RACE_CHART_COLORS, text_auto=".2%",
                 hover_data={"stops": ":,"}, labels={"subject_race": "", metric: metric.replace("_", " ")})
    fig.update_layout(showlegend=False, bargap=0.35)
    fig.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        rates.style.format({"stops": "{:,}", **{c: "{:.2%}" for c in rates.columns if c.endswith("rate")}}),
        hide_index=True, width="stretch",
    )

with tab_table:
    st.dataframe(summary["head"], width="stretch", hide_index=True)
    st.caption(f"First {len(summary['head']):,} of {summary['n']:,} filtered stops.")
