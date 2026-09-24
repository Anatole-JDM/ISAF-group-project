"""Load the Nashville stops file and build the labelled search sample.

The raw CSV is 1.04 GB. We stream it in chunks and keep only the 127,705 rows
where a search happened, because those are the only rows where the target is
observed. That filtering step IS the selective labels problem — see README.
"""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

from . import config as C

TRUEISH = {"TRUE", "True", "true", "1", "Y", "yes"}


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().isin(TRUEISH)


def _stringify_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Cast mixed-type object columns to string.

    Several columns (precinct, zone, reporting_area) mix strings with NaN floats
    in the raw file, which makes the parquet cache write fail. Normalising here
    keeps them categorical, which is what they are.
    """
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype("string")
    return out


# --------------------------------------------------------------------------- load
def load_searches(chunksize: int = 250_000, cache: bool = True) -> pd.DataFrame:
    """Return the labelled sample: one row per search, with the target observed.

    Verified shape: 127,705 searches out of 3,092,351 stops (4.13%).
    """
    if cache and C.LABELLED_PARQUET.exists():
        return pd.read_parquet(C.LABELLED_PARQUET)

    if not C.STOPS_ZIP.exists():
        raise FileNotFoundError(
            f"{C.STOPS_ZIP} missing — run `python scripts/download_data.py` first."
        )

    keep = []
    with zipfile.ZipFile(C.STOPS_ZIP) as zf, zf.open(C.STOPS_CSV_INNER) as fh:
        for chunk in pd.read_csv(fh, chunksize=chunksize, low_memory=False):
            keep.append(chunk[_truthy(chunk["search_conducted"])])

    df = pd.concat(keep, ignore_index=True)
    df = add_search_type(df)
    df = add_time_features(df)
    df = restrict_period(df)
    df = _stringify_objects(df)

    C.DATA_PROC.mkdir(parents=True, exist_ok=True)
    if cache:
        df.to_parquet(C.LABELLED_PARQUET, index=False)
    return df


# --------------------------------------------------------------------------- strata
def add_search_type(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve search_basis into a single, unambiguous search type.

    `search_basis` leaves 25,620 searches (20%) as "other". Audited against the
    raw flags, every one of them is arrest / warrant / inventory — all mechanical.
    Verified decomposition (2026-09-24):
        arrest 19,722 | arrest+inventory 2,886 | arrest+warrant 1,510
        inventory 965 | warrant 446 | arrest+warrant+inventory 73
        warrant+inventory 18                      -> 25,620, zero unresolved.

    Priority runs most-mechanical first, so `consent` is only assigned when NO
    mechanical basis applies. That makes "consent" mean *purely discretionary*,
    which is what the fairness argument requires. Note 69,430 rows carry
    raw_search_consent=TRUE but only 67,629 resolve to consent — the difference
    is consent co-occurring with a mechanical basis, which we do not count as
    discretionary.
    """
    out = df.copy()
    basis = out["search_basis"].astype("string").str.strip().str.lower()

    warrant = _truthy(out["raw_search_warrant"])
    arrest = _truthy(out["raw_search_arrest"])
    inventory = _truthy(out["raw_search_inventory"])
    plain = _truthy(out["raw_search_plain_view"]) | (basis == "plain view")
    pc = basis == "probable cause"
    consent = _truthy(out["raw_search_consent"]) | (basis == "consent")

    st = pd.Series("unresolved", index=out.index, dtype="object")
    for mask, label in [
        (consent, "consent"),          # assigned first, overwritten by anything mechanical
        (pc, "probable cause"),
        (plain, "plain view"),
        (inventory, "inventory"),
        (arrest, "arrest"),
        (warrant, "warrant"),
    ]:
        st[mask] = label

    out["search_type"] = st
    out["is_discretionary"] = out["search_type"].eq(C.SEARCH_TYPE_DISCRETIONARY)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out["date"], errors="coerce")
    out["date"] = d
    out["year"] = d.dt.year
    out["month"] = d.dt.month
    out["dow"] = d.dt.dayofweek
    hh = out["time"].astype("string").str.slice(0, 2)
    out["hour"] = pd.to_numeric(hh, errors="coerce")
    return out


def restrict_period(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the partial 2019 tail (data ends 24 March 2019)."""
    m = df["date"].between(C.DATE_MIN, C.DATE_MAX)
    return df.loc[m].copy()


def sklearn_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable dtypes to numpy-backed ones.

    sklearn's SimpleImputer finds missing values with `X != X`, which raises
    "boolean value of NA is ambiguous" on StringDtype / Int64 columns. The
    parquet cache round-trips as nullable dtypes, so convert at the boundary:
    string -> object with np.nan, nullable int/bool -> float64.
    """
    out = df.copy()
    for c in out.columns:
        dt = out[c].dtype
        if not pd.api.types.is_extension_array_dtype(dt):
            continue
        if pd.api.types.is_integer_dtype(dt) or pd.api.types.is_bool_dtype(dt):
            out[c] = out[c].astype("float64")
        else:
            out[c] = out[c].astype(object).where(out[c].notna(), np.nan)
    return out


# --------------------------------------------------------------------------- frame
def build_xy(
    df: pd.DataFrame,
    search_type: str | None = "consent",
    include_questionable: bool = False,
    include_protected: bool = True,
    y_coding: str = C.Y_NATIVE,
):
    """Return (X, y, meta) for one stratum.

    search_type=None pools every search type. Pooling is what produces the
    misleading headline result (see fairness.pooled_vs_stratified) — pass None
    only to demonstrate that, never as the primary analysis.
    """
    d = df if search_type is None else df.loc[df["search_type"].eq(search_type)]
    d = d.copy()

    y_raw = _truthy(d[C.TARGET]).astype(int)
    y = y_raw if y_coding == C.Y_NATIVE else 1 - y_raw

    # search_type / is_discretionary are DERIVED from the stratifiers. Leaving
    # them in X would hand the pooled model a near-perfect proxy for the search
    # type (mechanical searches hit at 27%, discretionary at 16%).
    drop = set(C.EXCLUDE_DEFAULT) | {
        C.TARGET, "date", "time", "location", "lat", "lng",
        "search_type", "is_discretionary",
    }
    if include_questionable:
        drop -= set(C.QUESTIONABLE)

    # RUN BOTH AND REPORT BOTH.
    # With race included, a contraband-optimising model searches WHITE drivers
    # more, because white drivers have the higher hit rate in discretionary
    # searches (25.5% vs 19.5% on the 2016+ test set) -- the reverse of actual
    # officer behaviour. That is a real result and it is the outcome test
    # expressed as a model. But "fairness through unawareness" is the obvious
    # objection, so the race-blind run is needed to show what race is doing.
    # Note that dropping race does NOT make the model race-neutral: precinct
    # and zone are strong proxies in a segregated city.
    if not include_protected:
        drop |= {c for c in C.PROTECTED if c != "subject_age"}

    X = sklearn_safe(d.drop(columns=[c for c in drop if c in d.columns], errors="ignore"))
    meta = d[["officer_id_hash", "precinct", "year", "search_type", C.PRIMARY_PROTECTED]].copy()
    return X, y, meta
