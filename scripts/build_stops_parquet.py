"""Build data/processed/stops.parquet: every stop, slim columns, for the app's Explore page.

    python scripts/download_data.py        # once
    python scripts/build_stops_parquet.py  # ~1 min

Uses the same definitions as the analysis (src/): the 2010-2018 period, and the
search type resolved by data.add_search_type (most mechanical basis wins, so
"consent" means purely discretionary). Stops with no usable coordinates are kept
(lat/lng set to NaN) so every count matches the analysis; they just aren't mapped.
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import data as D  # noqa: E402

OUTPUT = C.DATA_PROC / "stops.parquet"
# Davidson County box; points outside it are mis-geocoded (some land in other states).
BBOX = {"lat": (35.90, 36.50), "lng": (-87.15, -86.45)}

RAW_SEARCH_FLAGS = ["raw_search_consent", "raw_search_arrest", "raw_search_warrant",
                    "raw_search_inventory", "raw_search_plain_view"]
COLUMNS = [
    "date", "time", "lat", "lng", "precinct", "subject_age", "subject_race", "subject_sex",
    "violation", "arrest_made", "citation_issued", "warning_issued", "outcome",
    "contraband_found", "frisk_performed", "search_conducted", "search_basis", *RAW_SEARCH_FLAGS,
]
CATEGORIES = ["precinct", "subject_race", "subject_sex", "violation", "outcome", "search_type"]
BOOLEANS = ["arrest_made", "citation_issued", "warning_issued", "contraband_found",
            "frisk_performed", "search_conducted"]


def slim(chunk: pd.DataFrame) -> pd.DataFrame:
    searched = D._truthy(chunk["search_conducted"])
    chunk = D.add_search_type(chunk)
    chunk["search_type"] = chunk["search_type"].where(searched, "not searched")

    out = pd.DataFrame({
        "date": pd.to_datetime(chunk["date"], errors="coerce"),
        "hour": pd.to_numeric(chunk["time"].astype("string").str.slice(0, 2), errors="coerce").astype("Int8"),
        "lat": pd.to_numeric(chunk["lat"], errors="coerce").astype("float32"),
        "lng": pd.to_numeric(chunk["lng"], errors="coerce").astype("float32"),
        "subject_age": pd.to_numeric(chunk["subject_age"], errors="coerce").astype("Float32"),
    })
    outside = ~(out["lat"].between(*BBOX["lat"]) & out["lng"].between(*BBOX["lng"]))
    out.loc[outside, ["lat", "lng"]] = np.nan
    for col in CATEGORIES:
        out[col] = chunk[col].astype("string")
    for col in BOOLEANS:
        raw = chunk[col].astype("string").str.strip()
        out[col] = raw.map({"TRUE": True, "FALSE": False}).astype("boolean")
    return out


def main() -> None:
    if not C.STOPS_ZIP.exists():
        sys.exit(f"{C.STOPS_ZIP} missing — run `python scripts/download_data.py` first.")
    parts = []
    with zipfile.ZipFile(C.STOPS_ZIP) as zf, zf.open(C.STOPS_CSV_INNER) as fh:
        for chunk in pd.read_csv(fh, usecols=COLUMNS, dtype="string", chunksize=500_000):
            parts.append(slim(chunk))
            print(f"\r  {sum(map(len, parts)):,} rows", end="", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["date"].between(C.DATE_MIN, C.DATE_MAX)].reset_index(drop=True)
    for col in CATEGORIES:
        df[col] = df[col].astype("category")

    C.DATA_PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    searched = df["search_conducted"].fillna(False)
    print(f"\nWrote {len(df):,} stops ({searched.sum():,} searches, "
          f"{df['lat'].isna().sum():,} without usable coordinates) -> {OUTPUT.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
