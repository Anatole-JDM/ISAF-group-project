"""Extract only the stops where a search was conducted from the San Diego stops data."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
INPUT = HERE / "yg821jf8611_ca_san_diego_2020_04_01.csv" / "ca_san_diego_2020_04_01.csv"
OUTPUT = HERE / "ca_san_diego_searches.csv"

df = pd.read_csv(INPUT, low_memory=False)
searches = df[df["search_conducted"] == True]

searches.to_csv(OUTPUT, index=False)
print(f"Kept {len(searches):,} of {len(df):,} rows -> {OUTPUT.name}")
