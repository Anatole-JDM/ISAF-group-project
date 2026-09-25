# Data

The raw zip (`raw/`) is never committed. The two processed files the app reads are
committed so the app runs without a download step (see `.gitignore`):

- `processed/searches.parquet` — the labelled search sample (`src.data.load_searches`)
- `processed/stops.parquet` — all 2010-2018 stops, slim columns (`scripts/build_stops_parquet.py`)

## Source

Stanford Open Policing Project — Nashville, TN (Metropolitan Nashville PD).

- Landing page: https://openpolicing.stanford.edu/data/
- File: `yg821jf8611_tn_nashville_2020_04_01.csv.zip` (120 MB zipped, 1.04 GB CSV)
- Licence: Open Data Commons Attribution License

## Required citation

> E. Pierson, C. Simoiu, J. Overgoor, S. Corbett-Davies, D. Jenson, A. Shoemaker,
> V. Ramachandran, P. Barghouty, C. Phillips, R. Shroff, and S. Goel.
> "A large-scale analysis of racial disparities in police stops across the United States."
> *Nature Human Behaviour*, Vol. 4, 2020.

## Fetch

```bash
python scripts/download_data.py
```

## Verified shape (as of 2026-09-24)

- 3,092,351 stops, 2010-01-01 to 2019-03-24
- 127,705 searches (4.13%) — the only rows where `contraband_found` is observed
- search_basis: consent 67,629 / other 25,620 / plain view 18,443 / probable cause 16,013
