# Interpretability, Stability and Algorithmic Fairness — Group Project

**HEC Paris · MSc Data Science & AI for Business · Fall 2026 · Prof. Christophe Pérignon**

Scoring analysis on traffic-stop search decisions, comparing a white-box model,
a machine-learning model and a tabular foundation model across predictive
performance, interpretability, stability and fairness.

---

## The problem

**Client:** a police department (or its oversight board) deciding which traffic
stops warrant a search, under a fixed search capacity.

**Data:** Stanford Open Policing Project — Nashville MPD, 3,092,351 stops,
2010-01-01 to 2018-12-31 after dropping the partial 2019 tail.

**Target:** `contraband_found` — observed for the **127,705 stops where a search
occurred (4.13%)**.

### Object definitions (state these before any metric)

| Object | Definition |
|---|---|
| `Y` | 1 if contraband is found — what the officer is trying to predict |
| `Ŷ` | 1 if the model recommends a search |
| `D` | protected attribute: driver race |
| Favourable for the individual | **not being searched** |

The objective of the project is to build a binary classification model to be used by the MNPD to predict whether a vehicle search should be carried out on a given stop. 

The model would help identify fruitless vehicle searches thus saving the officer's time, as well as the driver's time, dignity, and community trust. Three different models will be built a white box (logistic regression), a black box (xgboost), and finally a tabular foundation model (TabPFN).

---

### The app
The user will be able to visualise the model's predictions as well as exploratory data using the following streamlit application.

#### App Description
There are ten different pages, all in the top bar, in the order of the argument:

| Page | What it shows |
|---|---|
| Dataset | a map of all 3.08M stops with filters (date, hour, weekday, race, sex, age, search type, violation, outcome, precinct) |
| Problem definition | the decision and objects (`Y`, `Ŷ`, `D`), selective labels, the hit-rate gap by legal basis, why pooling hides the disparity |
| White box models | the scorecard: description, metrics in every run, scores and calibration by race, coefficients, a decision tree |
| Black box models | the gradient-boosted trees, same layout |
| Foundation models | TabPFN, same layout (fitted only in `matched` mode) |
| Performance | accuracy and ranking agreement, value under a search budget, one stop, XPER, DeLong tests (opens on `matched`: all three models) |
| Stability | resampling, learning curve, distance among models, settings |
| Fairness testing | independence / separation / sufficiency and both Y codings, one tab per model, shared K, χ² / TOST, FPDP (opens on `matched`) |
| Officer features (comparison) | Act II: the officer's record, PLTR, XGBoost, pooled vs within-officer AUC, stability, fairness, economics, officer outcome test |
| Findings & conclusions | answers to the five questions, Act I and Act II, the recommendation (officer-level review) and the pilot |

The model pages, Performance and Fairness testing share the sidebar *Run* picker
(stratum × training mode) and read the cached runs in `outputs/`; without them they say
to run `python -m src.train`, and the other pages still work.

The Dataset page uses the same definitions as `src/`: the 2010-2018 period, and the
search type resolved by `data.add_search_type`. Its *Hit rate: consent vs. rest* tab runs
`fairness.pooled_vs_stratified` on the filtered stops, so the headline table can be
checked within a precinct, a period or a time of day. (Precinct 2 alone shows the same
reversal: consent −4.9 pp for Black drivers, non-consent +5.4 pp, pooled −0.2 pp.)

The map filters the stops in the browser, so it updates without reloading. On first start
the app writes a compact copy of the mapped stops to `app/static/stops/` (~11 MB,
git-ignored), which the browser downloads once; `.streamlit/config.toml` enables serving it.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train                 # fit once (a few minutes) -> outputs/, read by pages 4-7
streamlit run app/streamlit_app.py
```

The two processed files the app reads are committed (`data/processed/searches.parquet`,
`data/processed/stops.parquet`), so the app runs straight after cloning. To rebuild them
from the raw data:

```bash
python scripts/download_data.py          # raw zip -> data/raw/ (not committed)
rm data/processed/searches.parquet && python -c "from src import data; data.load_searches()"
python scripts/build_stops_parquet.py    # -> stops.parquet
```

`app/requirements.txt` lists only what the app imports; Streamlit Community Cloud uses
it (it looks next to the entry point first), so deploying the app does not install the
modelling stack.

## Publishing the app

The public app runs on Streamlit Community Cloud from the **`release`** branch
(main file `app/streamlit_app.py`, Python 3.12). Every push to `release` redeploys it
within a minute or two; pushes to `main` do not touch it, so work in progress never
breaks the public link.

`release` is `main` plus the model outputs: `outputs/` stays ignored on `main`, and is
committed on `release` only, so the published model pages have results. To publish:

```bash
git checkout release
git merge main
python -m src.train                # only if the models or data changed
git add -f outputs/ && git commit -m "Update model outputs"
git push
git checkout main
```
## Data licence

Open Data Commons Attribution License. Cite:

> E. Pierson, C. Simoiu, J. Overgoor, S. Corbett-Davies, D. Jenson, A. Shoemaker,
> V. Ramachandran, P. Barghouty, C. Phillips, R. Shroff, and S. Goel.
> "A large-scale analysis of racial disparities in police stops across the United
> States." *Nature Human Behaviour*, Vol. 4, 2020.
