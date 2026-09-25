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

The course convention (slide 238) is that `Y = 1` is the *favourable* outcome for
the individual. Here it is the opposite — `Y = 1` is favourable to the police.
Every fairness metric therefore reads backwards unless the inversion is declared.
`src/fairness.both_codings()` computes each metric under both, so the report can
show the same model is fair under one reading and unfair under the other. That is
the impossibility result in a real setting, not a textbook example.

Two metrics carry the argument:

- **FPR by race** — the share of *innocent* drivers the model would search.
  This is **predictive equality** (separation), and it is the harm that matters.
- **Hit rate / PPV by race** — the **outcome test** from the economics of
  discrimination. This is **sufficiency**.

---

## The spine: selective labels

Contraband is only observed for drivers who were searched, and **officers chose
that sample**. So:

- the training sample is not *stops*, it is *officer-selected stops*;
- if officers search some groups on weaker evidence, that group's recorded hit
  rate is depressed **by the selection itself**, and the model learns officer
  discretion rather than contraband risk;
- a model deployed on all stops would extrapolate off its training support.

This is why the recommendation to the client rests on trustworthiness, not AUC.

`officer_id_hash` makes the claim measurable rather than rhetorical: if grouped
CV by officer (`stability.officer_cv`) degrades sharply against a random split,
the model did not learn contraband risk.

---

## Headline finding (verified 2026-09-24)

Hit rate by race, split by whether the search was discretionary.
Reproduce with `src.fairness.pooled_vs_stratified(df)`:

| Stratum | white | black | black − white | hispanic | hispanic − white |
|---|---|---|---|---|---|
| consent (discretionary) | 20.84% | **15.68%** | **−5.16 pp** | **7.88%** | **−12.96 pp** |
| non-consent (mechanical) | 20.83% | **27.12%** | **+6.30 pp** | 15.63% | −5.20 pp |
| **POOLED** | 20.83% | 21.66% | **+0.82 pp** | 12.07% | −8.77 pp |

n = 127,122 searches, 2010-01-01 to 2018-12-31, zero unresolved search types.

The gap runs in **opposite directions** across strata for Black drivers, so
pooling cancels them and reports "no disparity". This is **effect modification**,
not Simpson's paradox — say so precisely.

It is not a composition artifact either: consent is 44.6% of white searches and
47.8% of Black ones.

**Definition matters, and it matters in your favour.** Using the raw
`search_basis == "consent"` field gives a −4.37 pp gap. Resolving search type
most-mechanical-first — so `consent` means *no* mechanical basis also applied —
drops 67,629 consent searches to 58,865 and widens the gap to **−5.16 pp**.
Purer discretion, larger disparity. Report both; the sensitivity is a finding.

### It replicates

| Jurisdiction | Pooled black−white | Consent-only black−white |
|---|---|---|
| Nashville | **+0.82 pp** | **−5.16 pp** |
| Illinois | −0.07 pp | −7.25 pp |
| North Carolina | −1.15 pp | −6.30 pp |

Three states, different years, different reporting systems, same reversal. Use
this as a robustness slide — it costs nothing and is the strongest evidence in
the deck.

---

<!-- RESULTS:START -->
## Results (consent stratum, train <2016 → test ≥2016)

Regenerate: `python -m src.train && python scripts/update_readme_results.py`

| Arm | n train | AUC | PR-AUC | Brier |
|---|---|---|---|---|
| scorecard | 49,752 | 0.5506 | 0.2466 | 0.1698 |
| gbm | 49,752 | 0.5642 | 0.2548 | 0.1697 |

n test = 9,113 · base rate 16.1% train → 21.4% test (a real shift across the regime split) · GBM backend `xgboost`.

### There is almost no signal, and it is the wrong signal

**Best AUC is 0.5642** — 0.0642 above chance. Decomposing that sliver:

| Ablation | AUC | Share of above-chance signal |
|---|---|---|
| Pre-search features only | 0.5642 | — |
| **+ officer identity** | 0.6009 | **57%** |
| **− driver race** | 0.5424 | **34%** |

Most of the model's discriminative power is *who stopped you* and *what you look like*. Almost none of it is anything about the situation.

Removing race does **not** make the model race-neutral: precinct and zone are strong proxies in a segregated city. Run both (`full` and `full_blind`) and report both.

### The selection inverts

A contraband-optimising model searches *white* drivers far more, because white drivers have the higher hit rate in discretionary searches — the reverse of actual officer behaviour. That is the outcome test expressed as a model. Check the Fairness tab at your chosen K; a group with zero selections yields an undefined PPV, which `group_rates` returns as NaN rather than a misleading zero.

### Recommendation to the client: do not deploy

Not because it is unfair *or* because it is inaccurate, but because all four dimensions point the same way — near-random discrimination, signal dominated by officer identity and race, large false-positive disparities, and negative net benefit under any defensible cost of searching an innocent driver.
<!-- RESULTS:END -->

---

## Traps (handled in `src/config.py`)

1. **Target leakage.** `contraband_drugs` and `contraband_weapons` are components
   of the target; `arrest_made`, `outcome`, `citation_issued` follow from it;
   `notes` is free text written afterwards. All excluded, each with a reason.
2. **Search-type pooling.** `search_basis` leaves 25,620 searches (20%) as
   "other". Audited against the raw flags, **all 25,620 resolve** to
   arrest / warrant / inventory — every one mechanical, none discretionary.
   `data.add_search_type()` resolves them, most-mechanical-first, so `consent`
   means *purely discretionary*.
3. **Judgement calls.** `frisk_performed`, `search_person`, `search_vehicle` are
   co-decided with the search. Excluded by default; flip `include_questionable`
   and document the effect.
4. **Thin subgroups.** Asian/PI has 323 consent searches. Report with a CI or
   fold into "other" — never a bare point estimate.

### Tabular foundation model arm

`TabPFNArm` defaults to **v2**, deliberately. The v3/v3.5 weights sit in GATED
HuggingFace repos and need **three** separate approvals:

1. a PriorLabs account,
2. a licence acceptance at <https://ux.priorlabs.ai> (Licenses tab), and
3. an 876 MB weights download.

The PriorLabs login flow handles the HuggingFace side itself — the download
completes even with unauthenticated HF requests. `Prior-Labs/TabPFN-v2-clf`
returns 200 unauthenticated, so v2 needs none of the above.

v2 is also the better citation for this report: it is the version in

> Hollmann et al., "Accurate predictions on small data with a tabular
> foundation model", *Nature* 637 (2025).

Its weights are under the Prior Labs License (Apache 2.0 + attribution);
the v3.5 weights are non-commercial.

Opting in to 3.5 is per-machine, so one person doing it does not force the
rest of the team through the licence flow:

```bash
export TABPFN_VERSION=V3_5
python -m src.train && python scripts/update_readme_results.py
```

The version actually fitted is recorded in every metrics JSON, so the report
cannot silently misstate which model produced the numbers.


---

## Layout

```
src/config.py      paths, column taxonomy, leakage exclusions with reasons
src/data.py        streaming load, search-type resolution, modelling frame
src/models.py      the three arms behind one interface
src/fairness.py    independence / separation / sufficiency + both Y codings
src/stability.py   temporal, officer and geographic splits; PSI; coefficient drift
src/economics.py   top-K net benefit, sensitivity, equity/efficiency frontier
src/train.py       fits the arms once, caches scores/metrics in outputs/ for the app
app/streamlit_app.py   the interactive deliverable (entry point + page navigation)
app/views/             one file per page (see The app below)
app/model_page.py      shared layout of the three model-family pages
app/stops_map/         the Explore page's map (deck.gl component that filters the stops in the browser)
app/common.py          shared by the pages: `src` import path, cached data and model runs, the Run picker, map component
scripts/download_data.py
scripts/build_stops_parquet.py   all stops, slim columns, for the Explore page
scripts/update_readme_results.py rewrites the Results section above from outputs/
```

### The app

Eight pages, all in the top bar, in the order of the argument:

| Page | What it shows |
|---|---|
| Dataset | a map of all 3.08M stops with filters (date, hour, weekday, race, sex, age, search type, violation, outcome, precinct) |
| Problem definition | the decision and objects (`Y`, `Ŷ`, `D`), selective labels, why pooling hides the disparity |
| White box models | the scorecard: description, metrics in every run, scores and calibration by race |
| Black box models | the gradient-boosted trees, same layout |
| Foundation models | TabPFN, same layout (fitted only in `matched` mode) |
| Performance | accuracy and ranking agreement, value under a search budget, one stop |
| Fairness testing | independence / separation / sufficiency and both Y codings, one tab per model, shared K |
| Findings & conclusions | the measured officer and race signal, the recommendation |

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

## Deliverables (due Monday 28 September, 9:40)

- [ ] Slide deck
- [ ] Code / notebook with the complete analysis
- [ ] Interactive application
- [ ] 15 min presentation + 10 min Q&A — **every member must be able to answer
      on any part**

## Data licence

Open Data Commons Attribution License. Cite:

> E. Pierson, C. Simoiu, J. Overgoor, S. Corbett-Davies, D. Jenson, A. Shoemaker,
> V. Ramachandran, P. Barghouty, C. Phillips, R. Shroff, and S. Goel.
> "A large-scale analysis of racial disparities in police stops across the United
> States." *Nature Human Behaviour*, Vol. 4, 2020.
