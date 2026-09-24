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

## Results (consent stratum, train <2016 → test ≥2016)

Reproduce with `python -m src.train`, then `streamlit run app/streamlit_app.py`.

| Arm | n train | AUC | PR-AUC | Brier |
|---|---|---|---|---|
| scorecard | 49,752 | 0.5518 | 0.2449 | 0.1678 |
| gbm | 49,752 | 0.5649 | 0.2514 | 0.1677 |

n test = 9,113 · base rate 16.1% train → 21.4% test
(a real base-rate shift across the regime split) · GBM backend `sklearn-histgb`.

### The finding: there is almost no signal, and it is the wrong signal

**Best AUC is 0.5649** — 0.0649 above chance. Decomposing that sliver:

| Ablation | AUC | Share of above-chance signal |
|---|---|---|
| Pre-search features only | 0.5649 | — |
| **+ officer identity** | 0.5955 | **47%** |
| **− driver race** | 0.5348 | **46%** |

Roughly half the model's discriminative power is *who stopped you*, and roughly
half is *what you look like*. Almost none of it is anything about the situation.

Removing race does **not** make the model race-neutral: precinct and zone are
strong proxies in a segregated city. Run both (`mode="full"` and
`mode="full_blind"`) and report both.

### The selection inverts

At K = 2,000 on the scorecard, selection rates are **white 51.4%, black 5.0%,
hispanic 0.0%**. A contraband-optimising model searches *white* drivers far more,
because white drivers have the higher hit rate in discretionary searches (25.5%
vs 19.5% on the test set) — the reverse of actual officer behaviour.

That is the outcome test expressed as a model, and it is a strong slide. It also
means a group with zero selections yields an undefined PPV; `group_rates` returns
NaN rather than a misleading zero. Do not report a bare point estimate there.

### Economics

Net benefit at K = 2,000 turns **negative once an innocent driver's search costs
more than ~0.25 of a justified search**. Under any defensible valuation,
deploying the model destroys value.

### Recommendation to the client: do not deploy

Not because it is unfair *or* because it is inaccurate, but because all four
dimensions point the same way — near-random discrimination, signal dominated by
officer identity and race, large false-positive disparities, and negative net
benefit. That is what "trustworthy AI rather than predictive performance alone"
looks like when you actually run the numbers.

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

---

## Layout

```
src/config.py      paths, column taxonomy, leakage exclusions with reasons
src/data.py        streaming load, search-type resolution, modelling frame
src/models.py      the three arms behind one interface
src/fairness.py    independence / separation / sufficiency + both Y codings
src/stability.py   temporal, officer and geographic splits; PSI; coefficient drift
src/economics.py   top-K net benefit, sensitivity, equity/efficiency frontier
app/streamlit_app.py   the interactive deliverable
scripts/download_data.py
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
streamlit run app/streamlit_app.py
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
