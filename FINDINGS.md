# Findings log

Running record of every empirical result, newest section last within each theme.
Each entry states how it was obtained so it can be re-run or challenged.

Data: Stanford Open Policing Project — Nashville MPD, 3,092,351 stops 2010-01-01
to 2019-03-24 (2019 dropped as a partial year). Target `contraband_found`,
observed only for the 127,122 searches. Scope: **consent searches only** (58,865),
justified by finding 1.

---

## 1. Pooling hides the disparity — the reason for the consent-only scope

Hit rate by race, split by whether the search was discretionary
(`src.fairness.pooled_vs_stratified`, 2026-09-24):

| Stratum | white | black | b−w | hispanic | h−w |
|---|---|---|---|---|---|
| consent (discretionary) | 20.84% | **15.68%** | **−5.16 pp** | **7.88%** | **−12.96 pp** |
| non-consent (mechanical) | 20.83% | **27.12%** | **+6.30 pp** | 15.63% | −5.20 pp |
| **POOLED** | 20.83% | 21.66% | **+0.82 pp** | 12.07% | −8.77 pp |

The Black–white gap runs in **opposite directions** across strata, so pooling
cancels them and reports "no disparity". This is **effect modification**, not
Simpson's paradox — the distinction matters and should be stated precisely.

Not a composition artifact: consent is 44.6% of white searches, 47.8% of Black,
46.0% of Hispanic.

**Definition sensitivity, in our favour.** Using raw `search_basis == "consent"`
gives −4.37 pp. Resolving search type most-mechanical-first — so `consent` means
*no* mechanical basis also applied — drops 67,629 to 58,865 and widens the gap to
−5.16 pp. Purer discretion, larger disparity.

### It replicates in two other jurisdictions

| Jurisdiction | pooled b−w | consent-only b−w |
|---|---|---|
| Nashville | **+0.82 pp** | **−5.16 pp** |
| Illinois | −0.07 pp | −7.25 pp |
| North Carolina | −1.15 pp | −6.30 pp |

Three states, different years and reporting systems, same reversal.

---

## 2. Search type resolves completely

`search_basis` leaves 25,620 searches (20%) as "other". Audited against the raw
flags, **all 25,620** are arrest / warrant / inventory — every one mechanical:

| signature | n |
|---|---|
| arrest | 19,722 |
| arrest + inventory | 2,886 |
| arrest + warrant | 1,510 |
| inventory | 965 |
| warrant | 446 |
| arrest + warrant + inventory | 73 |
| warrant + inventory | 18 |
| **total** | **25,620** |

Zero unresolved. Better than North Carolina (38% unresolvable) or Illinois (67%).

---

## 3. There is almost no signal

Consent searches, train <2016 → test ≥2016, all three arms on the **same 2,000-row
subsample** (2026-09-25, TabPFN via hosted inference):

| Arm | AUC | PR-AUC | Brier |
|---|---|---|---|
| scorecard (logistic) | 0.5329 | 0.2324 | 0.1718 |
| gbm (XGBoost) | 0.5478 | 0.2392 | 0.1824 |
| **TabPFN** | **0.5529** | 0.2409 | **0.1690** |

A state-of-the-art tabular foundation model lands within 0.02 AUC of logistic
regression. **"You didn't use a good enough model" is closed off empirically.**
TabPFN also has the best calibration of the three — it isn't failing, it is
correctly reporting that there is nothing to find.

On all available rows (49,752 train) the GBM reaches 0.5663.

---

## 4. What little signal exists is officer identity

| Ablation | AUC | share of above-chance signal |
|---|---|---|
| pre-search features only | 0.5663 | — |
| **+ officer identity** | 0.6005 | **52%** |
| **− driver race** | 0.5414 | ~37% |

Roughly half the model's discriminative power is *which officer stopped you*.
This is the selective-labels problem made measurable: contraband is observed only
for officer-selected searches, so the model learns discretion, not risk.

---

## 5. The features predict RACE better than they predict CONTRABAND

The decisive fairness result (2026-09-25). Predicting the driver's race
(black vs white, n=53,538 consent searches, 60.4% black), same temporal split:

| Predict **race** from | AUC |
|---|---|
| neighbourhood/ACS features only (41 cols) | **0.6915** |
| non-race stop features (precinct, zone, time, plate) | **0.6995** |
| *(predict **contraband** from the same features)* | *0.5329–0.5663* |

**Fairness through unawareness does not work here.** Remove `subject_race` and the
remaining features still recover race at ~0.70 AUC — markedly better than they
predict the outcome the model is supposed to be for.

### Corollary: the census features do not help predict contraband

| Mode | race | nbh | gbm AUC |
|---|---|---|---|
| `full` | ✓ | — | 0.5663 |
| `full_blind` | — | — | 0.5414 |
| `full_blind_nbh` | — | ✓ | **0.5346** |
| `full_nbh` | ✓ | ✓ | **0.5498** |

Adding 41 carefully-built ACS features makes the model **worse**, with and without
race. They are proxies for race, not for contraband: added variance, no signal to
offset it, and it fails to transfer across the 2016 regime split.

---

## 6. The selection inverts

At K = 2,000 on the scorecard, selection rates were white 51.4%, black 5.0%,
hispanic 0.0%. A contraband-optimising model searches *white* drivers far more,
because white drivers have the higher hit rate in discretionary searches — the
reverse of actual officer behaviour. This is the outcome test expressed as a model.

A group with zero selections yields an undefined PPV; `group_rates` returns NaN
rather than a misleading zero. Do not report a bare point estimate there.

---

## 7. Economics: deployment destroys value

Net benefit at K = 2,000 turns **negative** once an innocent driver's search is
valued above ~0.25 of a justified search. Under any defensible valuation, deploying
the model costs more than it returns. The false-positive price is a policy
judgement, not a data fact — ship the sensitivity table.

---

## 8. Methodological findings (things that would have silently broken the result)

| Issue | Effect if missed |
|---|---|
| `search_type` / `is_discretionary` derived from stratifiers reached `X` | near-perfect proxy for search type (27% mechanical vs 16% discretionary hit rate) |
| `class_weight="balanced"` on the scorecard | Brier 0.300 vs GBM 0.168 — would have failed the sufficiency/calibration audit for a reason unrelated to fairness |
| chunked `read_csv` inferred dtype **per chunk** | `5` vs `"5.0"` phantom categories; all dotted `zone` values fell in 2017-18, so one-hot **blanked geography for 29.7% of consent test rows** |
| `reason_for_stop` is a byte-identical duplicate of `violation` | L2 split every stop-reason coefficient across two identical blocks, corrupting the interpretability story |
| `year` was both a feature and the split variable | every test row scored outside its training support |

### Two pipelines converged independently

`scripts_claude/` (teammate) and `src/` reached the same consent definition
separately: 58,939 vs 58,865 rows (the difference is the 2019 tail), target rate
16.92% vs 16.9%, per-race counts within a handful (asian/PI 282 vs 282, other 133
vs 133). Both independently flagged the same 11 post-decision columns. The join on
`raw_row_number` matches all 58,865 rows with zero loss.

### The engineered features are leakage-free

`build_nbh_features.py` never reads `contraband_found` — it loads only
`stop_id, date, lat, lng, geo_outside_davidson_box, location` plus ACS tracts and
2010 census blocks. `build_plate_features.py` writes its feature parquet **before**
loading the outcome, which it uses only for a descriptive manifest. All 11 leakage
columns carry a `post_` prefix and can be dropped mechanically.

---

## 9. Recommendation to the client: do not deploy

Not because the model is unfair *or* because it is inaccurate, but because every
dimension points the same way:

- near-random discrimination (best AUC 0.5663, and a foundation model does no better)
- half the signal is officer identity, not driver risk
- the features predict race better than they predict contraband
- large false-positive disparities by race
- negative net benefit under any defensible cost of searching an innocent person

That is what "trustworthy AI rather than predictive performance alone" looks like
when the numbers are actually run.

---

## Reproduction

```bash
pip install -r requirements.txt tabpfn-client
export TABPFN_BACKEND=client TABPFN_TOKEN=...
python scripts/download_data.py
python -m src.train && python scripts/update_readme_results.py
streamlit run app/streamlit_app.py
```
