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
race.

**CORRECTED 2026-09-25 — the mechanism is temporal leakage, not "added variance".**
An earlier version of this entry attributed the drop to variance from extra
features. That was wrong. The real cause, verified independently:

| tripwire: predict WHICH SIDE OF THE 2016 SPLIT a row is on | AUC |
|---|---|
| **from the 41 nbh columns** | **0.9981** |
| from src's baseline features | 0.7121 |

ACS estimates are re-stamped per release, so every `nbh*` column describes the
**place-year**, not the place. Two columns make it explicit:

- `acs_vintage` is exactly `year - 1` (cross-tab perfectly diagonal)
- `acs_window_overlaps_stop` is True **only in 2010** (7,909 rows) — a pure year
  dummy with zero support on the test side

With a temporal split at 2016, every test row therefore carries vintage values
never seen in training. The model fits year-correlated structure that cannot
transfer. No single column causes it (each scores 0.55-0.65 alone); the *joint*
pattern of 41 vintage-stamped floats fingerprints the release year.

**Can they be repaired? Tested, 2026-09-25.** Re-running `build_nbh_features.py`
with a frozen vintage is impossible here (`opp_data/` is absent), but in-place
repair using only the shipped parquet was measured:

| repair | groups | median distinct years/group | tripwire AUC |
|---|---|---|---|
| none (as shipped) | — | — | **0.9983** |
| per-`location` median | 13,476 | 1 | 0.7629 |
| lat/lng 3dp median (~110 m) | 7,829 | 2 | 0.6806 |
| lat/lng 2dp median (~1.1 km) | 832 | 5 | **0.6044** |
| z-score within `acs_vintage` | — | — | 0.9985 |
| 2dp median + within-vintage z-score | — | — | 0.9984 |

Two conclusions:

1. **Normalisation does not work.** Z-scoring within vintage leaves the tripwire at
   0.9985. The year signal is not in the levels; it is in the joint 36-dimensional
   pattern, which per-column scaling preserves. Same reason a rank transform fails.
2. **Spatial averaging works only by destroying the feature.** Leakage falls
   monotonically as the grid coarsens, but the best result (0.6044) needs a ~1.1 km
   grid — and these are 800 m and 1,200 m radius measurements. Averaging an 800 m
   feature over a cell wider than its own radius no longer measures a neighbourhood;
   it measures a rough part of town, which `precinct` and `zone` already encode.

So the accurate statement is not "cannot be repaired" but: **you can trade leakage
for spatial resolution, and by the time the leakage is acceptable the feature has
no resolution left.** 0.60 is still well above chance and would keep contaminating a
temporal split.

The one clean subset is `nbh800_residents_2010` / `nbh1200_residents_2010` — 2010
decennial counts rather than ACS estimates, so they carry no vintage. Two columns,
measured to add nothing.

`config.NEVER_JOIN` blocks the contaminated columns from any new mode.

### RESOLVED 2026-09-25: the proper fix, and what it reveals

The teammate supplied `opp_data/features/nbh_location_vintage.parquet` — one row
per (location, ACS release), 568,044 rows = ~63,000 locations x 9 releases. Pinning
a single release and joining it to every stop regardless of year makes each value
`f(place)` instead of `f(place, year)`, keeping full 800m/1200m resolution.

| predict post-2016 from | AUC |
|---|---|
| nbh **as shipped** | 0.9983 |
| nbh **frozen at one release** | **0.59-0.62** |
| src's own baseline features, for scale | **0.7121** |

Frozen features are **less** year-informative than the features already in the
model. The residual ~0.6 is genuine covariate shift — stops fell from 444k (2012)
to 204k (2018) and the geographic mix moved — not vintage contamination.

**And the substantive answer is still no.**

| mode | scorecard | gbm | vs baseline |
|---|---|---|---|
| `full` (baseline) | 0.5516 | 0.5663 | — |
| `full_nbh` (contaminated) | 0.5377 | 0.5498 | −0.0165 |
| **`full_nbh_frozen`** | 0.5442 | **0.5636** | **−0.0027** |
| `full_time` | 0.5564 | 0.5733 | +0.0070 |
| `full_time_nbh_frozen` | 0.5495 | **0.5757** | +0.0094 |

Freezing recovers +0.0138 of the 0.0165 the contamination cost, so the diagnosis
was right and the repair works. But repaired, the features still add nothing:
`full_nbh_frozen` sits *below* the plain baseline, and the best combination beats
`full_time` by +0.0024 — **one tenth of the 0.024 context-resampling noise measured
in finding 10.2.**

This is a stronger result than either alternative. We did not conclude "census
features do not help" from a contaminated run; we found the contamination, fixed
it properly, and the answer was still no. **Neighbourhood socioeconomic composition
genuinely does not predict contraband** — consistent with everything else here.

**The race-proxy finding in 5 above is unaffected, and was conservative.** The
vintage contamination handicapped it. On a random split, with the temporal
confound removed:

| predict race from | temporal split | random split |
|---|---|---|
| nbh features only | 0.6915 | **0.7339** |
| non-race stop features | 0.6995 | **0.7184** |

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

## 10. TabPFN-specific experiments (`src/tabpfn_experiments.py`, 2026-09-25)

Hosted inference, consent searches, fixed 3,000-row evaluation subsample (identical
across every arm and configuration, so AUCs here sit slightly below the main run's
9,113-row numbers — sampling noise, not a discrepancy).

### 10.1 The learning curve is FLAT

| n_train | scorecard | gbm | tabpfn |
|---|---|---|---|
| 100 | 0.5211 | 0.5362 | 0.5189 |
| 250 | 0.5131 | 0.5202 | 0.5046 |
| 500 | 0.5338 | 0.5257 | 0.5348 |
| 1,000 | 0.5265 | 0.5090 | **0.5371** |
| 2,000 | 0.5214 | 0.5323 | **0.5460** |
| 5,000 | 0.5131 | 0.5365 | **0.5393** |

Fifty times more training data changes nothing. **The ceiling is absence of signal,
not lack of data** — with real signal and a data-starved model, AUC would climb with
n. TabPFN leads from n=1,000, consistent with its small-data claim, but see 10.2:
the margin is inside the noise.

### 10.2 The ranked list is essentially arbitrary — the deployment killer

Five different random training contexts of the same size, same test set:

| arm | AUC mean ± sd | spread | per-row pred sd | **top-200 Jaccard** |
|---|---|---|---|---|
| tabpfn | 0.5306 ± 0.0090 | 0.0242 | 0.0255 | **0.078** |
| gbm | 0.5258 ± 0.0041 | 0.0118 | 0.0943 | **0.071** |

Two things, both serious:

1. **Context variance exceeds between-model variance.** TabPFN's AUC moves 0.024
   just from *which* rows are in the training sample. The entire spread across the
   three model families is 0.020. So "TabPFN beat XGBoost" is not a finding — it is
   noise, and any single-seed model comparison here is unreliable.
2. **Top-200 Jaccard ≈ 0.07 for BOTH arms.** Resample the training data and roughly
   **93% of the 200 highest-risk stops change**. The top-K list *is* the deployed
   policy (see finding 7), so the policy is close to arbitrary. This is a stronger
   argument against deployment than any AUC, because it holds regardless of accuracy.

### 10.3 Equal scores, incompatible explanations

Permutation importance (AUC drop), and the Spearman rank correlation between arms:

| feature | gbm | scorecard | tabpfn |
|---|---|---|---|
| **subject_race** | 0.0025 | 0.0181 | **0.0382** |
| precinct | 0.0112 | 0.0141 | 0.0014 |
| zone | 0.0107 | −0.0022 | −0.0099 |
| month | 0.0131 | −0.0002 | 0.0008 |
| subject_age | 0.0105 | 0.0003 | 0.0055 |
| hour | 0.0102 | 0.0004 | 0.0011 |

| Spearman | gbm | scorecard | tabpfn |
|---|---|---|---|
| gbm | 1.000 | 0.079 | **−0.321** |
| scorecard | 0.079 | 1.000 | 0.758 |
| tabpfn | −0.321 | 0.758 | 1.000 |

Three models within 0.02 AUC of each other **disagree about what drives the
prediction** — GBM and TabPFN are *negatively* rank-correlated. Interpretability is
therefore not a property you can read off a single model: pick a different arm with
identical accuracy and you get a different story about why.

**And TabPFN leans hardest on race.** `subject_race` is its single largest feature
by a wide margin (0.0382), fifteen times the GBM's reliance (0.0025).

### 10.4 TabPFN is the best-calibrated arm, in every group

Brier within each racial group (lower is better):

| arm | black | hispanic | white |
|---|---|---|---|
| **tabpfn** | **0.1517** | **0.1771** | **0.1844** |
| scorecard | 0.1542 | 0.1802 | 0.1887 |
| gbm | 0.1631 | 0.1794 | 0.1997 |

Calibration is the sufficiency leg of the taxonomy, and TabPFN wins it outright.
Note Brier is sensitive to base rate, which differs by group, so compare *within*
a column, not across.

**The tension worth putting on a slide:** the best-calibrated arm is also the one
most dependent on the protected attribute. Sufficiency and independence pull in
opposite directions here, in one model, measurably.

### 10.5 More compute does not help

| thinking | AUC | Brier | seconds |
|---|---|---|---|
| off | 0.5360 | 0.1656 | 7.4 |
| medium | 0.5352 | 0.1682 | 34.9 |
| high | 0.5368 | 0.1670 | 24.3 |

Fit-time compute up to ~5x, AUC unchanged. More evidence there is nothing to find.

### 10.6 Deployability: inference cost scales with the TEST set

TabPFN learns in context, so it re-processes the training context for every test
row. Measured locally on CPU: **209s to score 9,113 rows** from a 500-row context,
roughly 4x that from 2,000. Cost is driven by how many people you score, not by how
much you train. A department scoring millions of stops needs GPUs or an external
API — and sending stop records to a third party is its own governance problem.

---

## 11. The one feature block that helps (2026-09-25)

Seven cyclical/calendar columns from `scripts_claude/build_time_features.py`
(`config.TIME_EXTRA`), joined on `raw_row_number`:

`hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `time_heaped`,
`is_federal_holiday`, `is_holiday_window`

| mode | scorecard | gbm | Δ gbm |
|---|---|---|---|
| `full` | 0.5516 | 0.5663 | — |
| **`full_time`** | **0.5564** | **0.5733** | **+0.0070** |

PR-AUC also improves, 0.2552 → 0.2702. The cyclical encodings are the only
genuinely *new* information in the teammate parquet — they make 23:00 and 00:00
adjacent, which src's integer `hour` cannot express. Everything else is either a
re-encoding of a column src already has (`plate_*` recodes
`vehicle_registration_state`) or vintage-contaminated (finding 5).

Report it honestly: +0.0070 on a 0.0663 above-chance signal is ~11% relative, and
the app still fires its "best AUC < 0.65 → do not deploy" banner. It does not
change the recommendation.

### Columns that must never be joined (`config.NEVER_JOIN`)

| column(s) | evidence |
|---|---|
| 11 × `post_*` | components of the target; `P(contraband \| post_contraband_drugs) = 1.000` |
| all 41 `nbh*` | split tripwire AUC 0.9981 (finding 5) |
| `acs_vintage` | exactly `year - 1` |
| `acs_window_overlaps_stop` | True only in 2010; pure year dummy |
| `plate_missing` | 0.13% pre-2017 vs 8-9% in 2017-18 — a recording change |
| `hour`, `month`, `day_of_week`, `minute_of_day` | duplicate encodings src already builds; coexisting clock encodings split the scorecard coefficient under L2 |

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
