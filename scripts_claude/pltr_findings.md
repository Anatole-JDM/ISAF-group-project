# Officer track record, officer disparity and PLTR - findings

Nashville consent-only searches 2010-2018 (58,865 searches, 17.0% find contraband).
Target: `contraband_found`. Every number below comes from the scripts listed at the end (runs of 27 Sep 2026).

---

## 1. The story in seven lines

1. Driver, place and time predict a successful search poorly (AUC 0.555-0.569): the team's ceiling.
2. **Who searches matters more than who is searched.** An officer's past search record, computed strictly
   from the past, lifts AUC by **+0.07** (to 0.63-0.64). The lift holds on officers never seen in training,
   in every test year, and over 5 random draws of held-out officers.
3. **Officers who search minority drivers disproportionately are the least successful searchers**, on
   minority AND on white drivers. The least successful fifth of searches is made by 140 officers who search
   Hispanic drivers twice as often as the others.
4. **A 19-term PLTR is a readable white box** that loses only ~0.01 AUC to XGBoost; its most stable terms are
   the officer's track record and his Hispanic search skew.
5. **Economics:** dropping the 20% of searches with the lowest score avoids 23% of fruitless searches and
   loses 10% of finds (random: 20% / 20%).
6. **Fairness has no free lunch.** Race-blind models select the three groups equally but, at the same score,
   minority drivers are 5-7 points less likely to carry contraband: blinding reproduces the officers' lower bar.
   Race-aware models are calibrated but select almost no Hispanic drivers - and using race to decide a
   search is not legally usable.
7. **Recommendation direction:** use the model to hold officers' search practices to account, not to score
   drivers.

---

## 2. What was built

| Script | Output | What it does |
|---|---|---|
| `build_officer_features.py` | `opp_data/features/officer_features.parquet` (local), joined into `data/nashville_consent_searches.parquet` by `build_master.py` | 25 officer features for each of the 3.09M stops, from that officer's earlier stops only |
| `analyze_officer_disparity.py` | `data/officer_disparity_findings.json` | officer-level outcome test, no model |
| `pltr.py`, `test_pltr.py` | - | Penalised Logistic Tree Regression (Dumitrescu, Hué, Hurlin & Tokpavi 2022) + 5 known-answer tests |
| `evaluate_officer_models.py` | `data/officer_model_results.json`, `data/pltr_terms.json`, `data/officer_model_predictions.parquet` | scorecard / XGBoost / PLTR, 4 feature sets, 2 splits |
| `officer_model_diagnostics.py` | `data/officer_model_diagnostics.json` | AUC by year, PLTR term stability, economics, officer counterfactual, calibration by race |
| `officer_holdout_stability.py` | `data/officer_holdout_stability.json` | officer holdout over 5 random draws of officers |

### Officer features (all strictly from the past)
- An event counts only if it happened strictly before the stop; same-minute stops never see each other;
  the stop's own outcome is never used. A brute-force recomputation of 400 random stops matched exactly.
- **Track record (previous 365 days):** stops, searches, finds, search rate, hit rate and consent hit rate,
  shrunk toward the force-wide past rate (20 pseudo-searches) so officers with few searches do not get
  extreme rates.
- **Today, before this stop:** stops, searches and finds already made, minutes since the first stop of the
  day, minutes since the last search, whether the last search found something.
- **Career:** days since the officer's first stop in the data (+ flags for officers active before 2010).
- **Disparity (all the officer's past):** how much more often the officer asks Black / Hispanic drivers for
  consent than white drivers (log ratio, per stop), and his hit-rate gap between those groups (outcome test).
  Shrunk toward the officer's own rate.
- **`off_hit_rate_same_race_past`:** the officer's past hit rate on drivers of this driver's race. Uses the
  driver's race: role `feature_race_derived`, never in a race-blind model.

Descriptive signal (2011-2018): hit rate by decile of the officer's past consent hit rate goes from
**5.9% to 31.9%**. A search right after a successful search succeeds 22.0% of the time vs 16.1% after a miss.

---

## 3. Officer disparity analysis (descriptive, no model)

Each officer is compared **with himself**: his consent-search rate per stop on Black (Hispanic) drivers vs
white drivers, and his hit rate on each group. That removes the officer's patrol area and habits from the
comparison. Officers with >= 30 white and >= 30 Black consent searches (163 officers, 33k searches);
>= 15 Hispanic (43 officers, 13k searches).

### Black vs white drivers
| Officers, by search skew (Black / white) | Skew range | Hit rate, all | on white | on Black |
|---|---|---|---|---|
| Least skewed 20% | 0.5-1.06x | **21.7%** | 25.1% | 19.1% |
| 2nd | 1.06-1.28x | 18.4% | 21.9% | 17.6% |
| 3rd | 1.28-1.55x | 18.1% | 21.1% | 17.9% |
| 4th | 1.56-1.98x | 15.6% | 19.3% | 15.6% |
| Most skewed 20% | 2.0-4.6x | **14.3%** | 18.5% | 13.5% |

- The median officer asks Black drivers for consent **1.41x** as often as white drivers; **81%** of officers
  are above 1.
- More skew goes with less success overall: Spearman **rho = -0.27 [-0.42, -0.11]**, including **on white
  drivers**: skewed officers search on a lower bar in general and apply it to Black drivers more often.
- The outcome gap is **systemic**: **77%** of officers find less on Black drivers than on white drivers
  (median gap -4.4 points); **28** have a significant negative gap vs ~4 expected by chance, 2 a significant
  positive one. Even the least skewed group shows a 6-point gap.
- **Concentration:** the most skewed 20% of officers do **24.5%** of consent searches of Black drivers but
  13.9% of those of white drivers.

### Hispanic vs white drivers (43 officers - wide intervals)
| Officers, by search skew (Hispanic / white) | Skew range | Hit rate, all | on white | on Hispanic |
|---|---|---|---|---|
| Least skewed 20% | 0.4-1.15x | **21.8%** | 24.5% | 12.3% |
| Most skewed 20% | 4.5-34x | **5.0%** | 9.2% | **1.7%** |

- Median skew **2.3x**; skew vs overall success **rho = -0.65 [-0.82, -0.42]**.
- The 9 most skewed officers carry out **half (49.7%)** of all Hispanic consent searches in the sample, and
  **98%** of their consent searches of Hispanic drivers find nothing.
- 91% of officers find less on Hispanic drivers; 8 significant vs ~1 expected by chance, 0 positive.

### How to say it
- "Consistent with a lower evidentiary bar", not "racist officers": the outcome test cannot prove intent
  (**infra-marginality**: if contraband is distributed differently across groups, hit rates can differ at the
  same bar).
- `contraband_found` blank was recorded as "nothing found": officers who record less look less successful.
- The within-officer comparison controls for the officer and roughly his area, not for what he saw.
- Officer IDs are hashed. Report groups of officers, **never a list of individuals**.

---

## 4. Does the track record improve prediction?

### Protocol
- Sample: consent-only searches 2010-2018 (58,865). Preprocessing and settings copied from the team's
  `src/` (median impute + scale / one-hot min 25; logistic `max_iter=2000`; XGBoost 400 trees, depth 5,
  lr 0.05, subsample and colsample 0.8; seed 42), so numbers are comparable.
- Feature sets: `full_time` (team's current best: age, sex, race, precinct, zone, reason for stop, plate
  state, hour, weekday, month + 7 time extras); `full_time_officer` (+ 15 officer features + the
  race-derived one); `full_time_officer_blind` (no driver race, sex or race-derived feature);
  `officer_only` (the 15 officer features alone).
- **Temporal split:** train 2010-2015 (49,752 searches, 16.1% hit), test 2016-2018 (9,113, 21.4% hit).
  Honest for history-based features, mimics deployment, comparable with the team. Regime change: far fewer
  searches after 2016, with a higher hit rate.
- **Officer holdout:** 20% of officers removed from training entirely; test = their 2016-2018 searches
  (1,845 on the main draw). Did the model learn a general rule or memorise individuals?
- AUC with bootstrap 95% CI; gains with a paired bootstrap CI (same resampled rows for both models).

### Results (AUC)
| Features | Model | Temporal split | Officer holdout |
|---|---|---|---|
| `full_time` (team's current) | scorecard | 0.555 [0.541, 0.569] | 0.551 [0.520, 0.584] |
| | XGBoost | 0.569 [0.556, 0.584] | 0.556 [0.523, 0.588] |
| `full_time_officer` | scorecard | 0.627 [0.613, 0.641] | 0.626 [0.595, 0.655] |
| | XGBoost | **0.640** [0.627, 0.652] | **0.643** [0.614, 0.669] |
| | PLTR (1-SE, 61 / 49 terms) | 0.636 [0.622, 0.650] | 0.633 [0.603, 0.661] |
| | PLTR sparse (19 / 25 terms) | 0.630 [0.616, 0.645] | 0.629 [0.597, 0.659] |
| `full_time_officer_blind` | scorecard | 0.623 [0.608, 0.637] | 0.616 [0.587, 0.646] |
| | XGBoost | 0.632 [0.618, 0.646] | 0.638 [0.605, 0.666] |
| | PLTR (1-SE, 52 / 21 terms) | 0.631 [0.617, 0.644] | 0.617 [0.588, 0.648] |
| | PLTR sparse (30 / 21 terms) | 0.628 [0.615, 0.642] | 0.617 [0.586, 0.647] |
| `officer_only` | scorecard | 0.622 [0.608, 0.635] | 0.618 [0.586, 0.648] |
| | XGBoost | 0.632 [0.618, 0.645] | 0.617 [0.587, 0.646] |

Paired AUC gains:
| | Temporal | Officer holdout |
|---|---|---|
| scorecard + officer vs `full_time` scorecard | +0.072 [+0.060, +0.085] | +0.074 [+0.047, +0.103] |
| XGBoost + officer vs `full_time` XGBoost | +0.070 [+0.056, +0.084] | +0.087 [+0.055, +0.120] |
| scorecard, race-blind + officer | +0.067 [+0.051, +0.084] | +0.065 [+0.031, +0.099] |
| XGBoost, race-blind + officer | +0.063 [+0.048, +0.079] | +0.082 [+0.043, +0.122] |
| PLTR + officer vs team's current scorecard | +0.081 [+0.066, +0.095] | +0.082 [+0.047, +0.115] |
| PLTR sparse (19 terms) + officer vs team's current scorecard | +0.075 [+0.060, +0.091] | +0.078 [+0.043, +0.114] |
| PLTR race-blind + officer vs team's current scorecard | +0.075 [+0.058, +0.094] | +0.065 [+0.023, +0.106] |

### Reading
1. **The largest gain found in the project (+0.07 AUC)**, identical for linear and tree models (not a model
   artefact), and **it does not shrink on unseen officers**: the model learned a general rule, not individuals.
2. **Officer features alone (0.62-0.63) do almost as well as everything together (0.63-0.64).** Driver, place
   and time add < 0.01 on top. The risk signal is mostly the officer, not the driver.
3. **Removing race costs 0.004-0.008 AUC** but changes fairness a lot (section 7).
4. **PLTR vs plain logistic:** the tree rules add ~0.01 (0.636 vs 0.627). **Cost of interpretability:** the
   19-term PLTR is 0.010 below XGBoost (0.630 vs 0.640).

---

## 5. Stability

### By test year (temporal split)
| Model | 2016 | 2017 | 2018 |
|---|---|---|---|
| team scorecard (`full_time`) | 0.554 | 0.566 | 0.546 |
| team XGBoost (`full_time`) | 0.578 | 0.558 | 0.567 |
| scorecard + officer | 0.620 | 0.647 | 0.608 |
| XGBoost + officer | 0.637 | 0.648 | 0.628 |
| PLTR + officer | 0.637 | 0.650 | 0.608 |
| PLTR sparse + officer | 0.625 | 0.651 | 0.605 |
| XGBoost race-blind + officer | 0.630 | 0.648 | 0.609 |
| officer features only, XGBoost | 0.624 | 0.655 | 0.609 |

- The officer lift is present **every year** (+0.05 to +0.09). All models are weakest in 2018 (the furthest
  from training); the linear models and PLTR lose more there (0.605-0.608) than XGBoost (0.628): a first
  sign of drift for a model trained once in 2015. Retrain yearly if deployed.

### Over 5 random draws of held-out officers (seeds 0-4; the main run is a 6th draw)
| Model | Mean AUC (min - max) | Gain vs team's model, mean (min - max) | Draws with gain CI > 0 |
|---|---|---|---|
| team scorecard | 0.546 (0.519 - 0.568) | - | - |
| team XGBoost | 0.571 (0.542 - 0.597) | - | - |
| scorecard + officer | 0.619 (0.602 - 0.667) | +0.073 (+0.034 - +0.121) | 5/5 |
| XGBoost + officer | 0.635 (0.595 - 0.695) | +0.063 (+0.031 - +0.118) | 4/5 |
| PLTR + officer | 0.625 (0.590 - 0.675) | +0.079 (+0.050 - +0.130) | 5/5 |
| PLTR sparse + officer | 0.618 (0.589 - 0.660) | +0.071 (+0.021 - +0.115) | 4/5 |
| XGBoost race-blind + officer | 0.631 (0.584 - 0.690) | +0.059 (+0.032 - +0.113) | 4/5 |

- **The gain is positive in every draw for every model**, but its size depends a lot on WHICH officers are
  held out (+0.03 to +0.12; AUC sd across draws 0.026-0.038). Say "+0.07 on average, between +0.03 and
  +0.12 depending on the officers", not "+0.087".
- Officers are heterogeneous: the track record helps most where officers differ most in skill.

### Stability of PLTR's terms
- Temporal vs officer holdout: only ~20% of variable combinations are shared (Jaccard 0.17-0.21), but the
  shared ones carry **51-62% of the model's importance**: the head of the model is stable, the tail is noise.
- **Selected in all 5 draws** (the stable core):
  - race-aware: officer consent hit rate, officer hit rate on this driver's race, officer's **Hispanic search
    skew**, driver race, precinct, zone;
  - race-blind: officer consent hit rate, the rule "low consent hit rate AND low overall hit rate", a rule
    combining the outcome of his last search and the time since his first stop of the day, officer's **Hispanic search skew**, precinct, zone.
- Report only the core as "what the model has learned"; the other terms change with the data.

---

## 6. The white box: PLTR with 19 terms (race-aware, temporal split)

Penalty chosen on training data only (the weakest penalty keeping <= 30 terms); AUC 0.630. Terms ranked by
importance (|coefficient| x spread = weight x coverage); odds ratios for linear terms are per standard
deviation.

| # | Term | Odds ratio | Searches concerned | Share of importance |
|---|---|---|---|---|
| 1 | officer's past consent hit rate (per SD) | 1.43 | all | 32% |
| 2 | officer's past hit rate on this driver's race <= 17.9% AND overall past hit rate <= 12.9% | 0.64 | 15% | 14% |
| 3 | driver Hispanic | 0.71 | 8% | 8% |
| 4 | driver white | 1.21 | 36% | 8% |
| 5 | experienced officer (> 569 days) AND first stop of the day (<= 5.5 min since first stop) | 1.24 | 21% | 8% |
| 6 | officer's previous search less than ~1h50 ago | 0.79 | 10% | 6% |
| 7 | male driver AND officer with <= 9 consent searches in the past year | 0.86 | 19% | 6% |
| 8 | early morning (~4:30-7:30) AND officer not among the busiest | 1.46 | 2.5% | 5% |
| 9 | officer's Hispanic search skew > 1.6x AND his Hispanic hit gap > -3.5 points | 0.93 | 21% | 3% |
| 10 | officer with a low search rate AND driver aged <= 19 | 1.09 | 9% | 2% |
| | 9 more terms (time of day, month, weekday, precinct, zone, reason for stop) | | | 7% together |

Reading: the model is mostly **about the officer**: his track record (terms 1-2), his rhythm (5, 6, 8) and
his experience; a search made shortly after another search succeeds less often - consistent with searches
made on momentum rather than evidence. The race-blind 30-term version has the same head (track record,
"low consent AND low overall hit rate" OR 0.60, first stop of the day, repeat searches) and uses the
officer's Hispanic search skew as a continuous term (OR 0.96 per SD).

---

## 7. Fairness

### Selection at a fixed budget (top 22% of test searches, temporal split)
Test-period hit rates: white 25.5% (3,368), Black 19.5% (4,947), Hispanic 15.1% (689).

| Model | Selected: white / Black / Hispanic | Hit rate of selected: white / Black / Hispanic |
|---|---|---|
| Team's scorecard (`full_time`) | 48.5% / 7.1% / **0%** | 27.3% / 19.8% / - |
| Team's XGBoost (`full_time`) | 40.5% / 12.1% / 2.3% | 29.2% / 23.7% / 43.8% |
| XGBoost + officer | 34.7% / 15.1% / 10.0% | 36.4% / 28.7% / 27.5% |
| PLTR + officer | 38.9% / 13.5% / 2.0% | 34.4% / 30.9% / 42.9% |
| PLTR sparse + officer | 36.4% / 14.9% / 4.1% | 33.6% / 29.9% / 25.0% |
| Scorecard, race-blind + officer | 27.4% / 18.9% / 18.6% | 36.4% / 27.9% / 25.8% |
| XGBoost, race-blind + officer | 26.0% / 19.7% / 18.6% | 38.3% / 28.1% / 24.2% |
| PLTR sparse, race-blind + officer | 26.9% / 19.2% / 19.0% | 36.0% / 29.4% / 26.7% |

### Calibration by race: at the same score, same chance of contraband?
Hit-rate gap vs white drivers at equal score (weighted over score fifths), temporal split:
| Model | Black - white | Hispanic - white |
|---|---|---|
| team scorecard (`full_time`) | -2.9 [-6.0, -0.1] | -4.6 [-16.3, +5.4] |
| scorecard + officer (race-aware) | 0.0 [-1.9, +2.0] | +3.5 [+0.2, +6.9] |
| PLTR sparse + officer (race-aware) | -0.9 [-2.8, +1.1] | +2.0 [-1.2, +5.1] |
| scorecard, race-blind + officer | **-4.7** [-6.4, -2.8] | **-6.5** [-9.4, -3.2] |
| XGBoost, race-blind + officer | **-4.8** [-6.4, -3.0] | **-7.6** [-10.5, -4.4] |
| PLTR sparse, race-blind + officer | **-4.6** [-6.1, -2.7] | **-7.2** [-9.9, -4.3] |

Calibration in the large, race-blind PLTR sparse: predicted vs observed hit rate white 22.2% vs 25.5%,
Black 20.6% vs 19.5%, Hispanic **18.8% vs 15.1%** - it over-predicts Hispanic drivers and under-predicts
white drivers.

### Reading - the central fairness finding
- **Race-blind models give equal selection rates but are not calibrated across groups:** at the same score,
  Black drivers are ~5 points and Hispanic drivers ~7 points less likely to carry contraband. The model
  over-scores minority drivers through proxies (zone, precinct, the officer's own skew) - it **reproduces
  the lower bar** the officers apply (section 3). Deployed, it would send more fruitless searches to
  minority drivers.
- **Race-aware officer models are calibrated** (Black gap 0.0, Hispanic slightly positive), but select few
  Hispanic drivers (2-10% vs 35-39% of white drivers) and use race as an input, which cannot lawfully decide
  a search (equal protection).
- **The team's current model gets neither:** it uses race, still over-scores Black drivers at equal score
  (-2.9 points) and selects no Hispanic driver at all.
- This is the classic impossibility: when base rates differ, equal selection and equal calibration cannot
  both hold. The data do not let a driver-scoring tool be both fair and lawful.

---

## 8. Economic performance (temporal split, 9,113 test searches: 1,946 finds, 7,167 fruitless)

### Keeping only the top X% of searches by score
| Keep | Model | Fruitless avoided | Finds lost | Hit rate of kept searches | Finds lost per 100 fruitless avoided |
|---|---|---|---|---|---|
| 80% | random | 1,433 (20%) | 389 (20%) | 21.4% | 27.2 |
| | team scorecard | 1,517 (21%) | 306 (16%) | 22.5% | 20.2 |
| | XGBoost + officer | 1,637 (23%) | **186 (10%)** | 24.1% | **11.4** |
| | PLTR sparse + officer | 1,632 (23%) | 191 (10%) | 24.1% | 11.7 |
| 50% | random | 3,584 (50%) | 973 (50%) | 21.4% | 27.2 |
| | team scorecard | 3,707 (52%) | 850 (44%) | 24.1% | 22.9 [21.0, 24.7] |
| | XGBoost + officer | 3,867 (54%) | 690 (35%) | 27.6% | 17.8 [16.5, 19.3] |
| | PLTR sparse + officer | 3,852 (54%) | 705 (36%) | 27.2% | 18.3 [16.9, 19.7] |

- The officer-aware models avoid fruitless searches at a **lower cost in finds** than the team's model: 22%
  lower when keeping half, 44% lower when keeping 80% - and at less than half the cost of dropping searches
  at random when the cut is modest (80% kept).

### If a search is only worth making above a break-even hit rate
(a find worth 1/c fruitless searches; c = b / (1 - b) for break-even hit rate b; value in "finds")
| Break-even hit rate | team scorecard: gain vs searching all | XGBoost + officer | PLTR sparse + officer |
|---|---|---|---|
| 15% | +12 (search 95%) | +112 (search 77%) | +102 (search 81%) |
| 20% | +89 (search 53%) | +302 (search 56%) | +277 (search 45%) |

The test period's overall hit rate (21.4%) sits near a 20% break-even: most of the value comes from not
making the worst searches; at a 20% break-even the officer-aware models find ~3x more value than the
team's model.

### Officer-level counterfactual
Test searches grouped by the officer's **past** consent hit rate (known before the search):
| Fifth | Officers | Officer's past consent hit rate | Hit rate | Drivers W / B / H | Officers' median skew B/W, H/W |
|---|---|---|---|---|---|
| 1 (least successful) | **140** | 4.2-16.0% | **12.2%** [9.8, 15.3] | 33% / 53% / **12.7%** | 1.90x, **1.54x** |
| 2 | 355 | | 18.8% | 34% / 59% / 6.2% | 1.98x |
| 3 (median) | 541 | | 20.9% [19.1, 23.1] | 36% / 57% / 5.6% | 1.87x |
| 4 | 509 | | 23.3% | 37% / 54% / 6.5% | 1.78x |
| 5 (most successful) | 228 | 25.2-59.3% | **31.5%** | 44% / 48% / 6.8% | 1.35x, 1.11x |

- The least successful fifth of searches (1,823) is made by **140 officers**, who search Hispanic drivers
  **twice as often** as the rest and are the most skewed toward Hispanic drivers.
- **If those searches had the median fifth's hit rate for the same finds: 764 fewer fruitless searches
  [380, 1,206] over 2016-2018 = 11% of all fruitless consent searches, ~500 of them of Black or Hispanic
  drivers.** Illustrative (assumes the same finds with fewer searches); bootstrap over officers.
- The most successful officers are also the least skewed (1.35x vs 1.9x).

---

## 9. Caveats that apply everywhere
- **Selective labels:** contraband is only observed when a search happened. "Avoided" means "among the
  searches that were made"; nothing is known about stops that were not searched.
- Consent searches only (the officer's discretionary decision); arrest, warrant, inventory and plain-view
  searches are excluded (cleaning protocol, step 5).
- `subject_race` is the officer's perception of race.
- Economics ignore the value of what is found, deterrence and the cost to the driver beyond the search;
  the break-even rate is an assumption, shown over a range.
- All models were trained on 2010-2015 (16.1% hit rate) and tested on 2016-2018 (21.4%): every group is
  under-predicted in level; compare groups, not levels.
- The officer features describe people. Officer IDs are hashed: report groups, never individuals.

## 10. Where this points for the recommendation
- **As a driver-scoring tool:** the best model (XGBoost + officer, AUC 0.64) is modest, and it forces a
  choice between an unlawful race-aware model and a race-blind one that over-scores minority drivers and
  would send them more fruitless searches. **Not recommended.**
- **As an officer-accountability tool:** the strongest, most stable signal is the officer. A 19-term PLTR
  (readable, +0.075 AUC over the team's scorecard, within 0.01 of XGBoost) or even the officer's track
  record alone can flag the officers whose searches mostly find nothing and who skew toward minority
  drivers. 140 officers account for the least successful fifth of searches; bringing them to the median
  would avoid ~11% of fruitless consent searches, two thirds of them of minority drivers.
- **Governance:** retrain yearly (2018 drift), monitor calibration by race, report officer groups and never
  individual scores publicly, and keep the officer's own history and the outcome test as review evidence,
  not as automatic sanctions (infra-marginality).

---

## Reproduce
```
python scripts_claude/build_officer_features.py     # a few minutes, brute-force check at the end
python scripts_claude/build_master.py               # master table, 123 columns
python scripts_claude/analyze_officer_disparity.py  # section 3
python scripts_claude/test_pltr.py                  # PLTR known-answer tests
python scripts_claude/evaluate_officer_models.py    # sections 4, 6, 7 (+ saves test predictions), ~25 min
python scripts_claude/officer_model_diagnostics.py  # sections 5 (by year, PLTR terms), 7 (calibration), 8
python scripts_claude/officer_holdout_stability.py --with-pltr   # section 5, ~55 min
```
