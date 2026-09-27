# Nashville traffic stops — features and data augmentation

**What this document covers:** every feature built so far, where its data comes from, exactly how it was computed, how it was validated, and what it cannot be trusted for. Features planned but not yet built are listed at the end.
**Status:** 2026-09-26, branch `main`.
**Companion documents:** `nashville_cleaning_protocol.md` (how the base table was cleaned) · `feature_engineering_plan.md` (design rationale) · `FINDINGS.md` (model results; section 5 and 11 test these features).

> **Read section 3.9 before using any `nbh*` column.** As built, the 41 neighbourhood columns jointly identify the year of the stop. With a split by time, that makes them a proxy for the period. The models use them only through a frozen-release repair, and even then they add nothing to predicting contraband.

---

## 1. Where things stand

| Block | Status | Output |
|---|---|---|
| Cleaned base table | ✅ built | `opp_data/processed/nashville_clean.parquet` — 3,088,286 stops (local) |
| Step 1 — Neighbourhood context (Census ACS) | ✅ built, validated — **use only frozen at one release (3.9)** | `opp_data/features/nbh_features.parquet` (local), `nbh_location_vintage.parquet` (local, needed for the frozen version) |
| Step 3 — Registration plate | ✅ built — not used by the models (recodes a column they already have) | `opp_data/features/plate_features.parquet` (local) |
| Step 6 — Time | ✅ built — **7 columns used by the models** (section 5) | `opp_data/features/time_features.parquet` (local) |
| Modelling table (all of the above, consent sample) | ✅ built | `data/nashville_consent_searches.parquet` + `data/nashville_columns.csv` (in git) |
| Step 2 — Who is stopped here vs. who lives here | ⏳ designed, not built | — |
| Step 5 — Policing activity nearby | ⏳ designed, not built | — |
| Step 4 — Road type | ⏸ parked by the team | — |

Every feature table has exactly one row per `stop_id` (3,088,286 rows), so they join one-to-one onto the cleaned table. The models in `src/` join on `raw_row_number`, which both pipelines keep: all 58,865 consent rows match.

### The modelling sample

The target is defined on **consent-only searches**: `search_conducted` and `raw_search_consent` are TRUE, and none of the other four raw search flags (arrest, warrant, inventory, plain view) is set. This excludes searches that also had a non-discretionary justification.

**Period: 2010-01-01 to 2018-12-31** (team decision, aligned with `src/config.py`). `data/nashville_consent_searches.parquet` still contains the 74 searches from January–February 2019 (58,939 rows): filter `date < 2019-01-01`.

| 2010–2018 | Consent-only searches | Contraband found |
|---|---|---|
| All | **58,865** | **16.9%** |
| Black | 32,311 | 15.7% |
| White | 21,227 | 20.8% |
| Hispanic | 4,645 | 7.9% |
| Asian / Pacific Islander | 282 | — (too few to report) |

`contraband_found` has no missing values in this sample. The two pipelines (`scripts_claude/` and `src/`) were built independently and produce the same 58,865 rows, group by group.

---

## 2. Pipeline and data sources

Run in this order (all outputs go under `opp_data/`, which is git-ignored):

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `clean_nashville.py` | raw OPP file (zip or extracted CSV) | `processed/nashville_clean.parquet` + manifest |
| 2 | `fetch_acs.py` | Census API (needs `CENSUS_API_KEY`) | `external/acs/acs5_tract_measures.parquet` + manifest |
| 3 | `fetch_block_pop.py` | Census API + block shapefile | `external/block_pop_2010_davidson.parquet` |
| 4 | `build_nbh_features.py` | 1–3 + tract shapefile | `features/nbh_features.parquet`, `nbh_location_vintage.parquet`, manifest |
| 5 | `build_plate_features.py` | 1 + state centres of population | `features/plate_features.parquet` + manifest |

External data, all from the US Census Bureau:

| Source | What it is | Used for |
|---|---|---|
| ACS 5-year estimates, releases 2006–2010 … 2014–2018 (9 releases) | survey-based tract statistics, published each December | all neighbourhood features |
| TIGER/Line 2010 census tracts, Tennessee (`tl_2010_47_tract10.zip`, 13.7 MB) | tract boundaries | locating stops, county line, neighbouring tracts |
| TIGER/Line 2010 census blocks, Davidson (`tl_2010_47037_tabblock10.zip`, 4.6 MB) | 10,775 blocks with interior points | where residents live inside a tract |
| 2010 Census SF1, table P001001, per block | exact block population (total 626,681 = official county count) | weighting |
| 2010 state centres of population (`CenPop2010_Mean_ST.txt`) | population-weighted centre of each state | plate distance |

---

## 3. Step 1 — Neighbourhood context

### 3.1 Why

A stop happens somewhere. The neighbourhood's make-up is both a plausible driver of police behaviour and a **proxy for the driver's race** (residential segregation). The features are needed for two separate purposes: as candidate predictors, and as the material for the course's step 2 — identifying which non-protected variables carry the unfairness.

### 3.2 Matching the data to the date of each stop (decision D1 = rule B)

ACS figures are 5-year averages. A stop in year *Y* uses **the release whose window ends in *Y* − 1**, so it only ever sees years before the stop:

| Stop year | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 |
|---|---|---|---|---|---|---|---|---|---|---|
| ACS release | 2006–10 | 2006–10 | 2007–11 | 2008–12 | 2009–13 | 2010–14 | 2011–15 | 2012–16 | 2013–17 | 2014–18 |

2010 is the only exception (its window includes the stop year); those 310,614 stops carry `acs_window_overlaps_stop = True`. All nine releases use the 2010 tract boundaries, so one geometry covers the whole period.

Using a release published after the stop date is **not target leakage** — the Census contains nothing about policing. It affects only how realistic the set-up is for a system deployed in real time.

### 3.3 Why a circle and not "the tract containing the stop" (decision D2)

Tract boundaries run down the middle of major roads, and most stops happen at road intersections. Measured on all stops:

| Distance to the nearest tract boundary | All stops | Intersections | Street addresses |
|---|---|---|---|
| ≤ 10 m | **29.1%** | **34.3%** | 3.7% |
| ≤ 100 m | 44.2% | 45.3% | 38.9% |

For roughly a third of stops, the containing tract is decided by geocoding error. The features are therefore computed over a **circle around the stop: 800 m (main), 1,200 m (robustness check)**.

### 3.4 How the circle is computed: population-weighted allocation

For a stop location ℓ, a circle of radius *r*, and each census tract *t*:

1. **Residents of tract *t* inside the circle** — the sum of the 2010 populations of the tract's blocks whose interior point lies within *r* of ℓ:
   *P*ₗₜ = Σ pop10(b) over blocks b of t within r of ℓ
2. **Weight** — the fraction of the tract's residents inside the circle:
   *w*ₗₜ = *P*ₗₜ / *P*ₜ, where *P*ₜ is the tract's total 2010 population
3. **Allocated counts** — every ACS count *C* (e.g. number of Black residents, number below poverty) is split in the same proportion:
   *C*ₗ = Σₜ *w*ₗₜ · *C*ₜ
4. **Shares are ratios of allocated counts**, never averages of tract percentages:
   share_black(ℓ) = Σₜ *w*ₗₜ · black(t) / Σₜ *w*ₗₜ · population(t)

Blocks with no residents (3,162 of 10,775: parks, industrial land, the airport) get zero weight automatically, so empty land never dilutes the result.

**Medians cannot be allocated** (a median of a circle is not a weighted average of medians). Income and age medians are therefore residents-weighted means of tract medians, marked `_approx`. The robust income feature is `income_pctile` (below).

**Stops just outside Davidson** (2,332 stops at 207 locations): they have no block populations, so they take the values of the tract that contains them (`nbh_geo_source = 'tract_pip'`).

### 3.5 Feature dictionary

Each feature exists twice: `nbh800_*` (main) and `nbh1200_*` (robustness). Ranges are for `nbh800_*` over all stops with features: minimum / median / maximum.

| Feature | Definition | ACS table | min / median / max |
|---|---|---|---|
| `pop_density` | allocated residents ÷ circle area (per km²) | B01003 | 0.4 / 1,171 / 4,187 |
| `share_black` | non-Hispanic Black ÷ population | B03002 | 0 / 0.27 / 0.98 |
| `share_white` | non-Hispanic white ÷ population | B03002 | 0.01 / 0.50 / 0.99 |
| `share_hispanic` | Hispanic (any race) ÷ population | B03002 | 0 / 0.06 / 0.57 |
| `share_asian` | non-Hispanic Asian ÷ population | B03002 | 0 / 0.02 / 0.31 |
| `poverty_rate` | below poverty line ÷ people with known poverty status | B17001 | 0 / 0.25 / 0.84 |
| `share_no_vehicle` | households with no vehicle ÷ occupied households | B25044 | 0 / 0.10 / 0.66 |
| `renter_share` | renter-occupied ÷ occupied housing units | B25003 | 0.01 / 0.57 / 1.00 |
| `unemployment_rate` | unemployed ÷ civilian labour force | B23025 (B23001 in 2006–10) | 0 / 0.09 / 0.42 |
| `unemployment_rel` | circle unemployment rate ÷ Davidson rate in the same release | as above | 0 / 1.15 / 6.84 |
| `share_bach_plus` | bachelor's degree or higher ÷ adults 25+ | B15003 (B15002 before 2012) | 0.02 / 0.27 / 0.86 |
| `residential_stability` | same house one year earlier ÷ population 1+ | B07003 | 0.51 / 0.77 / 0.98 |
| `share_drive_to_work` | commute by car, truck or van ÷ workers | B08301 | 0.26 / 0.90 / 1.00 |
| `income_pctile` | tract median household income ranked among Davidson's tracts within the release (0–1), then residents-weighted | B19013 | 0.01 / 0.35 / 1.00 |
| `median_income_approx` | residents-weighted mean of tract median incomes, nominal dollars | B19013 | 9,087 / 37,722 / 188,594 |
| `median_age_approx` | residents-weighted mean of tract median ages | B01002 | 14.6 / 32.5 / 55.7 |
| `income_weight_missing` | share of the circle's residents whose tract median income is suppressed | — | 0 / 0 / 0.19 |
| `residents_2010` | 2010 residents inside the circle | SF1 P001001 | 0 / 2,379 / 8,607 |

**Why two derived measures exist:**
- `income_pctile` rather than dollars: incomes are in each release's own year dollars, and a rank within the same release removes inflation and the business cycle without an external price index.
- `unemployment_rel` rather than the raw rate: 29.5% of the raw rate's variation is change over time at the same place (county unemployment went 7.5% → 8.8% → 4.6%), so the raw rate mostly tells a model *which year* a stop is in. The relative version keeps "high unemployment for its time and place".

### 3.6 Flags

| Flag | Meaning | Stops flagged |
|---|---|---|
| `nbh_geo_source` | `circle` / `tract_pip` (just outside Davidson) / `none` (no usable coordinates) | 2,879,269 / 2,332 / 206,685 |
| `acs_vintage` | ACS release used (window end year) | — |
| `acs_window_overlaps_stop` | 2010 stops, whose window includes the stop year | 310,614 |
| `geo_bucket` | location shared by ≥ 20 different address strings — a geocoder "bucket", mostly highway interchanges | 154,152 |
| `nbh800_circle_truncated` / `nbh1200_…` | circle crosses the county line (outside part has no block data) | 30,057 / 52,265 |
| `nbh800_low_residents` / `nbh1200_…` | fewer than 100 residents in the circle | 89,651 / 30,634 |

Circles with **no residents at all** leave every feature missing: 14,304 stops at 800 m, 1,596 at 1,200 m.

### 3.7 Validation

| Check | Result |
|---|---|
| Tracts per ACS release | 333 (161 Davidson + 172 neighbouring), identical set in all nine releases |
| Davidson population across releases | 612,884 (2006–10) → 684,017 (2014–18), smooth growth |
| Block populations | 10,775 blocks, total **626,681** = official 2010 count |
| Known-answer test: circle lying entirely inside one tract | expected 0.197666, got 0.197666 (10,170 such circles) |
| Residents allocated per tract sum to residents in the circle | asserted in the build |
| Shares outside [0, 1] | 0 |
| Employment fallback (2006–10) consistent with later releases | unemployment 7.5% (2006–10) → 8.1% (2007–11): no break at the table switch |

**Change over time at a fixed location** (share of each feature's variation, weighted by stops; decision D5 = no smoothing):

| `share_black` | `poverty_rate` | `income_pctile` | `residential_stability` | `unemployment_rate` | `unemployment_rel` |
|---|---|---|---|---|---|
| 2.5% | 7.4% | 7.9% | 14.6% | 29.5% | 21.2% |

For most features, differences between places dominate: changes over time are 2–8% of the variation. Smoothing across releases would add lag and blur real neighbourhood change, so it was not applied.

**This check was insufficient.** It measures each feature's change over time *separately*. It does not measure what a model can do with all 41 columns *together*, which is to recognise the year. See 3.9.

### 3.8 Limitations — state these in the deck

1. **Uniform composition within a tract.** The allocation assumes each tract's ACS make-up applies equally to all its blocks.
2. **Residents located as in 2010.** Davidson grew ~12% between 2010 and 2018, much of it in new developments (the Gulch, downtown). Blocks that were nearly empty in 2010 are under-weighted, increasingly so for later stops.
3. **Sampling noise in small counts.** Tracts have a median of only 143 unemployed people and 74 car-less households; the counts' coefficients of variation are **0.39** and **0.46** (vs 0.12–0.18 for renters, Black residents, degree holders). `unemployment_rel` removes the business cycle (29.5% → 21.2% of variation over time) but the remainder is mostly noise. These two are the least reliable neighbourhood features.
4. **Neighbourhood ≠ driver.** The circle describes where the stop happened, not who the driver is. This is weakest on interstates and parkways, where through-traffic dominates (decision D4, moot while the features are not model inputs).
5. **Neighbourhood racial make-up is a proxy for the driver's race.** Measured: the neighbourhood columns predict the driver's race at AUC 0.69–0.73 (section 3.9). They are therefore used in the fairness analysis, not as model inputs (D3).
6. **Coverage.** 6.7% of stops have no usable coordinates and get no neighbourhood features. A ZIP-code fallback was planned but not built.
7. **Medians** (`_approx`) are approximations; use `income_pctile` for income.
8. **As built, the columns identify the year** (section 3.9). Use them only frozen at a single release.

### 3.9 The columns identify the year: the flaw, the fix, and what it shows

*Found in review by leottawa (`FINDINGS.md`, section 5; commits `222f4e8` → `e14dba2` on `tawa`).*

**The flaw.** Rule B gives each stop the Census release of the year before it, so each value describes the **place in a given year**, not the place. Two columns make this explicit: `acs_vintage` is exactly `year − 1`, and `acs_window_overlaps_stop` is TRUE only in 2010. But the problem is not limited to them. Taken together, the 41 `nbh*` columns form a fingerprint of the release:

| Predict which side of the 2016 split a stop is on, from | AUC |
|---|---|
| the 41 `nbh*` columns, as built | **0.998** |
| the base features already in the model, for scale | 0.712 |

With a split by time (train before 2016, test from 2016), every test stop carries release values never seen in training, and the hit rate rises across the split (16.1% → 21.4%). A model can therefore learn year-related patterns that do not transfer. No single column causes it (each one alone scores 0.55–0.65); it is their combination. The per-feature check in 3.7 could not see this.

**Why the design missed it.** Rule B answered the question "does the feature use information from after the stop?" (it does not, and the columns contain no outcome data). It did not ask "can the feature reveal *when* the stop happened?", which is what matters once the evaluation splits by time.

**The fix.** Take one release for every stop, regardless of its year, from the location × release table (`opp_data/features/nbh_location_vintage.parquet`, 568,044 rows = ~63,000 locations × 9 releases). Each value then describes the place only. `src/config.py` pins the **2013 release** (`NBH_PIN_VINTAGE`; 2016 gives the same result).

| Predict post-2016 from | AUC |
|---|---|
| `nbh*` as built | 0.998 |
| **`nbh*` frozen at one release** | **0.59–0.62** |
| base features already in the model | 0.712 |

Frozen, the columns carry *less* information about the period than the features already in the model. The remaining ~0.6 is genuine change in where stops happened over time, not a release artefact.

**What it shows.**

| Model (XGBoost, AUC) | Without neighbourhood | + neighbourhood as built | + neighbourhood frozen |
|---|---|---|---|
| Base features | **0.5663** | 0.5498 (−0.0165) | 0.5636 (−0.0027) |

1. **The neighbourhood does not predict contraband.** Repaired, the features still add nothing. The small gain in some combinations (+0.002) is a tenth of the variation obtained just by changing the training sample (0.024).
2. **The neighbourhood predicts race.** From the neighbourhood columns alone, the driver's race (Black vs white) is predicted with AUC **0.69** (time split) to **0.73** (random split). This is better than any model predicts contraband. It is direct evidence that removing `subject_race` does not make a model race-blind.
3. **Decision D3 is therefore settled by the evidence:** the neighbourhood features are **not model inputs**. They are the evidence for the proxy analysis in the fairness section (course step 2: identify the variables that carry the unfairness).

**Practical consequence.** The frozen version needs `nbh_location_vintage.parquet` (88.7 MB), which is local only (`opp_data/` is not in git). Whoever reruns the frozen experiments needs that file shared outside GitHub, or has to rerun `build_nbh_features.py`.

---

## 4. Step 3 — Registration plate

### 4.1 Features

| Feature | Definition |
|---|---|
| `plate_state` | state of registration (50 states + DC); `missing` when not recorded |
| `plate_missing` | plate state not recorded — **data-quality flag, see 4.3** |
| `plate_out_of_state` | recorded and not TN; left missing (not FALSE) when unknown |
| `plate_group` | TN / border state (KY, VA, NC, GA, AL, MS, AR, MO — the 8 states touching Tennessee) / other state / missing |
| `plate_region` | TN / rest of the South / Midwest / Northeast / West / missing (Census regions) |
| `plate_distance_km` | great-circle distance from the plate state's 2010 centre of population to downtown Nashville (36.1627, −86.7816). Examples: TN 55 km, KY 230, GA 407, TX 1,141, NY 1,207, CA 2,922 |

The data contained exactly 51 codes plus missing: no foreign, government or invalid codes (the build asserts this).

### 4.2 What the data shows (descriptive only)

| Plate group | Stops | Search rate | Consent searches | Hit rate |
|---|---|---|---|---|
| TN | 2,879,812 (93.3%) | **4.18%** | 56,453 | 16.8% |
| Border state | 73,212 (2.4%) | 3.15% | 831 | 18.2% |
| Other state | 103,813 (3.4%) | 3.10% | 1,174 | 17.9% |
| Missing | 31,449 (1.0%) | 5.02% | 481 | 21.6% |

- **Out-of-state drivers are searched less than Tennessee drivers**, not more — the opposite of the "drug corridor" assumption. Their hit rates are similar.
- **33 of the 51 states have fewer than 30 consent searches.** `plate_state` itself is too thin to model; use `plate_group` or `plate_region`. Even the Northeast (105 searches) and West (165) are thin.

### 4.3 Two time artefacts

**`plate_missing` is a recording change, not a signal.** Missing plates are ~0.1–0.6% of stops until 2016, then **5.2% (2017), 6.3% (2018), 6.2% (2019)**. Within each period, missing and recorded plates have the same hit rate:

| Period | Plate recorded | Plate missing |
|---|---|---|
| 2010–16 | 16.4% (n = 54,066) | 15.7% (n = 70) |
| 2017+ | 22.9% (n = 4,392) | 22.6% (n = 411) |

The group's headline 21.6% exists only because 85% of missing-plate searches happen after 2017, when hit rates were higher across the board. **`plate_missing` must not be used as a predictor** — it is a "2017 or later" indicator.

**`plate_out_of_state` drifts upward:** 3.4% (2010) → 10.5% (2019), with a step between 2012 and 2013. It partly encodes the period; monitor it in the stability analysis.

### 4.4 Not built

`plate_state_cannabis_legal_at_stop` (recreational cannabis legal in the plate's state on the stop date) was designed as a hypothesis only. It is ethically loaded, would rest on ~2,000 out-of-state consent searches, and needs sourced legalisation dates. Not built; worth one sentence in the deck as a hypothesis deliberately not tested.

### 4.5 Use in the models

**Not used.** The models in `src/` already include `vehicle_registration_state`, and these columns only recode it. `plate_missing` is on the never-join list (`config.NEVER_JOIN`) because of the 2017 recording change.

---

## 4b. Step 6 — Time

Built by `build_time_features.py` from `date` and `time`. Times are local Nashville clock times, converted to UTC with daylight saving time before computing the sun's position.

| Feature | Definition |
|---|---|
| `hour`, `hour_missing`, `minute_of_day` | clock time; hour = −1 and a flag for the 5,465 stops without a time |
| `hour_sin`, `hour_cos` | cyclical time of day: makes 23:59 and 00:01 neighbours |
| `day_of_week`, `is_weekend`, `month`, `month_sin`, `month_cos` | calendar |
| `is_federal_holiday`, `is_holiday_window` | observed US federal holidays, and ±1 day |
| `is_dst` | daylight saving time in effect (65.3% of stops) |
| `sun_elevation_deg`, `light_period`, `is_dark` | sun's height; day / twilight / dark (dark = sun more than 6° below the horizon) |
| `minutes_after_civil_dusk`, `in_intertwilight` | for the veil-of-darkness test: the window 17:00–20:38, where the same clock time is light on some days and dark on others (535,077 stops) |
| `time_heaped` | time recorded at :00 or :30 (officer rounding; 4.66% vs 3.33% expected) |
| `dst_ambiguous`, `dst_nonexistent` | the repeated / skipped hour of a DST switch (500 / 1 stops) |

**Checks built into the script:** the sun's maximum height on the solstices and equinox matches its exact astronomical value (77.27° vs 77.28°, 30.40° vs 30.40°); no stop between 11:00 and 14:00 comes out dark, every stop between 01:00 and 04:00 does; sunset on 21 June 2015 computes to 20:07 CDT.

**Use in the models:** 7 columns (`config.TIME_EXTRA`): `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `time_heaped`, `is_federal_holiday`, `is_holiday_window`. They are the one feature block that improves the models: XGBoost AUC 0.5663 → **0.5733** (+0.007), PR-AUC 0.2552 → 0.2702 (`FINDINGS.md`, section 11). That is about 11% of a very small signal and does not change the recommendation. The other time columns duplicate encodings `src/` already builds and are on the never-join list.

---

## 5. Features available directly from the cleaned table

Not engineered, but usable as they are (see the cleaning protocol for their quality):

| Column | Notes |
|---|---|
| `subject_age` | kept exactly as recorded (242 under 15, 2,299 over 90) |
| `subject_sex` | male / female / missing (0.4%) |
| `subject_race` | protected attribute — for evaluation |
| `reason_for_stop` | 9 categories; mix drifts over time (registration 1.3% → 11.4%) |
| `precinct`, `zone`, `reporting_area` | police geography; missingness changes in 2017 (~19% → ~0.1%) |
| `date`, `time` | raw; the Step 6 features are derived from them (section 4b) |

**Excluded from any model (recorded after the search decision):** `search_basis`, all `raw_search_*` flags, `search_person`, `search_vehicle`, `frisk_performed`, `contraband_drugs`, `contraband_weapons`, `arrest_made`, `citation_issued`, `warning_issued`, `outcome`, `notes`. `officer_id_hash` is kept out of the features and used for grouped validation and the officer-level analysis.

---

## 6. Features that change meaning over time — summary

| Feature | Why it tracks time | Handling |
|---|---|---|
| **all 41 `nbh*` columns, jointly** | each stop gets the release of its year, so together they identify the year (AUC 0.998) | **freeze at one release** (section 3.9) |
| `acs_vintage` | exactly `year − 1` | never a feature |
| `acs_window_overlaps_stop` | TRUE only in 2010 | never a feature |
| `unemployment_rate` | business cycle | use `unemployment_rel` |
| `plate_missing` | recording change in 2017 | data-quality flag only |
| `plate_out_of_state` | share triples over the period | keep, monitor in stability analysis |
| `precinct` / `zone` / `reporting_area` | missingness drops in 2017 | missing as its own level; report the table |
| `reason_for_stop` | category mix drifts | keep, document the drift |
| consent share of searches, hit rate | 61% → 34%; 14% → 25% | rolling-origin validation, per-fold base rates |

---

## 7. Decisions log

| # | Decision | Taken |
|---|---|---|
| D1 | ACS release by rule B (window ending Y − 1) | team |
| D2 | population-weighted circle, not point-in-polygon | team, on the boundary evidence |
| D5 | no smoothing across releases | team, on the variance evidence |
| — | `unemployment_rel` added | team |
| — | robustness radius 1,200 m (400 m left 18.7% of stops with < 100 residents) | team |
| — | period 2010–2018 (the two months of 2019 would make a meaningless yearly fold) | team, aligned with `src/` |
| D6 | neighbourhood features frozen at the **2013 release** for any model use | leottawa, on the year-identification evidence (3.9) |
| D3 | neighbourhood racial make-up: **not a model input**; used as evidence in the proxy analysis | settled by the evidence (3.9): adds nothing to contraband, predicts race at 0.69–0.73 |
| — | time features: the 7 columns of `config.TIME_EXTRA` enter the models | leottawa (+0.007 AUC) |
| — | plate features: not used (recode `vehicle_registration_state`) | leottawa |
| D4 | neighbourhood features on interstate/parkway stops | moot while the features are not model inputs |
| — | cannabis feature | not built |

---

## 8. Still to build (optional — none is needed for the deliverables)

| Step | Features | Depends on |
|---|---|---|
| — | rebuild the neighbourhood features frozen at one release inside `build_nbh_features.py`, so the modelling table ships the safe version | a `--pin-vintage` option |
| 2 | recent stop composition by race within 800 m (previous 365 days, stop counts only); disparity ratio vs residents; driver–neighbourhood congruence (fairness analysis only) | Step 1 circles |
| 5 | stops within 800 m over the previous 30 / 365 days, distinct officers, share of night stops, trend | cleaned table only |
| 4 | road type, intersection vs address (parked) | cleaned table only |
| — | ZIP-code fallback for the 6.7% without coordinates | ZIP-level ACS fetch |
