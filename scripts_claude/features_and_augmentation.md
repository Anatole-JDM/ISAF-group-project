# Nashville traffic stops — features and data augmentation

**What this document covers:** every feature built so far, where its data comes from, exactly how it was computed, how it was validated, and what it cannot be trusted for. Features planned but not yet built are listed at the end.
**Status:** 2026-09-24, branch `cleaned-data`.
**Companion documents:** `nashville_cleaning_protocol.md` (how the base table was cleaned) · `feature_engineering_plan.md` (design rationale and decisions).

---

## 1. Where things stand

| Block | Status | Output |
|---|---|---|
| Cleaned base table | ✅ built | `opp_data/processed/nashville_clean.parquet` — 3,088,286 stops |
| Step 1 — Neighbourhood context (Census ACS) | ✅ built, validated | `opp_data/features/nbh_features.parquet` — 44 feature/flag columns |
| Step 3 — Registration plate | ✅ built | `opp_data/features/plate_features.parquet` — 6 features |
| Step 2 — Who is stopped here vs. who lives here | ⏳ designed, not built | — |
| Step 5 — Policing activity nearby | ⏳ designed, not built | — |
| Step 6 — Time (hour, weekday, darkness, holidays) | ⏳ designed, not built | — |
| Step 4 — Road type | ⏸ parked by the team | — |
| Single modelling table (all features joined) | ⏳ not built | — |

Every feature table has exactly one row per `stop_id` (3,088,286 rows), so they join one-to-one onto the cleaned table.

### The modelling sample

The target is defined on **consent-only searches**: `search_conducted` and `raw_search_consent` are TRUE, and none of the other four raw search flags (arrest, warrant, inventory, plain view) is set. This excludes searches that also had a non-discretionary justification.

| | Consent-only searches | Contraband found |
|---|---|---|
| All | **58,939** | **16.9%** |
| Black | 32,353 | 15.7% |
| White | 21,252 | 20.8% |
| Hispanic | 4,651 | 7.9% |
| Asian / Pacific Islander | 282 | 18.8% |

`contraband_found` has no missing values in this sample. **95.2%** of these searches (56,083) have neighbourhood features at 800 m.

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

### 3.8 Limitations — state these in the deck

1. **Uniform composition within a tract.** The allocation assumes each tract's ACS make-up applies equally to all its blocks.
2. **Residents located as in 2010.** Davidson grew ~12% between 2010 and 2018, much of it in new developments (the Gulch, downtown). Blocks that were nearly empty in 2010 are under-weighted, increasingly so for later stops.
3. **Sampling noise in small counts.** Tracts have a median of only 143 unemployed people and 74 car-less households; the counts' coefficients of variation are **0.39** and **0.46** (vs 0.12–0.18 for renters, Black residents, degree holders). `unemployment_rel` removes the business cycle (29.5% → 21.2% of variation over time) but the remainder is mostly noise. These two are the least reliable neighbourhood features.
4. **Neighbourhood ≠ driver.** The circle describes where the stop happened, not who the driver is. This is weakest on interstates and parkways, where through-traffic dominates (decision D4, open).
5. **Neighbourhood racial make-up is a proxy for the driver's race.** Whether it enters the model or only the fairness analysis is decision D3 (open).
6. **Coverage.** 6.7% of stops have no usable coordinates and get no neighbourhood features. A ZIP-code fallback was planned but not built.
7. **Medians** (`_approx`) are approximations; use `income_pctile` for income.

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

`plate_state_cannabis_legal_at_stop` (recreational cannabis legal in the plate's state on the stop date) was designed as a hypothesis only. It is ethically loaded, would rest on ~2,000 out-of-state consent searches, and needs sourced legalisation dates. It needs a team decision.

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
| `date`, `time` | raw; the Step 6 features will be derived from them |

**Excluded from any model (recorded after the search decision):** `search_basis`, all `raw_search_*` flags, `search_person`, `search_vehicle`, `frisk_performed`, `contraband_drugs`, `contraband_weapons`, `arrest_made`, `citation_issued`, `warning_issued`, `outcome`, `notes`. `officer_id_hash` is kept out of the features and used for grouped validation and the officer-level analysis.

---

## 6. Features that change meaning over time — summary

| Feature | Why it tracks time | Handling |
|---|---|---|
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
| D3 | neighbourhood racial make-up: model input, or fairness analysis only | **open** |
| D4 | neighbourhood features on interstate/parkway stops | **open** |
| — | cannabis feature | **open** |

---

## 8. Still to build

| Step | Features | Depends on |
|---|---|---|
| 2 | recent stop composition by race within 800 m (previous 365 days, stop counts only); disparity ratio vs residents; driver–neighbourhood congruence (fairness analysis only) | Step 1 circles |
| 6 | hour, missing-hour flag, weekday, month, weekend, holidays, darkness at the time and place of the stop | cleaned table only |
| 5 | stops within 800 m over the previous 30 / 365 days, distinct officers, share of night stops, trend | cleaned table only |
| 4 | road type, intersection vs address (parked) | cleaned table only |
| — | ZIP-code fallback for the 6.7% without coordinates | ZIP-level ACS fetch |
| — | one modelling table joining the cleaned table and all feature tables on `stop_id` | all of the above |
