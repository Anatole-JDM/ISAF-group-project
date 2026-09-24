# Nashville traffic stops — diagnostics and cleaning protocol

**Dataset:** Stanford Open Policing Project, `tn_nashville_2020_04_01.csv.zip`
**Scope:** 3,092,351 stops, 2010-01-01 to 2019-03-24, Metro Nashville Police Department
**Produced by:** `scripts_claude/diag_nashville.py` → `diag_nashville.json` (read-only profiling, no modelling)
**Status:** every rule below is justified by an observation in Part 1. Decisions that rest on judgement rather than evidence are listed separately in Part 5.

---

## Part 1 — What the diagnostics show

### 1.1 Row integrity

| Check | Result | Reading |
|---|---|---|
| Rows | 3,092,351 | — |
| Duplicates on (date, time, officer, age, sex, race, location) | **3** | OPP's own dedup already worked |
| `search_conducted` is NA | **39** | ambiguous after OPP's merge |
| `type` | 100% `vehicular` | no pedestrian stops to separate |

### 1.2 Target quality

| Check | Result |
|---|---|
| Searches | 127,705 |
| `contraband_found` TRUE while `search_conducted` FALSE | **0** |
| `contraband_found` NA among searches | **0** (OPP filled them — see caveat) |
| Monthly hit rate | range 13.3% – 30.9%, smooth upward trend, **no discontinuity** |
| Officers with ≥30 searches | 779 |
| …of those with a 0% hit rate | **20** (3,063 searches) |
| …with a 100% hit rate | 0 |
| Officer hit-rate quantiles | p05 2.3% · p25 14.3% · **p50 21.9%** · p75 29.2% · p95 38.4% |

**Caveat that cannot be resolved from this file:** OPP's readme states that when `contraband_found` was NA and a search occurred, it was set to FALSE. The raw column is not passed through, so the imputed share is unmeasurable here. The indirect evidence is reassuring — no contradictions with the drugs/weapons flags, no yearly break, only 2.4% of searches come from zero-hit officers — but this belongs in the limitations slide.

### 1.3 Protected attributes

| Variable | Finding |
|---|---|
| `subject_race` | white 1,670,873 (54.0%) · black 1,165,871 (37.7%) · hispanic 164,814 (5.3%) · asian/PI 41,668 (1.3%) · **unknown 36,878 (1.2%)** · other 10,397 · NA 1,850. Shares stable across years (black 37.0% in 2010 → ~38% mid-period) — no sign of a recoding break. |
| `subject_age` | 0.03% missing; **242 rows under 15**, **2,299 over 90**; p01 = 17, p50 = 34, p99 = 75; no spike at 0 or 99 |
| `subject_sex` | male 1,827,043 · female 1,252,486 · **NA 12,822 (0.4%)** |

### 1.4 Date and time

- Full coverage: **111 of 111 months present, no gaps.**
- Volume declines steadily: 310,622 (2010) → 444,146 (2012) → 204,217 (2018) → 14,235 (2019, partial).
- **End of series — checked in detail** (`diag_tail.py`). Every calendar day from 2018-10 to 2019-03 is present, including 2018-12-25 (22 stops). Stops per *active day*:

| Period | Stops/day |
|---|---|
| 2012 | 1,214 |
| 2015 | 979 |
| 2017 | 673 |
| 2018-10 | 444 |
| 2018-11 | 346 |
| 2018-12 | 152 |
| 2019-01 | 188 |
| 2019-02 | 157 |
| 2019-03 | 168 (24 days) |

  The drop in December 2018 is sharp but then **plateaus for four months**. A truncated extract tapers only at its end; a stable plateau indicates a real change in stop activity, consistent with the decline running since 2012. **Conclusion: the tail is not missing data.** The only genuinely partial period is **2019-03** (24 of 31 days, the extract ends 2019-03-24).
- `time` is **99.8% present** (much better than NC). Minute distribution is near-uniform (`:30` 2.4%, `:00` 2.3%) — **no rounding artefact**. Hour peaks at 16:00 (5.9%) and 23:00 (5.7%).

### 1.5 Geography

| Check | Result |
|---|---|
| `lat`/`lng` missing | 6.05% |
| Coordinates outside a Davidson County box (35.9–36.5 N, −87.2 to −86.4 W) | 20,394 (0.66% of all rows) |
| Distinct coordinates (4 dp) | 60,601 |
| **Share of stops at the 10 most frequent coordinates** | **9.92%** |
| Coordinates appearing exactly once | 43.6% |
| Decimal precision | mixed: 7 dp (2.07M), 14 dp (430k), 6 dp (331k), 5 dp (26k) |
| `precinct` / `zone` missing | 12.62% — 17.6% (2010), 7.2% (2012), 19.6% (2014–15), **0.1% from 2017** |
| `reporting_area` missing | 10.75%, same yearly pattern; 2,036 levels |

Two things stand out: **10% of stops share ten coordinates** (almost certainly default or centroid geocodes, not real stop locations), and geocoding completeness **changes regime in 2017**.

### 1.6 Categorical features

- **`violation` is a duplicate of `reason_for_stop`** — both have the same 9 levels with identical shares. Keep one.
- `reason_for_stop` mix **drifts materially**: registration 1.3% (2010) → 11.4% (2018); vehicle equipment 39.2% → 25.9%; investigative stop 3.2% → 1.3%.
- Officers: 2,295 total; median 545 stops each; 2,053 active pre-2017, 1,319 post-2017; **9.7% of post-2017 stops come from officers never seen before 2017**.
- `vehicle_registration_state`: TN 93.2%, missing 1.03% → out-of-state ≈ **5.8%**.

### 1.7 Sample definition — the most important finding

OPP's `search_basis` is a *derived* field. The raw flags tell a different story:

| Raw flag | Count among searches |
|---|---|
| `raw_search_consent` | 69,430 |
| `raw_search_arrest` | 33,856 |
| `raw_search_plain_view` | 18,443 |
| `raw_search_inventory` | 5,426 |
| `raw_search_warrant` | 2,966 |

Number of raw flags set per search: **0 → 16,013** · 1 → 95,521 · 2 → 14,120 · 3 → 1,859 · 4 → 177 · 5 → 15.

Two consequences:

1. **OPP's "probable cause" category (16,013) is exactly the set of searches with no raw flag at all.** It is a residual label, not a recorded justification. It should not be treated as a genuine probable-cause search.
2. **10,456 of the 69,430 consent-flagged searches also carry another flag** (mostly incident-to-arrest). OPP's priority rule assigns them elsewhere, giving `search_basis == 'consent'` = 67,629. A clean discretionary sample should require consent **and no other flag**: ≈ **58,974 searches**.

**Composition drift:** the consent share of searches falls 61.4% (2010) → 43.2% (2016) → 33.6% (2018). A temporal split therefore changes the population, not only the period.

### 1.8 Drift inside the consent sample

| Year | n | Hit % | Male % | Mean age | Black % | Most common stop reason |
|---|---|---|---|---|---|---|
| 2010 | 9,041 | 14.0 | 82.7 | 30.2 | 52.7 | vehicle equipment |
| 2012 | 10,184 | 17.5 | 82.0 | 30.8 | 54.6 | vehicle equipment |
| 2014 | 9,371 | 18.4 | 77.9 | 31.5 | 57.2 | moving violation |
| 2016 | 5,069 | 20.6 | 79.0 | 31.9 | 53.3 | moving violation |
| 2018 | 2,778 | 24.9 | 78.9 | 32.5 | 54.8 | moving violation |

The base rate rises from 14.0% to 24.9% across the period. **Any single train/test split by time inherits an ~11-point base-rate shift.**

Hit rate by race within consent searches: **white 21.5%** (n=24,580) · **black 17.2%** (n=36,979) · **hispanic 8.6%** (n=5,308).

Group sizes per year (consent): black 4,766 → 1,521; white 3,207 → 1,026; **hispanic 944 (2010) → 206 (2018)**. Hispanic cells get thin after 2016.

---

## Part 2 — Cleaning protocol

### 2.0 Changelog — decisions taken by the team (2026-09-23)

**Applied:**

| # | Action | Rows / columns affected | Decided by |
|---|---|---|---|
| A1 | Drop rows where `search_conducted` is NA | **39 rows** | team |
| A2 | Drop exact duplicates on (date, time, officer, age, sex, race, location) | **3 rows** | team |
| A3 | Drop the column `violation` (identical to `reason_for_stop`, same 9 levels, same shares) | **1 column** | team |
| A4 | Drop 2019-03 (partial month: 24 of 31 days, extract ends 2019-03-24) | **4,023 rows** | evidence, `diag_tail.py` |

**Row count after cleaning: 3,092,351 − 39 − 3 − 4,023 = 3,088,286 stops.**

**Explicitly rejected:**

| # | Proposed | Decision |
|---|---|---|
| R1 | Bound `subject_age` to [15, 90] | **Rejected — ages kept exactly as recorded.** The 242 rows under 15 and 2,299 over 90 stay untouched. |
| R2 | Cut the series at 2018-10-31 as "incomplete" | **Rejected — the premise was wrong.** Every day is present; the late-2018 drop is a real activity change, not missing data (Part 1.4). |

**Still open** (listed in Part 5 / Part 6): treatment of `unknown` race, the consent-exclusivity filter, the geography decision pending the top-10-coordinate check.

### Step 1 — Row-level exclusions

Applied as A1, A2 and A4 above. No filter on `type` — all stops are vehicular.

### Step 2 — Variable-level handling

| Variable | Rule | Status |
|---|---|---|
| `violation` | **Dropped** — duplicate of `reason_for_stop` | applied (A3) |
| `subject_age` | **Kept as recorded**, no bounds, no flags | applied (R1) |
| `subject_sex` | Keep `male` / `female`; NA (12,822) becomes its own level `unknown_sex` | proposed |
| `subject_race` | Keep white, black, hispanic, asian/pacific islander. `unknown` (36,878), `other` (10,397) and NA (1,850) **excluded from fairness comparisons but kept in training**, so the training sample matches deployment | **open** |
| `time` | Keep as recorded (99.8% present). Derive `hour`; where missing, `hour = -1` plus an explicit `hour_missing` indicator — **never median imputation** | proposed |
| `reason_for_stop` | Keep all 9 levels; NA (8,020) → `unknown_reason`. Document the drift in Part 1.6 rather than correcting it | proposed |
| `vehicle_registration_state` | Derive `out_of_state` (TN vs not). Missing (1.03%) → separate level, not FALSE | proposed |
| `lat` / `lng` | See Step 3 | **blocked** on the top-10-coordinate check |
| `precinct`, `zone`, `reporting_area` | Missing → explicit `unknown` level. **Do not** impute: missingness is time-dependent (Step 4) | proposed |
| `officer_id_hash` | Not a model feature (Part 3). Retained for grouped validation and heterogeneity analysis | proposed |

### Step 3 — Geography (needs one verification first)

1. Flag coordinates outside the Davidson box (20,394) as missing.
2. **Before using coordinates, inspect the 10 most frequent points (9.92% of all stops).** If they are precinct buildings or default geocodes, exclude those coordinates (set to missing) rather than letting the model learn a fictitious hotspot. *This check is not yet run.*
3. Round coordinates to **4 decimals (~11 m)** to normalise the mixed precision (7 / 14 / 6 dp).
4. Preferred geographic feature: **`reporting_area`** (2,036 levels) or a census-tract join, not raw lat/lng, because the areas are stable across the geocoding regime change while precision is not.

### Step 4 — The 2017 geocoding regime change

`precinct`, `zone` and `reporting_area` go from ~19% missing (2014–16) to ~0.1% (2017+). A model trained before 2017 and tested after would see a different missingness structure, and "geography present" would act as a proxy for period. Two acceptable options:

- **(a)** Restrict the geographic analysis to **2017-01-01 onwards**, where coverage is near-complete; or
- **(b)** Keep the full period, use `unknown` as a level, and **report the missingness-by-year table** so the jury can see the artefact.

Recommendation: **(b)** for the main model, **(a)** for the spatial/census analysis.

### Step 5 — Sample definitions

**Primary sample — discretionary (consent-only) searches:**

```
search_conducted == TRUE
AND raw_search_consent == TRUE
AND raw_search_arrest == FALSE
AND raw_search_warrant == FALSE
AND raw_search_inventory == FALSE
AND raw_search_plain_view == FALSE
AND date <  2019-03-01
```

≈ **58,900 rows** *(est.; 58,974 before dropping the partial month, which holds ~150 consent searches)*. Target `contraband_found`, base rate ≈ 18–19%.

This is stricter than `search_basis == 'consent'` (67,629) and is justified by Part 1.7: 10,456 consent searches carry a second, non-discretionary justification.

**Secondary sample — all searches** (127,705) with a `basis` stratifier, for robustness. OPP's `probable cause` label must be renamed **`no_basis_recorded`**, since it is the no-flag residual.

**Audit sample — all stops** (~3.07M after Step 1) with target `search_conducted`, for the model of officer discretion.

### Step 6 — Validation design

The base rate moves 14.0% → 24.9% and the consent share moves 61% → 34%. A single 80/20 time split would conflate three things: period, population and base rate. Protocol:

1. **Rolling-origin validation** across the period: train 2010–2013 → test 2014; train 2010–2014 → test 2015; and so on to 2018. Report AUC per fold.
2. **Final held-out period: 2017-01-01 to 2018-10-31** (~5,600 consent searches), used once.
3. Report **bootstrap 95% confidence intervals** on every AUC — with test folds of a few thousand rows and a ~20% base rate, differences under roughly 0.03 AUC will not be distinguishable.
4. Group folds by `officer_id_hash` in a **second** validation run, to check whether performance depends on having seen the officer in training (9.7% of post-2017 stops come from unseen officers).
5. Report the **base rate of every fold** next to its AUC.

---

## Part 3 — Leakage: variables excluded from X

Recorded at or after the search decision, so unusable as predictors:

`search_basis`, all `raw_search_*` flags, `search_person`, `search_vehicle`, `frisk_performed`, `contraband_drugs`, `contraband_weapons`, `arrest_made`, `citation_issued`, `warning_issued`, `outcome`, `notes`.

`officer_id_hash` is excluded on different grounds: a model that predicts contraband from *who made the stop* is modelling discretion, not risk. It stays as a grouping variable for validation and for the heterogeneity analysis.

**Candidate feature set (X):** `hour` + `hour_missing`, day of week, month, year, `subject_age` (+ missing flag), `subject_sex`, `reason_for_stop`, `out_of_state`, `reporting_area` (or precinct/zone), optionally census-tract attributes.
**Protected attributes (D), for evaluation:** `subject_race`, `subject_sex`, `subject_age`.

Note `subject_sex` and `subject_age` appear in both lists. That is deliberate and must be stated explicitly in the deck: they are legitimate predictors *and* protected attributes, which is precisely the tension the course's mitigation step addresses.

---

## Part 4 — Statistical power for the fairness tests

Consent sample, per year: black 4,766 → 1,521 · white 3,207 → 1,026 · **hispanic 944 → 206**.

- Pooled over the period, all three groups support group-level metrics comfortably (black ≈ 37k, white ≈ 24.6k, hispanic ≈ 5.3k before the exclusivity filter).
- **Year-by-year** fairness metrics are unreliable for Hispanic drivers after 2016. Either pool years for that group or report confidence intervals wide enough to show it.

---

## Part 5 — Decisions that rest on judgement, not evidence

Settled (Part 2.0): ages kept as recorded; `violation` dropped; duplicates and NA-outcome rows removed; the series kept in full except the partial month 2019-03.

Still to be decided, and defended in the presentation:

1. **Keeping `unknown` race in training but out of the fairness comparison** — the alternative (dropping the rows) makes the training sample differ from the deployment population.
2. **Consent-only exclusivity filter** — a stricter definition than OPP's, costing ~8,700 searches in exchange for a cleaner decision definition.
3. **Excluding `officer_id_hash` from X** — a modelling ethics choice, not a technical one.
4. **Option (b) in Step 4** — keeping pre-2017 geography with an `unknown` level rather than restricting the period.
5. **How to treat the post-2018-11 activity regime** — the stop rate falls to roughly a third and stays there. Keep it as ordinary data, or treat it as a separate regime in the stability analysis?

---

## Part 6 — Open items before modelling

1. **Inspect the top-10 coordinates** (9.92% of stops) — not yet run; blocks the geography decision in Step 3.
2. Confirm the exact row counts after Steps 1 and 5 (all figures marked *est.*).
3. Decide whether to add the census-tract join (requires TIGER tract boundaries; OPP publishes no shapefile for Nashville).
