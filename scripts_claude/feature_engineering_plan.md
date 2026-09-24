# Nashville — feature engineering and augmentation plan

**Base data:** cleaned stops per `nashville_cleaning_protocol.md` Part 2.0 — **3,088,286 stops**; consent-exclusive searches = **58,939**.
**Status:** design only. Nothing below has been built. Every step lists its blind spots and how they are handled.
**Evidence:** `diag_geo.py` → `diag_geo.json` (read-only), plus HEAD requests to census.gov (no download).

---

## Part 0 — What the geography and plate profiling showed

### 0.1 The top-10 coordinates are mostly real hotspots, not default geocodes

This corrects the protocol's earlier hypothesis. Most of the most frequent points carry **one consistent intersection string** across all ten years:

| Coordinate | Stops | Location string (share of that point) |
|---|---|---|
| 36.0796, −86.7264 | 14,975 | `HARDING PL & NOLENSVILLE PIKE` (14,970 of 14,975) |
| 36.2173, −86.6933 | 9,964 | `BRILEY PKWY N & MCGAVOCK PIKE` (9,961) |
| 36.0454, −86.7133 | 8,788 | `NOLENSVILLE PIKE & OLD HICKORY BLVD` (8,781) |
| 36.1165, −86.6976 | 8,167 | `MCGAVOCK PIKE & MURFREESBORO PIKE` (8,162) |
| 36.1077, −86.6769 | 7,656 | `DELL PKWY & MURFREESBORO PIKE` (7,656, one string only) |

These are genuine enforcement hotspots on major arterials. **But some points are geocoder buckets:**

| Coordinate | Stops | Distinct strings | Problem |
|---|---|---|---|
| 36.0790, −86.9551 | 10,424 | **181** | `I 40 W & JEFFERSON ST` and `I 40 W & BROADWAY` (downtown) snapped to a point ~15 km west |
| 36.1538, −86.6855 | 11,798 | 68 | several Briley Pkwy interchanges collapsed together |
| 36.1370, −86.7250 | 9,644 | 70 | I-40 and Murfreesboro Pike locations collapsed |
| 36.1221, −86.7024 | 32,024 | 58 | Briley Pkwy / Murfreesboro Pike / Two Rivers Pkwy |

**Interstate and parkway stops are geocoded imprecisely; arterial intersections are geocoded well.**

### 0.2 Coordinate quality

- **20,377 coordinates fall outside Davidson County**, spanning latitude 31.9–40.3 and longitude −115.1 to −74.8 (Arizona to New Jersey), with one wrong point repeated many times (35.7276, −86.9605). These are geocoding errors → set to missing.
- **Missingness changes regime in 2014:** 2.5–3.2% missing in 2010–2013, **8.3–10.7% in 2014–2018**. Missing geography is therefore correlated with period.
- The mix of decimal precisions (7 / 15 / 14 / 6 dp) is **constant across years** — several geocoding sources in parallel, not a pipeline change.

### 0.3 The `location` string is a feature source in itself

100,122 distinct values, format `STREET A & STREET B, CITY, TN, ZIP`.

| Pattern | % of stops | % of searches | Consent hit rate (n) |
|---|---|---|---|
| any interstate (I-24/40/65/440) | 8.07 | 5.66 | **14.4%** (2,886) |
| Briley Pkwy | 4.54 | 2.97 | **22.7%** (1,222) |
| Ellington Pkwy | 0.82 | 0.75 | 16.9% (425) |
| intersection (`&`) | 77.11 | 77.32 | 17.1% (45,563) |
| street address (starts with a number) | 16.51 | 17.38 | 18.3% (10,062) |

It also contains a **ZIP code**, usable as a fallback geography when coordinates are missing.

### 0.4 Registration state

- 52 codes, **all valid** (no code has fewer than 20 stops).
- **TN share falls from 96.4% (2010) to 84.3% (2018)** — out-of-state plates quadruple over the period. Time-dependent feature.

| Plate group | Stops | Search rate | Consent n | Consent hit rate |
|---|---|---|---|---|
| TN | 2,879,812 | 4.18% | 56,453 | 16.8% |
| border states (KY, AL, GA, MS, AR, MO, NC, VA) | 73,212 | **3.15%** | 831 | 18.2% |
| other states | 103,813 | **3.10%** | 1,174 | 17.9% |
| **missing** | 31,449 | **5.02%** | 481 | **21.6%** |

Out-of-state drivers are searched **less** than Tennessee drivers, not more — the opposite of the "drug corridor" prior. A missing plate state carries the highest search and hit rates.

---

## Step 1 — Census / ACS neighbourhood context

### 1.1 Temporal matching — the core design decision

ACS 5-year estimates are released each December for the preceding five-year window. Every vintage from 2006–2010 to 2014–2018 uses **2010 tract boundaries**, so one geometry file covers the whole study period.

| ACS 5-year vintage | Released |
|---|---|
| 2005–2009 | Dec 2010 *(2000 tract boundaries)* |
| 2006–2010 | Dec 2011 |
| 2007–2011 | Dec 2012 |
| 2008–2012 | Dec 2013 |
| 2009–2013 | Dec 2014 |
| 2010–2014 | Dec 2015 |
| 2011–2015 | Dec 2016 |
| 2012–2016 | Dec 2017 |
| 2013–2017 | Dec 2018 |
| 2014–2018 | Dec 2019 |

Three possible rules for a stop in year *Y*:

| Rule | Vintage used | Pros | Cons |
|---|---|---|---|
| **A. Contemporaneous** | window centred on *Y* (ends *Y*+2) | best description of the neighbourhood that year | uses data describing the future; 2018–19 stops need vintages on 2020 boundaries (2016–2020, 2017–2021), so a crosswalk |
| **B. Reference-lagged** *(recommended)* | window ending *Y*−1 | describes only the years before the stop; single 2010 geometry | some vintages were published after the stop date |
| **C. Publication-aware** | latest vintage released before the stop date | exactly what a deployed system could have known | 2010 stops have no tract-level ACS on 2010 boundaries; needs a 2000→2010 crosswalk |

**Rule B mapping:** 2010 → 2006–2010 · 2011 → 2006–2010 · 2012 → 2007–2011 · 2013 → 2008–2012 · 2014 → 2009–2013 · 2015 → 2010–2014 · 2016 → 2011–2015 · 2017 → 2012–2016 · 2018 → 2013–2017 · 2019 → 2014–2018.
Only 2010 is imperfect (its window includes the stop year); those rows get a flag `acs_window_overlaps_stop = 1`.

**Blind spot — anachronism is not target leakage.** ACS contains no policing data, so a vintage published after the stop cannot leak the outcome. Rule B's cost is realism for a deployed system, not contamination of the target. This distinction should be stated in the deck.

### 1.2 Variables

All from ACS 5-year detailed tables, at tract level:

| Concept | Table | Derived feature |
|---|---|---|
| Population | B01003 | total population; density = population / land area (TIGER `ALAND10`) |
| Race / ethnicity | B03002 | shares: non-Hispanic white, non-Hispanic Black, Hispanic, non-Hispanic Asian — aligned with OPP's categories, where Hispanic is exclusive |
| Income | B19013 | median household income |
| Poverty | B17001 | share below the poverty line |
| Vehicles | B25044 | share of households with no vehicle |
| Tenure | B25003 | renter share |
| Employment | B23025 | unemployment rate |
| Education | B15003 (B15002 before 2012) | share with a bachelor's degree or more |
| Age | B01002 | median age |
| Residential stability | B07003 | share in the same house one year earlier |
| Commuting | B08301 | share driving to work |

### 1.3 Blind spots and how each is handled

| # | Blind spot | Handling |
|---|---|---|
| 1 | **Table definitions change across vintages** (education moved from B15002 to B15003 in the 2012 vintage) | Verify every variable code against each vintage's `variables.json` before download; fail loudly on a mismatch |
| 2 | **Inflation** — each vintage reports income in its own end-year dollars | Convert to constant 2018 dollars with CPI-U-RS, **and** compute the within-vintage percentile rank across Davidson tracts |
| 3 | **Tract-level sampling error** — margins of error are large for small groups | Keep features with large denominators; carry the coefficient of variation; flag tracts with population under 500 |
| 4 | **Non-residential tracts** (airport, industrial, downtown core, universities) | Flag low-population and high group-quarters tracts; their ratios are unstable |
| 5 | **Stops sit on tract boundaries.** Census tract boundaries frequently follow major roads, and 77% of stops are at road intersections | Measure the share of stops within 100 m of a boundary **after** download. If material, use a buffer (below) as the primary method |
| 6 | **Geocoding errors** (0.2) | Out-of-county points → missing; flag bucketed coordinates (≥20 distinct strings at one point) |
| 7 | **Missing coordinates are period-dependent** (3% → 8–11% from 2014) | ZIP from the `location` string as a fallback (ZCTA-level ACS), with a flag recording which geography was used |
| 8 | **The stop location is not the driver's home.** Tract context describes the place, not the person — especially on interstates and parkways | Interact neighbourhood features with road type (Step 4). Proposed: no neighbourhood context for limited-access highways |
| 9 | **Proxy risk.** With residential segregation, tract race composition is a strong proxy for driver race | Team decision (D3 below): in X, or only in the fairness analysis |
| 10 | **CRS mismatch** — OPP coordinates in WGS84, TIGER in NAD83 | Difference under 2 m, negligible; recorded for completeness |

### 1.4 Join method

- **Secondary: point-in-polygon** — assign each stop to the 2010 tract containing it.
- **Primary, if blind spot 5 proves material: 800 m buffer** around the stop, with tract attributes averaged by the share of the buffer's area falling in each tract. This avoids an arbitrary assignment when the stop sits on the road dividing two tracts.

The Tennessee statewide tract file (not just Davidson) is used so stops just across the county line still resolve.

### 1.5 Validation checks after the build

1. Share of stops with a tract assigned, by year.
2. Share of stops within 100 m of a tract boundary.
3. Tract population and race shares compared to published Davidson County totals.
4. Features constant within tract and vintage; year-to-year changes plausible (no tract jumping from 10% to 60% Black between vintages).

---

## Step 2 — Relative measures ("who is stopped here vs. who lives here")

*Depends on Step 1; designed now, built after Step 1 is validated.*

### 2.1 Features

- **Area disparity index** for group *g* in area *a*: share of *g* among stops in *a* ÷ share of *g* among residents of *a* (ACS).
- **Driver–neighbourhood congruence**: the share of the driver's own race among the area's residents — the "out of place" measure from the policing literature (a Black driver in a 90%-white tract scores 0.07).

### 2.2 Blind spots

| # | Blind spot | Handling |
|---|---|---|
| 1 | **Temporal leakage** — stop shares computed on the full data include future stops | Stop composition from a **trailing 365-day window strictly before the stop date**, excluding the stop itself |
| 2 | **Circularity** — built from search or hit outcomes, the index encodes past policing of the outcome | Built from **stop counts only**, never searches or contraband |
| 3 | **Benchmark problem** — residents are not the driving population (commuters, through-traffic) | State it explicitly. Offer the veil-of-darkness comparison as the literature's standard alternative benchmark |
| 4 | **Small numbers** — few stops of a group in an area make ratios explode | Empirical-Bayes shrinkage toward the county rate; minimum-count flag |
| 5 | **The congruence feature is a function of race** | It cannot sit in a race-blind X. It belongs to the fairness analysis (course step 2: identifying the variables that generate unfairness) |

---

## Step 3 — Registration-plate features

| Feature | Definition | Note |
|---|---|---|
| `plate_group` | TN / border state / other state / missing | **Missing kept as its own level** — it has the highest search rate (5.02%) and hit rate (21.6%) |
| `plate_distance_km` | distance from the plate state's population centroid to Nashville | continuous alternative to the grouping |
| `plate_region` | Census region of the plate state | coarse, stable |
| `plate_state_cannabis_legal_at_stop` | recreational cannabis legal in the plate state **on the stop date** (e.g. CO and WA from Dec 2012, CA from Nov 2016; exact effective dates to be sourced per state) | time-matched augmentation |

**Blind spots:**
1. **Drift** — the TN share falls 96.4% → 84.3%, so plate features carry period information. Monitor in the stability analysis.
2. **Thin cells** — only ~2,000 out-of-state consent searches in total. Keep groupings coarse; state-level features will be noisy.
3. **Ethics** — profiling by plate origin is itself a contested practice. The cannabis feature in particular encodes an assumption about origin states; include it only as a tested hypothesis, not by default.

---

## Steps 4–6 — to be detailed after Steps 1–3

- **Step 4 — Road context**: road type from the `location` string (interstate / parkway / arterial "PIKE" / residential address), location kind (intersection vs address), bucketed-geocode flag. The data already shows different consent hit rates by road type (interstate 14.4%, Briley Pkwy 22.7%).
- **Step 5 — Density and activity**: stops per area per trailing period, officer concentration, share of night stops — all on trailing windows before the stop date, from stop counts only.
- **Step 6 — Temporal crossing**: hour × area, weekday × hour, daylight vs dark (sunset-matched), holidays.

---

## Approvals needed before Step 1 is built

1. **Install** `geopandas` (pulls `shapely`, `pyproj`, `pyogrio`) via pip — none is installed.
2. **Download** `tl_2010_47_tract10.zip`, Tennessee 2010 census tracts, from www2.census.gov — **13.73 MB**.
3. **Query** the Census API (api.census.gov) — 9 vintages × ~11 tables, Tennessee tracts; small JSON responses, no API key needed at this volume.

## Decisions needed from the team

- **D1** — temporal rule: A, **B (recommended)**, or C.
- **D2** — join method: point-in-polygon, or 800 m buffer if boundary proximity is material.
- **D3** — tract race composition in X, or only in the fairness analysis.
- **D4** — neighbourhood features on interstates and parkways: keep with a road-type interaction, or set to missing.
