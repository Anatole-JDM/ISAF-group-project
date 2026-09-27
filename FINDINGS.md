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
| non-consent (**mixed**, see §1.1) | 20.83% | **27.12%** | **+6.30 pp** | 15.63% | −5.20 pp |
| **POOLED** | 20.83% | 21.66% | **+0.82 pp** | 12.07% | −8.77 pp |

The Black–white gap runs in **opposite directions** across strata. This is
**effect modification**, not Simpson's paradox — the distinction matters and
should be stated precisely.

### With 95% bootstrap CIs (`fairness.pooled_vs_stratified_ci`, 2,000 draws)

| stratum | race | n | hit rate | gap vs white (pp) | excludes 0 |
|---|---|---|---|---|---|
| consent | black | 32,311 | 15.68% [15.30, 16.08] | **−5.15 [−5.80, −4.49]** | ✓ |
| consent | hispanic | 4,645 | 7.88% [7.13, 8.65] | **−12.95 [−13.89, −12.02]** | ✓ |
| non-consent | black | 35,352 | 27.12% [26.67, 27.62] | **+6.30 [+5.61, +6.94]** | ✓ |
| non-consent | hispanic | 5,457 | 15.63% [14.64, 16.57] | −5.22 [−6.33, −4.17] | ✓ |
| **POOLED** | black | 67,663 | 21.66% [21.36, 21.98] | **+0.83 [+0.35, +1.31]** | ✓ |
| **POOLED** | hispanic | 10,102 | 12.07% [11.44, 12.71] | −8.75 [−9.46, −8.02] | ✓ |

**The Black reversal is beyond doubt.** Consent [−5.80, −4.49] and non-consent
[+5.61, +6.94] do not overlap and sit on opposite sides of zero, 11.46 pp apart.
It cannot be dismissed as noise.

**But do not quote 11.46 pp as discretionary-vs-mechanical.** The non-consent
group is only 52.4% mechanical. Against the genuinely mechanical bases alone
(arrest + warrant + inventory, +1.77 pp) the separation is **6.93 pp**. That is
the number to defend. See §1.1.

**The pooled result is worse than "no disparity".** It is +0.83 [+0.35, +1.31] —
statistically significant and pointing the WRONG WAY. A pooled analysis would not
merely miss the disparity; it would report with confidence that Black drivers are
searched on marginally *better* evidence than white drivers. That is a stronger
indictment of pooling than a null result would be.

**Nuance: only the Black–white comparison reverses.** Hispanic gaps are negative in
both strata (−12.95 consent, −5.22 non-consent) with no sign flip, though the
consent gap is 2.5x larger. Say "the Black–white comparison reverses", not "the
gaps reverse".

Not a composition artifact: consent is 44.6% of white searches, 47.8% of Black,
46.0% of Hispanic.

**Definition sensitivity — report both numbers, not the flattering one.**

| definition of "consent" | n | b−w |
|---|---|---|
| `search_basis == "consent"` only | 67,431 | −4.41 pp |
| any consent flag (raw flag OR basis) | 69,218 | **−4.60 pp** ← the conservative number |
| resolved most-mechanical-first (our sample) | 58,836 | **−5.16 pp** ← the headline |

Resolving most-mechanical-first — so `consent` means *no* other legal authority
also applied — removes **10,386** consent-flagged searches. An earlier version of
this document said 8,764, carried over from a stale count of 67,629 resolved rows;
the true pre-purification count is 69,251 (69,218 with reportable race).

**The removal is not neutral, and we do not claim it is.** Those 10,386 searches
hit at **28.66%** against 16.91% for the ones we keep, and **76.35%** of them ended
in arrest against 7.70%. The exclusion is on the legal *basis*, not on the outcome
— a search carrying arrest, warrant or inventory authority did not require consent,
whatever it yielded — but that basis is outcome-correlated in practice, so the
purification is not free. The gap within the removed rows is −0.56 pp.

**Concede plain view.** 1,247 of the removals are re-labelled `plain view`, which is
discretionary and whose flag is set by what was already seen. It should not have
overridden `consent`. Both gaps have the same sign and support the same conclusion,
so the scope decision stands on either — but **−4.60 pp is the number to quote when
challenged.**

### It replicates in two other jurisdictions

| Jurisdiction | pooled b−w | consent-only b−w |
|---|---|---|
| Nashville | **+0.82 pp** | **−5.16 pp** |
| Illinois | −0.07 pp | −7.25 pp |
| North Carolina | −1.15 pp | −6.30 pp |

Three states, different years and reporting systems, same reversal.

### 1.1 What the comparison group actually contains

"Non-consent" is not a synonym for "mechanical". Decomposed by legal basis
(`scripts/deck_stats.py`, recomputed from `data/processed/searches.parquet`):

| legal basis | n | white hit | Black hit | b−w | officer discretion? |
|---|---|---|---|---|---|
| probable cause | 15,877 | 61.21% | 45.31% | **−15.90 pp** | YES |
| consent *(our sample)* | 58,836 | 20.84% | 15.68% | **−5.16 pp** | YES |
| warrant | 2,908 | 14.99% | 14.50% | −0.50 pp | no |
| arrest | 31,358 | 19.42% | 21.21% | +1.80 pp | no |
| inventory | 1,447 | 14.98% | 23.44% | +8.47 pp | no |
| plain view | 16,634 | 5.91% | 14.44% | +8.54 pp | YES |

- **Truly mechanical** (arrest + warrant + inventory): n = 35,713, **+1.77 pp**.
- **Still discretionary** (probable cause + plain view): n = 32,511, +11.17 pp —
  **47.7%** of the comparison group.

n throughout this table is the count each gap is computed on — drivers recorded
white, Black or Hispanic — so it is slightly below the raw row count (35,713 vs
35,733 mechanical). The modelling sample is all 58,865 resolved consent searches.

**This strengthens the mechanism rather than weakening it.** The two bases that
require an officer to judge *this driver* are both negative, and probable cause —
15,877 searches at a 49.57% base rate, far better powered than consent — runs
**−15.90 pp**, three times the headline in the same direction. The claim
"officers apply a lower evidentiary bar to Black drivers where they have
discretion" is supported twice, not once.

**What was wrong was the label, not the finding.** The contrast is consent versus
non-consent, not discretionary versus mechanical. Plain view is the exception that
proves the point: it is discretionary but positive, because the basis is recorded
*because* something was already seen.

Probable cause is fitted in the repo (`outputs/metrics__probable_cause__full.json`,
gbm 0.5593 on 5,809 test rows at a 48.2% base rate) and was never written up.

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

Officer identity is worth 52% as much as **everything else combined** — every
driver characteristic, location and time feature put together. Officer
heterogeneity is large and real.

Note the phrasing carefully: this is NOT "52% of the model's accuracy comes from
officers". The model is 0.0663 above chance; adding officer identity adds a further
0.0342.

`officer_id_hash` is excluded from every deployed model (`config.IDENTIFIERS`). It
is added only in this controlled ablation. It would be wrong to deploy: at the
moment of decision the officer is *constant across his own choice set*, so it
cannot discriminate between the drivers he is choosing among — zero decision value
whatever it does to AUC.

### 4.1 But the model is NOT memorising officers (2026-09-25)

Leave-one-officer-group-out, 5-fold `GroupKFold` so no officer appears in both
train and test, against a random-split control:

| split | AUC | folds |
|---|---|---|
| grouped by officer (**unseen** officers) | 0.5922 ± 0.0143 | .584 .602 .615 .577 .582 |
| random (officers seen in both) | 0.6013 ± 0.0099 | .621 .599 .596 .594 .597 |
| **gap** | **−0.0092** | inside the ±0.014 fold noise |

1,477 officers, median 10 searches each; the top 10 officers are 10.2% of the
sample. (Both figures exceed the 0.5663 temporal-split number because random and
grouped CV do not span the 2016 regime shift — compare the two rows to each other,
not to the headline.)

**This test came back negative, and it corrects an overstatement.** `stability.py`
predicted that if performance collapsed on held-out officers, the model had learned
discretion rather than risk. It did not collapse. So:

- ✅ Officers differ substantially in outcome rates — knowing *which* officer is
  informative.
- ❌ The model is **not** officer-specific. Its learned stop-features → contraband
  mapping transfers to officers it has never seen.

Both are true simultaneously. The defensible claim is about officer *heterogeneity*,
not about the model memorising individuals. Earlier drafts of this document said
"the model learns discretion, not risk" — that is stronger than the evidence
supports and has been corrected here.

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

Computed by `scripts/deck_stats.py` on the matched run, K = 2,000 of 9,113 test
searches (`outputs/economics_matched.json`):

| arm | TP | FP | precision | break-even FP price |
|---|---|---|---|---|
| tabpfn | 509 | 1,491 | 25.45% | **0.341** |
| gbm | 501 | 1,499 | 25.05% | 0.334 |
| scorecard | 460 | 1,540 | 23.00% | 0.299 |
| *random ranking* | *427* | *1,573* | *21.35%* | *0.272* |

The break-even price is the value of one innocent driver's search at which net
benefit hits zero, expressed as a multiple of the value of one justified search.

**The best arm buys 0.069 of headroom over ranking at random.** Above a
false-positive price of 0.341 — that is, once searching an innocent driver costs
more than about a third of what catching a real one is worth — deployment destroys
value. At K = 2,000 TabPFN finds 82 more hits than random, at the cost of 1,491
innocent searches. The false-positive price is a policy judgement, not a data fact,
so we ship the sensitivity rather than picking the number.

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

Between n=100 and n=5,000, fifty times more data changes nothing.

**PARTIALLY REVISED 2026-09-27 — at full scale TabPFN does improve.** Scott asked for
a full-training-size run and was right to. On the FULL 9,113-row test set (not the
3,000-row subsample used for the curve above, so these do not splice directly):

| n_train | AUC | Brier | seconds |
|---|---|---|---|
| 5,000 | 0.5662 | 0.1680 | 10.4 |
| 10,000 | 0.5602 | 0.1696 | 7.5 |
| 20,000 | 0.5651 | 0.1689 | 10.5 |
| **49,752 (all)** | **0.5730** | 0.1683 | 25.3 |

*Same test set: scorecard 0.5516 · gbm 0.5663 · tabpfn@2,000 0.5529*

Two things this overturns:

1. **TabPFN does not fail at scale.** 49,752 rows in 25s via hosted inference, no
   context limit. The earlier CPU difficulty was inference cost, not a ceiling.
2. **TabPFN draws level with XGBoost on the base feature set** — 0.5730 against
   0.5663, matching gbm+time (0.5733). Nominally ahead; see §24, the gap is not
   statistically distinguishable, so this is "no longer behind", not "best".

Stated honestly: the curve is **flat to 5,000, then rises modestly to full scale**,
+0.020 over the n=2,000 result. The intermediate points are non-monotonic (~±0.006
wobble), and the total gain sits at the edge of the 0.024 context-resampling spread
in 10.2. It is a real but small effect, and it does not disturb the central finding:
the best model anyone has fitted on this problem reaches AUC 0.573.

**It does sharpen the deployment argument.** The best-performing arm is also the
least explainable (occlusion only, finding 16) and the most race-dependent
(permutation importance 0.0382, fifteen times the GBM's — finding 10.3). Performance,
interpretability and fairness point at different models.

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
   policy (see finding 7), so the policy is close to arbitrary.

**RECONCILED 2026-09-27 — always normalise by chance.** A teammate's XGBoost run
reported a top-k overlap of 0.811, which looks like a flat contradiction of 0.078.
It is not: the two used different K and N, and Jaccard has a chance floor that
depends on both. Normalised:

| measurement | K | N | observed | chance | **x chance** |
|---|---|---|---|---|---|
| this finding | 200 | 3,000 | 0.078 | 0.034 | **2.3x** |
| like-for-like, base features | 200 | 9,113 | 0.025 | 0.011 | **2.3x** |
| like-for-like, **+ officer features** | 200 | 9,113 | 0.066 | 0.011 | **6.0x** |
| teammate's run | 2,005 | 9,113 | 0.811 | 0.123 | **6.6x** |

His K is 22% of the test set, where two *random* top-k lists already overlap at
Jaccard 0.123. Ours is 2%, where chance is 0.011. Once normalised the two runs
**agree**: base features sit at 2.3x chance, officer features at ~6x under either
protocol. Report the multiple, not the raw Jaccard, or the number means nothing.

Note also that his perturbation is 10 model SEEDS on a fixed training set, while
ours resamples the TRAINING DATA. Seed variation is the weaker perturbation, so the
two are measuring different kinds of stability even after normalisation.

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

### 10.4 TabPFN is the best-calibrated arm — and still worse than a constant

Brier within each racial group (lower is better):

| arm | black | hispanic | white |
|---|---|---|---|
| **tabpfn** | **0.1517** | **0.1771** | **0.1844** |
| scorecard | 0.1542 | 0.1802 | 0.1887 |
| gbm | 0.1631 | 0.1794 | 0.1997 |

Calibration is the sufficiency leg of the taxonomy, and TabPFN wins it outright
**among the three arms**. Note Brier is sensitive to base rate, which differs by
group, so compare *within* a column, not across.

**But "best of three" is not "good".** Against the no-skill floor — predicting that
group's base rate for every driver in it — every arm loses, in every group:

| group | n | base rate | no-skill floor | tabpfn | scorecard | gbm |
|---|---|---|---|---|---|---|
| black | 4,947 | 0.1949 | **0.1569** | 0.1579 | 0.1608 | 0.1710 |
| hispanic | 689 | 0.1509 | **0.1282** | 0.1377 | 0.1392 | 0.1413 |
| white | 3,368 | 0.2550 | **0.1900** | 0.1926 | 0.1957 | 0.2085 |

Pooled, the floor is 0.1679 against tabpfn 0.1690, scorecard 0.1718, gbm 0.1824 —
**negative skill on calibration for all three.** They do carry ranking information:
PR-AUC beats its 0.2135 floor by +0.019 to +0.027, which is why a top-K policy is
conceivable at all. Do not call any of these arms well calibrated without the floor
beside it.

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

## 12. Fairness metrics with confidence intervals (2026-09-25)

`fairness.group_rates_ci` — percentile bootstrap, 2,000 draws within each group,
decision rule held fixed. gbm / `full_time`, top-K = 2,000 of 9,113.

| race | n | selected | selection rate | TPR | PPV | FPR |
|---|---|---|---|---|---|---|
| white | 3,368 | 1,356 | 40.3% [38.6, 41.9] | 46.6% [43.3, 50.1] | 29.5% [27.0, 32.0] | 38.1% [36.3, 40.0] |
| black | 4,947 | 611 | 12.4% [11.5, 13.3] | 15.8% [13.5, 18.1] | 24.9% [21.5, 28.2] | 11.5% [10.6, 12.5] |
| **hispanic** | 689 | **12** | 1.7% [0.9, 2.8] | 4.8% [1.0, 9.4] | **42.1% [11.1, 72.7]** | 1.2% [0.3, 2.2] |

**The hispanic PPV is ±30.8pp and must not be reported as a point estimate.** Only
12 hispanic drivers are selected at K=2,000, so the interval spans 11% to 73%.
Earlier drafts quoted "41.7%" as if it were a finding; it is not.

### Which metrics survive a small subgroup depends on the denominator

| metric | conditions on | hispanic n | usable |
|---|---|---|---|
| PPV | those **selected** | 12 | ✗ ±30.8pp |
| TPR | those **with contraband** | ~104 | ~ ±4.2pp |
| **FPR** | those **without contraband** | ~585 | ✓ **±0.9pp** |
| selection rate | **all** drivers in the group | 689 | ✓ ±0.95pp |

This is a fortunate alignment rather than a lucky one: **predictive equality (FPR)
is both the metric that matters most for the harm argument and the one that
survives the small sample**, because it conditions on the large group (innocent
drivers) rather than the small one (those the model picked). The hispanic FPR
claim — 1.2% [0.3, 2.2] against white 38.1% [36.3, 40.0] — is solid.

General rule for the deck: report FPR and selection rate for every group; report
PPV and TPR for white and black only, with intervals; and say why.

---

## 13. XPER — three quarters of the signal is race (2026-09-27)

The instructor's own method (his research, 2 of the 8 set readings): decompose the
performance metric itself into per-feature Shapley contributions.

    AUC = phi_0 + sum_j phi_j,   phi_0 = 0.5

Computed **exactly** — all 2^10 = 1,024 coalitions, each a re-estimation on that
feature subset (the deck's option 1). No sampling, no package. Efficiency identity
holds to machine precision: gap = -3.5e-17.

| feature | φⱼ | share of signal |
|---|---|---|
| **`subject_race`** | **+0.03796** | **+73.5%** |
| `violation` | +0.00529 | +10.2% |
| `subject_age` | +0.00499 | +9.7% |
| `precinct` | +0.00214 | +4.1% |
| `vehicle_registration_state` | +0.00176 | +3.4% |
| `subject_sex` | +0.00153 | +3.0% |
| `month` | +0.00118 | +2.3% |
| `dow` | +0.00008 | +0.2% |
| `hour` | −0.00113 | −2.2% |
| `zone` | −0.00215 | −4.2% |

**AUC 0.5516 = 0.5 + 0.0516, and race is 0.0380 of that 0.0516.** The model is close
to a race detector with a small amount of noise attached. Two features carry negative
φ — they make out-of-sample AUC *worse* than the coalitions without them.

This is the sharpest statement of the project's thesis, in the instructor's own
notation, and it converges with finding 5 (the features predict race at 0.72-0.73
and contraband at 0.57).

## 14. Opening the white box (2026-09-27)

`src.interpret.coefficient_table`. Base rate 16.1% gives an AME multiplier of
p(1-p) = **0.133**, so every coefficient shrinks ~7.5x in probability terms — the
deck's slide-34 point, concretely.

| feature | coef | odds ratio | AME (pp) |
|---|---|---|---|
| **`subject_race_hispanic`** | **−0.850** | **0.428** | **−11.28** |
| `subject_race_unknown` | −0.669 | 0.512 | −8.89 |
| `zone_117` | −0.647 | 0.524 | −8.59 |
| `vehicle_registration_state_SC` | +0.523 | 1.688 | +6.95 |
| `subject_sex_male` | −0.487 | 0.614 | −6.47 |

The largest single coefficient in a model built to predict contraband is a race
indicator.

### A decision tree beats the scorecard

| max_depth | AUC | leaves |
|---|---|---|
| 2 | 0.5539 | 4 |
| 3 | 0.5588 | 8 |
| **4** | **0.5605** | **16** |
| 5 | 0.5592 | 32 |

A **16-leaf tree (0.5605) beats the logistic scorecard (0.5516)** and closes most of
the gap to gradient boosting (0.5663). The most interpretable model available is
competitive — the argument for the white box, made with numbers rather than asserted.

## 15. Fairness as a hypothesis test (2026-09-27)

The course frames fairness as a **test statistic**, not a confidence interval.

**χ² test of independence**, selection × race at K=2,000, white vs black:

    chi2 = 978.0,  dof 1,  p = 1.1e-214,  n = 8,315
    selection: white 41.1%, black 11.5%, gap -29.6pp

Statistical parity is rejected outright.

**TOST equivalence.** χ² on large n rejects nearly anything, which is the deck's own
limitation 2. Equivalence testing inverts the burden: how wide a tolerance δ before
the model would *certify* as fair? The deck never assigns δ a value, so we sweep it:

    theta_hat = -0.2964,  se = 0.0096
    delta 0.05  z_L -25.63  z_U +36.03   not equivalent
    delta 0.10  z_L -20.43  z_U +41.23   not equivalent
    delta 0.20  z_L -10.03  z_U +51.63   not equivalent
    delta 0.30  z_L  +0.37  z_U +62.03   not equivalent
    delta 0.32                            EQUIVALENT

**You would have to declare a 32 percentage-point selection gap acceptable before
this model certifies as fair.** That converts "unfair" from an adjective into a
magnitude, using the instructor's own framework.

## 16. Local explanations, and why the three arms differ (2026-09-27)

`src.explain`, wired into each arm's page.

| arm | method | exact? |
|---|---|---|
| scorecard | Shapley in closed form, `coef_j (x_j − E[x_j])` | **exact** |
| gbm | TreeSHAP, `xgboost predict(pred_contribs=True)` | **exact** |
| tabpfn | occlusion — replace with background, measure the change | **approximate only** |

TabPFN exposes no attribution of its own, so its explanation is an estimate whose
error cannot be bounded. Interpretability is not binary: the white box answers for
free, the tree answers with a special algorithm, and the foundation model only
estimates. That is a measurable cost of the black box.

On one held-out stop the three arms disagree locally exactly as they disagree
globally (Spearman −0.321, finding 10.3): the scorecard says `zone` pushed the score
up (+0.311), TabPFN says it pushed it down (−0.008).

---

## 17. The officer ablation at full scale — a clean 2x2 (2026-09-27)

Scott asked for TabPFN with the officer features. Run as an **ablation**, both arms,
all 49,752 training rows, full 9,113-row test set:

| features | arm | encoded cols | AUC | Brier |
|---|---|---|---|---|
| without officer | gbm | 117 | 0.5663 | 0.1695 |
| without officer | tabpfn | 117 | 0.5730 | 0.1683 |
| **with officer** | gbm | 539 | 0.6005 | 0.1676 |
| **with officer** | **tabpfn** | 539 | **0.6018** | **0.1670** |

| arm | Δ | share of above-chance signal |
|---|---|---|
| tabpfn | +0.0288 | +39% |
| gbm | +0.0342 | +52% |

1,477 officers; 421 have ≥25 training searches and survive `min_frequency=25`, the
rest collapse to "infrequent". TabPFN ingested 539 columns without difficulty (67s).

**Three observations.**

1. **With officer identity the model families converge.** Without it they sit 0.007
   apart; with it, 0.001. The information is in the feature, not the algorithm.
2. **TabPFN gains proportionally less** (+39% vs +52%) because it already extracted
   more from the base features — so the officer signal is partly redundant with what
   a stronger model finds anyway.
3. **The ceiling is still 0.60.** That is the highest number anywhere in this
   project: foundation model, every training row, plus officer identity. Ten points
   above chance.

### Why it stays out of the deployed model

This is an ablation, not a recommendation, and the reason is not fairness squeamishness:

**At the moment of decision the officer is CONSTANT across their own choice set.**
Officer Smith is choosing among the drivers *he* stopped. His identity is identical
for every option in front of him, so it cannot discriminate between them. A feature
that does not vary across the decision-maker's options has **zero decision value**,
whatever it does to test-set AUC.

Supporting reasons: it models the officer's hit rate rather than the driver's risk;
it launders past selection behaviour into future justification; and deploying it
would route searches toward high-hit-rate officers whose rates reflect whom they
chose to search in the first place.

The general principle, worth stating in the presentation:

> **A feature can improve AUC and still have no decision value, if it does not vary
> across the options the decision-maker actually faces. Test-set performance and
> deployment usefulness are different things.**

It would become legitimate under a *different* decision — a supervisor choosing which
officers' consent practices to audit. Different question, different model, and
arguably the more useful one given finding 4.

---

## 18. Officer BEHAVIOUR features — the strongest model, and a real correction (2026-09-27)

A teammate built 25 time-aware officer features into the consent table: rolling
365-day windows, empirical-Bayes shrunk (k=20), 2010 held out as a warm-up year, and
independently brute-force verified against 400 random stops. These are **officer
behaviour**, not officer identity, and the distinction matters:

| | officer ID | officer behaviour |
|---|---|---|
| form | 1,477-level categorical | `off_consent_hit_rate_365d_shrunk`, `off_searches_today_before`, `off_last_search_hit`, `off_hit_gap_black_white`, ... |
| a NEW officer | **no value** — unseen level | **works from day one** |
| what it encodes | individual memorisation | a transferable behavioural pattern |

### Results

| model | test AUC |
|---|---|
| `full_time` base | 0.5713 |
| **`full_time_officer`** | **0.6417** |
| `full_time_officer_blind` (race removed) | 0.6362 |

Independently replicated here: base 0.5663 -> **+officer 0.6425**, matching his
0.6417 to four decimals. **+0.070 AUC — double what officer ID gave (+0.034)**, and
the blind variant keeps 0.636, so it is not race in disguise.

### It transfers to unseen officers

This is what officer ID cannot do. His officer-holdout split has
`test_officers_seen_in_training: 0`, and GroupKFold gives:

    random k-fold        0.6798
    officer groupkfold   0.6687      gap only -0.011

### Corrections to earlier findings

1. **Finding 10.2 is narrowed.** "The ranked list is arbitrary" was measured on a
   near-signal-free model. With officer features, ranking stability roughly triples
   relative to chance (2.3x -> 6.0x). The arbitrariness was largely a property of a
   model with nothing to rank on, not of the problem. At K=200 roughly 88% of the
   list still churns, so the concern is softened rather than removed.
2. **Finding 4's framing was too broad.** The "constant across the officer's own
   choice set" objection is correct for a STOP-LEVEL decision — Officer Smith's own
   hit rate is identical for every driver in front of him. But it does not make the
   features invalid: for a supervisor deciding which officers' consent practices to
   audit, or which stops get review, they vary and carry real decision value. The
   objection is about *which decision is being modelled*, not about the features.

### What it adds to the fairness argument

`off_hit_gap_black_white` and `off_log_search_ratio_black_white` measure **each
officer's own racial disparity** as a feature. And the descriptive spread is large:
officer consent hit rates run **5.9% in the lowest decile to 31.9% in the highest**,
a 5.4x range. Officer heterogeneity is not a nuisance term — it is most of what is
predictable here, which is exactly the selective-labels thesis.

---

## 19. The full matrix — and race stops mattering (2026-09-27)

All three modes x both arms, full training set (49,752), full test set (9,113):

| mode | cols | gbm | **tabpfn** | Δ |
|---|---|---|---|---|
| `full_time` | 17 | 0.5733 | **0.5798** | +0.0065 |
| `full_time_officer` | 42 | 0.6459 | **0.6529** | +0.0070 |
| `full_time_officer_blind` | 40 | 0.6428 | **0.6498** | +0.0070 |

Brier follows the same order: TabPFN best in every row (0.1591 vs 0.1608 in the
best configuration). The gbm column replicates the teammate's XGBoost run within
0.004 (his 0.5713 / 0.6417 / 0.6362).

> **⚠ CORRECTED BY §24.** The +0.007 ordering below is **inside the noise**. The
> DeLong paired test puts TabPFN vs XGBoost at p = 0.69. Do not present this as a
> ranking, and do not say TabPFN "wins".

TabPFN is nominally ahead in every configuration by a consistent +0.007 — an
ordering that does not survive a significance test. The observation worth keeping is
the *contrast* with officer *ID*, where the arms converged to 0.001 apart: ID is 421
sparse one-hot columns where model family matters little, while behaviour is 25
dense numerics where any difference between families would show. Even there, the
difference is not statistically distinguishable.

### The finding: race matters because nothing else did

| model | with race | race removed | cost of removing race |
|---|---|---|---|
| base | 0.5663 | 0.5414 | **−0.0249** |
| **+ officer behaviour** | 0.6529 | 0.6498 | **−0.0031** |

**An 8x reduction.** XPER (finding 13) measured race at 73.5% of the above-chance
signal in the base model. Once the model has genuine behavioural features about how
the search decision is actually made, removing race costs almost nothing — at
*higher* accuracy.

This converts the fairness result from a negative into a constructive one:

> The model leaned on race because it had nothing else to lean on. Give it real
> signal about the decision process and race becomes close to irrelevant.

**Caveats before this goes on a slide.** The group fairness metrics (FPR gaps,
selection rates, chi-square, TOST) in findings 12 and 15 were computed on the BASE
model and have not been recomputed here — a race-blind model can still produce
disparate selection through proxies, and `off_hit_gap_black_white` is itself a
race-derived feature. 0.65 is also still a weak model. What this shows is that the
*dependence on the protected attribute* collapses, not that the model is fair.

---

## 20. Fairness recomputed on the officer model — Test / Identify / Mitigate (2026-09-27)

The metrics in findings 12 and 15 were computed on the BASE model. Recomputed on the
officer models, gbm, top-K = 2,000 of 9,113:

| model | AUC | sel W | sel B | sel gap | FPR gap | χ² | TOST δ* |
|---|---|---|---|---|---|---|---|
| base (`full_time`) | 0.5733 | 40.3% | 12.4% | −27.9pp | −26.6pp | 863 | 0.30 |
| `+ officer` | 0.6459 | 35.2% | 14.9% | −20.3pp | −16.4pp | 463 | 0.22 |
| **`+ officer, race-blind`** | **0.6428** | 27.9% | 18.6% | **−9.4pp** | **−6.0pp** | **100** | **0.11** |

**Every fairness metric improves monotonically while accuracy rises.** Selection gap
3x smaller, FPR gap 4.4x smaller, χ² 8.6x smaller, and the equivalence margin needed
to certify falls from 0.30 to 0.11. Going race-blind is nearly free: **0.003 AUC for
a fall from −16.4pp to −6.0pp in the false-positive gap.**

### This completes the course's Test → Identify → Mitigate spine

| step | evidence |
|---|---|
| **TEST** | χ² = 863, FPR gap −26.6pp, TOST needs δ = 0.30 (findings 12, 15) |
| **IDENTIFY** | XPER: `subject_race` is 73.5% of the above-chance signal (finding 13) |
| **MITIGATE** | model the DECISION PROCESS, not the driver, and drop race → FPR gap −6.0pp, χ² = 100, **+0.070 AUC** |

The mitigation costs no accuracy. It gains it.

### The conclusion this replaces

Earlier sections of this document concluded "do not deploy" from a near-random,
race-dominated model. That conclusion was correct **for that model**. The corrected
reading:

> The model leaned on race because it had nothing else. It was being asked to predict
> contraband from the DRIVER, when almost all the available signal is in the DECISION
> — who is searching, how often, with what recent track record. Given features that
> describe the decision process, accuracy rises, reliance on the protected attribute
> collapses 8x, and every group-fairness metric improves at the same time.

### Still not fair

χ² = 100 remains overwhelmingly significant, and δ = 0.11 means you would still have
to accept an 11-point selection gap to certify. The FPR gap of −6.0pp is real. The
model is **dramatically better, not fair** — the recommendation is "this is the
direction", not "this is fixed".

---

## 21. Adversarial self-critique — two tests a grader would run (2026-09-27)

### 21.1 Is the "race-blind" model actually blind? — DEFUSED

**8 of the 25 officer features are race-derived**: `off_hit_gap_black_white`,
`off_hit_gap_hisp_white`, `off_log_search_ratio_black_white`,
`off_log_search_ratio_hisp_white`, `off_consent_black_past`,
`off_consent_white_past`, `off_consent_hisp_past`, `off_hit_rate_same_race_past`.

Dropping `subject_race` while keeping those would re-admit race through eight side
doors — the exact proxy trap documented in finding 5. Tested by removing all eight:

| config | AUC | sel gap | FPR gap | χ² | TOST δ* |
|---|---|---|---|---|---|
| base | 0.5733 | −27.9pp | −26.6pp | 863 | 0.30 |
| `officer` | 0.6459 | −20.3pp | −16.4pp | 463 | 0.22 |
| `officer_blind` (8 still in) | 0.6428 | −9.4pp | −6.0pp | 100 | 0.11 |
| **`officer_TRULY_blind`** | **0.6403** | **−7.5pp** | **−4.8pp** | **65** | **0.10** |

Removing all eight costs **0.0025 AUC and makes the model fairer still**. The
mitigation in finding 20 is **not** a proxy artefact — it survives the hardest
version of the test. Use `officer_TRULY_blind` as the headline model.

### 21.2 Is the gain BETWEEN officers or WITHIN? — CONFIRMED, and it matters

`off_consent_hit_rate_365d_shrunk` is a moving average of Y itself. Strictly prior,
so not leakage — but does the model discriminate between DRIVERS, or merely between
OFFICERS? Within an officer those features are constant, so any between-officer gain
has no value for a stop-level decision.

AUC computed **inside each officer's own stops** (officers with ≥10 test searches and
both classes present, weighted by n), against pooled AUC:

| model | pooled | **within-officer** | gap |
|---|---|---|---|
| base | 0.5733 | 0.5510 | +0.022 |
| `officer` | 0.6459 | **0.5657** | **+0.080** |
| `officer_TRULY_blind` | 0.6403 | **0.5597** | **+0.081** |

**About 80% of the +0.070 headline gain is between-officer variance.** For the
decision an officer actually faces — which of *my* stops to search — the model
improves from 0.5510 to 0.5657, a gain of **+0.015, not +0.070**.

The model largely learns **which officers find contraband**, not **which drivers
carry it**.

This is not fatal, but it changes what the model is for:

> The deployable use is **supervisory** — which officers' consent practices warrant
> review — not stop-level triage. At the stop level the honest number is 0.566, and
> the earlier "constant across the officer's own choice set" objection (finding 17)
> was right. Finding 18 walked that objection back too far.

Always report the within-group metric when the features are group-level aggregates.
A pooled AUC flatters any model whose features vary mainly between decision-makers.

---

## 22. FPDP and PDP — the instructor's mitigation method, run (2026-09-27)

### 22.1 FPDP: no single feature repairs parity

The deck's Fairness Partial Dependence Plot: freeze feature X_A at one value for
every row, rescore with the already-fitted model, recompute the fairness statistic.
A "candidate variable" is one whose freezing lifts the p-value above 0.05.

Run on `officer_TRULY_blind` (χ² = 65, p = 5.8e-16 unfrozen), 32 features, each
frozen at its median / 10th / 90th percentile (mode for categoricals), best result kept:

| frozen feature | χ² | p | repairs parity? | χ² drop |
|---|---|---|---|---|
| `off_consent_hit_rate_365d_shrunk` | 23.4 | 1.3e-06 | **no** | 42.1 |
| `off_experience_days` | 31.5 | 2.0e-08 | no | 34.0 |
| `hour_sin` | 33.6 | 6.7e-09 | no | 31.9 |
| `zone` | 35.8 | 2.2e-09 | no | 29.7 |
| `off_stops_today_before` | 40.3 | 2.2e-10 | no | 25.2 |

**0 of 32 features repair parity.** The best single intervention takes χ² from 65 to
23, still p = 1.3e-06.

This is a **stronger** result than the deck's own German Credit example, where
freezing `Telephone` restored parity at near-zero accuracy cost. Here:

> The unfairness is not localisable in any one feature. It is distributed across the
> whole feature set, so no single-variable mitigation can repair it — which is a
> direct argument that the disparity is structural rather than incidental.

### 22.2 PDP: the model is not flat

Partial dependence spread (max − min predicted probability):

| feature | spread |
|---|---|
| `off_consent_hit_rate_365d_shrunk` | **0.150** |
| `off_minutes_since_first_stop_today` | 0.093 |
| `subject_age` | 0.068 |
| `off_hit_rate_365d_shrunk` | 0.052 |
| `off_stops_today_before` | 0.051 |

The officer's own recent hit rate moves the predicted probability by 15 percentage
points across its range — by far the dominant effect, consistent with XPER and with
finding 21.2.

Caveat the deck itself raises (PDP's independence assumption): `precinct` and `zone`
are strongly dependent in a segregated city, so their PD curves average over
combinations that do not occur.

## 23. Acknowledged limitations — what we did NOT do

Stated explicitly rather than left to be discovered:

1. **The threshold test was not run.** Selective labels are diagnosed throughout but
   never corrected. The remedy exists and is *in this repository* —
   `opp/lib/threshold_test.R` and `opp/stan/threshold_test_hierarchical_identified.stan`,
   by the authors of the data (Simoiu, Corbett-Davies & Goel). It requires R + Stan
   and was out of scope for the time available. Every hit-rate comparison here is
   therefore subject to inframarginality: equal average hit rates are consistent with
   different search thresholds when the underlying risk distributions differ.
2. **No multiple-testing correction.** Dozens of tests are reported without
   Bonferroni or FDR adjustment. The headline effects are far too large for this to
   matter (χ² = 863, p = 1e-214), but the smaller comparisons should be read with it
   in mind.
3. **Base-rate drift is unaddressed.** 16.1% train to 21.4% test. The models are
   calibrated on one base rate and evaluated on another; Brier and calibration-by-race
   both inherit this.
4. **The 2016 split is motivated, not tested.** It is chosen for the Driving While
   Black report and the documented MNPD practice change. No sensitivity analysis over
   alternative split years was run.
5. **Global surrogate not fitted.** Taught in the deck (sl. 84-86) with a fidelity
   number; not attempted here.
6. **There is no validation set.** Every model-selection decision in this document —
   tree depth, TabPFN context size, which feature blocks to keep, the vintage pin, K
   — was made by reading the same 9,113-row temporal test set that produces every
   reported number. Limitation 2 covers multiple testing of p-values; this is the
   separate problem that the point estimates are maxima over repeated reads. **Every
   AUC difference below 0.01 in this document should be read as inside the noise**
   and decided on other grounds. The officer block at +0.070 (p = 6.6e−23) is the
   only performance claim an order of magnitude clear of it.
7. **Provenance: the officer models are not reproducible from this repository.**
   `officer_TRULY_blind` is the headline model of §21.1, §22.1 and §24.2, but the
   string `TRULY` appears in no file except this one. `src.data.build_xy` has no
   officer flag, and nothing in the repo writes `outputs/critique_tests.json`,
   `outputs/fairness_officer_model.json` or `outputs/officer_ablation_full.json` —
   those runs were done interactively and only their outputs were committed. The
   reproducible path today is `scripts_claude/evaluate_officer_models.py`, which
   replicates the same design with 16 officer features and gives 0.6397 / 0.6324
   against src's 25-feature 0.6459 / 0.6428. State this before being asked; it takes
   four seconds to check.

---

## 24. Are the AUC differences significant? DeLong + clustered bootstrap (2026-09-27)

Until now this document compared AUCs with no standard error on any DIFFERENCE.
The arms are scored on the SAME test rows, so their AUCs are correlated and an
unpaired comparison is invalid. `src/significance.py` implements DeLong (1988) and
an officer-clustered bootstrap.

### 24.1 The three-arm comparison — no arm is best

Matched n=2,000, same subsample, paired DeLong. Regenerate with
`python3 scripts/run_significance.py` → `outputs/significance__consent__matched.json`:

| comparison | AUC a | AUC b | diff | p |
|---|---|---|---|---|
| gbm vs **tabpfn** | 0.5478 | 0.5529 | −0.0051 | **0.413 n.s.** |
| scorecard vs gbm | 0.5329 | 0.5478 | −0.0149 | 0.032 * |
| scorecard vs tabpfn | 0.5329 | 0.5529 | −0.0200 | <0.001 *** |

> **Superseded numbers.** An earlier version of this table gave gbm vs tabpfn as
> −0.0026 (p 0.691) and scorecard vs gbm as −0.0173 (p 0.019), and included a
> decision-tree arm. Those came from an interactive run and did not reproduce from
> `outputs/scores__consent__matched.parquet`; the tree is not in that file at all.
> The table above is generated by the script named here, and the marginal AUCs now
> agree with `metrics__consent__matched.json` exactly. **The conclusion is unchanged:
> the two black boxes are indistinguishable and the scorecard is significantly worse
> than both.** If you quoted the decision-tree comparison anywhere, drop it.

**"TabPFN is the best arm" is NOT supported.** It is indistinguishable from XGBoost
(p = 0.41). Only the logistic scorecard is significantly worse than the two black
boxes, by 0.015 to 0.020 AUC — the measurable price of interpretability here, and a
small one.

This corrects finding 19, which reported TabPFN winning "every configuration" by
+0.007. That ordering is inside the noise. **Do not rank the arms.**

The interpretability argument stands on the scorecard gap: 0.015-0.020 AUC is what
an exact, auditable, printable explanation costs against a black box on this data.

### 24.2 The officer models — the gain is real, and race is free

Full training, paired DeLong:

| comparison | diff | 95% CI | p |
|---|---|---|---|
| base vs `officer` | −0.0726 | [−0.0870, −0.0581] | **6.6e−23 ***** |
| base vs `officer_TRULY_blind` | −0.0669 | [−0.0829, −0.0510] | **2.2e−16 ***** |
| `officer` vs `officer_blind` | +0.0031 | [−0.0031, +0.0093] | 0.332 n.s. |
| **`officer` vs `officer_TRULY_blind`** | **+0.0056** | [−0.0019, +0.0132] | **0.144 n.s.** |

**Removing `subject_race` AND all 8 race-derived officer features is statistically
free** — the confidence interval on the cost includes zero.

### 24.3 Clustered by officer — it survives

Rows are not independent: 1,477 officers, the top 10 accounting for 10.2% of
searches. Resampling OFFICERS rather than rows, 400 draws:

| comparison | diff | clustered 95% CI | p |
|---|---|---|---|
| `officer` − base | +0.0722 | [+0.0516, +0.0947] | 0.000 |
| `TRULY_blind` − `officer` | −0.0054 | [−0.0137, +0.0026] | 0.180 |
| `TRULY_blind` − base | +0.0667 | [+0.0456, +0.0879] | 0.000 |

The clustered interval is **wider** than DeLong's ([+0.0516,+0.0947] vs
[+0.0581,+0.0870]), as it must be when the effective sample is closer to 1,477 than
to 58,865 — and it still excludes zero. The officer-feature gain is not an artefact
of treating correlated rows as independent.

**The defensible summary:** the model arms are statistically indistinguishable from
each other; what is significant is the FEATURE SET, and going fully race-blind costs
nothing measurable.

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
