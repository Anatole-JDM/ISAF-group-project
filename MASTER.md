# MASTER — Nashville consent searches: the red line

One document the whole team works from: the story, the numbers and the words we use. Every figure names
the file that produces it; where two runs disagree, §12 says which one we use. Written 27 Sep 2026 (evening),
from everything in `main`, the `release` branch, `FINDINGS.md` and the team's model sections.

Marks: ✅ reproducible from a script in `main` · ⚠ run interactively, not reproducible from the repo ·
⏳ still to run

---

## 0. The story in ten lines

1. **The client** (MNPD, hypothetical) asked for a score telling officers which traffic stops are worth a
   consent search, to cut fruitless searches under limited capacity.
2. **We built it three ways** (white box, XGBoost, TabPFN) on 58,865 consent searches, trained on 2010–2015
   and tested on 2016–2018.
3. **With what is known about the stop and the driver, no model predicts the outcome usefully.** AUC is
   0.53–0.58 for all three families, none beats a constant prediction on calibration, and the best buys
   little over random ranking.
4. **What little the models learn is mostly race.** Race is 73.5% of the white box's above-chance signal, and
   the same features predict the driver's race (AUC 0.69–0.73) better than contraband. The foundation model,
   the most accurate, is also the most race-driven: at the team's search budget it would flag 57% of white
   drivers and 1.7% of Black drivers.
5. **Why?** Contraband is only observed when an officer chose to search (selective labels), so the outcome is
   produced by the search decision. Given the searching officer's own past record, every model family gains
   +0.07 AUC, on officers never seen in training, in every year.
6. **That gain is entirely between officers.** Inside one officer's own searches (the decision he actually
   faces) the white box is at 0.50 and XGBoost at 0.555. The record tells you *which officer*, never *which
   driver*.
7. **So the officer record is our measuring instrument, not a deployable input.** It shows that a consent
   search's outcome is explained more by *who searches* than by *who is searched*.
8. **The records show a pattern.** Officers who ask minority drivers for consent disproportionately are the
   least successful searchers, on minority *and* white drivers. Nine officers carry out a quarter of all
   consent searches of Hispanic drivers, and 98% of those find nothing.
9. **Every model trained on these outcomes inherits the problem.** Race-aware, it selects almost no minority
   drivers. Race-blind, it over-scores them by 5–8 points at equal score. No single feature repairs parity.
10. **Recommendation.** Do not put a search score in officers' hands. Deploy the white box inside an
    officer-level consent-search review for supervisors. Bringing the least successful fifth of searches to
    the median hit rate would have meant about 760 fewer fruitless searches in 2016–2018 for the same finds,
    two thirds of them of Black or Hispanic drivers.

---

## 1. The brief, the client, the question

### 1.1 What the course asks (`Project ISAF 2026_2027.pdf`)
- A **scoring analysis** on a binary target, for a **hypothetical client**.
- **Three model types:** a white box (e.g. logistic regression), a machine-learning model (e.g. XGBoost), a
  tabular foundation model.
- **For each model, four dimensions:**
  - predictive performance, **statistical and economic**;
  - interpretability: the **main drivers and individual predictions**;
  - stability;
  - fairness.
- **Go beyond metrics:** discuss the **trade-offs** across the three models.
- **A clear recommendation of which model to deploy in production, and why,** reflecting the requirements
  of **trustworthy AI** rather than performance alone.
- **Deliverables:** slides, the code, and an app the client can use to test the models.
- **Format and grading:** 15 min talk + 10 min Q&A, and every member must be able to answer on any part.
  Technical 5 · presentation 5 · Q&A 5 · slides & code/app 10.

### 1.2 The client and what they asked (role play)
**Client:** Metro Nashville Police Department (MNPD) command staff; its civilian oversight is the second
audience.

**Their question:**
> "Consent searches are our most discretionary searches: no warrant, no probable cause, the driver agrees.
> 83% find nothing. Each fruitless search costs officer time, the driver's time and dignity, and community
> trust, and our searches have documented racial disparities. Can a scoring model tell our officers which
> stops are worth a consent request?"

**Success criteria (ours to state, theirs to accept):**
- fewer fruitless searches;
- finds preserved;
- no discrimination;
- explainable to a driver, a court and the oversight board.

**Our answer in one sentence:** no model can make that roadside call from the information available. But the
same analysis shows where fruitless searches come from (the search practice, officer by officer), and a
white-box officer-level review can act on it (§9).

---

## 2. Data, scope, target

| | Value | Source |
|---|---|---|
| Data | Stanford Open Policing Project, Nashville (Pierson et al., *Nature Human Behaviour* 2020), ODC-By licence | `README.md` |
| Stops 2010–2018 | 3,078,074 after our cleaning (src's cleaning: 3,078,116) | `opp_data/processed/nashville_clean.parquet` |
| Searches 2010–2018 | 127,121 | same |
| **Consent-only searches 2010–2018 (the sample)** | **58,865**, 16.9% find contraband, 1,477 officers | `data/nashville_consent_searches.parquet` ✅ |
| Drivers | Black 32,311 (54.9%) · white 21,227 (36.1%) · Hispanic 4,645 (7.9%) · other/unknown 682 | same |
| Train, 2010–2015 | 49,752 searches, 16.1% hit | `data/officer_model_results.json` ✅ |
| Test, 2016–2018 | 9,113 searches, 21.4% hit: white 3,368 at 25.5% · Black 4,947 at 19.5% · Hispanic 689 at 15.1% | same |

**Why consent searches only.** The outcome test depends on the legal basis of the search (`FINDINGS.md` §1):

| Legal basis | Searches | Black − white hit rate |
|---|---|---|
| Probable cause (officer judges this driver) | 15,877 | **−15.9 pp** |
| **Consent (officer judges this driver; our sample)** | 58,836 | **−5.2 pp** |
| Arrest + warrant + inventory (mechanical) | 35,713 | +1.8 pp |
| All searches pooled | — | +0.8 pp: the wrong sign |

- **Pooling misleads.** It would report, with confidence, that Black drivers are searched on marginally
  *better* evidence.
- **It replicates** in Illinois (−7.3 pp) and North Carolina (−6.3 pp).
- **What "consent" means here:** consent recorded AND no arrest, warrant, inventory or plain-view authority
  (cleaning protocol, step 5).
- **That definition removes 10,386 searches, and the removal isn't neutral:** they hit at 28.7% and 76% ended
  in arrest. So report the gap both ways, −5.16 pp (pure) and −4.60 pp (all consent-flagged), and concede
  plain view.

**Target and coding.**
- Y = 1: contraband found. Ŷ = 1: the model would search. D: driver race, as perceived by the officer.
- The favourable outcome for the driver is *not being searched*, the reverse of the course convention.
  Fairness metrics are therefore reported under both codings (`src/fairness.both_codings`).

**Selective labels, the fact that shapes everything.** Contraband is observed only when an officer chose to
search. Every model learns from officers' decisions, and every number is conditional on a search having been
made.

**Split.**
- **Temporal:** train 2010–2015, test 2016–2018. No learning from the future (the officer record is
  history-based), and it mimics deployment.
- **The test years follow a change in practice:** consent searches fell from ~9,000 a year (2011–2014) to
  2,167 in 2018, and the hit rate rose from 16.1% to 21.4%.
- **Officer holdout:** 20% of officers never seen in training.

---

## 3. What we built (feature work), and what it bought

| Block | Content | Predicts contraband? | Note |
|---|---|---|---|
| Stop & driver | age, sex, race, precinct, zone, reason for stop, plate state, hour, weekday, month | AUC 0.55–0.57 | the task as posed (Act I) |
| Time | 7 cyclical/calendar features | +0.007, the only block that helps in Act I | `FINDINGS.md` §11 |
| Neighbourhood (ACS) | 800 m / 1,200 m population-weighted circles, release before the stop year | as shipped, a *year fingerprint* (predicts the side of the 2016 split at AUC 0.998); frozen at one release, fixed, and adds nothing | predicts the driver's race at 0.69–0.73 |
| Plate | state, region, distance | re-encodes plate state; `plate_missing` is a 2017 recording change | never a predictor |
| **Officer record** | 25 features from the searching officer's own earlier stops (§5.1) | +0.07 pooled AUC, every family | Act II: a measuring instrument, not a deployable input |

**Leakage controls:**
- The 11 post-decision columns carry a `post_` prefix and are blocked.
- Officer features only use events strictly before the stop: same-minute stops never see each other, and the
  stop's own outcome is never counted. Checked by brute force on 400 random stops.
- PLTR's cross-validation re-learns its rules inside each fold. On pure noise it gives CV AUC 0.505.

---

## 4. Act I — the task as posed: can stop and driver information predict a successful consent search?

### Performance
**Matched comparison:** every model trained on the same 2,000 rows and tested on the same 9,113 searches
(`src/`; outputs on `release`):

| | White box (logistic) | XGBoost | TabPFN | Constant prediction |
|---|---|---|---|---|
| AUC | 0.533 | 0.548 | 0.553 | 0.500 |
| PR-AUC | 0.232 | 0.239 | 0.241 | 0.214 |
| Brier (lower is better) | 0.172 | 0.182 | 0.169 | **0.168** |

- **Significance (DeLong):** XGBoost vs TabPFN −0.005 (p = 0.41, **not significant**). The white box is below
  both (−0.015, p = 0.03; −0.020, p < 0.001).
- **All 49,752 training rows (src):** white box 0.552 (0.556 with time features) · XGBoost 0.566 (0.573) ·
  TabPFN 0.573 (0.580).
- **Our pipeline, time features:** white box 0.555 [0.541, 0.569], XGBoost 0.569 [0.556, 0.584] ✅.

**It is not the model:**
- TabPFN's learning curve is flat from 100 to 5,000 rows and gains only 0.02 at full scale.
- Five times more compute changes nothing.
- Resampling TabPFN's training context moves its AUC by 0.024, more than the whole 0.020 spread between the
  three families.

### Economic performance
Keeping the top 2,000 of the 9,113 test searches (K ≈ 22%):

| | Finds | Fruitless | Hit rate | Break-even cost ratio* |
|---|---|---|---|---|
| Random | 427 | 1,573 | 21.4% | 0.27 |
| White box | 460 | 1,540 | 23.0% | 0.30 |
| XGBoost | 501 | 1,499 | 25.1% | 0.33 |
| TabPFN | 509 | 1,491 | 25.5% | 0.34 |

\*The break-even cost ratio is the cost of searching an innocent driver, as a fraction of the value of a find,
above which keeping the top K destroys value. It equals hit rate / (1 − hit rate).
- The best model buys 0.07 of headroom over random: 82 extra finds for 1,491 innocent searches.
- The price is a policy judgement: we show the sensitivity and don't pick it.

### What the models learn
- **XPER** (the instructor's method, exact over 1,024 coalitions): **driver race is 73.5% of the white box's
  above-chance AUC.** Its largest coefficient is "driver Hispanic" (odds ratio 0.43, −11 pp).
- **The same stop features predict the driver's race** at AUC 0.69–0.73, better than they predict contraband.
  Dropping race does not remove it.
- **The selection inverts.** Asked to maximise finds, every model searches white drivers most. The white box
  at the top 22% selects white 48.5%, Black 7.1%, Hispanic 0%, because consent searches of white drivers
  found more. It is the outcome test, expressed as a model.
- **The three families disagree on why:** XGBoost's and TabPFN's feature importances are negatively
  correlated (Spearman −0.32).

### Stability
Resample the training data and ~93% of the 200 highest-scored searches change. The top-200 overlap is only
2.3× what chance gives, for XGBoost and TabPFN alike.

### Verdict of Act I
**As posed, the task cannot be done responsibly with this data:**
- near-chance ranking;
- no calibration skill;
- an unstable list;
- a signal that is mostly race.

This is a result, not a failure: the information that would identify contraband at the roadside is not in
the record.

---

## 5. Act II — where a search's outcome comes from: the officer record as a measuring instrument

### 5.1 What the officer record is
For every stop, 25 features computed from **the searching officer's own earlier stops only**
(`scripts_claude/build_officer_features.py`, strictly past, verified on 400 random stops):
- **Track record over the past 365 days:** stops, searches, finds, search rate, hit rates. Rates are shrunk
  toward the force-wide rate so small samples aren't extreme.
- **The day so far:** stops, searches and finds before this stop; minutes since the first stop of the day and
  since the last search; whether the last search found something.
- **Career length.**
- **The officer's own racial disparity over his past:** how much more often he asks Black / Hispanic drivers
  for consent than white drivers, his hit-rate gap between those groups, and his past hit rate on drivers of
  this driver's race.

**It is behaviour, not identity:** it works for an officer never seen in training. Officer *identity* as a
category adds only +0.034 and is useless for a new officer.

**Two versions are in use:** Scott's and Julia's models use 16 of the 25; src's models (Leo) use all 25, 8 of
which are derived from the race of past drivers (§12).

### 5.2 What it shows
| Model | Stop & driver only | + officer record | Gain |
|---|---|---|---|
| White box (logistic / PLTR, 19 rules) | 0.555 | 0.627 / 0.630 | +0.07 |
| XGBoost | 0.569 (Julia: 0.571) | 0.640 (Julia: 0.642; src: 0.646) | +0.07 |
| TabPFN (src, all 25 officer features ⚠) | 0.580 | 0.653 | +0.07 |

- **The gain is significant.** Paired gain for XGBoost: +0.070 [+0.056, +0.084]. Resampling officers instead
  of searches: +0.072 [+0.052, +0.095]. DeLong p = 6.6 × 10⁻²³: the only performance claim in the project an
  order of magnitude clear of the noise.
- **It holds everywhere tested:**
  - on **officers never seen in training** (XGBoost 0.643, PLTR 0.629; GroupKFold 0.669 vs 0.680 random);
  - in **every test year** (+0.05 to +0.09);
  - over **5 other random draws** of held-out officers (gain +0.03 to +0.12, mean +0.07).
- **The officer record alone reaches 0.62–0.63:** almost everything predictable is about who searches.
- **Descriptively,** the hit rate runs from 5.9% to 31.9% across deciles of the officer's past consent hit
  rate.

### 5.3 What it does not do — and why that decides everything
An officer decides among **his own** stops. So the number that measures decision value is the AUC **inside
each officer's own searches** ✅ (`data/officer_model_diagnostics.json` §F). It covers the 209 officers with
≥ 10 test searches (7,005 searches); CIs resample officers.

| Model | Pooled AUC | **Within-officer AUC** | Score variance between officers |
|---|---|---|---|
| White box, stop & driver only | 0.555 | 0.538 [0.518, 0.559] | 24% |
| XGBoost, stop & driver only | 0.569 | 0.543 [0.524, 0.561] | 24% |
| **White box (PLTR, 19 rules) + officer record** | 0.630 | **0.500 [0.480, 0.519]** | 60% |
| XGBoost + officer record | 0.640 | 0.555 [0.534, 0.574] | 52% |
| Logistic, officer record only | 0.622 | **0.443 [0.422, 0.463]** | 68% |

src's XGBoost, same measure: 0.551 → 0.566.

- **The whole gain is between officers.** Inside an officer's own searches, the officer-aware white box is a
  coin flip.
- **The officer record alone does worse than a coin flip there.** That's consistent with regression to the
  mean in an officer's rolling record (a hypothesis).
- **The record tells you which officer, never which driver.**

### 5.4 The team line on the officer record (proposed; the debate is in §10.3)
1. **We use it to explain, not to predict.** Prediction is the measuring instrument: comparing models with and
   without the record measures how much of a search's outcome is attributable to the searcher rather than the
   searched. The answer (more than everything about the driver, place and time together) is the finding.
2. **It is not leakage and not a trick.**
   - It is strictly past, verified, and holds on unseen officers.
   - "A good officer finds more" is right about the direction, and it is exactly the point: if success
     depended on the driver, the searcher's record would add little. It adds more than the driver.
3. **It is never an input to a deployed driver score.**
   - Two identical drivers would get different scores depending on who stopped them: the score would measure
     the officer.
   - It would steer searches toward officers whose past hit rates reflect whom they chose to search.
   - It has no decision value at the roadside (§5.3).
4. **"Record", not "quality".**
   - A high past hit rate mixes selection threshold, skill, assignment (specialised units, beats; not in the
     data) and recording habits.
   - Say "officers whose past searches rarely found anything", never "bad officers".
5. **It changes the fairness reading (§7).** Once the model knows who searched, race adds almost nothing.
   Removing race *and every race-derived feature* costs 0.006 AUC (p = 0.14, not significant), against 0.025
   for removing race from the driver-only model (src XGBoost). The racial signal of Act I largely stood in
   for search practice.

### 5.5 What the officers' records say (descriptive, no model) ✅
Source: `data/officer_disparity_findings.json`, diagnostics §D and §G. Each officer is compared **with
himself** (his own consent-search rates and hit rates by driver race), which removes his beat and habits from
the comparison.

- **Skew.** The median officer asks Black drivers for consent **1.4×** as often per stop as white drivers, and
  Hispanic drivers **2.3×**. 81% and 84% of officers are above 1.
- **The more skewed the officer, the less he finds, on white drivers too.**
  - Spearman −0.27 [−0.42, −0.11] (Black/white skew) and −0.65 [−0.82, −0.42] (Hispanic/white skew).
  - Least vs most skewed fifth: overall hit rate 21.7% vs 14.3% (Black/white skew), 21.8% vs 5.0%
    (Hispanic/white skew).
- **The gap is systemic.** 77% of officers find less on Black than on white drivers (28 significant vs ~4
  expected by chance); 91% find less on Hispanic drivers (8 vs ~1).
- **Concentration.** 9 officers (0.6% of those who made consent searches) carry out **a quarter of all
  consent searches of Hispanic drivers** (1,146 of 4,645), and **98%** of those find nothing.
- **Counterfactual.** Group test searches by the officer's past record:
  - The least successful fifth (1,823 searches) is made by **140 of the 763 officers** who searched in
    2016–2018.
  - It hits **12.2%** [9.8, 15.3], vs 20.9% for the median fifth.
  - It searches Hispanic drivers twice as often (12.7% vs ~6%).
  - At the median hit rate for the same finds, that means **764 fewer fruitless searches** [380, 1,206] over
    2016–2018: 11% of all fruitless consent searches, about 500 of them of Black or Hispanic drivers.
- **Practice patterns** (from the white box's rules and §G):
  - A search at the officer's **first stop of the calendar day** hits 20.9% vs 15.8% later. That holds in
    every time band and for low- and high-volume officers alike. A third of these are after midnight, so it
    is not "start of shift".
  - A search **within ~2 hours of the officer's previous search** has lower odds (0.79).
  - Reading: later, repeated searches look more speculative. That's a hypothesis for training, not a causal
    claim.
- **Say "consistent with a lower evidentiary bar", never "racist officers".** The outcome test can't prove
  intent (infra-marginality), and blank contraband fields were recorded as "nothing found".

---

## 6. The three models on the four dimensions (as the brief asks)

Two columns per model: **Act I** (the task as posed: stop and driver information) and **Act II** (with the
officer record: diagnostic, not deployable).

### 6.1 White box: logistic scorecard (Act I) → PLTR (Act II)
**Owners:** Leo (scorecard, `src/`), Scott (PLTR: `scripts_claude/pltr.py`, section `reports/pltr/README.md`).

**Why two:** the scorecard is the course's reference white box. PLTR (Dumitrescu, Hué, Hurlin & Tokpavi 2022;
slides 66–71) adds threshold rules and interactions while staying readable, which is worth it once there is
a signal to model.

| Dimension | Act I: scorecard | Act II: PLTR, 19 rules |
|---|---|---|
| Performance | AUC 0.533 matched, 0.552–0.556 full. Significantly below both black boxes (−0.015 / −0.020). Brier worse than a constant | AUC 0.630 [0.615, 0.643] (61 rules: 0.636). 0.010 below XGBoost; +0.075 [+0.060, +0.091] over the Act I scorecard. Unseen officers 0.629. **Within officer 0.500** |
| Economics | Top 2,000: 460 finds / 1,540 fruitless; break-even 0.30 | Top 22%: hit rate 32.0%, 3.12 searches per find (4.68 if all searched). But the ranking is between officers (§5.3) |
| Main drivers | "Driver Hispanic" is the largest coefficient (OR 0.43). XPER: race = 73.5% of the signal | 81% of importance involves the officer (64% the officer alone). Past consent hit rate (OR 1.43 per SD, 32% of importance); "poor record on this race AND overall" (OR 0.64, 15% of searches); driver Hispanic 0.71, white 1.21; experienced officer at his first stop of the day 1.24; previous search < ~110 min 0.79; Hispanic search skew > 1.6× 0.93 |
| One decision | Exact: coefficient × (x − mean) | Exact: the model is a sum of rules in log-odds. ⏳ 3 cases (`pltr_whitebox_report.py --fit`) |
| Stability | — | 5 officer draws 0.618 ± 0.026. By year 0.625 / 0.651 / 0.605 (drift in 2018). The head is stable (shared terms carry 62% of importance), the tail is not. Selected in all 5 draws: the officer's consent hit rate and driver race (plus the officer's Hispanic skew in the 61-rule and race-blind versions). ⏳ bootstrap coefficients |
| Fairness (top 22%) | Selects white 48.5% / Black 7.1% / Hispanic 0% | Race-aware: calibrated (equal-score gap −0.9 / +2.0 pp, n.s.), but selects 36% / 15% / 4%. Race-blind: 27% / 19% / 19%, FPR gaps −6 / −7 pp, but at equal score minority drivers are 4.6 / 7.2 pp less likely to carry contraband |
| Deployability | Trivial, printable, exact | Trivial, 19 readable rules, exact |

### 6.2 XGBoost
**Owners:** Julia (Act II: `scripts_claude/xgboost_officer_analysis.py`, `xgboost_surrogate.py`,
`reports/xgboost_officer/README.md`); Act I in `src/` (Leo).

| Dimension | Act I | Act II |
|---|---|---|
| Performance | AUC 0.548 matched, 0.566–0.573 full. Tied with TabPFN (p = 0.41). Worst Brier (0.182) | AUC 0.642 [0.628, 0.655]; race-blind 0.636. Unseen officers 0.643 (GroupKFold 0.669 vs random 0.680). **Within officer 0.555** |
| Economics | 501 finds / 1,499 fruitless; break-even 0.33 | Top 22%: hit rate 33.7%, 2.97 searches per find. 7,108 searches avoided, 1,270 of 1,946 finds lost |
| Main drivers | Flat permutation importances (precinct, zone, month, age, hour ≈ 0.01 each; race only 0.0025) | 62% of SHAP attribution is the officer record. Top: officer consent hit rate, minutes since first stop of the day, officer hit rate on this race, driver race, officer hit rate. Depth-3 surrogate tree: "officer's consent record, then: is the driver white?" (R² 0.39; a linear surrogate reaches 0.62) |
| One decision | Exact TreeSHAP | Highest score (82%): a white woman stopped by an officer with a 57% past consent hit rate (+0.81 log-odds from that alone). Nothing was found |
| Stability | Top-200 list 2.3× chance under resampling | 10 seeds 0.640 ± 0.002; 20 bootstraps 0.635 [0.628, 0.641]. Importance ranking correlates 0.94 across bootstraps but 0.85 across eras. Strong drift of the top officer features (PSI 0.75–0.95). 5 officer draws 0.635 (0.595–0.695) |
| Fairness (top 22%) | Selects 40% / 12% / 2%; FPR 38 / 12 / 1%; χ² 863 | Selects 35% / 15% / 10%, calibrated in every group. Race-blind: 26% / 20% / 19%, FPR gap −5 pp, but hit-rate gap among flagged −8.6 pp (sufficiency fails). src "truly blind": FPR gap −4.8 pp, χ² 65 |
| Deployability | Standard; exact SHAP; drift monitoring needed | Same |

### 6.3 TabPFN (tabular foundation model)
**Owner:** Leo (`src/models.py`, `src/tabpfn_experiments.py`, `FINDINGS.md` §10, §17–19).

| Dimension | Act I | Act II |
|---|---|---|
| Performance | AUC 0.553 matched (not significantly above XGBoost), 0.573–0.580 full. Best Brier (0.169), yet still worse than a constant (0.168). **Within an officer's own searches: 0.533 [0.512, 0.553]**, like the white box (0.521) and XGBoost (0.519) in the same run ✅ | AUC 0.653 (src, all 25 officer features) ⚠ run interactively; only AUC and Brier were kept. Without driver race but with its 8 race-derived officer features: 0.650. That is **not** race-blind: the fully race-blind version was run for XGBoost only. Within-officer not computed |
| Economics | 509 finds / 1,491 fruitless; break-even 0.34, the best: 82 more finds than random | Not computed |
| Main drivers | No native attribution. Permutation importance: **race is its largest feature (0.038, 15× XGBoost's)**. Disagrees with XGBoost on the drivers (Spearman −0.32). Its scores are compressed (sd 0.044 vs 0.139 for XGBoost), so race decides who is in the top K. ⏳ No global surrogate yet (optional, from the saved scores) | Not computed |
| One decision | Occlusion only: approximate, and its error cannot be bounded | — |
| Stability | Resampling the training context moves AUC by 0.024, more than the 0.020 spread between families. Top-200 list 2.3× chance | Not computed |
| Fairness | **The most race-driven selection of the three.** At K = 2,000 it flags **56.5% of white drivers, 1.7% of Black drivers, 0% of Hispanic drivers** (χ² 3,314, against 1,081 for the white box and 322 for XGBoost); FPR 56.4% vs 1.7% ✅ `data/matched_arms_diagnostics.json`. Best-calibrated arm within groups, but below the no-skill floor in every group. At equal score, Black drivers −5.9 pp [−10.7, −1.9] | Not computed (no predictions were saved). Removing driver race alone costs 0.003, with the 8 race-derived features still in |
| Deployability | Inference cost grows with the number of stops scored (209 s on CPU for 9,113 rows). The hosted API sends stop records to a third party: a data-governance issue for a police force | Same |

### 6.4 Trade-offs side by side
| Dimension | White box | XGBoost | TabPFN | What decides |
|---|---|---|---|---|
| Statistical performance | Lowest in Act I (−0.015 to −0.020, significant); −0.010 in Act II | Tied with TabPFN | Tied with XGBoost | Nothing separates the black boxes; the white box's cost is small |
| Economic performance | Break-even 0.30 | 0.33 | 0.34 | All barely above random (0.27) |
| Interpretability | Exact: the model **is** its explanation | Exact SHAP; a readable surrogate captures only 39% | Approximate only | White box |
| Stability | Head stable; drift in 2018 | Seed-stable; explanations shift across eras | Most sensitive to its training context | White box / XGBoost |
| Fairness | All three fail the same way, because the labels carry the disparity (§7) | same | The worst: flags 56.5% of white drivers, 1.7% of Black, 0% of Hispanic (χ² 3,314) | None: the fix is upstream |
| Governance | Runs anywhere, auditable, printable | Standard | GPU or third-party API | White box |

**Lesson:** performance does not separate the three models. Interpretability, stability and governance do,
and fairness fails for all three for one reason that no model can fix.

---

## 7. Fairness — test, identify, mitigate, and why no model fixes it
- **Test the decisions.** Discretionary searches of minority drivers find less: consent −5.2 pp (Black vs
  white) and −13.0 pp (Hispanic vs white); probable cause −15.9 pp. Mechanical searches do not (+1.8 pp).
  Within officers, 77% find less on Black drivers.
- **Test the models (Act I).**
  - *Independence fails, backwards:* models trained to find contraband select white drivers (χ² 863). A
    30-point selection gap would have to be declared acceptable before the model certifies as fair (TOST).
  - *Separation fails:* FPR gap −27 pp.
  - *Sufficiency* comes closest.
- **Identify.** XPER: race is 73.5% of the white box's signal. Proxies recover race at AUC 0.69–0.73, so
  dropping race does not remove it.
- **Mitigate.**
  - Dropping race alone doesn't work: proxies keep it.
  - With the officer record, race becomes nearly redundant and the gaps shrink (src XGBoost): FPR gap −27 pp
    (Act I) → −16 pp (+ officer record) → −5 pp (+ officer record, fully race-blind); χ² 863 → 463 → 65.
  - Fairness partial dependence: **no single feature restores parity (0 of 32).**
- **The impossibility, measured.** Base rates differ (test: white 25.5%, Black 19.5%, Hispanic 15.1%), so a
  model cannot both select groups equally *and* be equally accurate for each.
  - Race-aware white box and XGBoost: calibrated, but they select few minority drivers.
  - Race-blind versions: they select nearly equally, but over-score minority drivers at equal score (about 5
    points for Black drivers, 7–8 for Hispanic drivers).
- **Reading.** The disparity is in the labels: the outcomes were produced by search decisions that apply a
  lower bar to minority drivers (§5.5). A model trained on them must either reproduce that pattern or
  mis-score minority drivers. The fix is upstream, in search practice, which is exactly what an officer-level
  review addresses.

---

## 8. What a model can legitimately do here

| Use | Decision it supports | Verdict | Why |
|---|---|---|---|
| Roadside triage | "Should I ask this driver for consent?" | ✗ | Within an officer's own stops the best model reaches only 0.54–0.56 AUC; without the officer record its signal is mostly race; the ranked list is unstable. Race-aware scoring is not lawful, and race-blind scoring over-scores minority drivers |
| Justifying a search after the fact | "The model said 30%" | ✗ never | It would launder a discretionary decision; consent is the legal basis, not a probability |
| Deciding where to patrol | zones, shifts | ✗ not supported | Place and neighbourhood barely predict contraband; predictive-patrol feedback loops |
| **Reviewing consent-search practice** | **which officers' practices a supervisor reviews, coaches or retrains** | ✓ | Officer records differ 5× (5.9% vs 31.9%), persist, transfer to new officers and tie to racial skew; the value is quantified (§5.5) |
| Training content | what to teach | ✓ | Readable patterns: repeat searches, skewed searching |
| Oversight reporting | public accountability, policy evaluation | ✓ | Outcome tests by legal basis and race, monitored year on year |

---

## 9. Recommendation to the client

### 9.1 The answer
1. **Do not deploy a search score to officers:** none of the three models, with or without the officer record.
   In Act I it is near chance and mostly race; in Act II it has no value inside an officer's own decision.
2. **Deploy the white box inside an officer-level consent-search review,** run by supervisors, not at the
   roadside. For each officer, each quarter:
   - consent-search rate;
   - hit rate, shrunk, with an interval;
   - consent-search ratio by driver race;
   - hit-rate gap by race (his own outcome test);
   - a benchmark: a race-blind white-box expectation of the hit rate for the stops he made (where, when,
     why). ⏳ To build and validate in the pilot.
3. **Why the white box:**
   - the black boxes are not significantly different from each other, and their edge over the white box is
     small (0.010–0.020 AUC) and buys nothing in an officer-level review;
   - a review of people's professional conduct must be exact and contestable: the officer, his union and the
     oversight board must be able to read why he was flagged;
   - it is stable where it matters, runs anywhere, and sends no data to a third party.

### 9.2 Expected value
- If the least successful fifth of consent searches reached the median hit rate for the same finds: **~764
  fewer fruitless searches** in 2016–2018 [380, 1,206], 11% of all fruitless consent searches, **about two
  thirds of them of Black or Hispanic drivers**.
- They were made by 140 of the 763 officers who searched in the period: a reviewable number.

### 9.3 Trustworthy-AI requirements (EU High-Level Expert Group, 2019)
| Requirement | Roadside search score | Officer-level review (white box) |
|---|---|---|
| Human agency & oversight | pre-empts a discretionary legal decision | a supervisor decides; the model only flags |
| Technical robustness | near-chance, unstable top-K list | officer records persist, transfer and hold every year; drift → yearly retraining |
| Privacy & data governance | scores every driver | aggregates per officer; hashed IDs in the analysis; no third-party API |
| Transparency | race drives the signal; TabPFN can't be explained | 19 readable rules; published method |
| Non-discrimination & fairness | reproduces or inverts the disparity | targets the practice that produces it |
| Societal well-being | more searches of the wrong people | fewer fruitless searches, most of them of minority drivers |
| Accountability | who answers for a score? | review protocol, appeal for officers, oversight reporting |

**EU lens, if asked.** Under the EU AI Act, AI used by law enforcement to assess individuals is high-risk, and
risk assessments of individuals based solely on profiling are prohibited. A driver score would sit at or over
that line. (Check the exact article wording before putting it on a slide.)

### 9.4 Conditions (a 6-month pilot)
- **Human review only.** Flags go to a human review, never to automatic sanctions. Officers see, and can
  contest, their own report.
- **Statistical safeguards.** Intervals and shrinkage on every officer rate. Flag only when the pattern is
  clear, controlling false discoveries across ~1,500 officers.
- **Monitor:**
  - fruitless consent searches per 1,000 stops;
  - consent-search ratios and hit-rate gaps by race;
  - finds per search;
  - complaints.
- **Retrain yearly:** drift appears by 2018, and the officer features drift (PSI > 0.25).
- **Next analyses:**
  - the threshold test (Simoiu, Corbett-Davies & Goel; code already in `opp/`), which the outcome test cannot
    replace (infra-marginality);
  - whether assignment (units, beats) explains part of the officer differences.

---

## 10. Narrative guardrails and Q&A

### 10.1 Words
| Say | Don't say |
|---|---|
| "the task as posed cannot be done responsibly with this data" | "our model predicts whether a search will succeed" |
| "the outcome is explained more by who searches than by who is searched" | "officer features improve the model" (as a product claim) |
| "the officer record", "officers whose past searches rarely found anything" | "good cops / bad cops", "officer quality" |
| "consistent with a lower evidentiary bar" | "racist officers", "the officer discriminates" |
| "the racial signal stood in for search practice" | "race predicts contraband" |
| "inside an officer's own stops, the model is at chance" | 0.64 without the 0.50–0.56 next to it |
| "we recommend the white box, for officer-level review" | "deploy none" on its own (the brief asks which model) |

### 10.2 Likely questions and agreed answers
| Question | Answer |
|---|---|
| Aren't officer features cheating, or leakage? | They use only events strictly before the stop: same-minute stops are excluded and the stop's own outcome is never counted. Verified by brute force on 400 stops, and they hold on officers never seen in training. And we don't deploy them. |
| Obviously good officers find more. What's the insight? | The direction is obvious; the size isn't. The officer's record predicts more than everything about the driver, place and time together, and hit rates run from 5.9% to 31.9% across deciles of officers' records. That tells the client where fruitless searches come from. |
| Your best model has 0.64. Why not deploy it? | 0.64 is pooled across officers. Inside one officer's decisions it is 0.50–0.56: it ranks officers, not drivers. |
| AUC 0.55: did you fail? | Three model families, heavy feature engineering, a foundation model and all the data all land within 0.02. It's the information, not the model, and what the models do learn is race. That's the result the client needs before buying a tool. |
| Why consent searches only? | The outcome test depends on the legal basis; pooling reverses the sign (§2). |
| Why not just drop race? | Proxies recover race at AUC 0.69–0.73; no single feature restores parity (FPDP 0/32); and race-blind models over-score minority drivers at equal score. |
| Can you call these officers racist? | No. The outcome test can't prove intent (infra-marginality). We report groups, never individuals, and the threshold test is the next step. |
| Isn't reviewing officers with a model its own fairness problem? | Yes: officers are subjects too. Hence shrinkage and intervals, false-discovery control, human review, the right to see and contest one's report, no automatic sanctions, and a benchmark for where and when they work. |
| Why is "minutes since first stop of the day" so important? An artefact? | We checked: first-stop searches hit 20.9% vs 15.8%, in every time band and officer-volume group. A third are after midnight, so it means "first stop of the calendar day", not "start of shift". It isn't leakage (it uses only earlier stops). Reading: the first search is more targeted, later ones more speculative (a hypothesis). |
| Why the white box if XGBoost scores higher? | The black boxes don't differ significantly from each other; the white box costs 0.015–0.020 AUC in Act I and 0.010 in Act II. For reviewing people's conduct, exactness and contestability are requirements. |
| Is the foundation model also at chance inside an officer's own searches? | Yes. In the matched run (same training rows, same test set), within an officer: TabPFN 0.533 [0.512, 0.553], white box 0.521, XGBoost 0.519. And it is the most race-driven: at K = 2,000 it flags 56.5% of white drivers and 1.7% of Black drivers. |
| Are the TabPFN officer results reproducible? | Act I, yes (`src/`). Act II TabPFN numbers were run interactively (`FINDINGS.md` limitation 7). The reproducible Act II path is `scripts_claude/evaluate_officer_models.py` (white box and XGBoost). ⏳ TabPFN to be added there. |
| Why a time split and not a random one? | No learning from the future, officers persist across years, and it mimics deployment. A random split gives XGBoost 0.683, which is optimistic. |
| What about stops that were not searched? | Selective labels: we can't know, and every result is conditional on a search. The officer-level review doesn't need that counterfactual. |
| Why do race-aware models select almost no Hispanic drivers? | Consent searches of Hispanic drivers found contraband least often (15.1% vs 25.5% for white drivers in the test years), the footprint of the lower bar. A model maximising finds learns to avoid them. A disparity in the "favourable" direction is still race doing the work. |
| What would MNPD do on Monday? | Pilot the quarterly consent-search review for supervisors, publish the method to the oversight board, and use the patterns in training. |

### 10.3 The debate we had, and why we landed here
- **Option A: present the officer-aware model as the best predictor (0.64).** Rejected:
  - no decision value at the roadside (§5.3);
  - unfair to drivers (the score depends on who stopped them);
  - it invites the "good cop" objection with no answer.
- **Option B: drop the officer record.** Rejected: it throws away the evidence that explains *why* the task
  fails and *where* the fairness problem sits.
- **Option C (adopted):** keep the task as posed front and centre (Act I, three models, four dimensions). Use
  the officer record to *explain* where outcomes come from (Act II). Deploy only at the officer level, with
  the white box.
- **The risk of C** is that a jury reads it as "changing the question to escape a failed model". The answer is
  in the order: we do the task completely first, and the pivot comes from a measured result (§5.3), not from
  a preference.

---

## 11. Presentation spine (15 minutes)

| # | Beat | Message | Evidence | Figure / table | Min |
|---|---|---|---|---|---|
| 1 | The question | "Can a model tell an officer whom to search?" | — | title | 0.5 |
| 2 | Client & stakes | 83% of consent searches find nothing; disparities are documented | §1.2 | — | 1 |
| 3 | Data & scope | consent only; the outcome test by legal basis; selective labels | §2 | `outputs/deck/reversal.png` (release) | 1.5 |
| 4 | Three models, task as posed | all near chance, none beats a constant; black boxes tied | §4 | floors + DeLong tables | 1.5 |
| 5 | Economics | break-even 0.30–0.34 vs 0.27 random | §4 | `outputs/deck/economics.png` | 1 |
| 6 | What they learn | race is 73.5% of the signal; features predict race better; the selection inverts, and most for TabPFN (57% of white drivers flagged, 1.7% of Black); the list is unstable | §4, §6.3 | `outputs/deck/xper.png` | 1.5 |
| 7 | The pivot | the officer record adds +0.07 to every family, on unseen officers | §5.2 | table §5.2 | 1 |
| 8 | **The decisive slide** | inside an officer's own stops: 0.50–0.56. Which officer, never which driver | §5.3 | table §5.3 | 1 |
| 9 | What the white box says | 19 rules, 81% about the officer; XGBoost agrees (62%) | §6.1–6.2 | `reports/pltr/terms_top10.png` | 1.5 |
| 10 | What the records say | skewed officers find less; 9 officers = a quarter of Hispanic consent searches; 98% fruitless | §5.5 | disparity table | 1.5 |
| 11 | Fairness | the impossibility, measured; no feature fixes it; the fix is upstream | §7 | `reports/pltr/fairness_by_race.png` | 1 |
| 12 | Recommendation | the white box for officer-level review, not a roadside score; 764 fewer fruitless searches; trustworthy-AI table | §8–9 | use-case matrix + §9.3 | 1.5 |
| 13 | Limits & next | selective labels, infra-marginality → threshold test; pilot conditions | §9.4 | — | 0.5 |
| | | | | **Total** | **15** |

Backups: the legal-basis decomposition, every arm against its floor, the fairness ladder, PLTR's full rules,
Julia's surrogate tree, the first-stop check (§5.5), within-officer AUC by model.

---

## 12. Alignment register — conflicts found, and what we use

| # | Conflict | Where | Decision / action |
|---|---|---|---|
| 1 | Stop and search counts differ: 3,092,351 stops and 127,705 searches ("to 2018-12-31"), 127,122, 3,078,116, 3,078,074 / 127,121 | `README.md`, `FINDINGS.md`, deck, ours | Say **"3.08 million stops, 127,121 searches, 58,865 consent searches (2010–2018)"**. Fix `README.md` |
| 2 | "The white box" is the scorecard (deck) or PLTR (section 2) | deck vs `reports/pltr/README.md` | Both, one per act (§6.1). ⏳ Optional: PLTR on stop & driver features, so the white box is one model in both acts |
| 3 | Two officer-feature sets (16 vs 25) and three XGBoost numbers (0.640 / 0.642 / 0.646) | ours / Julia / src | Act II numbers from Scott's pipeline (reproducible). Julia's 0.642 in the XGBoost section (same pipeline, library versions). src's 0.646 only next to TabPFN's 0.653 |
| 4 | TabPFN Act II: no code (the only trace is a 6-row JSON of AUC and Brier on `release`); all 25 officer features vs 16 in the other rows; its "blind" keeps 8 race-derived features; the hosted checkpoint and training size are not recorded | `FINDINGS.md` limitation 7, `outputs/tabpfn_officer_behaviour.json` (release) | **Decided (27 Sep): keep Leo's results, no rerun.** Say: "same data, split, test set and officer-record features; the TabPFN run uses all 25 of them, where XGBoost on the same 25 gives 0.646 vs 0.640 with 16." Don't present its 0.650 as race-blind. TabPFN's Act I fairness and within-officer AUC now come from its saved scores (`scripts_claude/matched_arms_diagnostics.py`) |
| 5 | Three definitions of "race-blind" | ours / Julia: no race, sex or same-race feature; src "blind": no race; src "truly blind": no race-derived feature at all | Headline = **no race and no race-derived feature**. ⏳ Rerun PLTR that way (drop the 4 skew/gap features) |
| 6 | Deck: "If a model must ship: the scorecard, race-blind, with officer behaviour features" | `scripts/deck_content.py` | Contradicts §5.3; replace with §9 |
| 7 | Deck: "Give it the officer, and race stops mattering" (Mitigate) | deck | Reframe as an explanation (§5.4, point 5), not a deployable fix |
| 8 | Deck: "Deploy none of the three" | deck | The brief asks which model: "the white box, for officer-level review; no search score" (§9) |
| 9 | `FINDINGS.md` §9 still gives the base-model "do not deploy" reasons; limitation 5 says no surrogate was fitted (Julia fitted one) | `FINDINGS.md` | Leo to update, or mark as superseded by MASTER |
| 10 | "9 officers carry out half of all Hispanic consent searches" | `scripts_claude/pltr_findings.md`, earlier chat | **Corrected:** a quarter of all (half of those by the 43 officers analysed). Fixed in `pltr_findings.md` |
| 11 | PLTR term 5 / XGBoost's #2 read as "start of shift" | `reports/pltr/README.md`, Julia's note | **Corrected:** first stop of the *calendar* day (a third after midnight); the effect holds in every time band (§5.5, diagnostics §G). Update the wording |
| 12 | Two economics framings (break-even cost ratio vs break-even hit rate) | `FINDINGS.md` §7 vs ours | Same quantity (ratio = rate / (1 − rate)); use the ratio. Act II search-level economics are between officers, so use the officer counterfactual as the economic case |
| 13 | K = 2,000 vs top 22% = 2,005 | src vs ours / Julia | Say "top ~22% (K ≈ 2,000)" |
| 14 | Deck numbers come from `outputs/deck_stats.json`, which is git-ignored on `main` and exists only on `release`. `main` and `release` have diverged: `release` has the Officers and Stability app pages; `main` has the PLTR white-box files and this document | git | Merge `main` into `release` before rebuilding the app; decide where the new deck is built |
| 15 | The app's officer page: "a comparison, not the deployable model" | `app/views/officers.py` (release) | Consistent with this document; keep |

---

## 13. Where everything lives

Owners are read from the git history; correct them if wrong.

| What | File | Owner |
|---|---|---|
| Cleaning protocol | `scripts_claude/nashville_cleaning_protocol.md` | Scott |
| Features (built, fixed, dropped) | `scripts_claude/features_and_augmentation.md` | Scott |
| Master table (123 columns) | `data/nashville_consent_searches.parquet`, `data/nashville_columns.csv` | Scott |
| Act I, three arms, fairness tests, XPER, FPDP, DeLong | `src/`, `FINDINGS.md`, outputs on `release` | Leo |
| Act I, matched run: fairness at K = 2,000, calibration, within-officer AUC for all three arms (incl. TabPFN) | `scripts_claude/matched_arms_diagnostics.py` → `data/matched_arms_diagnostics.json` (reads `outputs/scores__consent__matched.parquet` from `release`) | Scott |
| Officer record | `scripts_claude/build_officer_features.py` | Scott |
| Officer disparity analysis | `scripts_claude/analyze_officer_disparity.py` → `data/officer_disparity_findings.json` | Scott |
| Act II models (white box, XGBoost) | `scripts_claude/evaluate_officer_models.py` → `data/officer_model_results.json`, `pltr_terms.json`, `officer_model_predictions.parquet` | Scott |
| Diagnostics (by year, economics, counterfactual, calibration, within-officer AUC, first stop) | `scripts_claude/officer_model_diagnostics.py` → `data/officer_model_diagnostics.json` | Scott |
| Stability over officer draws | `scripts_claude/officer_holdout_stability.py` → `data/officer_holdout_stability.json` | Scott |
| White-box section and figures | `reports/pltr/`, `scripts_claude/pltr_whitebox_report.py` | Scott |
| XGBoost section and figures | `reports/xgboost_officer/`, `scripts_claude/xgboost_officer_analysis.py`, `xgboost_surrogate.py` | Julia |
| App | `app/` (`release` branch: published version) | Anatole |
| Current deck | `reports/ISAF_presentation.pptx`, `scripts/deck_content.py` | Leo |

Reproduce Act II from the master table:
```
python scripts_claude/evaluate_officer_models.py        # ~25 min
python scripts_claude/officer_model_diagnostics.py      # ~2 min, no fitting
python scripts_claude/officer_holdout_stability.py --with-pltr   # ~55 min
python scripts_claude/pltr_whitebox_report.py [--fit]   # figures; --fit ~20 min
python scripts_claude/xgboost_officer_analysis.py       # Julia, ~2 min
```
