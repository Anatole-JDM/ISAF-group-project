# 2. White-box model: Penalised Logistic Tree Regression (PLTR)

Owners: [ ] · Code: `main` — `scripts_claude/pltr.py` (model), `scripts_claude/evaluate_officer_models.py` (evaluation), `scripts_claude/officer_model_diagnostics.py`, `scripts_claude/officer_holdout_stability.py`, `scripts_claude/pltr_whitebox_report.py` (this section); tests `scripts_claude/test_pltr.py`; full write-up `scripts_claude/pltr_findings.md` · Status: done except the items marked **[--fit]** (run `python scripts_claude/pltr_whitebox_report.py --fit`, ~20 min)

The white box is the **19-term PLTR** on the same 33 inputs and the same time split as the XGBoost arm (race-aware `full_time_officer`). Its race-blind version is the mitigation (2.5). Reference points: plain logistic scorecard with the same inputs (AUC 0.627) and the team's current scorecard without officer features (0.555).

## 2.1 Specification

| Item | Value |
|---|---|
| Algorithm / library / version | PLTR, Dumitrescu, Hué, Hurlin & Tokpavi (2022), course slides 66-71. Own implementation (`pltr.py`) on scikit-learn 1.8.0, Python 3.14. Stage 1: rules from short decision trees (depth 1 per variable, depth 2 per pair, min 200 searches per leaf); stage 2: logistic regression on [inputs + rules] with an adaptive lasso (liblinear). |
| Preprocessing (encoding, scaling, missing values) | Linear part identical to the team scorecard: numerics median-imputed and standardised; categoricals most-frequent-imputed + one-hot (levels < 25 rows grouped), then one-hot columns with < 200 training rows removed (31 removed: a zone seen 11 times otherwise got the largest coefficient). Trees: numerics median-imputed, categoricals out-of-fold smoothed target encoding; every categorical rule is translated back into a list of categories. |
| Features used (number + list or link) | 33 inputs, same as XGBoost: age, sex, race, precinct, zone, reason for stop, plate state, hour, weekday, month + 7 time features + 16 officer features (`evaluate_officer_models.py`, `OFF` + `OFF_RACE`). Candidate terms: 106 linear columns + 644 rules; **19 selected (7 linear, 12 rules)**. |
| Regularisation, hyperparameters, how they were chosen | Adaptive lasso (weights 1/abs(pilot ridge coefficient), gamma = 1, pilot C = 1). Penalty on an 11-value grid (C 0.003-1) by an **honest** 5-fold CV on the training years: rules, encodings and weights re-learned inside each fold (learning them once leaked labels: CV 0.755 vs test 0.566 on synthetic data). One-standard-error rule → C = 0.1, 61 terms. **White box = weakest penalty keeping <= 30 terms, chosen on training data only → C = 0.032, 19 terms.** |
| Class imbalance handling | None, as in the other arms (base rate 16.1% train, 21.4% test). |
| Training size | 49,752 consent searches, 2010-2015. |
| Seeds if used | 42 (CV folds, trees). Deterministic given the data; variability assessed over 5 draws of held-out officers and 50 bootstrap resamples (2.4). |

## 2.2 Performance

| Metric | Validation (5-fold CV, train years) | Test 2016-2018 (9,113) | 95% CI |
|---|---|---|---|
| AUC | 0.655 | **0.630** | 0.615-0.643 |
| PR-AUC | – | 0.319 (base rate 0.214) | 0.300-0.338 |
| Brier score | – | 0.161 (no-skill 0.168) | 0.157-0.166 |
| AUC on random split (vs time split) | – | **[--fit]** | – |

For reference, same test set: 61-term PLTR (1-SE) 0.636 [0.622, 0.649]; logistic scorecard + officer 0.627; **XGBoost + officer 0.640**; team scorecard without officer features 0.555. White box vs team scorecard: **+0.075 [+0.060, +0.091]** (paired bootstrap). Officer holdout (officers never seen in training): 0.629 [0.597, 0.659]. CV > test because the CV folds mix years and the hit rate rises after 2016 (drift, not leakage: on pure noise the CV gives 0.505).

**Economic** (top 22% flagged = the team's budget, 2,005 of 9,113): 7,108 searches avoided (78%), 1,304 of 1,946 finds lost (67%) · **cost per find 3.12 searches** (vs 4.68 if all searched; hit rate 32.0% vs 21.4%; XGBoost 2.97) · milder cut: dropping only the lowest-scored 20% avoids 1,632 fruitless searches and loses 191 finds (10%) · gains curve: `reports/pltr/gains_curve.png`

## 2.3 Interpretability

**Global** — the 10 most important terms (importance = |coefficient| x spread, i.e. weight x coverage; linear odds ratios per standard deviation); together 92% of the importance:

| # | Term | Odds ratio | Searches concerned | Reading |
|---|---|---|---|---|
| 1 | Officer's past-year consent hit rate | 1.43 per SD | all | the officer's track record is the main driver (32% of importance) |
| 2 | Officer's past hit rate on this driver's race <= 17.9% AND his overall past hit rate <= 12.9% | 0.64 | 15% | officers who rarely find anything keep finding nothing |
| 3 | Driver Hispanic | 0.71 | 8% | lower hit rate on Hispanic drivers (history of looser searches) |
| 4 | Driver white | 1.21 | 36% | higher hit rate on white drivers |
| 5 | Officer experience > 569 days AND <= 5.5 min since his first stop of the day | 1.24 | 21% | first searches of an experienced officer's shift succeed more |
| 6 | Officer's previous search < ~110 min ago | 0.79 | 10% | repeat searches succeed less (momentum rather than evidence) |
| 7 | Male driver AND officer with <= 9 consent searches in the past year | 0.86 | 19% | occasional searchers do worse on men |
| 8 | Hour ~4:30-7:30 AND officer not among the busiest | 1.46 | 2.5% | early-morning searches succeed more |
| 9 | Officer's Hispanic/white search skew > 1.6x AND his Hispanic hit gap > -3.5 pts | 0.93 | 21% | officers who over-search Hispanic drivers find less |
| 10 | Officer search rate <= 25.5% AND driver aged <= 19 | 1.09 | 9% | selective officers searching teenagers |

The remaining 9 terms (time, month, weekday, precinct, zone, reason for stop) carry 7% together. Figure: `reports/pltr/terms_top10.png`.

**Local** — 3 stops explained (a hit, a miss, the borderline case at the cut-off); PLTR is additive in log-odds, so each term's contribution is exact: **[--fit]**

**Course methods used:** PLTR itself (interpretable non-linear model, slides 66-71); odds ratios and average marginal effects (slides 35-36); importance = weight x coverage; exact additive local decomposition **[--fit]**; stability of selection (2.4). No SHAP/LIME needed: the model is its own explanation.

**Finding in one sentence:** 81% of the white box's importance lies in terms that involve the officer (64% in terms about the officer alone) — like XGBoost, it predicts which officer searches well, and it says so in 19 readable rules.

## 2.4 Stability

| Test | Result |
|---|---|
| AUC across seeds (mean ± sd) | Training is deterministic; across **5 draws of held-out officers**: 0.618 ± 0.026 (min 0.589, max 0.660); gain over the team scorecard +0.071 (+0.021 to +0.115), CI above 0 in 4/5 draws. Bootstrap test-AUC spread: **[--fit]** |
| Coefficients across bootstrap resamples (sign changes?) | **[--fit]** (50 resamples, rules fixed: selection frequency and sign changes per term). Across data splits: the officer's consent hit rate and driver race are selected in all 5 officer draws. |
| AUC per year (rolling windows) | 2016: 0.625 [0.607, 0.645] · 2017: 0.651 [0.631, 0.675] · 2018: 0.605 [0.577, 0.630] (XGBoost 2018: 0.628) — drift by 2018, retrain yearly |
| Rank correlation of variable importance across resamples | **[--fit]**. Across splits (time vs officer holdout): 21% of variable combinations shared, but they carry 62% of the importance — the head is stable, the tail is not; report only the core |

## 2.5 Fairness

Top 22% flagged, test 2016-2018 (white 3,368 · Black 4,947 · Hispanic 689). Gap = group − white, 95% bootstrap CI.

| Metric | White | Black | Hispanic | Gap / test |
|---|---|---|---|---|
| Share flagged "search" | 36.4% | 14.9% | 4.1% | ratio vs white 0.41 / 0.11; −21.5 pp [−23.2, −19.6] / −32.3 pp [−34.5, −30.2] |
| True positive rate | 48.0% | 22.8% | 6.7% | −25.1 pp [−29.6, −20.9] / −41.2 pp [−46.7, −35.4] |
| False positive rate | 32.4% | 13.0% | 3.6% | −19.5 pp [−21.3, −17.4] / −28.8 pp [−31.1, −26.4] |
| Hit rate among those flagged | 33.6% | 29.9% | 25.0% | −3.7 pp [−8.1, 0.0] / −8.6 pp [−23.7, +8.7] |
| Calibration (predicted − observed) | −1.2 pp | +0.2 pp | −1.3 pp | at equal score: Black −0.9 pp [−2.8, +1.1], Hispanic +2.0 pp [−1.2, +5.1] → calibrated |

**Variables driving the gap (proxies):** driver race directly (2 terms, 16% of importance) and the rule on the officer's past hit rate on the driver's race (14%): 31% of the importance involves race. Zone, precinct and the officer's own skew act as proxies once race is removed.

**Mitigation tried and cost in performance:** race-blind PLTR (no race, sex or race-derived officer feature; 30 terms). Cost **0.002 AUC** (0.630 → 0.628). Share flagged 26.9% / 19.2% / 19.0% (ratios 0.72 / 0.71); FPR gap −19.5 → −6.2 pp (Black), −28.8 → −6.7 pp (Hispanic); TPR 38.0% / 29.1% / 33.7%. **But calibration breaks:** at equal score Black drivers are 4.6 pp [2.7, 6.1] and Hispanic drivers 7.2 pp [4.3, 9.9] less likely to carry contraband; it over-predicts Hispanic drivers by +3.7 pp and under-predicts white drivers by −3.3 pp. Hit rate among flagged 36.0% / 29.4% / 26.7%. Equal flagging and equal calibration cannot both hold when base rates differ (impossibility result).

## 2.6 For the slides

**Finished figures:** `reports/pltr/terms_top10.png`, `reports/pltr/gains_curve.png`, `reports/pltr/fairness_by_race.png`

**3 key messages:**
1. A 19-rule model anyone can read reaches AUC 0.630 — 0.01 below XGBoost and +0.075 over the team's scorecard: here interpretability costs almost nothing.
2. The rules describe officers, not drivers: track record, shift rhythm, repeat searches — and officers who over-search Hispanic drivers find less.
3. Race-aware it is calibrated but flags almost no minority drivers (and uses race); race-blind it flags groups almost equally for 0.002 AUC but over-scores minority drivers by 5-7 points. Not deployable to score drivers; usable to review officers.

## 2.7 Q&A

| Likely question | Agreed answer |
|---|---|
| Why PLTR and not plain logistic regression? | Same inputs, +0.009 AUC (0.636 vs 0.627 for the 61-term version) from threshold effects and interactions logistic regression cannot express, and each rule stays readable with an odds ratio. It is the course's answer to "interpretable but non-linear". |
| Why 19 terms and not the best CV model? | The CV-best model has 86 terms; the one-standard-error rule gives 61 (AUC 0.636). The 19-term version, chosen on training data by size, loses 0.006 AUC and can be read on one slide. |
| CV AUC 0.655 but test 0.630 — leakage? | No. CV folds mix 2010-2015; the test years have a higher hit rate (drift). The CV is honest: rules are re-learned in each fold, and on pure noise it gives 0.505 and keeps 0 terms. |
| Isn't using driver race unlawful? | Yes; the race-aware model is a diagnostic. The race-blind version is the deployable candidate, and it fails calibration (2.5). |
| Why are some odds ratios "per SD"? | Numeric inputs are standardised in the linear part (same as the team scorecard); rules are 0/1, so their odds ratio is for the rule being true. |

**Known weakness:** only the head of the model is stable (about 20% of variable combinations repeat across splits, carrying ~60% of the importance), so present the core terms only; PLTR and logistic lose more than XGBoost in 2018 (0.605 vs 0.628) — retrain yearly; term 5 ("first stop of the day") shares Julia's possible recording artefact for `off_minutes_since_first_stop_today`, to check with Scott; the disparity features are computed from the race of the officer's past drivers.
