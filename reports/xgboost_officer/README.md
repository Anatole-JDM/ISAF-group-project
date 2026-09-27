# XGBoost with the officer features — stability, fairness, interpretation

Reproduce: `python scripts_claude/xgboost_officer_analysis.py` (about 2 minutes on 4 CPU cores, no GPU).
Raw numbers: `data/xgboost_officer_analysis.json`. Figures: this folder.

Setup is identical to `scripts_claude/evaluate_officer_models.py` (it is imported, not copied): consent
searches 2010-2018 (58,865), train < 2016 (49,752), test ≥ 2016 (9,113), same preprocessing and
hyperparameters, seed 42. Three XGBoost models:

| Model | Features | Test AUC |
|---|---|---|
| `full_time` | team baseline, no officer features | 0.571 |
| `full_time_officer` | + 16 officer features (**model under study**) | **0.642** |
| `full_time_officer_blind` | without driver race, sex and `off_hit_rate_same_race_past` | 0.636 |

(Scott's run reports 0.640 for the same model; the 0.002 gap is library-version noise and sits well
inside the seed spread below.)

---

## 1. Stability

**Performance is stable.**

| Check | Result |
|---|---|
| AUC by test year (2016 / 2017 / 2018) | 0.641 / 0.652 / 0.625 — the gain over baseline holds every year |
| 10 random seeds | AUC 0.640 ± 0.002; scores correlate ≥ 0.94; ≥ 81% of the top-22% selection is shared |
| 20 bootstrap refits of the training set | AUC 0.635, 95% range 0.628–0.641 |
| Unseen officers (GroupKFold vs random KFold, training years) | 0.669 vs 0.680 — gap only **0.011** |

The last line matters. Officer features describe an officer's *history*, so they transfer to
officers the model has never seen. This is not the model memorising individual officers.

**The explanation is mostly stable, but shifts across eras.**

- Across bootstrap refits, the importance ranking correlates at 0.94 on average (min 0.89). The top
  4 features (`off_consent_hit_rate_365d_shrunk`, `off_hit_rate_same_race_past`,
  `off_minutes_since_first_stop_today`, `subject_race`) are always in the top 5
  (`stability_importance_bootstrap.png`).
- Training on 2010-2012 vs 2013-2015 gives rankings that correlate at only 0.85, with 6 of the top 10
  shared. `subject_sex` moves 17 places and `off_stops_365d` moves 12. The model's reasons depend on
  the period it learned from.

**Drift is large on the officer features themselves.** PSI train → test:
`off_experience_days` 0.95, `off_hit_rate_365d_shrunk` 0.83, `off_consent_hit_rate_365d_shrunk` 0.75,
`off_consent_365d` 0.37 (> 0.25 = material shift). The score itself has PSI 0.24, right at the
threshold. Partly mechanical, since experience grows with time. Still, the most important features
are also the ones that drift most, so the model needs monitoring if deployed.

**Distance among models.** Adding officer features changes *who* is flagged far more than it changes
AUC. Only 50% of the top-22% selection overlaps between `full_time` and `full_time_officer` (score
Spearman 0.48). Two models with close AUCs are not interchangeable for a search policy.

## 2. Fairness (top 22% of test searches flagged, 95% bootstrap CIs)

| Model | Group | Selection rate | False positive rate | Hit rate of selected |
|---|---|---|---|---|
| `full_time` | white | 40.4% | 37.9% | 30.0% |
| | black | 12.2% | 11.5% | 24.6% |
| | hispanic | 2.2% | 1.2% | 53% (CI 28–81%, n tiny) |
| `full_time_officer` | white | 35.2% | 29.8% | 36.8% |
| | black | 14.8% | 12.9% | 30.0% |
| | hispanic | 9.6% | 8.7% | 22.7% |
| `full_time_officer_blind` | white | 26.0% | 21.8% | 37.7% |
| | black | 19.5% | 17.2% | 29.2% |
| | hispanic | 19.3% | 16.8% | 26.3% |

(`fairness_by_race.png`)

- **Independence / separation.** Every model flags white drivers most, the reverse of officer
  behaviour. Officer features narrow the gap. The black/white selection ratio goes from 0.30 to 0.42,
  and the black − white FPR gap from −26 pp to −17 pp. The blind model narrows it further (ratio 0.75,
  FPR gap −5 pp).
- **Sufficiency** is violated in every model. Among flagged drivers, the black hit rate is 6.8 pp below
  the white one (officer model) and 8.6 pp below it (blind model). The two criteria cannot both hold
  here, which is the impossibility result again.
- **Calibration by group** improves a lot with officer features. The mean predicted rate minus the
  observed rate goes from −5.1 / −4.4 / −8.0 pp (white / black / hispanic) to +0.2 / −0.4 / −1.8 pp.
  Without them, the baseline under-predicts the post-2016 rise in hit rate for everyone.
- **Removing race does not remove race.** `off_hit_rate_same_race_past` is computed from the driver's
  race. For Hispanic drivers its mean SHAP is −0.21 log-odds, on top of −0.54 from `subject_race`
  itself. That is why the blind model also drops it.

The course coding (Y = 1 favourable = not searched) flips the reading of these rates; see
`src/fairness.both_codings()` for the convention used in the report.

## 3. Interpretation (exact TreeSHAP, log-odds, one-hot levels summed per column)

- **62% of the model's total attribution goes to officer features**
  (`interpretation_global_importance.png`). Top 5: officer's past consent hit rate, minutes since the
  officer's first stop that day, officer's past hit rate on drivers of this race, driver race,
  officer's overall past hit rate.
- **Directions are monotone and intuitive** (`interpretation_beeswarm.png`). An officer with a better
  past hit rate gets a higher score. The Spearman correlation between value and SHAP is 0.97 for the
  consent hit rate and 0.94 for the same-race hit rate. More experience gives a higher score (0.85).
  Across quintiles of the officer's past consent hit rate, the observed hit rate rises from 12% to 32%,
  and the SHAP value rises from −0.15 to +0.43.
- **One pattern to check with Scott:** `off_minutes_since_first_stop_today`. Searches in the officer's
  first ~3 minutes of the day (40% of test rows) hit at 26%. Searches 3–80 minutes in hit at 14%. The
  model uses this heavily (rank 2–3). This is not outcome leakage, since the feature only uses earlier
  stops. But it could be a recording artefact, such as a shift start logged as the first stop, and
  should be understood before being presented.
- **Local explanations** (in the JSON). The highest-scored driver (82%) was a white woman stopped by an
  officer with a 57% past consent hit rate. That single feature contributes +0.81 log-odds. No
  contraband was found.

## 4. Global surrogate

Reproduce: `python scripts_claude/xgboost_surrogate.py` → `data/xgboost_surrogate.json`, `surrogate_tree.png`.

A decision tree is fitted to **XGBoost's predicted probability** on the training rows (not to the
true outcome), then checked on the 2016-2018 test rows:

| Surrogate | Leaves | R² vs XGBoost | Rank corr. | Top-22% overlap | AUC vs truth |
|---|---|---|---|---|---|
| tree, depth 2 | 4 | 0.32 | 0.55 | 50% | 0.601 |
| **tree, depth 3** | 8 | **0.39** | **0.60** | **52%** | **0.617** |
| tree, depth 4 | 16 | 0.47 | 0.64 | 56% | 0.624 |
| tree, depth 6 | 53 | 0.55 | 0.72 | 59% | 0.629 |
| linear regression | – | 0.62 | 0.76 | 65% | 0.632 |
| *XGBoost itself* | | | | | *0.642* |

**The depth-3 tree** (`surrogate_tree.png`) reads as: *first, how good is this officer's consent-search
track record (≤ 18%, 18–24%, > 24%)? Then, is the driver white?* It splits on driver race twice. 91% of
its split importance is officer features, and the rest is `subject_race_white`. Its 8 leaves are well
calibrated on the test years: each leaf's mean score matches its observed hit rate within 2.2 pp. The
top leaf (officer consent hit rate > 24%, search within 6.5 minutes of the officer's first stop)
scores 36% and hits 37%.

**The limit is fidelity.** Even 53 leaves reproduce only 55% of XGBoost's score variance, and the
readable 8-leaf tree agrees on only half of the drivers XGBoost would search. A tree like this is fine
to *describe* the black box on a slide, but not to *stand in for it*. A plain linear model is the most
faithful surrogate (R² 0.62), and on its own it gets within 0.01 AUC of XGBoost.
The trees and interactions add little predictive value here.

## What this means for the argument

Officer features raise AUC by 0.07, stay stable across seeds, years and unseen officers, and fix the
calibration drift. But they make the model even more a model of **who searches**, not of **who carries
contraband**: 62% of the attribution is officer history. This fits the selective-labels thesis in the
main README. The signal we can learn is officer skill and selection, not driver risk.
