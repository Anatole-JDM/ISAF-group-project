"""White box models — the scorecard arm (logistic regression)."""
import model_page

model_page.render(
    "scorecard", "White box models",
    description="""
**Scorecard: logistic regression on one-hot-encoded pre-search features.** Readable to a
caseworker: the coefficients are the explanation.

- `class_weight` is left at None, not "balanced". Balanced weighting improves ranking on an
  imbalanced target but destroys calibration, and calibration *is* the sufficiency dimension
  of the fairness taxonomy.
- For the full WOE/IV scorecard banks deploy, swap the preprocessor for optbinning's
  `BinningProcess`: monotonic bins and points per attribute.
""",
    missing_hint="Run `python -m src.train` to fit it.",
    explain_todo="TODO for the group: show the signed coefficients (the scorecard's global "
                 "explanation) and per-stop contributions here.",
)
