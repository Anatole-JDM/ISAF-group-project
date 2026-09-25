"""Black box models — the gradient-boosted trees arm (XGBoost)."""
import model_page

model_page.render(
    "gbm", "Black box models",
    description="""
**Gradient-boosted trees: XGBoost** (400 trees, depth 5, learning rate 0.05), falling back to
scikit-learn's `HistGradientBoosting` when XGBoost is not installed. The backend used is
recorded with each run (`gbm_backend` below).

The machine-learning arm: more flexible than the scorecard, with no built-in global
explanation.
""",
    missing_hint="Run `python -m src.train` to fit it.",
    explain_todo="TODO for the group: attach SHAP explanations here (`shap.TreeExplainer`), "
                 "global importance and per-stop.",
)
