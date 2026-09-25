"""Tabular foundation models — the TabPFN arm."""
import model_page

model_page.render(
    "tabpfn", "Tabular foundation models",
    description="""
**TabPFN**: a pre-trained transformer that predicts in a single forward pass, conditioning on
the training rows as context (Hollmann et al., *Nature* 637, 2025). Default checkpoint **V2**,
which is ungated; opt in to V3.5 per machine with `TABPFN_VERSION=V3_5`.

Its cost grows roughly quadratically with the training context, so it trains on a subsample
(`TABPFN_MAX_TRAIN`, default 5,000) and only in the **matched** mode, where every arm trains
on that same subsample so the three-way comparison stays fair. It has no native global
explanation: the honest black box of the comparison.
""",
    missing_hint="TabPFN is fitted only in the `matched` training mode, and only where the "
                 "`tabpfn` package is installed. Install it, then run `python -m src.train` "
                 "(set `TABPFN_MAX_TRAIN=2000` for a faster CPU run).",
    explain_todo="TODO for the group: local explanations via `tabpfn-extensions` (SHAP-based).",
)
