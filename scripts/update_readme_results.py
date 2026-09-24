"""Regenerate the Results section of README.md from outputs/.

Run after `python -m src.train` so the README never drifts from the artifacts.
Rewrites everything between the RESULTS markers; edit anything outside them.

    python scripts/update_readme_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _load(name):
    p = C.OUTPUTS / name
    return json.loads(p.read_text()) if p.exists() else None


def build() -> str:
    full = _load("metrics__consent__full.json")
    blind = _load("metrics__consent__full_blind.json")
    sig = _load("officer_signal.json")
    if not full:
        return "_No cached runs. Run `python -m src.train`._"

    arms = full["arms"]
    rows = "\n".join(
        f"| {a['arm']} | {a['n_train']:,} | {a['auc']:.4f} | {a['pr_auc']:.4f} | {a['brier']:.4f} |"
        for a in arms
    )
    a0 = arms[0]
    af = max(a["auc"] for a in arms)
    L = [
        "## Results (consent stratum, train <2016 → test ≥2016)",
        "",
        "Regenerate: `python -m src.train && python scripts/update_readme_results.py`",
        "",
        "| Arm | n train | AUC | PR-AUC | Brier |",
        "|---|---|---|---|---|",
        rows,
        "",
        f"n test = {a0['n_test']:,} · base rate {a0['base_rate_train']:.1%} train → "
        f"{a0['base_rate_test']:.1%} test (a real shift across the regime split) · "
        f"GBM backend `{full['gbm_backend']}`.",
    ]
    if full.get("failed_arms"):
        L += ["", "**Arms that did not run:**", ""]
        L += [f"- `{k}` — {v}" for k, v in full["failed_arms"].items()]

    L += ["", "### There is almost no signal, and it is the wrong signal", "",
          f"**Best AUC is {af:.4f}** — {af - 0.5:.4f} above chance. Decomposing that sliver:", "",
          "| Ablation | AUC | Share of above-chance signal |", "|---|---|---|"]
    if sig:
        L.append(f"| Pre-search features only | {sig['baseline']:.4f} | — |")
        L.append(f"| **+ officer identity** | {sig['with_officer']:.4f} | "
                 f"**{sig['above_chance_uplift']:.0%}** |")
    if blind:
        ab = max(a["auc"] for a in blind["arms"])
        L.append(f"| **− driver race** | {ab:.4f} | **{(af - ab) / (af - 0.5):.0%}** |")
    L += [
        "",
        "Most of the model's discriminative power is *who stopped you* and *what you "
        "look like*. Almost none of it is anything about the situation.",
        "",
        "Removing race does **not** make the model race-neutral: precinct and zone are "
        "strong proxies in a segregated city. Run both (`full` and `full_blind`) and "
        "report both.",
        "",
        "### The selection inverts",
        "",
        "A contraband-optimising model searches *white* drivers far more, because white "
        "drivers have the higher hit rate in discretionary searches — the reverse of "
        "actual officer behaviour. That is the outcome test expressed as a model. Check "
        "the Fairness tab at your chosen K; a group with zero selections yields an "
        "undefined PPV, which `group_rates` returns as NaN rather than a misleading zero.",
        "",
        "### Recommendation to the client: do not deploy",
        "",
        "Not because it is unfair *or* because it is inaccurate, but because all four "
        "dimensions point the same way — near-random discrimination, signal dominated by "
        "officer identity and race, large false-positive disparities, and negative net "
        "benefit under any defensible cost of searching an innocent driver.",
    ]
    return "\n".join(L)


def main() -> None:
    p = Path(__file__).resolve().parents[1] / "README.md"
    s = p.read_text()
    body = f"{START}\n{build()}\n{END}"
    if START in s and END in s:
        s = s[: s.index(START)] + body + s[s.index(END) + len(END):]
    else:
        s = s.replace("\n---\n\n## Traps", f"\n---\n\n{body}\n\n---\n\n## Traps")
    p.write_text(s)
    print(f"README results updated from {C.OUTPUTS}")


if __name__ == "__main__":
    main()
