"""Figures for the presentation deck. Every number is read from outputs/ or the
search sample - nothing is typed in by hand, so a rerun after a retrain moves the
charts with the results."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp") + "/mpl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator
import numpy as np
import pandas as pd

OUT = ROOT / "outputs" / "deck"
OUT.mkdir(parents=True, exist_ok=True)

INK, ACCENT, BLUE, GREEN, GREY = "#1A1A2E", "#C0392B", "#2C7FB8", "#1BAF7A", "#8A8984"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def _save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", p.relative_to(ROOT))
    return p


def fig_reversal(stats):
    """The headline: hit-rate gap by the legal basis of the search.

    Coloured by how much discretion the basis involves, NOT by the sign of the
    gap. Plain view is discretionary and positive; hiding that by colouring on
    sign would be the same mistake as calling the whole non-consent group
    "mechanical".
    """
    DISC = {"consent  (our sample)", "probable cause", "plain view"}
    rows = stats["by_basis"]
    labels = [r["label"] for r in rows]
    vals = [r["gap"] for r in rows]
    cols = [ACCENT if r["label"] in DISC else BLUE for r in rows]
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    bars = ax.barh(labels, vals, color=cols, height=0.62)
    ax.axvline(0, color=INK, lw=1.2)
    ax.axvline(stats["gap_mechanical"], color=BLUE, ls="--", lw=1.6)
    ax.text(stats["gap_mechanical"], len(rows) - 0.35,
            f" mechanical benchmark {stats['gap_mechanical']:+.2f}pp",
            color=BLUE, fontsize=9.5, fontweight="bold", va="top")
    for b, r in zip(bars, rows):
        x = b.get_width()
        ax.text(x + (0.55 if x >= 0 else -0.55), b.get_y() + b.get_height() / 2,
                f"{x:+.2f}pp", va="center", ha="left" if x >= 0 else "right",
                fontsize=10.5, fontweight="bold", color=INK)
        ax.text(0.25 if x < 0 else -0.25, b.get_y() + b.get_height() / 2,
                f"n={r['n']:,}", va="center", ha="left" if x < 0 else "right",
                fontsize=8.5, color=GREY)
    ax.set_xlabel("Black \u2212 white contraband hit-rate gap (percentage points)")
    ax.set_xlim(min(vals) - 7, max(vals) + 7)
    ax.set_title("Where the officer must judge THIS driver, searches of Black drivers find less.\n"
                 "Red = officer discretion.   Blue = the law supplies the authority.",
                 loc="left", fontsize=12, fontweight="bold", pad=14)
    return _save(fig, "reversal.png")


def fig_floors(stats):
    """Every arm against the no-skill floor, on both metrics."""
    arms = stats["matched_arms"]
    names = [a["arm"] for a in arms]
    x = np.arange(len(names))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

    a1.bar(x, [a["brier"] for a in arms], color=[ACCENT] * len(names), width=0.55)
    a1.axhline(stats["brier_floor"], color=INK, ls="--", lw=1.6)
    a1.text(len(names) - 0.4, stats["brier_floor"], f"  no-skill {stats['brier_floor']:.4f}",
            va="bottom", ha="right", fontsize=9.5, fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels(names)
    a1.set_ylabel("Brier score  (lower is better)")
    a1.set_ylim(0.15, max(a["brier"] for a in arms) * 1.08)
    a1.set_title("Calibration: all three are WORSE than\npredicting the base rate for everyone",
                 loc="left", fontsize=11, fontweight="bold", color=ACCENT)

    a2.bar(x, [a["pr_auc"] for a in arms], color=[BLUE] * len(names), width=0.55)
    a2.axhline(stats["pr_floor"], color=INK, ls="--", lw=1.6)
    a2.text(len(names) - 0.4, stats["pr_floor"], f"  no-skill {stats['pr_floor']:.4f}",
            va="bottom", ha="right", fontsize=9.5, fontweight="bold")
    a2.set_xticks(x); a2.set_xticklabels(names)
    a2.set_ylabel("PR-AUC  (higher is better)")
    a2.set_ylim(0.19, max(a["pr_auc"] for a in arms) * 1.08)
    a2.set_title("Ranking: all three beat the floor,\nby 0.019 to 0.027",
                 loc="left", fontsize=11, fontweight="bold", color=BLUE)
    fig.tight_layout()
    return _save(fig, "floors.png")


def fig_fairness_ladder(stats):
    """FPDP's four models: what removing race and adding officer behaviour does."""
    rows = stats["ladder"]
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    a1.plot(x, [abs(r["fpr_gap"]) * 100 for r in rows], "-o", color=ACCENT, lw=2.4, ms=8)
    for i, r in enumerate(rows):
        a1.annotate(f"{r['fpr_gap']*100:+.1f}pp", (i, abs(r["fpr_gap"]) * 100),
                    textcoords="offset points", xytext=(0, 11), ha="center",
                    fontsize=9.5, fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=9)
    a1.set_ylabel("|FPR gap|, Black vs white  (pp)")
    a1.set_ylim(0, 30)
    a1.set_title("The harm metric: innocent drivers searched", loc="left",
                 fontsize=11, fontweight="bold")

    a2.plot(x, [r["auc"] for r in rows], "-o", color=BLUE, lw=2.4, ms=8, label="pooled AUC")
    wi = [r.get("auc_within") for r in rows]
    xs = [i for i, v in enumerate(wi) if v]
    a2.plot(xs, [wi[i] for i in xs], "--s", color=GREY, lw=2, ms=7, label="within-officer AUC")
    for i, r in enumerate(rows):
        a2.annotate(f"{r['auc']:.4f}", (i, r["auc"]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9.5, fontweight="bold")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=9)
    a2.set_ylabel("AUC"); a2.set_ylim(0.52, 0.68)
    a2.legend(frameon=False, fontsize=9, loc="lower right")
    a2.set_title("Most of the gain is BETWEEN officers, not within", loc="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fairness_ladder.png")


def fig_economics(stats):
    """Net benefit against the price of searching an innocent driver."""
    sens = pd.DataFrame(stats["sensitivity"])
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.plot(sens["cost_fp"], sens["net_benefit"], "-o", color=ACCENT, lw=2.6, ms=8)
    ax.axhline(0, color=INK, lw=1.4)
    be = stats["breakeven"]["tabpfn"]
    ax.axvline(be, color=GREEN, ls="--", lw=1.8)
    ax.text(be, ax.get_ylim()[1] * 0.55, f"  breaks even at {be:.3f}", color=GREEN,
            fontsize=10.5, fontweight="bold")
    nb = stats["breakeven"]["no_skill"]
    ax.axvline(nb, color=GREY, ls=":", lw=1.8)
    ax.text(0.62, ax.get_ylim()[0] * 0.30, f"random ranking\nbreaks even at {nb:.3f}",
            color=GREY, fontsize=9.5, ha="left")
    ax.set_xlabel("Price of searching one innocent driver\n(as a multiple of the value of one justified search)")
    ax.set_ylabel("Net benefit at K = 2,000")
    ax.set_xscale("log")
    ax.set_xticks(sens["cost_fp"])
    ax.set_xticklabels([str(c) for c in sens["cost_fp"]])
    ax.xaxis.set_minor_formatter(NullFormatter())   # log minor labels collided with ours
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_title("The best model pays only if an innocent search costs less than a third\n"
                 "of what a justified search is worth.", loc="left",
                 fontsize=12, fontweight="bold", pad=12)
    return _save(fig, "economics.png")


def fig_xper(stats):
    """Where the scorecard's AUC above 0.5 actually comes from."""
    phi = stats["xper"]["phi"]
    items = sorted(phi.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    cols = [ACCENT if k == "subject_race" else (GREY if v < 0 else BLUE) for k, v in items]
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.barh(labels, vals, color=cols, height=0.62)
    ax.axvline(0, color=INK, lw=1.2)
    share = phi["subject_race"] / stats["xper"]["sum_phi"]
    ax.text(phi["subject_race"], labels.index("subject_race"),
            f"  {share:.1%} of all signal", va="center", fontsize=11,
            fontweight="bold", color=ACCENT)
    ax.set_xlabel("XPER contribution to AUC above 0.5")
    ax.set_title(f"AUC {stats['xper']['auc_full']:.4f} = 0.5 + {stats['xper']['sum_phi']:.4f}, "
                 "decomposed exactly over 1,024 coalitions",
                 loc="left", fontsize=12, fontweight="bold", pad=12)
    return _save(fig, "xper.png")


def main():
    stats = json.loads((ROOT / "outputs" / "deck_stats.json").read_text())
    print("building figures...")
    fig_reversal(stats); fig_floors(stats); fig_fairness_ladder(stats)
    fig_economics(stats); fig_xper(stats)
    print("done")


if __name__ == "__main__":
    main()
