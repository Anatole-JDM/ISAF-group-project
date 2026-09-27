"""Collect every number the deck quotes into one JSON, from the artifacts and the
search sample. The deck reads only this file, so no figure and no slide carries a
hand-typed number, and a retrain moves both together."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import data as D
from src import economics as E

OUT = ROOT / "outputs"
K = 2000


def _gap(d: pd.DataFrame) -> tuple[float, int, float, float]:
    r = d.groupby("subject_race")["contraband_found"].agg(["size", "mean"])
    w = r.loc["white", "mean"] * 100 if "white" in r.index else np.nan
    b = r.loc["black", "mean"] * 100 if "black" in r.index else np.nan
    return b - w, int(r["size"].sum()), w, b


def sample_stats() -> dict:
    """The scope slide and the reversal chart. Recomputed, never quoted from prose."""
    df = D.add_search_type(D.load_searches())
    for c in ("contraband_found", "arrest_made"):
        if c in df:
            df[c] = D._truthy(df[c]).astype(float)

    basis = df["search_basis"].astype("string").str.strip().str.lower()
    flagged = D._truthy(df["raw_search_consent"]) | (basis == "consent")
    kept = df[df["search_type"] == "consent"]
    removed = df[flagged & (df["search_type"] != "consent")]
    nc = df[df["search_type"] != "consent"]
    mech = nc[nc["search_type"].isin(["arrest", "warrant", "inventory"])]
    disc = nc[nc["search_type"].isin(["probable cause", "plain view"])]

    by = []
    for t, lab in [("consent", "consent  (our sample)"), ("probable cause", "probable cause"),
                   ("plain view", "plain view"), ("arrest", "arrest"),
                   ("inventory", "inventory"), ("warrant", "warrant")]:
        d = df[df["search_type"] == t]
        if not len(d):
            continue
        g, n, w, b = _gap(d)
        by.append({"label": lab, "gap": float(g), "n": n, "white": float(w), "black": float(b),
                   "hit": float(d["contraband_found"].mean() * 100)})

    return {
        "n_searches": int(len(df)),
        "n_flagged": int(flagged.sum()),
        "n_consent": int(len(kept)),
        "n_removed": int(len(removed)),
        "hit_kept": float(kept["contraband_found"].mean()),
        "hit_removed": float(removed["contraband_found"].mean()),
        "arrest_removed": float(removed["arrest_made"].mean()),
        "gap_pure": float(_gap(kept)[0]),
        "gap_all_flagged": float(_gap(df[flagged])[0]),
        "gap_nonconsent": float(_gap(nc)[0]),
        "gap_mechanical": float(_gap(mech)[0]), "n_mechanical": int(len(mech)),
        "gap_still_disc": float(_gap(disc)[0]), "n_still_disc": int(len(disc)),
        "share_disc_in_comparison": float(len(disc) / len(nc)),
        "gap_pooled": float(_gap(df)[0]),
        "by_basis": sorted(by, key=lambda r: r["gap"]),
    }


def main() -> None:
    s = {}
    s.update(sample_stats())

    m = json.loads((OUT / "metrics__consent__matched.json").read_text())
    s["matched_arms"] = [{k: a[k] for k in ("arm", "auc", "pr_auc", "brier", "n_train", "n_test")}
                         for a in m["arms"]]
    s["matched_n"] = m.get("matched_n")
    s["split_year"] = m.get("split_year")

    sc = pd.read_parquet(OUT / "scores__consent__matched.parquet")
    y = sc["y"].astype(int).to_numpy()
    base = float(y.mean())
    s["test_n"] = int(len(y))
    s["base_rate"] = base
    s["brier_floor"] = base * (1 - base)
    s["pr_floor"] = base

    s["breakeven"] = {"no_skill": base / (1 - base)}
    for arm in ("scorecard", "gbm", "tabpfn"):
        nb = E.net_benefit_at_k(y, sc[f"score_{arm}"].to_numpy(), K)
        s["breakeven"][arm] = nb["tp"] / nb["fp"]
        s.setdefault("topk", {})[arm] = {"tp": nb["tp"], "fp": nb["fp"],
                                         "precision": nb["tp"] / nb["k"]}
    s["K"] = K
    s["sensitivity"] = E.sensitivity(y, sc["score_tabpfn"].to_numpy(), K)[
        ["cost_fp", "tp", "fp", "net_benefit"]].to_dict("records")

    ct = json.loads((OUT / "critique_tests.json").read_text())
    s["critique_tests"] = ct
    s["ladder"] = [
        {"label": "base\n(no officer)", "key": "base"},
        {"label": "+ officer\n(id + behaviour)", "key": "officer"},
        {"label": "+ officer\nrace-blind", "key": "officer_blind"},
        {"label": "behaviour only\nno id, no race", "key": "officer_TRULYblind"},
    ]
    for row in s["ladder"]:
        d = ct[row["key"]]
        row.update({"auc": d["auc"], "fpr_gap": d["fpr_gap"], "sel_gap": d["sel_gap"],
                    "chi2": d["chi2"], "tost": d["tost"], "n_cols": d["n_cols"],
                    "auc_within": d.get("auc_within_officer")})

    s["xper"] = json.loads((OUT / "xper_scorecard.json").read_text())
    fp = json.loads((OUT / "fpdp_pdp.json").read_text())
    s["fpdp"] = {"n_features": len(fp["fpdp"]), "base_chi2": fp["base_chi2"],
                 "n_repairing": sum(1 for r in fp["fpdp"] if r.get("repairs")),
                 "best": sorted(fp["fpdp"], key=lambda r: r["chi2"])[:3]}

    (OUT / "deck_stats.json").write_text(json.dumps(s, indent=1))
    print(f"wrote outputs/deck_stats.json  ({len(json.dumps(s)):,} chars)")
    print(f"  consent {s['n_consent']:,} of {s['n_flagged']:,} flagged | "
          f"gap {s['gap_pure']:+.2f} pure / {s['gap_all_flagged']:+.2f} all")
    print(f"  mechanical {s['gap_mechanical']:+.2f}pp | probable cause "
          f"{[r['gap'] for r in s['by_basis'] if r['label']=='probable cause'][0]:+.2f}pp")
    print(f"  brier floor {s['brier_floor']:.4f} | arms "
          f"{[round(a['brier'],4) for a in s['matched_arms']]}")


if __name__ == "__main__":
    main()
