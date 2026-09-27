"""Produce the DeLong significance artifact for FINDINGS.md section 24 and the deck.

Section 24 previously had no producing code - the numbers were computed in an
interactive session and only quoted. This script closes that gap: it recomputes
the paired tests from the cached scores and writes outputs/significance_matched.json.

    python3 scripts/run_significance.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.metrics import roc_auc_score

from src import significance as SG

OUT = ROOT / "outputs"


def main(stratum: str = "consent", mode: str = "matched") -> dict:
    sc = pd.read_parquet(OUT / f"scores__{stratum}__{mode}.parquet")
    y = sc["y"].astype(int).to_numpy()
    arms = [c.replace("score_", "") for c in sc.columns if c.startswith("score_")]
    scores = {a: sc[f"score_{a}"].to_numpy() for a in arms}

    res = {
        "stratum": stratum, "mode": mode, "n_test": int(len(y)),
        "base_rate": float(y.mean()),
        "auc": {a: float(roc_auc_score(y, scores[a])) for a in arms},
        "pairs": SG.delong_test(y, scores),
    }
    (OUT / f"significance__{stratum}__{mode}.json").write_text(json.dumps(res, indent=1))

    print(f"{stratum}/{mode}  n = {len(y):,}  arms: {', '.join(arms)}")
    print(f"\n{'pair':<26} {'AUC a':>7} {'AUC b':>7} {'diff':>9} {'p':>8}  verdict")
    for r in res["pairs"]:
        verdict = "significant" if r["p_value"] < 0.05 else "not significant"
        print(f"{r['a'] + ' vs ' + r['b']:<26} {r['auc_a']:>7.4f} {r['auc_b']:>7.4f} "
              f"{r['diff']:>+9.4f} {r['p_value']:>8.3f}  {verdict}")
    print(f"\nwrote outputs/significance__{stratum}__{mode}.json")
    return res


if __name__ == "__main__":
    main(*(sys.argv[1:3] or ["consent", "matched"]))
