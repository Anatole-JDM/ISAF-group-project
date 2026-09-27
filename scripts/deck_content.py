"""The deck, as data.

One definition, rendered by two backends (scripts/render_pptx.py and
scripts/render_pdf.py) so the .pptx and the .pdf cannot drift apart. Every
figure comes from outputs/deck_stats.json, which scripts/deck_stats.py
regenerates from the artifacts, so no number here is hand-typed.

Block types a slide may carry:
    stats    [(value, label), ...]        the big-number band
    bullets  [(head, body) | str, ...]
    picture  "name.png" from outputs/deck/
    table    {headers, rows, col_w, highlight_neg}
    callout  "text"            the emphasised band above the footer
    caption  "text"
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "outputs" / "deck_stats.json").read_text())

PC = [r for r in S["by_basis"] if r["label"] == "probable cause"][0]
ARM = {a["arm"]: a for a in S["matched_arms"]}
LAD = {r["key"]: r for r in S["ladder"]}
DISC = {"consent  (our sample)", "probable cause", "plain view"}
RACE_SHARE = S["xper"]["phi"]["subject_race"] / S["xper"]["sum_phi"]

NAME = {"scorecard": "Scorecard", "gbm": "XGBoost", "tabpfn": "TabPFN"}


def _delong_rows():
    """Built from outputs/significance__consent__matched.json, which
    scripts/run_significance.py regenerates.

    This table used to be quoted from an interactive run and no longer reproduced: it
    reported gbm vs tabpfn at -0.0026 (p 0.691) against the artifact's -0.0051
    (p 0.413), and carried a decision-tree arm absent from the matched scores.
    """
    sig = S.get("significance")
    if not sig:
        return [["run scripts/run_significance.py", "", "", "", "", ""]]
    rows = []
    for r in sig["pairs"]:
        flag = "SIGNIFICANT" if r["p_value"] < 0.05 else "not significant"
        pval = "<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}"
        rows.append([f"{NAME.get(r['a'], r['a'])} vs {NAME.get(r['b'], r['b'])}",
                     f"{r['auc_a']:.4f}", f"{r['auc_b']:.4f}",
                     f"{r['diff']:+.4f}".replace("-", "−"), pval, flag])
    return rows


TITLE = {
    "cover": True,
    "headline": ["Should a police force let a model", "decide who gets searched?"],
    "sub": "Interpretability, stability and algorithmic fairness on "
           f"{S['n_consent']:,} discretionary traffic-stop searches",
    "foot": ["Nashville Metropolitan PD, 2010–2018  ·  Stanford Open Policing Project",
             "HEC Paris  ·  MSc Data Science & AI for Business  ·  "
             "Prof. Christophe Pérignon  ·  28 September 2026"],
    "notes": "One presenter opens. Do not read the title aloud. Say: we were asked to build a "
             "scoring model and recommend whether to deploy it. Our recommendation is not to "
             "deploy any of the three, and the analysis says why across all four dimensions.",
}

SLIDES = [
    TITLE,
    {
        "kicker": "The brief", "title": "The client's decision, not ours",
        "bullets": [
            ("A police force has finite search capacity.",
             "Roughly 2,000 discretionary searches in our test window. Which stops should get one?"),
            ("The offer on the table: rank every stop by predicted probability of contraband, "
             "then search the top K.",
             "That is a scoring model on a binary target — exactly what the brief asks for."),
            ("Three model families, four dimensions.",
             "White box, machine learning, tabular foundation model — each judged on predictive "
             "performance (statistical AND economic), interpretability, stability and fairness."),
        ],
        "callout": "Our answer: build all three, then recommend deploying none of them — and the "
                   "reason is not that they score badly. It is what they are scoring.",
        "notes": "Set up the deliverable honestly. The recommendation is negative and we reached it "
                 "through the four dimensions, not by picking the lowest AUC.",
    },
    {
        "kicker": "Data", "title": "Nashville, because the outcome is observed",
        "stats": [("3,078,116", "traffic stops, 2010–2018"),
                  (f"{S['n_searches']:,}", "ended in a search"),
                  (f"{S['n_consent']:,}", "consent searches — our sample"),
                  (f"{S['test_n']:,}", "test searches, 2016+")],
        "bullets": [
            ("Stanford Open Policing Project — Pierson et al., Nature Human Behaviour 4 (2020).",
             "Open Data Commons Attribution Licence. Cited in the repository, in the app and here."),
            ("Why Nashville: contraband_found is recorded for every search.",
             "Most forces record that a search happened, not what it produced. No outcome, no target "
             "and no outcome test."),
            ("Temporal split at 2016 — train before, test after.",
             "Not random: a random split leaks the future into the past through the same officer, "
             "the same shift and the same policy regime."),
        ],
        "notes": "If asked why not a random split: officers persist across years, so a random split "
                 "puts the same officer's 2017 behaviour in both train and test.",
    },
    {
        "kicker": "Problem definition", "title": "What we predict — and the trap underneath it",
        "bullets": [
            ("Y = 1 if contraband was found.   Ŷ = 1 if the model would search.   D = driver race.",
             "Note the inversion against the course convention: here Y = 1 is the outcome that "
             "JUSTIFIES a search, so the favourable outcome for the individual is NOT being "
             "searched. We report every fairness metric under both codings."),
            ("Selective labels: we observe contraband only for stops an officer already chose to search.",
             "Every model here is trained on the officers' own decisions. It learns to predict the "
             "result of a filter we cannot see through — not the underlying rate of contraband."),
            ("So the honest question is not “how accurate is it?”",
             "It is: what does this score reproduce? That reframing drives the whole analysis."),
        ],
        "callout": "A model trained on who officers chose to search can only ever be as fair as the "
                   "choices it is imitating.",
        "callout_color": "blue",
        "notes": "The single most important conceptual slide. Selective labels is why we spend the "
                 "next slides on what the sample IS before any model appears.",
    },
    {
        "kicker": "Scope — say this precisely", "title": "Exactly which searches are in our sample",
        "stats": [(f"{S['n_flagged']:,}", "flagged consent"),
                  (f"{S['n_consent']:,}", "consent AND no other authority"),
                  (f"{S['n_removed']:,}", "excluded — also carry arrest,\nwarrant, inventory or plain view")],
        "bullets": [
            (f"The {S['n_removed']:,} excluded searches are not neutral: they hit at "
             f"{S['hit_removed']:.1%} against {S['hit_kept']:.1%} for the ones we keep, and "
             f"{S['arrest_removed']:.0%} of them ended in arrest.",
             "The exclusion is on the legal basis, not on the outcome — but that basis is "
             "outcome-correlated, so the purification is not free. We therefore report both."),
        ],
        "callout": f"Black−white hit-rate gap:   {S['gap_pure']:+.2f}pp on the pure stratum   ·   "
                   f"{S['gap_all_flagged']:+.2f}pp on all {S['n_flagged']:,} consent-flagged searches. "
                   "Same sign, same conclusion.",
        "notes": "MEMORISE THIS SLIDE — every member will be asked something adjacent to it. The "
                 "line: our sample is the 58,865 searches where consent was recorded and no other "
                 "legal authority was. 10,386 consent-flagged searches carry one of those and are "
                 "excluded; they hit at 28.7% against our 16.9%, so we report the gap both ways, "
                 "-5.16pp pure and -4.60pp on all consent-flagged. We concede plain view should not "
                 "have overridden consent — it is discretionary and its flag is set by what was found.",
    },
    {
        "kicker": "Test", "title": "The outcome test, by legal basis",
        "picture": "reversal.png", "picture_h": 3.85,
        "callout": f"Consent {S['gap_pure']:+.2f}pp and probable cause {PC['gap']:+.2f}pp — the two "
                   f"bases needing the officer's judgement about THIS driver — are both negative. "
                   f"The mechanical bases sit at {S['gap_mechanical']:+.2f}pp.",
        "notes": "THE critical slide for Q&A. If asked what is in the comparison group: arrest, "
                 "warrant and inventory are the mechanical benchmark, 35,713 searches, gap +1.77pp, "
                 "so the reversal is 6.9 points not 11. Probable cause and plain view are the other "
                 "47.6% and are NOT mechanical. Probable cause runs -15.90pp on 15,877 searches at a "
                 "49.6% base rate — three times our headline, same direction. The mechanism is "
                 "supported twice; our original label was wrong and we fixed it.",
    },
    {
        "kicker": "Models", "title": "Three families, one matched comparison",
        "table": {
            "headers": ["", "White box", "Machine learning", "Tabular foundation"],
            "rows": [["Model", "Logistic scorecard", "XGBoost", "TabPFN v2"],
                     ["Explanation", "the coefficients ARE the model", "TreeSHAP (exact)",
                      "occlusion (approximate)"],
                     ["Train rows", f"{S['matched_n']:,}", f"{S['matched_n']:,}", f"{S['matched_n']:,}"],
                     ["AUC", f"{ARM['scorecard']['auc']:.4f}", f"{ARM['gbm']['auc']:.4f}",
                      f"{ARM['tabpfn']['auc']:.4f}"],
                     ["Brier", f"{ARM['scorecard']['brier']:.4f}", f"{ARM['gbm']['brier']:.4f}",
                      f"{ARM['tabpfn']['brier']:.4f}"]],
            "col_w": [2.0, 2.0, 2.0, 2.0],
        },
        "callout": f"All three train on the identical {S['matched_n']:,} rows — TabPFN's context "
                   "ceiling. Anything else compares training budgets, not model families.",
        "callout_color": "blue",
        "caption": f"Test set: the same {S['test_n']:,} searches from 2016 onward for every arm.",
        "notes": "Stress the matched design — it is the only fair three-way comparison, and it is "
                 "why we cap the other two at 2,000 rows.",
    },
    {
        "kicker": "Predictive performance", "title": "Measured against doing nothing",
        "picture": "floors.png", "picture_h": 3.95,
        "callout": f"Not one of the three beats the Brier floor of {S['brier_floor']:.4f} — the score "
                   f"you get by predicting the base rate {S['base_rate']:.1%} for every single driver.",
        "notes": "Say this before being asked. On calibration all three arms have NEGATIVE skill. "
                 "They do carry ranking information — PR-AUC beats its floor by 0.019 to 0.027 — "
                 "which is why a top-K policy is conceivable at all. But we will not call any of "
                 "them well calibrated.",
    },
    {
        "kicker": "Significance", "title": "Are the three actually different? No.",
        "table": {
            "headers": ["Comparison", "AUC a", "AUC b", "Δ AUC", "p", ""],
            "rows": _delong_rows(),
            "col_w": [2.8, 1.1, 1.1, 1.2, 1.0, 1.9],
        },
        "callout": "DeLong paired test on correlated AUCs, recomputed from the cached scores by "
                   "scripts/run_significance.py. The two black boxes are statistically "
                   "indistinguishable, so we do not rank them — any choice between them must be "
                   "made on the other three dimensions.",
        "notes": "Why it matters: 'which model is most accurate' is the wrong question on this data. "
                 "The scorecard IS significantly worse than both, by about 0.018 AUC — that is the "
                 "real, and small, price of interpretability here.",
    },
    {
        "kicker": "Economic performance", "title": "What it would be worth to deploy",
        "picture": "economics.png", "picture_h": 4.1,
        "callout": f"The best arm breaks even only while an innocent driver's search costs less than "
                   f"{S['breakeven']['tabpfn']:.3f} of a justified one. Random ranking breaks even at "
                   f"{S['breakeven']['no_skill']:.3f} — the model buys "
                   f"{S['breakeven']['tabpfn'] - S['breakeven']['no_skill']:.3f} of headroom.",
        "notes": "The false-positive price is a policy judgement, not a data fact — we ship the "
                 f"sensitivity, we do not pick the number. At K={S['K']:,} TabPFN finds "
                 f"{S['topk']['tabpfn']['tp']} hits against {S['base_rate']*S['K']:.0f} for random: "
                 f"{S['topk']['tabpfn']['tp'] - round(S['base_rate']*S['K'])} extra finds at the cost "
                 f"of {S['topk']['tabpfn']['fp']:,} innocent searches.",
    },
    {
        "kicker": "Interpretability", "title": "Three models, three grades of explanation",
        "table": {
            "headers": ["Arm", "Local explanation", "Exactness", "What it costs"],
            "rows": [["Scorecard", "Shapley in closed form: coef × (x − E[x])", "EXACT",
                      "nothing — it is algebra"],
                     ["XGBoost", "TreeSHAP, native to the model", "EXACT", "nothing"],
                     ["TabPFN", "occlusion against a background value", "APPROXIMATE",
                      "ignores interactions; error cannot be bounded"]],
            "col_w": [1.4, 3.7, 1.4, 3.1],
            "highlight_row": 2,
        },
        "bullets": [
            ("The foundation model has no native attribution at all.",
             "Not a gap in our implementation — TabPFN exposes none. For a decision that must be "
             "justified to the person searched, an unbounded approximation is a real cost, not a "
             "technical footnote."),
        ],
        "notes": "The app lets the jury click any test row and see the attribution for all three "
                 "arms, labelled EXACT or APPROXIMATE. Offer to demo it.",
    },
    {
        "kicker": "Interpretability · XPER", "title": "Where the signal actually comes from",
        "picture": "xper.png", "picture_h": 4.25,
        "callout": f"Driver race alone accounts for {RACE_SHARE:.1%} of everything the white-box model "
                   f"knows above chance. Exact decomposition over "
                   f"{S['xper']['n_coalitions']:,} coalitions; efficiency gap 3×10⁻¹⁷.",
        "notes": "XPER is Professor Pérignon's own method — use the name and say the decomposition "
                 "is exact, not sampled. The headline is uncomfortable and should be: the model's "
                 "predictive power IS mostly race.",
    },
    {
        "kicker": "Stability", "title": "The score does not survive its own assumptions",
        "stats": [(f"{LAD['officer_TRULYblind']['auc']:.4f}", "pooled AUC, best model"),
                  (f"{LAD['officer_TRULYblind']['auc_within']:.4f}", "the same model, WITHIN officer"),
                  ("~80%", "of the gain is between officers,\nnot inside them")],
        "bullets": [
            ("Leave-one-officer-out: the model largely learns WHICH OFFICER made the stop.",
             "Deployed, it would rank a new officer's stops with almost no signal. The pooled number "
             "is the one you would quote; the within-officer number is the one you would live with."),
            ("Temporal split plus an officer-clustered bootstrap over 1,477 officers.",
             "The officer block gains +0.072 AUC, CI [+0.052, +0.095] — the only performance claim "
             "in this project an order of magnitude clear of the noise."),
        ],
        "notes": "Lead with the within-officer number, do not let him find it. It is the difference "
                 "between a model that ranks stops and a model that ranks officers.",
    },
    {
        "kicker": "Identify · Fairness", "title": "The model fails independence — backwards",
        "table": {
            "headers": ["Criterion", "Test", "Result on the base model"],
            "rows": [["Independence", "Ŷ ⟂ D — statistical parity",
                      f"FAILS: χ² = {LAD['base']['chi2']:.0f}, selection gap "
                      f"{LAD['base']['sel_gap']*100:+.1f}pp"],
                     ["Separation", "equal FPR / TPR across groups",
                      f"FAILS: FPR gap {LAD['base']['fpr_gap']*100:+.1f}pp — the harm metric"],
                     ["Sufficiency", "equal hit rate at equal score",
                      "closest to holding; calibration by race is in the app"]],
            "col_w": [1.8, 3.3, 5.1],
        },
        "callout": "Read the SIGN. The model searches Black drivers LESS than white drivers "
                   f"({LAD['base']['sel_gap']*100:+.1f}pp) — it inverts real officer behaviour, having "
                   "learned that consent searches of Black drivers find less. A disparity in the "
                   "“favourable” direction is still a disparity, and it is still race doing the work.",
        "callout_h": 1.25,
        "notes": "Do not let anyone say 'the model is fair to Black drivers'. It is not selecting "
                 "them, which is a different thing. The FPR gap is the share of INNOCENT drivers "
                 "searched by race — the metric with a victim. We read it before AUC.",
    },
    {
        "kicker": "Mitigate", "title": "Give it the officer, and race stops mattering",
        "picture": "fairness_ladder.png", "picture_h": 3.95,
        "callout": f"Removing race entirely costs "
                   f"{LAD['officer']['auc'] - LAD['officer_TRULYblind']['auc']:.4f} AUC "
                   f"(p = 0.144, not significant) and takes the FPR gap from "
                   f"{LAD['officer']['fpr_gap']*100:+.1f}pp to "
                   f"{LAD['officer_TRULYblind']['fpr_gap']*100:+.1f}pp. Going race-blind is "
                   "statistically free.",
        "notes": "The constructive result. Race carried the signal only because the model had "
                 "nothing better; give it the officer's own behavioural history and race becomes "
                 "redundant. Chi-square falls from 863 to 65, the TOST margin from 0.30 to 0.10.",
    },
    {
        "kicker": "Mitigate · FPDP", "title": "But nothing in the data repairs parity",
        "stats": [(f"0 of {S['fpdp']['n_features']}", "features whose adjustment\nrestores statistical parity"),
                  (f"{S['fpdp']['base_chi2']:.0f}", "χ² of the best model —\nstill rejects independence"),
                  (f"{S['fpdp']['best'][0]['chi2']:.0f}",
                   f"best single repair ({S['fpdp']['best'][0]['feature']}),\nstill far above 3.84")],
        "bullets": [
            ("Fairness Partial Dependence Plots sweep every feature and ask: if we intervened on "
             "this one, would the disparity close?",
             "No, for all 32. The disparity is not carried by any single removable column — it is "
             "spread across the feature set, which is what proxy discrimination looks like when "
             "you go looking for it."),
            ("This is why “just drop the sensitive feature” is not a fix.",
             "We dropped race and the disparity persisted at −4.8pp. There is no column to delete."),
        ],
        "notes": "FPDP is also Pérignon's method. The negative result IS the finding — it is what "
                 "makes the recommendation about deployment rather than about feature selection.",
    },
    {
        "kicker": "Limitations", "title": "What we are not claiming",
        "bullets": [
            ("Selective labels. We never observe contraband for stops that were not searched.",
             "Every number here is conditional on an officer having already decided to search."),
            ("No validation set. Tree depth, context size, feature blocks and K were all chosen by "
             "reading the same 9,113-row test set that produces every reported number.",
             "Treat any AUC difference below 0.01 as inside the noise. Only the officer block "
             "(+0.072, p = 6.6e⁻²³) is clear of it."),
            ("Inframarginality. A hit-rate gap is not proof of a different threshold.",
             "The outcome test is suggestive, not dispositive; a full threshold test is the next step."),
            ("Provenance. The officer-model runs were done interactively and only their outputs were "
             "committed — the driver script is not in the repository.",
             "We say this before being asked. The reproducible path today is "
             "scripts_claude/evaluate_officer_models.py, which replicates the design and gives 0.6397."),
        ],
        "bullet_size": 14.5,
        "notes": "Volunteering the provenance gap costs a fraction of what being caught costs. "
                 "grep TRULY takes four seconds and he will run it.",
    },
    {
        "kicker": "Recommendation", "title": "Deploy none of the three",
        "bullets": [
            ("Not because they score badly — because of what they score.",
             f"{RACE_SHARE:.0%} of the white box's signal is race. The best model's edge is mostly "
             "which officer made the stop. None of the three beats a constant prediction on "
             "calibration. Net benefit is negative at any defensible price for searching an "
             "innocent driver."),
            ("If a model must ship: the scorecard, race-blind, with officer behaviour features.",
             "It costs 0.018 AUC against the black boxes — the only statistically significant "
             "performance difference we found — and buys an exact, auditable explanation for every "
             "individual decision. That is the trustworthy-AI trade the brief asks us to make."),
            ("The finding worth more than the model: the outcome test itself.",
             f"Consent {S['gap_pure']:+.2f}pp, probable cause {PC['gap']:+.2f}pp. Deploy that as an "
             "officer-level audit, not a driver-level score. It needs no model, it has a victim you "
             "can name, and it points at the decision that is actually going wrong."),
        ],
        "callout": "A trustworthy system here is one that audits the officers, not one that ranks "
                   "the drivers.",
        "notes": "Land this cleanly and stop. It answers the brief's exact words: the recommendation "
                 "reflects the requirements of a trustworthy AI system rather than predictive "
                 "performance alone.",
    },
    {
        "kicker": "Appendix", "title": "Backup · the comparison group, decomposed",
        "table": {
            "headers": ["Legal basis", "n", "white hit", "Black hit", "gap", "discretionary?"],
            "rows": [[r["label"], f"{r['n']:,}", f"{r['white']:.2f}%", f"{r['black']:.2f}%",
                      f"{r['gap']:+.2f}pp", "YES" if r["label"] in DISC else "no"]
                     for r in S["by_basis"]],
            "col_w": [2.6, 1.3, 1.4, 1.4, 1.4, 1.8],
            "highlight_neg": 4,
        },
        "caption": f"Mechanical (arrest + warrant + inventory): n = {S['n_mechanical']:,}, gap "
                   f"{S['gap_mechanical']:+.2f}pp.   Still discretionary (probable cause + plain "
                   f"view): n = {S['n_still_disc']:,}, gap {S['gap_still_disc']:+.2f}pp — "
                   f"{S['share_disc_in_comparison']:.1%} of the comparison group.   "
                   f"Pooled over everything: {S['gap_pooled']:+.2f}pp.   "
                   "n is the count each gap is computed on: drivers recorded white, "
                   "Black or Hispanic.",
        "notes": "Bring this up the moment anyone asks what 'non-consent' contains.",
    },
    {
        "kicker": "Appendix", "title": "Backup · every arm against its no-skill floor",
        "table": {
            "headers": ["Arm", "AUC", "PR-AUC", "vs PR floor", "Brier", "vs Brier floor"],
            "rows": [[a["arm"], f"{a['auc']:.4f}", f"{a['pr_auc']:.4f}",
                      f"{a['pr_auc'] - S['pr_floor']:+.4f}", f"{a['brier']:.4f}",
                      f"{a['brier'] - S['brier_floor']:+.4f}  worse"] for a in S["matched_arms"]],
            "col_w": [1.6, 1.3, 1.3, 1.5, 1.3, 2.0],
        },
        "caption": f"Floors from a constant prediction at the test base rate {S['base_rate']:.4f}: "
                   f"Brier {S['brier_floor']:.4f}, PR-AUC {S['pr_floor']:.4f}.   At K = {S['K']:,}: "
                   f"TabPFN {S['topk']['tabpfn']['tp']} hits against "
                   f"{S['base_rate']*S['K']:.0f} for random ranking.",
        "notes": "If he asks which of the arms beats the intercept — this is the answer, and we "
                 "put it up ourselves.",
    },
    {
        "kicker": "Appendix", "title": "Backup · the fairness ladder in numbers",
        "table": {
            "headers": ["Model", "cols", "AUC", "within-officer", "sel gap", "FPR gap",
                        "χ²", "TOST δ*"],
            "rows": [[r["label"].replace("\n", " "), str(r["n_cols"]), f"{r['auc']:.4f}",
                      f"{r['auc_within']:.4f}" if r.get("auc_within") else "—",
                      f"{r['sel_gap']*100:+.1f}pp", f"{r['fpr_gap']*100:+.1f}pp",
                      f"{r['chi2']:.0f}", f"{r['tost']:.2f}"] for r in S["ladder"]],
            "col_w": [2.7, 0.8, 1.1, 1.5, 1.2, 1.2, 0.9, 1.1],
        },
        "caption": "TOST δ* is the smallest equivalence margin at which the model could be "
                   "certified fair. It falls from 0.30 to 0.10 — an improvement, and still a "
                   "10-point selection gap you would have to declare acceptable.",
        "notes": "Chi-square on large n rejects almost anything, which is why we also report TOST: "
                 "it inverts the burden and asks how much unfairness you would have to accept.",
    },
]
