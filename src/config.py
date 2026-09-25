"""Central configuration: paths, column taxonomy, and the leakage exclusions.

Every modelling decision that could be challenged in the Q&A lives here, with a
written reason. If a jury member asks "why did you drop that column?", the answer
should be readable in this file.
"""
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

STOPS_URL = (
    "https://stacks.stanford.edu/file/druid:yg821jf8611/"
    "yg821jf8611_tn_nashville_2020_04_01.csv.zip"
)
STOPS_ZIP = DATA_RAW / "tn_nashville.csv.zip"
STOPS_CSV_INNER = "tn_nashville_2020_04_01.csv"
LABELLED_PARQUET = DATA_PROC / "searches.parquet"

# The team's canonical consent-search file, built by scripts_claude/. Carries 41
# ACS-derived neighbourhood features at 800m and 1200m radii. Verified clean:
# build_nbh_features.py never reads contraband_found (it loads only stop_id,
# date, lat, lng, geo_outside_davidson_box, location plus census data), so these
# cannot leak the outcome. Joins to our frame on raw_row_number with all 58,865
# consent rows matching; their extra 74 are the 2019 tail we drop.
NBH_PARQUET = ROOT / "data" / "nashville_consent_searches.parquet"
NBH_JOIN_KEY = "raw_row_number"
NBH_PREFIX = "nbh"

# --------------------------------------------------------------------------- target
TARGET = "contraband_found"

# The sample is *searches*, not stops. contraband_found is observed only when a
# search happened (127,705 of 3,092,351 stops = 4.13%). This is the selective
# labels problem and it is the spine of the project, not a footnote.
SAMPLE_DEFINITION = "search_conducted == True"

# 2019 ends on 24 March. A partial year distorts any temporal split.
DATE_MIN, DATE_MAX = "2010-01-01", "2018-12-31"

# --------------------------------------------------------------------------- protected
PROTECTED = ["subject_race", "subject_sex", "subject_age"]
PRIMARY_PROTECTED = "subject_race"

# Groups with enough consent-search volume to report. Verified 2026-09-24 on the
# RESOLVED search_type (mechanical bases override consent), 2010-2018:
#   black 32,311 | white 21,227 | hispanic 4,645 | asian/PI 282 | unknown 238 | other 133
# Asian/PI is too thin for stable subgroup metrics — report it with a CI or fold
# it into "other", but do not present a bare point estimate.
RACE_REPORTABLE = ["white", "black", "hispanic"]
RACE_TOO_THIN = ["asian/pacific islander", "other", "unknown"]

# --------------------------------------------------------------------------- LEAKAGE
# Columns that encode the outcome or follow from it. Including any of these
# produces a meaningless AUC near 1.0. This is the single most likely way for
# this project to fail silently.
LEAKAGE = {
    "contraband_drugs":             "component of the target",
    "contraband_weapons":           "component of the target",
    "arrest_made":                  "consequence of finding contraband",
    "citation_issued":              "post-stop outcome",
    "warning_issued":               "post-stop outcome",
    "outcome":                      "post-stop outcome; encodes arrest/citation/warning",
    "raw_verbal_warning_issued":    "raw duplicate of outcome",
    "raw_written_warning_issued":   "raw duplicate of outcome",
    "raw_traffic_citation_issued":  "raw duplicate of outcome",
    "raw_misd_state_citation_issued": "raw duplicate of outcome",
    "notes":                        "free text written after the stop; can name the contraband",
}

# Not leakage, but NOT features either: these define the strata we analyse
# separately. Using them as predictors would fold the pooling problem back in.
STRATIFIERS = [
    "search_basis",
    "raw_search_consent", "raw_search_arrest", "raw_search_warrant",
    "raw_search_inventory", "raw_search_plain_view",
]

# Constant on the modelling sample (we filter to searches), so uninformative.
CONSTANT_ON_SAMPLE = ["search_conducted"]

# Identifiers / grouping keys. officer_id_hash is a *grouping* variable for
# clustered errors and leave-one-officer-out — treating it as a feature would
# let the model memorise officers, which is the thing we want to detect.
IDENTIFIERS = ["raw_row_number", "officer_id_hash"]

# --------------------------------------------------------------------------- judgement calls
# These are decided at the same moment as the search, not after it, so they are
# not strictly leakage — but they describe the search rather than the driver.
# THE GROUP MUST MAKE A WRITTEN CALL ON EACH AND JUSTIFY IT IN THE REPORT.
# Default here is to exclude; flip to include and document the effect.
QUESTIONABLE = {
    "frisk_performed":       "co-decided with the search; arguably officer suspicion, not driver risk",
    "search_person":         "describes the search, not the pre-search evidence",
    "search_vehicle":        "describes the search, not the pre-search evidence",
    "raw_driver_searched":   "raw duplicate of search_person",
    "raw_passenger_searched":"raw duplicate of search_person",
    "raw_suspect_ethnicity": "raw duplicate of subject_race — reintroduces D directly",
}

# Dropped after inspecting cardinality on the real file (2026-09-24).
DROP_EXTRA = {
    "type":           "constant on this data — every row is a vehicular stop",
    "reporting_area": "geographic ID with ~1,700 levels; zone and precinct already "
                      "carry geography at a usable granularity, and one-hot encoding "
                      "this would swamp TabPFN's feature budget",
    # FIX: verified identical to `violation` on all 127,122 rows (zero mismatches,
    # identical value_counts). Keeping both splits every stop-reason coefficient
    # across two identical one-hot blocks under L2, halving each and making the
    # scorecard's "main drivers" story wrong -- which is the interpretability
    # dimension we are graded on.
    "reason_for_stop": "byte-identical duplicate of `violation`; keeping both halves "
                       "every stop-reason coefficient under L2 regularisation",
    # FIX: `year` is the train/test split variable. Train is year<2016, so every
    # test row has a `year` value with ZERO training support. Trees cannot split
    # on it usefully and the linear arm extrapolates off the end of its range.
    # month/dow/hour carry the within-period seasonality without this problem.
    "year":            "this is the split variable — test years have no training "
                       "support, so it cannot be a feature as well",
}

EXCLUDE_DEFAULT = (
    list(LEAKAGE) + STRATIFIERS + CONSTANT_ON_SAMPLE
    + IDENTIFIERS + list(QUESTIONABLE) + list(DROP_EXTRA)
)

# --------------------------------------------------------------------------- features
CANDIDATE_FEATURES = [
    "subject_age", "subject_sex", "subject_race",
    "precinct", "zone", "reporting_area",
    "type", "violation", "reason_for_stop",
    "vehicle_registration_state",
    "hour", "dow", "month", "year",        # engineered in features.py
]

# --------------------------------------------------------------------------- search types
# search_basis has 25,620 "other" (20% of searches). Resolve against the raw
# flags before stratifying — an unresolved fifth of the sample is a pooling
# problem waiting to happen.
SEARCH_TYPE_DISCRETIONARY = "consent"
SEARCH_TYPE_ORDER = ["consent", "probable cause", "plain view", "arrest", "warrant", "inventory", "unresolved"]

# --------------------------------------------------------------------------- Y convention
# Course convention (slide 238): Y = 1 is the FAVORABLE outcome for the individual.
# Here the natural coding is Y = contraband_found, where Y = 1 is favorable for
# the officer and UNFAVORABLE for the driver. Every fairness metric therefore
# reads backwards unless the inversion is stated.
#
# fairness.py computes metrics in both codings so the report can show that the
# same model is "fair" under one and unfair under the other.
FAVORABLE_FOR_INDIVIDUAL = "not_searched"
Y_NATIVE = "contraband_found"      # 1 = contraband found
Y_COURSE = "no_contraband"         # 1 = innocent (favorable to the driver)
