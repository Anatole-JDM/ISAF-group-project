"""Fetch ACS 5-year tract data for the neighbourhood features (feature plan, Step 1).

Pure fetcher: downloads raw estimates + margins of error, resolves variable codes
per vintage, and saves them. No feature is computed here - weighting, inflation
and percentiles happen in the build step, so the raw pull stays auditable.

Temporal rule B (team decision D1): a stop in year Y uses the 5-year window
ending in Y-1; 2010 stops use 2006-2010 (flagged downstream). Vintages needed,
named by the window's end year as the Census API does: 2010 ... 2018.

Geography: all tracts of Davidson County plus its six neighbours, so an 800 m
circle crossing the county line still finds its tracts. All these vintages use
2010 tract boundaries, matching tl_2010_47_tract10.

Usage
  python scripts_claude/fetch_acs.py --check   # metadata only: resolve every code, fetch no data
  python scripts_claude/fetch_acs.py           # full fetch
Requires CENSUS_API_KEY for the data requests (--check works without it):
free key at https://api.census.gov/data/key_signup.html. Never commit the key.

Outputs (git-ignored, under opp_data/external/acs/)
  variables_{v}.json            API variable dictionary per vintage (provenance)
  acs5_tract_measures.parquet   one row per (vintage, tract GEOID): each measure + its MOE
  fetch_manifest.json           per vintage: which codes resolved each measure, row counts, sanity checks
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'opp_data' / 'external' / 'acs'
VINTAGES = list(range(2010, 2019))            # window end years 2010..2018
STATE = '47'                                  # Tennessee
COUNTIES = {'037': 'Davidson', '021': 'Cheatham', '147': 'Robertson', '165': 'Sumner',
            '189': 'Wilson', '149': 'Rutherford', '187': 'Williamson'}
API = 'https://api.census.gov/data/{v}/acs/acs5'
MAX_VARS = 45                                 # API limit is 50 incl. NAME
# ACS annotation values that mean "not available" - never real data
SENTINELS = {-999999999, -888888888, -666666666, -555555555, -333333333, -222222222}


def norm(label):
    """Labels change format across vintages ('Total' / 'Estimate!!Total' / 'Total:')."""
    s = label.lower().replace('estimate!!', '').replace(':', '')
    return re.sub(r'\s+', ' ', s).strip()


# Each measure lists alternatives, tried in order. An alternative is either
#   {'codes': {code: [keywords that must appear in its label]}}  -> summed
#   {'group': table, 'match': regex on the normalised label}     -> all matches summed
MEASURES = {
    'pop_total':       [{'codes': {'B01003_001': ['total']}}],
    'race_total':      [{'codes': {'B03002_001': ['total']}}],
    'nh_white':        [{'codes': {'B03002_003': ['not hispanic', 'white alone']}}],
    'nh_black':        [{'codes': {'B03002_004': ['not hispanic', 'black']}}],
    'nh_asian':        [{'codes': {'B03002_006': ['not hispanic', 'asian alone']}}],
    'hispanic':        [{'codes': {'B03002_012': ['hispanic or latino']}}],
    'med_hh_income':   [{'codes': {'B19013_001': ['median household income']}}],
    'pov_universe':    [{'codes': {'B17001_001': ['total']}}],
    'pov_below':       [{'codes': {'B17001_002': ['below poverty level']}}],
    'veh_universe':    [{'codes': {'B25044_001': ['total']}}],
    'veh_none':        [{'codes': {'B25044_003': ['owner occupied', 'no vehicle'],
                                   'B25044_010': ['renter occupied', 'no vehicle']}}],
    'tenure_total':    [{'codes': {'B25003_001': ['total']}}],
    'renter':          [{'codes': {'B25003_003': ['renter occupied']}}],
    # B23001 fallback: ages 16-64 have an 'in labor force!!civilian' level, ages 65+ do not
    # ('65 to 69 years!!in labor force!!unemployed'). Both must be matched, or the fallback
    # covers 16-64 only while B23025 covers 16+.
    'lf_civilian':     [{'codes': {'B23025_003': ['civilian labor force']}},
                        {'group': 'B23001', 'match': r'in labor force!!civilian$'
                                                     r'|(65 to 69 years|70 to 74 years|75 years and over)!!in labor force$'}],
    'unemployed':      [{'codes': {'B23025_005': ['unemployed']}},
                        {'group': 'B23001', 'match': r'in labor force!!civilian!!unemployed$'
                                                     r'|(65 to 69 years|70 to 74 years|75 years and over)!!in labor force!!unemployed$'}],
    'edu_total':       [{'codes': {'B15003_001': ['total']}},
                        {'codes': {'B15002_001': ['total']}}],
    'edu_bach_plus':   [{'codes': {'B15003_022': ["bachelor's degree"], 'B15003_023': ["master's degree"],
                                   'B15003_024': ['professional school degree'], 'B15003_025': ['doctorate degree']}},
                        {'group': 'B15002', 'match': r"(bachelor's|master's|professional school|doctorate) degree$"}],
    'median_age':      [{'codes': {'B01002_001': ['median age']}}],
    'mob_total':       [{'codes': {'B07003_001': ['total']}}],
    'mob_same_house':  [{'codes': {'B07003_004': ['same house 1 year ago']}}],
    'commute_total':   [{'codes': {'B08301_001': ['total']}}],
    'commute_car':     [{'codes': {'B08301_002': ['car, truck, or van']}}],
}
MEDIANS = {'med_hh_income', 'median_age'}      # cannot be summed; single code only


class MissingApiKey(RuntimeError):
    pass


def get(url, params, tries=4):
    safe = {k: ('***' if k == 'key' else v) for k, v in params.items()}   # never print the key
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            # The API answers a missing/invalid key with a redirect to an HTML page and HTTP 200.
            if 'missing_key' in r.url:
                raise MissingApiKey(
                    'Census API: no key was sent. Data requests need a key: get one at '
                    'https://api.census.gov/data/key_signup.html, then set CENSUS_API_KEY.')
            if 'invalid_key' in r.url:
                raise MissingApiKey(
                    'Census API: the key was rejected as invalid. New keys only work after you click the '
                    'activation link in the confirmation email (allow a few minutes after clicking). '
                    'Also check CENSUS_API_KEY has no quotes or spaces.')
            if r.status_code == 200 and 'html' not in r.headers.get('Content-Type', ''):
                return r
            err = f"HTTP {r.status_code}, {r.headers.get('Content-Type')}: {r.text[:200]}"
        except requests.RequestException as e:
            err = str(e).replace(params.get('key', '\0'), '***')
        time.sleep(2 ** i)
    raise RuntimeError(f'failed {url} {safe}: {err}')


def load_variables(v):
    path = OUT / f'variables_{v}.json'
    if not path.exists():
        r = get(API.format(v=v) + '/variables.json', {})
        path.write_text(r.text, encoding='utf8')
    raw = json.loads(path.read_text(encoding='utf8'))['variables']
    return {k[:-1]: norm(d.get('label', '')) for k, d in raw.items() if k.endswith('E')}


def resolve(v, labels):
    """measure -> (codes, alternative index). Raises if no alternative resolves."""
    out, problems = {}, []
    for m, alts in MEASURES.items():
        for i, alt in enumerate(alts):
            if 'codes' in alt:
                ok = all(c in labels and all(k in labels[c] for k in kws)
                         for c, kws in alt['codes'].items())
                if ok:
                    out[m] = (list(alt['codes']), i)
                    break
            else:
                hits = sorted(c for c, lab in labels.items()
                              if c.startswith(alt['group'] + '_') and re.search(alt['match'], lab))
                if hits:
                    out[m] = (hits, i)
                    break
        else:
            problems.append(m)
        if m in MEDIANS and m in out and len(out[m][0]) != 1:
            problems.append(f'{m} (median resolved to {len(out[m][0])} codes)')
    if problems:
        raise RuntimeError(f'vintage {v}: could not resolve {problems}')
    return out


def fetch_vintage(v, res, key):
    codes = sorted({c for cs, _ in res.values() for c in cs})
    fields = [c + 'E' for c in codes] + [c + 'M' for c in codes]
    frames = []
    for cty in COUNTIES:
        parts = []
        for i in range(0, len(fields), MAX_VARS):
            params = {'get': ','.join(fields[i:i + MAX_VARS]), 'for': 'tract:*',
                      'in': f'state:{STATE} county:{cty}'}
            if key:
                params['key'] = key
            rows = get(API.format(v=v), params).json()
            d = pd.DataFrame(rows[1:], columns=rows[0])
            parts.append(d.set_index(['state', 'county', 'tract']))
            time.sleep(0.2)
        frames.append(pd.concat(parts, axis=1))
    raw = pd.concat(frames).reset_index()
    raw['GEOID'] = raw['state'] + raw['county'] + raw['tract']
    vals = raw.drop(columns=['state', 'county', 'tract']).set_index('GEOID').apply(pd.to_numeric, errors='coerce')
    vals = vals.mask(vals.isin(SENTINELS))

    out = pd.DataFrame(index=vals.index)
    for m, (cs, _) in res.items():
        est = vals[[c + 'E' for c in cs]]
        moe = vals[[c + 'M' for c in cs]]
        # a sum is missing if any component is missing - no silent partial sums
        out[m] = est.sum(axis=1, min_count=len(cs)).where(est.notna().all(axis=1))
        # Census approximation for the MOE of a sum: root of the sum of squares
        out[m + '_moe'] = np.sqrt((moe ** 2).sum(axis=1, min_count=len(cs)))
    out.insert(0, 'county', out.index.str[2:5])
    out.insert(0, 'vintage', v)
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='resolve codes only, fetch no data')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    key = os.environ.get('CENSUS_API_KEY')
    if not args.check and not key:
        sys.exit('CENSUS_API_KEY is not set. Data requests need a key: '
                 'https://api.census.gov/data/key_signup.html')

    manifest = {'rule': 'B: stop year Y uses the window ending Y-1; 2010 stops use 2006-2010',
                'vintages': VINTAGES, 'counties': COUNTIES,
                'started_utc': datetime.now(timezone.utc).isoformat(), 'per_vintage': {}}
    resolved = {}
    for v in VINTAGES:
        labels = load_variables(v)
        res = resolve(v, labels)
        resolved[v] = res
        manifest['per_vintage'][v] = {
            'window': f'{v - 4}-{v}',
            'measures': {m: {'codes': cs, 'alternative': i} for m, (cs, i) in res.items()}}
        fallbacks = {m: len(cs) for m, (cs, i) in res.items() if i > 0}
        print(f'{v - 4}-{v}: all {len(res)} measures resolved'
              + (f'; fallback tables used for {fallbacks}' if fallbacks else ''), flush=True)

    if args.check:
        (OUT / 'fetch_manifest_check.json').write_text(json.dumps(manifest, indent=1))
        print('check only - no data fetched')
        return

    frames = []
    for v in VINTAGES:
        t = fetch_vintage(v, resolved[v], key)
        dav = t[t['county'] == '037']
        checks = {
            'n_tracts_total': int(len(t)),
            'n_tracts_davidson': int(len(dav)),
            'davidson_population': int(dav['pop_total'].sum()),
            'race_parts_exceed_total': int(((t[['nh_white', 'nh_black', 'nh_asian', 'hispanic']].sum(axis=1))
                                            > t['race_total'] + 0.5).sum()),
            'tracts_zero_population': int((t['pop_total'] == 0).sum()),
            'missing_values_per_measure': {m: int(t[m].isna().sum()) for m in MEASURES if t[m].isna().any()},
        }
        manifest['per_vintage'][v]['checks'] = checks
        print(f"{v - 4}-{v}: {checks['n_tracts_total']} tracts ({checks['n_tracts_davidson']} Davidson), "
              f"Davidson pop {checks['davidson_population']:,}", flush=True)
        frames.append(t)

    allv = pd.concat(frames, ignore_index=True)
    allv.to_parquet(OUT / 'acs5_tract_measures.parquet', index=False)
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['rows'] = int(len(allv))
    (OUT / 'fetch_manifest.json').write_text(json.dumps(manifest, indent=1))
    print(f"wrote acs5_tract_measures.parquet ({len(allv):,} rows) and fetch_manifest.json")


if __name__ == '__main__':
    sys.exit(main())
