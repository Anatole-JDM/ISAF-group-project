"""Step 1 - neighbourhood (ACS) features for every Nashville stop.

Method (feature_engineering_plan.md, Step 1; decisions D1 = rule B, D2 = population-weighted circle)
  1. Each stop location gets a circle of radius r (800 m primary, 400 m robustness).
  2. The 2010 census blocks whose interior point falls in the circle say how many of each
     tract's residents live inside it:  w[loc, tract] = pop10(tract, inside circle) / pop10(tract).
  3. Each tract's ACS counts are allocated to the circle with those weights (dasymetric
     interpolation), and shares are computed as ratios of allocated counts - never as
     averages of tract percentages.
  4. The ACS release is matched to the stop date by rule B: stop year Y -> window ending Y-1
     (2010 stops -> 2006-2010, flagged).
  Medians (income, age) cannot be allocated; they are population-weighted means of tract
  medians and are marked `_approx`. The robust income feature is `income_pctile`: the tract's
  percentile rank among Davidson tracts within the same release, then weighted the same way.
  `unemployment_rel` = circle unemployment rate / Davidson rate in the same release. The raw
  rate moves with the business cycle (29.5% of its variance is change over time at a fixed
  location), so it would act as a date proxy; the relative version keeps only the local signal.
  Team decisions: D5 = no smoothing across releases; robustness radius 1200 m.

Assumptions (state them in the deck): within a tract, residents are spread like the 2010
block populations, and the tract's ACS composition is uniform across its blocks.

Stops without usable coordinates (missing, or outside Davidson's bounding box) get no
features and nbh_geo_source = 'none'. Stops located in a neighbouring county get the
features of the tract that contains them (nbh_geo_source = 'tract_pip').

Inputs  : opp_data/processed/nashville_clean.parquet, opp_data/external/acs/acs5_tract_measures.parquet,
          opp_data/external/{tl_2010_47037_tabblock10.zip, block_pop_2010_davidson.parquet, tl_2010_47_tract10.zip}
Outputs : opp_data/features/nbh_features.parquet            one row per stop_id
          opp_data/features/nbh_location_vintage.parquet    one row per (location, release) - for checks and maps
          opp_data/features/nbh_manifest.json               coverage, flags, validation, D5 noise diagnostic
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'opp_data' / 'external'
STOPS = ROOT / 'opp_data' / 'processed' / 'nashville_clean.parquet'
ACS = EXT / 'acs' / 'acs5_tract_measures.parquet'
BLOCKS = EXT / 'tl_2010_47037_tabblock10.zip'
BLOCK_POP = EXT / 'block_pop_2010_davidson.parquet'
TRACTS = EXT / 'tl_2010_47_tract10.zip'
OUT = ROOT / 'opp_data' / 'features'

METRIC = 'EPSG:32616'           # UTM 16N, metres
RADII = [800, 1200]              # primary first; 1200 m robustness (400 m left 18.7% of stops with <100 residents)
LOW_RESIDENTS = 100              # circles with fewer 2010 residents are flagged
BUCKET_MIN_STRINGS = 20          # a coordinate shared by >= 20 distinct location strings is a geocoder bucket
FIRST_VINTAGE, LAST_VINTAGE = 2010, 2018

# ratio features: name -> (numerator measure, denominator measure)
RATIOS = {
    'share_black': ('nh_black', 'race_total'),
    'share_white': ('nh_white', 'race_total'),
    'share_hispanic': ('hispanic', 'race_total'),
    'share_asian': ('nh_asian', 'race_total'),
    'poverty_rate': ('pov_below', 'pov_universe'),
    'share_no_vehicle': ('veh_none', 'veh_universe'),
    'renter_share': ('renter', 'tenure_total'),
    'unemployment_rate': ('unemployed', 'lf_civilian'),
    'share_bach_plus': ('edu_bach_plus', 'edu_total'),
    'residential_stability': ('mob_same_house', 'mob_total'),
    'share_drive_to_work': ('commute_car', 'commute_total'),
}
COUNT_MEASURES = sorted({m for pair in RATIOS.values() for m in pair} | {'pop_total'})
MEDIAN_MEASURES = {'median_income_approx': 'med_hh_income', 'median_age_approx': 'median_age',
                   'income_pctile': 'income_pctile'}


def log(msg):
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)


# ---------------------------------------------------------------- inputs
def load_stops():
    s = pd.read_parquet(STOPS, columns=['stop_id', 'date', 'lat', 'lng', 'geo_outside_davidson_box', 'location'])
    s['year'] = s['date'].dt.year
    s['acs_vintage'] = s['year'].sub(1).clip(FIRST_VINTAGE, LAST_VINTAGE).astype('int16')
    s['acs_window_overlaps_stop'] = s['year'].eq(2010)
    usable = s['lat'].notna() & s['lng'].notna() & ~s['geo_outside_davidson_box'].fillna(False)
    s['loc_key'] = pd.Series(pd.NA, index=s.index, dtype='string')
    s.loc[usable, 'loc_key'] = (s.loc[usable, 'lat'].round(6).astype(str) + ','
                                + s.loc[usable, 'lng'].round(6).astype(str))
    return s


def load_acs():
    a = pd.read_parquet(ACS)
    dav = a[a['county'] == '037']
    bad = dav[COUNT_MEASURES].isna().sum()
    if bad.any():   # counts must be complete for Davidson; only medians may be missing
        raise ValueError(f'missing count measures in Davidson tracts: {bad[bad > 0].to_dict()}')
    # income percentile rank among Davidson tracts, within each release
    a['income_pctile'] = np.nan
    a.loc[dav.index, 'income_pctile'] = dav.groupby('vintage')['med_hh_income'].rank(pct=True)
    return a


def load_blocks():
    b = gpd.read_file(f'zip://{BLOCKS}', ignore_geometry=True)[['GEOID10', 'INTPTLAT10', 'INTPTLON10']]
    b = b.merge(pd.read_parquet(BLOCK_POP), on='GEOID10', how='left', validate='1:1')
    assert b['pop10'].notna().all()
    b = b[b['pop10'] > 0].copy()                  # zero-population blocks carry no weight
    b['tract'] = b['GEOID10'].str[:11]
    pts = gpd.GeoSeries(gpd.points_from_xy(b['INTPTLON10'].astype(float), b['INTPTLAT10'].astype(float)),
                        crs='EPSG:4269').to_crs(METRIC)
    return b.reset_index(drop=True), np.asarray(pts.values)


# ---------------------------------------------------------------- core
def circle_weights(loc_pts, blk_pts, blocks, tract_index, radius):
    """Sparse matrices over (location, tract): residents inside the circle, and weight w."""
    tree = shapely.STRtree(blk_pts)
    li, bi = tree.query(loc_pts, predicate='dwithin', distance=radius)
    blk_pop = blocks['pop10'].to_numpy(dtype='float64')
    blk_ti = tract_index.loc[blocks['tract']].to_numpy()             # block -> tract column, computed once
    shape = (len(loc_pts), len(tract_index))
    P = sp.coo_matrix((blk_pop[bi], (li, blk_ti[bi])), shape=shape).tocsr()   # residents of tract t inside circle
    tract_pop = np.bincount(blk_ti, weights=blk_pop, minlength=shape[1])
    W = P.multiply(1.0 / tract_pop[np.newaxis, :]).tocsr()             # fraction of tract t inside circle
    return P, W, tract_pop


def davidson_unemployment(acs_v):
    dav = acs_v[acs_v['county'] == '037']
    return float(dav['unemployed'].sum() / dav['lf_civilian'].sum())


def features_for_radius(P, W, acs_v, tract_ids, radius):
    """All features for every location, for one ACS release."""
    M = acs_v.set_index('GEOID').reindex(tract_ids)
    alloc = pd.DataFrame(W @ M[COUNT_MEASURES].to_numpy(dtype='float64'), columns=COUNT_MEASURES)
    f = pd.DataFrame(index=alloc.index)
    area_km2 = np.pi * (radius / 1000) ** 2
    f['pop_density'] = alloc['pop_total'] / area_km2
    for name, (num, den) in RATIOS.items():
        d = alloc[den]
        f[name] = np.where(d > 0, alloc[num] / d.where(d > 0), np.nan)
    f['unemployment_rel'] = f['unemployment_rate'] / davidson_unemployment(acs_v)
    circle_res = np.asarray(P.sum(axis=1)).ravel()
    for name, col in MEDIAN_MEASURES.items():
        v = M[col].to_numpy(dtype='float64')
        has = ~np.isnan(v)
        num = P @ np.where(has, v, 0.0)
        den = P @ has.astype('float64')
        f[name] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        if name == 'median_income_approx':   # share of the circle's residents whose tract median is missing
            f['income_weight_missing'] = np.where(circle_res > 0, 1 - den / np.where(circle_res > 0, circle_res, 1), np.nan)
    f['residents_2010'] = circle_res
    f.loc[circle_res == 0, [c for c in f.columns if c != 'residents_2010']] = np.nan
    return f.astype('float32')


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stops = load_stops()
    acs = load_acs()
    blocks, blk_pts = load_blocks()
    tracts = gpd.read_file(f'zip://{TRACTS}')[['GEOID10', 'COUNTYFP10', 'geometry']].to_crs(METRIC)
    county = tracts.loc[tracts['COUNTYFP10'] == '037'].union_all()
    tract_ids = pd.Index(sorted(blocks['tract'].unique()), name='GEOID')
    tract_index = pd.Series(np.arange(len(tract_ids)), index=tract_ids)
    log(f'stops {len(stops):,}; with usable coordinates {stops["loc_key"].notna().sum():,}; '
        f'populated blocks {len(blocks):,} in {len(tract_ids)} tracts')

    # unique locations
    locs = (stops.dropna(subset=['loc_key'])
            .groupby('loc_key').agg(lat=('lat', 'first'), lng=('lng', 'first'),
                                    n_loc_strings=('location', 'nunique')).reset_index())
    g = gpd.GeoSeries(gpd.points_from_xy(locs['lng'], locs['lat']), crs='EPSG:4326').to_crs(METRIC)
    loc_pts = np.asarray(g.values)
    locs['in_davidson'] = shapely.contains_xy(county, *shapely.get_coordinates(loc_pts).T)
    locs['dist_county_edge_m'] = shapely.distance(loc_pts, county.boundary)
    locs['geo_bucket'] = locs['n_loc_strings'] >= BUCKET_MIN_STRINGS
    # locations outside Davidson: the tract that contains them
    out_idx = np.flatnonzero(~locs['in_davidson'].to_numpy())
    pip = gpd.sjoin(gpd.GeoDataFrame(geometry=g.iloc[out_idx].reset_index(drop=True), crs=METRIC),
                    tracts[['GEOID10', 'geometry']], how='left', predicate='within')
    pip = pip[~pip.index.duplicated()].sort_index()     # align with out_idx order
    locs['pip_tract'] = pd.Series(pd.NA, index=locs.index, dtype='string')
    locs.loc[out_idx, 'pip_tract'] = pip['GEOID10'].to_numpy()
    log(f'unique locations {len(locs):,} ({len(out_idx):,} outside Davidson)')

    stops['acs_vintage'] = stops['acs_vintage'].astype('int64')
    vintages = [int(x) for x in sorted(stops['acs_vintage'].unique())]
    frames, weights = [], {}
    for r in RADII:
        P, W, tract_pop = circle_weights(loc_pts, blk_pts, blocks, tract_index, r)
        weights[r] = (P, W)
        # identity check: sum over tracts of w * tract_pop = residents in circle
        assert np.allclose(np.asarray((W.multiply(tract_pop[np.newaxis, :])).sum(axis=1)).ravel(),
                           np.asarray(P.sum(axis=1)).ravel())
        log(f'r={r} m: circle weights built, {P.nnz:,} (location, tract) pairs')
        for v in vintages:
            f = features_for_radius(P, W, acs[acs['vintage'] == v], tract_ids, r)
            f = f.add_prefix(f'nbh{r}_')
            f['loc_key'] = locs['loc_key'].to_numpy()
            f['acs_vintage'] = v
            frames.append((r, v, f))
    # assemble (location, vintage) table with both radii side by side
    per_v = {}
    for r, v, f in frames:
        per_v.setdefault(v, []).append(f.set_index(['loc_key', 'acs_vintage']))
    lv = pd.concat([pd.concat(fs, axis=1) for fs in per_v.values()]).reset_index()

    # locations outside Davidson -> containing tract's own shares (no circle), both radii
    if len(out_idx):
        a = acs.set_index(['GEOID', 'vintage'])
        rows = []
        for i in out_idx:
            t = locs.at[i, 'pip_tract']
            for v in vintages:
                if pd.isna(t) or (t, v) not in a.index:
                    continue
                m = a.loc[(t, v)]
                rec = {'loc_key': locs.at[i, 'loc_key'], 'acs_vintage': v}
                for r in RADII:
                    rec[f'nbh{r}_pop_density'] = np.nan            # no circle -> no density
                    for name, (num, den) in RATIOS.items():
                        rec[f'nbh{r}_{name}'] = m[num] / m[den] if m[den] > 0 else np.nan
                    rec[f'nbh{r}_unemployment_rel'] = (rec[f'nbh{r}_unemployment_rate']
                                                       / davidson_unemployment(acs[acs['vintage'] == v]))
                    rec[f'nbh{r}_median_income_approx'] = m['med_hh_income']
                    rec[f'nbh{r}_median_age_approx'] = m['median_age']
                    rec[f'nbh{r}_income_pctile'] = np.nan            # percentile defined among Davidson tracts
                    rec[f'nbh{r}_residents_2010'] = np.nan
                rows.append(rec)
        if rows:
            pip_df = pd.DataFrame(rows)
            lv = lv.set_index(['loc_key', 'acs_vintage'])
            pip_df = pip_df.set_index(['loc_key', 'acs_vintage'])
            lv.loc[pip_df.index, pip_df.columns] = pip_df.astype('float32')
            lv = lv.reset_index()

    # location-level flags
    flags = locs[['loc_key', 'in_davidson', 'dist_county_edge_m', 'geo_bucket', 'n_loc_strings']].copy()
    for r in RADII:
        flags[f'nbh{r}_circle_truncated'] = flags['in_davidson'] & (flags['dist_county_edge_m'] < r)
    lv = lv.merge(flags, on='loc_key', how='left')
    for r in RADII:
        lv[f'nbh{r}_low_residents'] = lv[f'nbh{r}_residents_2010'] < LOW_RESIDENTS
    lv.to_parquet(OUT / 'nbh_location_vintage.parquet', index=False)
    log(f'location x release table: {len(lv):,} rows')

    # ---------------------------------------------------------------- per stop
    keep = [c for c in lv.columns if c.startswith('nbh')] + ['geo_bucket']
    out = stops[['stop_id', 'loc_key', 'acs_vintage', 'acs_window_overlaps_stop']].merge(
        lv[['loc_key', 'acs_vintage', 'in_davidson'] + keep], on=['loc_key', 'acs_vintage'], how='left')
    out['nbh_geo_source'] = np.select(
        [out['loc_key'].isna(), out['in_davidson'].eq(True), out['in_davidson'].eq(False)],
        ['none', 'circle', 'tract_pip'], default='none')
    out = out.drop(columns=['in_davidson', 'loc_key'])
    assert len(out) == len(stops) and out['stop_id'].is_unique
    out.to_parquet(OUT / 'nbh_features.parquet', index=False)
    log(f'per-stop table: {len(out):,} rows, {out.shape[1]} columns')

    # ---------------------------------------------------------------- validation + D5 diagnostic
    man = {'built_utc': datetime.now(timezone.utc).isoformat(), 'radii_m': RADII,
           'rule': 'B: window ending Y-1; 2010 stops use 2006-2010',
           'rows': int(len(out)), 'unique_locations': int(len(locs))}
    man['geo_source_counts'] = out['nbh_geo_source'].value_counts().to_dict()
    man['geo_source_share_by_year'] = (pd.crosstab(stops['year'], out['nbh_geo_source'], normalize='index')
                                       .round(4).to_dict(orient='index'))
    man['flags'] = {
        'acs_window_overlaps_stop': int(out['acs_window_overlaps_stop'].sum()),
        'geo_bucket': int(out['geo_bucket'].fillna(False).sum()),
        **{f'nbh{r}_circle_truncated': int(out[f'nbh{r}_circle_truncated'].fillna(False).sum()) for r in RADII},
        **{f'nbh{r}_low_residents': int(out[f'nbh{r}_low_residents'].fillna(False).sum()) for r in RADII},
        **{f'nbh{r}_all_features_missing_with_coords': int((out['nbh_geo_source'].eq('circle')
                                                            & out[f'nbh{r}_share_black'].isna()).sum()) for r in RADII},
    }
    feat_cols = [c for c in out.columns if c.startswith('nbh800_') and out[c].dtype.kind == 'f']
    man['nbh800_ranges'] = {c: [round(float(out[c].min()), 4), round(float(out[c].median()), 4),
                                round(float(out[c].max()), 4)] for c in feat_cols}
    shares = [f'nbh{r}_{n}' for r in RADII for n in RATIOS]
    man['shares_outside_0_1'] = int(((out[shares] < -1e-6) | (out[shares] > 1 + 1e-6)).sum().sum())

    # sanity: a circle lying entirely inside one tract must reproduce that tract's shares
    P800, _ = weights[800]
    single = np.flatnonzero((np.diff(P800.indptr) == 1) & locs['in_davidson'].to_numpy())
    if len(single):
        i = single[0]
        t = tract_ids[P800.indices[P800.indptr[i]]]
        v = vintages[-1]
        row = acs[(acs['GEOID'] == t) & (acs['vintage'] == v)].iloc[0]
        got = lv[(lv['loc_key'] == locs.at[i, 'loc_key']) & (lv['acs_vintage'] == v)]['nbh800_share_black'].iloc[0]
        man['single_tract_check'] = {'n_single_tract_circles': int(len(single)), 'tract': t,
                                     'expected': round(float(row['nh_black'] / row['race_total']), 6),
                                     'got': round(float(got), 6)}

    # D5: release-to-release jumps at a fixed location, circle vs tract
    z = lv[lv['in_davidson']].pivot(index='loc_key', columns='acs_vintage', values='nbh800_share_black')
    jumps = z.diff(axis=1).abs().max(axis=1).dropna()     # circles with no residents have no share
    w = stops.dropna(subset=['loc_key']).groupby('loc_key').size().reindex(jumps.index).fillna(0)
    def wq(x, wt, q):
        o = np.argsort(x.to_numpy()); cx = np.cumsum(wt.to_numpy()[o]); return float(x.to_numpy()[o][np.searchsorted(cx, q * cx[-1])])
    man['D5_noise_nbh800_share_black_max_release_jump'] = {
        'median_over_locations': round(float(jumps.median()), 4),
        'p95_over_locations': round(float(jumps.quantile(0.95)), 4),
        'median_stop_weighted': round(wq(jumps, w, 0.5), 4),
        'p95_stop_weighted': round(wq(jumps, w, 0.95), 4),
        'reference_tract_level_pop_ge_1000': {'median': 0.053, 'p95': 0.132},
    }
    # how much of each feature's variation is change over time at a fixed location (stop-weighted)
    wloc = stops.dropna(subset=['loc_key']).groupby('loc_key').size()
    dec = {}
    for f in ['share_black', 'poverty_rate', 'income_pctile', 'residential_stability',
              'unemployment_rate', 'unemployment_rel']:
        c = f'nbh800_{f}'
        d = lv.loc[lv['in_davidson'], ['loc_key', c]].dropna()
        wt = d['loc_key'].map(wloc).fillna(0).to_numpy()
        x = d[c].to_numpy(dtype='float64')
        mu = d.groupby('loc_key')[c].transform('mean').to_numpy()
        tot = np.average((x - np.average(x, weights=wt)) ** 2, weights=wt)
        dec[f] = round(float(np.average((x - mu) ** 2, weights=wt) / tot), 4)
    man['nbh800_share_of_variance_over_time'] = dec
    (OUT / 'nbh_manifest.json').write_text(json.dumps(man, indent=1, default=str))
    log('manifest written')
    print(json.dumps({k: man[k] for k in ['geo_source_counts', 'flags', 'shares_outside_0_1',
                                           'single_tract_check', 'D5_noise_nbh800_share_black_max_release_jump',
                                           'nbh800_share_of_variance_over_time']
                      if k in man}, indent=1, default=str))


if __name__ == '__main__':
    main()
