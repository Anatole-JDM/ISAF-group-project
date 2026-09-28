"""Step 6 - time features for every Nashville stop.

Features (one row per stop_id):
  hour, hour_missing              clock hour 0-23 (-1 when time is missing, with the flag)
  minute_of_day                   0-1439, <NA> when missing
  hour_sin, hour_cos              cyclical encoding of the time of day (for the logistic model)
  time_heaped                     minute is :00 or :30 - officers round some times (doubled counts)
  day_of_week, is_weekend         Monday = 0; Saturday/Sunday
  month, month_sin, month_cos     calendar month + cyclical encoding
  year                            kept for splits and stability analysis; NOT a model feature by default
  is_federal_holiday              US federal holiday (observed date)
  is_holiday_window               within one day of a federal holiday (holiday travel)
  is_dst                          daylight saving time in effect at the stop
  sun_elevation_deg               sun's height above the horizon at the stop time, Nashville
  light_period                    day (sun above -0.833 deg) / twilight (-6 to -0.833) / dark (below -6)
  is_dark                         light_period == 'dark' (after civil dusk, before civil dawn)
  minutes_after_civil_dusk        clock minutes relative to that day's civil dusk (negative = before)
  in_intertwilight                clock time between the year's earliest and latest civil dusk -
                                  the window used by the veil-of-darkness test
  dst_ambiguous, dst_nonexistent  local time fell in the repeated / skipped hour of a DST switch

How darkness is computed
  Recorded times are local Nashville clock times. Each is converted to UTC with the
  America/Chicago rules (DST included). The sun's elevation then follows the NOAA solar-position
  equations, at a single point (downtown Nashville): across Davidson County the sun's position
  differs by about 2 minutes of time, negligible next to the 30-minute rounding of some records.
  Repeated autumn hour (01:00-01:59): read as standard time - it is dark either way.
  Skipped spring hour (02:00-02:59): shifted forward - also dark either way.

Checks (the build stops if they fail)
  Known astronomical answers: the sun's maximum height at Nashville's latitude is
  90 - (lat - 23.44) = 77.28 deg on the June solstice, 90 - (lat + 23.44) = 30.40 deg on the
  December solstice, 90 - lat = 53.84 deg at the March equinox.
  No stop between 11:00 and 14:00 is 'dark'; every stop between 01:00 and 04:00 is.

Inputs : opp_data/processed/nashville_clean.parquet
Outputs: opp_data/features/time_features.parquet, opp_data/features/time_manifest.json
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parents[1]
STOPS = ROOT / 'opp_data' / 'processed' / 'nashville_clean.parquet'
OUT = ROOT / 'opp_data' / 'features'
TZ = 'America/Chicago'
LAT, LON = 36.1627, -86.7816          # downtown Nashville
SUNSET_DEG, CIVIL_DEG = -0.833, -6.0  # standard sunset (refraction + solar disc), civil twilight


# ---------------------------------------------------------------- solar position (NOAA)
def solar_terms(unix_seconds):
    """Declination (deg) and equation of time (min) for UTC instants."""
    jd = np.asarray(unix_seconds, dtype='float64') / 86400.0 + 2440587.5
    T = (jd - 2451545.0) / 36525.0
    L0 = np.mod(280.46646 + T * (36000.76983 + T * 0.0003032), 360.0)
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    Mr = np.radians(M)
    C = (np.sin(Mr) * (1.914602 - T * (0.004817 + 0.000014 * T))
         + np.sin(2 * Mr) * (0.019993 - 0.000101 * T) + np.sin(3 * Mr) * 0.000289)
    omega = np.radians(125.04 - 1934.136 * T)
    lam = np.radians(L0 + C - 0.00569 - 0.00478 * np.sin(omega))
    eps0 = 23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60
    eps = np.radians(eps0 + 0.00256 * np.cos(omega))
    decl = np.degrees(np.arcsin(np.sin(eps) * np.sin(lam)))
    y = np.tan(eps / 2) ** 2
    L0r = np.radians(L0)
    eot = 4 * np.degrees(y * np.sin(2 * L0r) - 2 * e * np.sin(Mr) + 4 * e * y * np.sin(Mr) * np.cos(2 * L0r)
                         - 0.5 * y * y * np.sin(4 * L0r) - 1.25 * e * e * np.sin(2 * Mr))
    return decl, eot


def sun_elevation(unix_seconds, lat=LAT, lon=LON):
    decl, eot = solar_terms(unix_seconds)
    utc_min = np.mod(np.asarray(unix_seconds, dtype='float64') / 60.0, 1440.0)
    tst = np.mod(utc_min + eot + 4 * lon, 1440.0)           # true solar time, minutes
    ha = np.radians(tst / 4.0 - 180.0)                      # hour angle
    la, de = np.radians(lat), np.radians(decl)
    cz = np.clip(np.sin(la) * np.sin(de) + np.cos(la) * np.cos(de) * np.cos(ha), -1, 1)
    return 90.0 - np.degrees(np.arccos(cz))


def event_utc_minutes(day_start_unix, elev_deg, lat=LAT, lon=LON):
    """UTC minutes after 00:00 UTC of each date when the sun sets through elev_deg."""
    decl, eot = solar_terms(day_start_unix + 18 * 3600)     # terms near Nashville's solar noon
    noon = 720 - 4 * lon - eot
    la, de = np.radians(lat), np.radians(decl)
    cos_h = (np.sin(np.radians(elev_deg)) - np.sin(la) * np.sin(de)) / (np.cos(la) * np.cos(de))
    return noon + 4 * np.degrees(np.arccos(np.clip(cos_h, -1, 1)))


def to_unix(ts_utc):
    return (ts_utc - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().to_numpy()


def known_answer_tests():
    out = {}
    for day, expected in [('2015-06-21', 90 - (LAT - 23.44)), ('2015-12-22', 90 - (LAT + 23.44)),
                          ('2015-03-20', 90 - LAT)]:
        grid = pd.date_range(f'{day} 00:00', periods=1440, freq='min', tz=TZ).tz_convert('UTC')
        got = float(sun_elevation(to_unix(pd.Series(grid))).max())
        out[day] = {'expected_max_elevation': round(expected, 2), 'computed': round(got, 2)}
        assert abs(got - expected) < 0.6, f'solar check failed on {day}: {got:.2f} vs {expected:.2f}'
    return out


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tests = known_answer_tests()
    s = pd.read_parquet(STOPS, columns=['stop_id', 'date', 'time'])
    n = len(s)

    hh = pd.to_numeric(s['time'].str.slice(0, 2), errors='coerce')
    mm = pd.to_numeric(s['time'].str.slice(3, 5), errors='coerce')
    mod = hh * 60 + mm
    has_time = mod.notna()

    f = pd.DataFrame({'stop_id': s['stop_id']})
    f['hour_missing'] = ~has_time
    f['hour'] = hh.fillna(-1).astype('int8')
    f['minute_of_day'] = mod.astype('Int16')
    ang = 2 * np.pi * mod / 1440.0
    f['hour_sin'] = np.sin(ang).astype('float32')
    f['hour_cos'] = np.cos(ang).astype('float32')
    f['time_heaped'] = (mm.isin([0, 30]) & has_time)
    f['day_of_week'] = s['date'].dt.dayofweek.astype('int8')
    f['is_weekend'] = f['day_of_week'] >= 5
    f['month'] = s['date'].dt.month.astype('int8')
    f['month_sin'] = np.sin(2 * np.pi * (f['month'] - 1) / 12).astype('float32')
    f['month_cos'] = np.cos(2 * np.pi * (f['month'] - 1) / 12).astype('float32')
    f['year'] = s['date'].dt.year.astype('int16')

    hol = USFederalHolidayCalendar().holidays(s['date'].min() - pd.Timedelta(days=2),
                                              s['date'].max() + pd.Timedelta(days=2))
    day = s['date'].dt.normalize()
    f['is_federal_holiday'] = day.isin(hol)
    near = set(hol) | set(hol + pd.Timedelta(days=1)) | set(hol - pd.Timedelta(days=1))
    f['is_holiday_window'] = day.isin(near)

    # ---- local clock time -> UTC, with DST
    local = s['date'] + pd.to_timedelta(mod, unit='min')
    amb = local.dt.tz_localize(TZ, ambiguous='NaT', nonexistent='shift_forward').isna() & has_time
    nonex = local.dt.tz_localize(TZ, ambiguous=np.zeros(n, dtype=bool), nonexistent='NaT').isna() & has_time
    loc = local.dt.tz_localize(TZ, ambiguous=np.zeros(n, dtype=bool), nonexistent='shift_forward')
    utc = loc.dt.tz_convert('UTC')
    f['dst_ambiguous'], f['dst_nonexistent'] = amb, nonex
    offset_h = (loc.dt.tz_localize(None) - utc.dt.tz_localize(None)).dt.total_seconds() / 3600
    f['is_dst'] = pd.array(np.where(has_time, offset_h == -5, pd.NA), dtype='boolean')

    # ---- darkness
    elev = np.where(has_time, sun_elevation(np.nan_to_num(to_unix(utc))), np.nan)
    f['sun_elevation_deg'] = elev.astype('float32')
    lp = np.select([elev >= SUNSET_DEG, elev >= CIVIL_DEG, elev < CIVIL_DEG], ['day', 'twilight', 'dark'], default='')
    f['light_period'] = pd.Series(np.where(has_time, lp, None), dtype='string').astype('category')
    f['is_dark'] = pd.array(np.where(has_time, elev < CIVIL_DEG, pd.NA), dtype='boolean')

    # ---- civil dusk per date, in local clock minutes -> veil-of-darkness helpers
    dates = pd.Series(np.sort(day.unique()))
    d_utc0 = dates.dt.tz_localize('UTC')
    dusk_utc = d_utc0 + pd.to_timedelta(event_utc_minutes(to_unix(d_utc0), CIVIL_DEG), unit='min')
    dusk_local = dusk_utc.dt.tz_convert(TZ)
    dusk_min = pd.Series((dusk_local.dt.hour * 60 + dusk_local.dt.minute).to_numpy(), index=dates.to_numpy())
    sunset_utc = d_utc0 + pd.to_timedelta(event_utc_minutes(to_unix(d_utc0), SUNSET_DEG), unit='min')
    sunset_local = sunset_utc.dt.tz_convert(TZ)
    f['minutes_after_civil_dusk'] = (mod - day.map(dusk_min)).astype('Int16')
    lo, hi = int(dusk_min.min()), int(dusk_min.max())
    f['in_intertwilight'] = mod.between(lo, hi) & has_time

    assert len(f) == n and f['stop_id'].is_unique
    f.to_parquet(OUT / 'time_features.parquet', index=False)

    # ---- sanity checks + report
    dark_by_hour = f[has_time].groupby('hour')['is_dark'].mean().round(3)
    assert dark_by_hour.loc[11:13].max() == 0, 'midday stops classified dark'
    assert dark_by_hour.loc[1:3].min() == 1, 'night stops not classified dark'

    def hm(x):
        return f'{int(x) // 60:02d}:{int(x) % 60:02d}'

    sol = {}
    for dstr in ['2015-06-21', '2015-12-21']:
        i = dates[dates == pd.Timestamp(dstr)].index[0]
        sol[dstr] = {'sunset_local': str(sunset_local.iloc[i].strftime('%H:%M %Z')),
                     'civil_dusk_local': str(dusk_local.iloc[i].strftime('%H:%M %Z'))}
    man = {
        'built_utc': datetime.now(timezone.utc).isoformat(), 'rows': int(n),
        'solar_known_answer_tests': tests, 'solstice_sunset_and_dusk': sol,
        'time_missing': int((~has_time).sum()),
        'time_heaped_pct': round(100 * float(f['time_heaped'][has_time].mean()), 2),
        'time_heaped_expected_if_uniform_pct': round(100 * 2 / 60, 2),
        'dst_ambiguous': int(amb.sum()), 'dst_nonexistent': int(nonex.sum()),
        'is_dst_pct': round(100 * float(f['is_dst'].astype('float').mean()), 1),
        'light_period_counts': {('missing_time' if pd.isna(k) else str(k)): int(v)
                                for k, v in f['light_period'].value_counts(dropna=False).items()},
        'dark_share_by_hour': {int(k): float(v) for k, v in dark_by_hour.items()},
        'intertwilight_window_local': [hm(lo), hm(hi)],
        'in_intertwilight': int(f['in_intertwilight'].sum()),
        'federal_holiday_stops': int(f['is_federal_holiday'].sum()),
        'holiday_window_stops': int(f['is_holiday_window'].sum()),
        'not_model_features_by_default': ['year', 'minutes_after_civil_dusk', 'in_intertwilight',
                                          'dst_ambiguous', 'dst_nonexistent', 'time_heaped'],
    }
    (OUT / 'time_manifest.json').write_text(json.dumps(man, indent=1, default=str))
    print(f'wrote time_features.parquet ({n:,} rows, {f.shape[1] - 1} columns)')
    print(json.dumps({k: man[k] for k in ['solar_known_answer_tests', 'solstice_sunset_and_dusk', 'time_missing',
                                           'time_heaped_pct', 'dst_ambiguous', 'dst_nonexistent',
                                           'light_period_counts', 'intertwilight_window_local',
                                           'in_intertwilight', 'federal_holiday_stops']}, indent=1, default=str))
    print('dark share by hour:', man['dark_share_by_hour'])


if __name__ == '__main__':
    main()
