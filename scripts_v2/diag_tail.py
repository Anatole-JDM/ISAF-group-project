"""Check whether the end of the Nashville series is incomplete or a genuine decline.

Read-only: loads `date` (+ a few columns) and reports monthly stop counts, daily
counts for the final months, and how many distinct days each month contains.
"""
import zipfile, io
import pandas as pd
import pyarrow as pa
from pyarrow import csv as pacsv

P = ('C:/Users/smaye/OneDrive/Bureau/X-HEC/HEC_courses/algo_fairness/'
     'ISAF_gp/ISAF-group-project/opp_data/tn_nashville_2020_04_01.csv.zip')
use = ['date', 'search_conducted', 'subject_race', 'reason_for_stop']
z = zipfile.ZipFile(P)
with z.open(z.infolist()[0]) as f:
    df = pacsv.read_csv(
        f, read_options=pacsv.ReadOptions(block_size=1 << 26),
        convert_options=pacsv.ConvertOptions(
            include_columns=use, column_types={c: pa.string() for c in use})).to_pandas()

d = pd.to_datetime(df['date'], errors='coerce')
df['ym'] = d.dt.to_period('M').astype(str)
df['day'] = d.dt.date

m = df.groupby('ym').agg(stops=('date', 'size'), days_present=('day', 'nunique'))
m['days_in_month'] = [pd.Period(x).days_in_month for x in m.index]
m['stops_per_active_day'] = (m['stops'] / m['days_present']).round(0)

print('=== monthly counts, 2017-01 onward ===')
print(m.loc['2017-01':].to_string())

print('\n=== monthly stops per active day, full series (yearly mean) ===')
m['year'] = [x[:4] for x in m.index]
print(m.groupby('year')['stops_per_active_day'].mean().round(0).to_string())

print('\n=== daily counts, 2018-10 onward ===')
tail = df[df['ym'] >= '2018-10'].groupby('day').size()
print(tail.to_string())
