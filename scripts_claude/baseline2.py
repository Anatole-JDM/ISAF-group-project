import zipfile, io, numpy as np, pandas as pd
import pyarrow as pa, pyarrow.compute as pc
from pyarrow import csv as pacsv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
BASE='C:/Users/smaye/OneDrive/Bureau/X-HEC/HEC_courses/algo_fairness/ISAF_gp/ISAF-group-project/opp_data/'
def T(c): return pc.fill_null(pc.equal(pc.utf8_lower(c),'true'),False)
def load(path, cols, mode, frac=1.0, seed=0):
    z=zipfile.ZipFile(path); info=z.infolist()[0]
    with z.open(info) as f: have=next(io.TextIOWrapper(f,'utf8',errors='replace')).strip().split(',')
    use=[c for c in cols if c in have]; parts=[]; rng=np.random.default_rng(seed)
    with z.open(info) as f:
        rdr=pacsv.open_csv(f, read_options=pacsv.ReadOptions(block_size=1<<26),
            convert_options=pacsv.ConvertOptions(include_columns=use, column_types={c:pa.string() for c in use}))
        for b in rdr:
            t=b
            if mode=='consent':
                t=t.filter(T(t.column('search_conducted')))
                if t.num_rows: t=t.filter(pc.fill_null(pc.equal(t.column('search_basis'),'consent'),False))
            if t.num_rows==0: continue
            df=t.to_pandas()
            if frac<1.0: df=df.sample(frac=frac, random_state=rng.integers(1e9))
            parts.append(df)
    return pd.concat(parts, ignore_index=True)
def prep(df, statecode=None):
    df=df.copy()
    d=pd.to_datetime(df['date'], errors='coerce')
    df['year']=d.dt.year; df['month']=d.dt.month; df['dow']=d.dt.dayofweek
    if 'time' in df: df['hour']=pd.to_numeric(df['time'].str.slice(0,2), errors='coerce')
    if 'subject_age' in df: df['age']=pd.to_numeric(df['subject_age'], errors='coerce')
    for c in ['lat','lng']:
        if c in df: df[c]=pd.to_numeric(df[c], errors='coerce')
    if 'vehicle_registration_state' in df and statecode:
        df['out_of_state']=(df['vehicle_registration_state']!=statecode).astype(int)
    return df.dropna(subset=['year'])
def fit(tag, df, cat, num, split_year, ycol='y'):
    cat=[c for c in cat if c in df.columns]; num=[c for c in num if c in df.columns]
    tr=df[df.year<split_year]; te=df[df.year>=split_year]
    if len(tr)<1000 or len(te)<500: print(f"{tag}: too small"); return
    X=cat+num
    pre=ColumnTransformer([('c',Pipeline([('i',SimpleImputer(strategy='constant',fill_value='M')),('o',OneHotEncoder(handle_unknown='ignore',min_frequency=30))]),cat),
                           ('n',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),num)])
    lr=Pipeline([('p',pre),('m',LogisticRegression(max_iter=1000))]).fit(tr[X],tr[ycol])
    a_lr=roc_auc_score(te[ycol], lr.predict_proba(te[X])[:,1])
    Xb=tr[X].copy(); Xt=te[X].copy()
    for c in cat:
        Xb[c]=Xb[c].astype('category'); Xt[c]=pd.Categorical(Xt[c],categories=Xb[c].cat.categories)
    gb=XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, enable_categorical=True,
                     tree_method='hist', eval_metric='auc', n_jobs=8).fit(Xb,tr[ycol])
    p=gb.predict_proba(Xt)[:,1]; a_gb=roc_auc_score(te[ycol],p)
    print(f"  {tag:46} n_tr={len(tr):>9,} base={tr[ycol].mean():5.1%}  LR={a_lr:.3f}  XGB={a_gb:.3f}")
    return te,p

NASH=['date','time','lat','lng','precinct','zone','subject_age','subject_race','subject_sex','reason_for_stop',
      'vehicle_registration_state','search_basis','contraband_found','search_conducted','outcome','arrest_made']
NCC=['date','time','county_name','department_name','subject_age','subject_race','subject_sex','reason_for_stop',
     'search_basis','contraband_found','search_conducted','outcome','arrest_made']

print("=== TARGET 1: contraband | consent search (NC) — ablation ===", flush=True)
nc=prep(load(BASE+'nc_statewide_2020_04_01.csv.zip', NCC,'consent')); nc=nc[nc.year>=2002]
nc['y']=nc.contraband_found.str.lower().eq('true').astype(int)
fit('A full (incl. county+department)', nc, ['subject_sex','reason_for_stop','county_name','department_name'], ['age','hour','dow','month'], 2013)
fit('B no agency/geo (driver+stop only)', nc, ['subject_sex','reason_for_stop'], ['age','hour','dow','month'], 2013)
fit('C agency/geo only', nc, ['county_name','department_name'], ['month'], 2013)

print("\n=== TARGET 2: search_conducted | all stops (audit model) ===", flush=True)
ncs=prep(load(BASE+'nc_statewide_2020_04_01.csv.zip', NCC,'all', frac=0.15, seed=1)); ncs=ncs[ncs.year>=2002]
ncs['y']=ncs.search_conducted.str.lower().eq('true').astype(int)
fit('NC: P(search) [race excluded from X]', ncs, ['subject_sex','reason_for_stop','county_name','department_name'], ['age','hour','dow','month'], 2013)

print("\n=== TARGET 3: Nashville alternatives ===", flush=True)
na=prep(load(BASE+'tn_nashville_2020_04_01.csv.zip', NASH,'all', frac=0.35, seed=2), statecode='TN')
na['y']=na.search_conducted.str.lower().eq('true').astype(int)
fit('TN: P(search) [race excluded]', na, ['subject_sex','reason_for_stop','precinct','zone'], ['age','hour','dow','month','lat','lng','out_of_state'], 2017)
na2=na[na.outcome.isin(['citation','warning'])].copy(); na2['y']=(na2.outcome=='citation').astype(int)
fit('TN: P(citation vs warning)', na2, ['subject_sex','reason_for_stop','precinct','zone'], ['age','hour','dow','month','lat','lng','out_of_state'], 2017)
na3=na.copy(); na3['y']=na3.arrest_made.str.lower().eq('true').astype(int)
fit('TN: P(arrest)', na3, ['subject_sex','reason_for_stop','precinct','zone'], ['age','hour','dow','month','lat','lng','out_of_state'], 2017)
print('ALL DONE', flush=True)
